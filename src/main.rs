mod audio;
mod bus;
mod log;
mod post;
mod ppi;
mod rtc;
mod scc;
mod slot;
mod vdp;

use audio::Audio;
use bus::Bus;
use post::{Post, ShaderMode};
use std::num::Wrapping;
use std::sync::Arc;
use vdp::Vdp;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

// Native `std::time::Instant` panics on wasm32-unknown-unknown because the
// browser sandbox has no monotonic clock. `web-time` is a drop-in replacement
// that uses `performance.now()` under the hood.
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
use z80emu::{host::TsCounter, Cpu, Z80NMOS};

/// T-states per emulated frame at NTSC (3.579545 MHz / 60 Hz). One call to
/// `execute_with_limit` consumes this many cycles before we raise the next
/// VBLANK and let the game's interrupt handler run.
const FRAME_TSTATES: i32 = 59_659;

/// Visible scan-lines per frame, sized for V9938's 212-line mode (R9
/// bit 7 = LN). MSX1 software (and MSX2 in 192-line mode) only uses the
/// top 192 of these — but most MSX2 cartridges with a status bar (KV2,
/// Vampire Killer, Metal Gear, etc.) enable 212 lines so the score area
/// fits below the playfield. Snapshotting / line-IRQ checks loop over
/// all 212 so the per-line shader state is in place when 212-mode kicks
/// in; 192-mode software just has the extra 20 entries idle.
const VISIBLE_LINES: u32 = 212;

/// T-states budget per scanline. Real NTSC V9938 uses 228 T-states per line
/// (3.579545 MHz / 59.94 Hz / 262 lines = 228.02). Hardcoding 228 — instead
/// of `FRAME_TSTATES / 262` which integer-truncates to 227 — keeps the
/// scanline boundaries aligned with where real hardware fires line IRQs,
/// which matters for beam-racing software (Quarth's split-screen scroll,
/// per-line palette tricks, etc.).
const SCANLINE_TSTATES: i32 = 228;

/// MSX target frame rate. We pace CPU emulation against wall-clock time so
/// the emulator runs at the right speed regardless of host display refresh
/// (30 Hz, 60 Hz, 144 Hz, whatever).
const MSX_HZ: f64 = 60.0;

/// Maximum wall-clock seconds we'll try to catch up after a stall (e.g. the
/// window was dragged or backgrounded). Anything longer is discarded — better
/// to skip than to freeze the UI grinding through hundreds of MSX frames.
const MAX_CATCHUP_SECS: f64 = 0.1;

/// Load a cartridge ROM from the command line argument (native) or the
/// `?rom=<path>` URL parameter (web). Returns `None` when nothing is
/// specified — the emulator then boots straight into BASIC.
#[cfg(not(target_arch = "wasm32"))]
fn load_cartridge_rom() -> Option<Vec<u8>> {
    // Walk argv looking for the first *positional* argument — the cartridge
    // path. Recognised flags (`--shader VALUE`, `--shader=VALUE`) are
    // consumed so they don't get mistaken for the ROM filename. Unknown
    // `--…` arguments are skipped silently; we're deliberately keeping the
    // CLI tiny and dependency-free.
    let mut args = std::env::args().skip(1);
    let path = loop {
        let arg = args.next()?;
        if arg == "--shader" || arg == "--mapper" {
            let _ = args.next(); // skip the flag's value
            continue;
        }
        if arg.starts_with("--") {
            continue; // covers --shader=… / --mapper=… / unknown flags
        }
        break arg;
    };
    match std::fs::read(&path) {
        Ok(bytes) => {
            eprintln!("loaded cartridge: {} ({} bytes)", path, bytes.len());
            Some(bytes)
        }
        Err(e) => {
            eprintln!("failed to read cartridge ROM {}: {}", path, e);
            std::process::exit(1);
        }
    }
}

/// Mapper override: `--mapper <name>` / `--mapper=<name>` on the command
/// line, `?mapper=<name>` on the web. Applies to the boot cartridge and
/// every drag-and-drop swap in the session. Unknown names warn and fall
/// back to auto-detection; valid names come from `CartridgeMapper::NAMES`
/// so new mappers join automatically.
fn forced_mapper() -> Option<slot::CartridgeMapper> {
    #[cfg(not(target_arch = "wasm32"))]
    let choice: Option<String> = {
        let mut args = std::env::args().skip(1);
        let mut found = None;
        while let Some(arg) = args.next() {
            if let Some(value) = arg.strip_prefix("--mapper=") {
                found = Some(value.to_string());
            } else if arg == "--mapper" {
                found = args.next();
            }
        }
        found
    };
    #[cfg(target_arch = "wasm32")]
    let choice: Option<String> = web_sys::window()
        .and_then(|w| w.location().href().ok())
        .and_then(|href| web_sys::Url::new(&href).ok())
        .and_then(|url| url.search_params().get("mapper"));

    let choice = choice?;
    match slot::CartridgeMapper::parse(&choice) {
        Some(mapper) => Some(mapper),
        None => {
            let msg = format!(
                "unknown mapper '{}' — valid: {}; using auto-detection",
                choice,
                slot::CartridgeMapper::name_list()
            );
            #[cfg(not(target_arch = "wasm32"))]
            eprintln!("{}", msg);
            #[cfg(target_arch = "wasm32")]
            web_sys::console::warn_1(&format!("[msx_rs] {}", msg).into());
            None
        }
    }
}

/// Pick the system ROM set. `MSX_BIOS=nms8245` (or a path to a directory
/// holding `MSX2.ROM` + `MSX2EXT.ROM`) boots the real Philips NMS-8245
/// BIOS; anything else — including unset — boots the embedded C-BIOS.
/// Falls back to C-BIOS with a warning when the files can't be read, so a
/// missing ROM set never breaks startup.
#[cfg(not(target_arch = "wasm32"))]
fn load_machine_roms() -> bus::MachineRoms {
    let Some(choice) = std::env::var_os("MSX_BIOS") else {
        return bus::MachineRoms::CBios;
    };
    let dir = match choice.to_str() {
        Some("nms8245") => std::path::PathBuf::from("assets/NMS8245"),
        Some(path) => std::path::PathBuf::from(path),
        None => return bus::MachineRoms::CBios,
    };
    let main = std::fs::read(dir.join("MSX2.ROM"));
    let ext = std::fs::read(dir.join("MSX2EXT.ROM"));
    match (main, ext) {
        (Ok(main), Ok(ext)) if main.len() == 0x8000 && ext.len() == 0x4000 => {
            eprintln!("bios: NMS-8245 ROMs from {}", dir.display());
            bus::MachineRoms::Nms8245 { main, ext }
        }
        (Ok(m), Ok(e)) => {
            eprintln!(
                "bios: {} ROM sizes wrong (main {} ext {}), falling back to C-BIOS",
                dir.display(), m.len(), e.len()
            );
            bus::MachineRoms::CBios
        }
        (m, e) => {
            eprintln!(
                "bios: could not read ROMs from {} ({}), falling back to C-BIOS",
                dir.display(),
                m.err().or(e.err()).map(|e| e.to_string()).unwrap_or_default()
            );
            bus::MachineRoms::CBios
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn load_cartridge_rom() -> Option<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let win = web_sys::window()?;
    let href = win.location().href().ok()?;
    let url = web_sys::Url::new(&href).ok()?;
    let rom_path = url.search_params().get("rom")?;

    web_sys::console::log_1(&format!("[msx_rs] fetching cartridge: {}", rom_path).into());

    let response: web_sys::Response = JsFuture::from(win.fetch_with_str(&rom_path))
        .await
        .expect("fetch failed")
        .dyn_into()
        .expect("fetch did not return a Response");
    if !response.ok() {
        panic!(
            "cartridge fetch returned HTTP {}: {}",
            response.status(),
            rom_path
        );
    }
    let buffer: js_sys::ArrayBuffer = JsFuture::from(response.array_buffer().expect("array_buffer"))
        .await
        .expect("array_buffer await failed")
        .dyn_into()
        .expect("not an ArrayBuffer");
    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
    web_sys::console::log_1(
        &format!("[msx_rs] cartridge loaded: {} bytes", bytes.len()).into(),
    );
    Some(bytes)
}

/// Web variant of `load_machine_roms`: `?bios=nms8245` (or `?bios=<dir>`)
/// fetches `MSX2.ROM` + `MSX2EXT.ROM` over HTTP, relative to the page —
/// same convention as `?rom=` for cartridges. Any failure (missing param,
/// 404, wrong size) falls back to the embedded C-BIOS with a console
/// warning, so a broken URL never blanks the page.
#[cfg(target_arch = "wasm32")]
async fn load_machine_roms() -> bus::MachineRoms {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    async fn fetch_bytes(path: &str) -> Option<Vec<u8>> {
        let win = web_sys::window()?;
        let response: web_sys::Response = JsFuture::from(win.fetch_with_str(path))
            .await
            .ok()?
            .dyn_into()
            .ok()?;
        if !response.ok() {
            return None;
        }
        let buffer: js_sys::ArrayBuffer =
            JsFuture::from(response.array_buffer().ok()?).await.ok()?.dyn_into().ok()?;
        Some(js_sys::Uint8Array::new(&buffer).to_vec())
    }

    let Some(choice) = web_sys::window()
        .and_then(|w| w.location().href().ok())
        .and_then(|href| web_sys::Url::new(&href).ok())
        .and_then(|url| url.search_params().get("bios"))
    else {
        return bus::MachineRoms::CBios;
    };
    let dir = if choice == "nms8245" {
        "assets/NMS8245".to_string()
    } else {
        choice
    };
    let main = fetch_bytes(&format!("{}/MSX2.ROM", dir)).await;
    let ext = fetch_bytes(&format!("{}/MSX2EXT.ROM", dir)).await;
    match (main, ext) {
        (Some(main), Some(ext)) if main.len() == 0x8000 && ext.len() == 0x4000 => {
            web_sys::console::log_1(
                &format!("[msx_rs] bios: NMS-8245 ROMs from {}", dir).into(),
            );
            bus::MachineRoms::Nms8245 { main, ext }
        }
        _ => {
            web_sys::console::warn_1(
                &format!(
                    "[msx_rs] bios: could not load NMS-8245 ROMs from {}, falling back to C-BIOS",
                    dir
                )
                .into(),
            );
            bus::MachineRoms::CBios
        }
    }
}

/// Initial post-process shader, selected by `?shader=sharp|crt` on the web
/// or `--shader sharp|crt` on the command line. Defaults to `Sharp`.
#[cfg(not(target_arch = "wasm32"))]
fn initial_shader_mode() -> ShaderMode {
    // Walk argv looking for `--shader <value>` or `--shader=value`. We don't
    // pull in clap for two CLI flags — this stays out of the way and keeps
    // the cartridge positional arg unambiguous.
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--shader=") {
            return ShaderMode::parse(value).unwrap_or(ShaderMode::Sharp);
        }
        if arg == "--shader" {
            if let Some(value) = args.next() {
                return ShaderMode::parse(&value).unwrap_or(ShaderMode::Sharp);
            }
        }
    }
    ShaderMode::Sharp
}

#[cfg(target_arch = "wasm32")]
fn initial_shader_mode() -> ShaderMode {
    let Some(win) = web_sys::window() else { return ShaderMode::Sharp };
    let Ok(href) = win.location().href() else { return ShaderMode::Sharp };
    let Ok(url) = web_sys::Url::new(&href) else { return ShaderMode::Sharp };
    url.search_params()
        .get("shader")
        .and_then(|v| ShaderMode::parse(&v))
        .unwrap_or(ShaderMode::Sharp)
}

/// Host KeyCode → MSX keyboard-matrix position (row, column).
///
/// Mapping follows a QWERTY layout. Some MSX-specific keys (DEAD, GRAPH,
/// CODE, STOP, SEL) don't have a clean host equivalent; the table picks
/// plausible substitutes (Alt for GRAPH/CODE, Pause for STOP). The numpad
/// rows (9, 10) are skipped — almost nothing actually uses them.
fn map_key(code: KeyCode) -> Option<(u8, u8)> {
    use KeyCode::*;
    Some(match code {
        // Row 0: 0..7
        Digit0 => (0, 0), Digit1 => (0, 1), Digit2 => (0, 2), Digit3 => (0, 3),
        Digit4 => (0, 4), Digit5 => (0, 5), Digit6 => (0, 6), Digit7 => (0, 7),
        // Row 1: 8 9 - = \ [ ] ;
        Digit8 => (1, 0), Digit9 => (1, 1),
        Minus => (1, 2), Equal => (1, 3), Backslash => (1, 4),
        BracketLeft => (1, 5), BracketRight => (1, 6), Semicolon => (1, 7),
        // Row 2: ' ` , . / DEAD A B   (DEAD has no host key)
        Quote => (2, 0), Backquote => (2, 1), Comma => (2, 2), Period => (2, 3),
        Slash => (2, 4), KeyA => (2, 6), KeyB => (2, 7),
        // Row 3: C..J
        KeyC => (3, 0), KeyD => (3, 1), KeyE => (3, 2), KeyF => (3, 3),
        KeyG => (3, 4), KeyH => (3, 5), KeyI => (3, 6), KeyJ => (3, 7),
        // Row 4: K..R
        KeyK => (4, 0), KeyL => (4, 1), KeyM => (4, 2), KeyN => (4, 3),
        KeyO => (4, 4), KeyP => (4, 5), KeyQ => (4, 6), KeyR => (4, 7),
        // Row 5: S..Z
        KeyS => (5, 0), KeyT => (5, 1), KeyU => (5, 2), KeyV => (5, 3),
        KeyW => (5, 4), KeyX => (5, 5), KeyY => (5, 6), KeyZ => (5, 7),
        // Row 6: SHIFT CTRL GRAPH CAPS CODE F1 F2 F3
        ShiftLeft | ShiftRight => (6, 0),
        ControlLeft | ControlRight => (6, 1),
        AltLeft => (6, 2),       // GRAPH
        CapsLock => (6, 3),
        AltRight => (6, 4),      // CODE
        F1 => (6, 5), F2 => (6, 6), F3 => (6, 7),
        // Row 7: F4 F5 ESC TAB STOP BS SEL RET
        F4 => (7, 0), F5 => (7, 1),
        Escape => (7, 2), Tab => (7, 3),
        Pause => (7, 4),         // STOP
        Backspace => (7, 5),
        Enter => (7, 7),
        // Row 8: SPACE HOME INS DEL ←  ↑  ↓  →
        Space => (8, 0),
        Home => (8, 1), Insert => (8, 2), Delete => (8, 3),
        ArrowLeft => (8, 4), ArrowUp => (8, 5),
        ArrowDown => (8, 6), ArrowRight => (8, 7),
        _ => return None,
    })
}

#[derive(Debug)]
enum RenderError {
    Timeout,
    Outdated,
    Lost,
    Validation,
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    cpu: Z80NMOS,
    clock: TsCounter<i32>,
    bus: Bus,
    /// Wall-clock timestamp of the last `step_to_realtime` call. Used to
    /// figure out how many MSX frames to emulate this render tick.
    last_step: Instant,
    /// Fractional-frame accumulator. Wall time is converted into MSX-frame
    /// units; whole frames are consumed and the remainder rolls over to the
    /// next tick. This is the right way to stay rate-locked when the
    /// invocation interval is variable (e.g. when `about_to_wait` fires
    /// multiple times per real frame because of redraw queueing).
    msx_frame_accumulator: f64,
    /// Audio output. On native this keeps the cpal stream alive; on web it
    /// owns the AudioContext + ScriptProcessorNode + callback. Pub so the
    /// keyboard handler can call `resume()` for the browser's autoplay unlock.
    audio: Audio,
    /// Post-process pipeline: owns the 320×240 intermediate texture the VDP
    /// renders into, plus the upscale shaders.
    post: Post,
    /// Currently active post-process shader. Toggled at runtime via Alt+S.
    shader_mode: ShaderMode,
    /// Crash-hunting aid (MSX_PCTRACE=2): ring buffer of the most recent
    /// instruction PCs. Dumped when the PC falls back to the reset vector
    /// region (< 0x10) from running code — i.e. an unexpected reboot — so
    /// the trace shows the wild-jump path that led there.
    pc_ring: Vec<u16>,
    pc_ring_idx: usize,
    pc_prev: u16,
}

impl State {
    async fn new(window: Arc<Window>, cartridge_rom: Option<Vec<u8>>) -> Self {
        // On native, `inner_size` reflects the OS-window size accurately.
        // On web, `inner_size` reads the canvas's CSS layout box — which
        // can be 0 until the browser has done a layout pass after appending
        // the canvas. The drawing buffer (`canvas.width`/`canvas.height`)
        // is what wgpu actually needs, and we already set those explicitly
        // in `resumed`, so prefer that source on the web build.
        #[cfg(not(target_arch = "wasm32"))]
        let size = window.inner_size();
        #[cfg(target_arch = "wasm32")]
        let size = {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().expect("winit web window has no canvas");
            winit::dpi::PhysicalSize::new(canvas.width(), canvas.height())
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        });

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!("[msx_rs] surface format: {:?}", surface_format).into(),
        );

        // Mailbox > Fifo for input latency: render runs at full speed and
        // the *latest* frame is shown at each vsync, instead of `present()`
        // blocking for a frame that might already be stale. Fall back to
        // Fifo when Mailbox isn't supported (some macOS/Linux setups).
        let present_mode = [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Fifo]
            .iter()
            .copied()
            .find(|m| surface_caps.present_modes.contains(m))
            .unwrap_or(wgpu::PresentMode::Fifo);
        eprintln!("present mode: {:?}", present_mode);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // The VDP renders into the post-process intermediate texture, not the
        // surface directly — its pipeline target format must match that
        // texture (which we keep equal to the surface format for colour-space
        // consistency).
        let vdp = Vdp::new(&device, surface_format);
        let post = Post::new(&device, surface_format);
        let shader_mode = initial_shader_mode();
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!("[msx_rs] shader: {}", shader_mode.label()).into(),
        );
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!("shader: {}", shader_mode.label());

        let audio = Audio::new();
        // System ROM choice: `MSX_BIOS=nms8245` env var on native,
        // `?bios=nms8245` query parameter on web; C-BIOS otherwise.
        #[cfg(not(target_arch = "wasm32"))]
        let machine = load_machine_roms();
        #[cfg(target_arch = "wasm32")]
        let machine = load_machine_roms().await;

        let bus = Bus::new(
            vdp,
            Arc::clone(&audio.psg),
            Arc::clone(&audio.scc),
            cartridge_rom,
            machine,
            forced_mapper(),
        );

        // Default::default() on Z80NMOS gives all-zero registers — including
        // SP = 0, which is wrong: a real Z80 reset sets SP = 0xFFFF. Without
        // this explicit reset(), the first CALL/PUSH wraps SP into ROM and
        // the BIOS spins forever in RST 38H. The Cpu trait's reset() is the
        // source of truth for the real boot state.
        let mut cpu = Z80NMOS::default();
        cpu.reset();
        let clock = TsCounter::<i32>::default();

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            cpu,
            clock,
            bus,
            last_step: Instant::now(),
            msx_frame_accumulator: 0.0,
            audio,
            post,
            shader_mode,
            pc_ring: vec![0u16; 64],
            pc_ring_idx: 0,
            pc_prev: 0,
        }
    }

    /// Hot-swap the cartridge and reset the system around it. `None` ejects
    /// the cartridge → next boot lands in BASIC. `Some(bytes)` plugs the ROM
    /// in, with mapper auto-detected by `Bus::swap_cartridge`.
    ///
    /// Order matters: silence audio → wipe VDP → swap slot → reset CPU. If
    /// we reset the CPU before clearing VRAM, the BIOS init writes would
    /// race with our wipe. Pattern table residue from the previous game
    /// would briefly bleed through.
    fn load_cartridge(&mut self, rom: Option<Vec<u8>>) {
        // Audio thread reads PSG/SCC behind locks — the swap_cartridge and
        // scc.reset() calls take those locks themselves, so we just kick the
        // SCC here (PSG is reset inside swap_cartridge).
        {
            let mut scc = self.audio.scc.lock().unwrap();
            scc.reset();
        }
        self.bus.vdp.reset();
        self.bus
            .swap_cartridge(rom, std::sync::Arc::clone(&self.audio.scc));
        // Z80NMOS::default() leaves SP = 0 which makes the BIOS spin in RST
        // 38h — same gotcha as cold boot, so we go through reset() here too.
        self.cpu = Z80NMOS::default();
        self.cpu.reset();
        self.clock = TsCounter::<i32>::default();
        self.msx_frame_accumulator = 0.0;
    }

    /// Run the CPU for one MSX frame, stepping per scanline so V9938
    /// line-interrupts can fire mid-frame and software-driven per-line
    /// register changes (split-screen scroll, multiple SATs) actually
    /// land where they're meant to.
    ///
    /// Layout (NTSC-ish, ~71 K T-states per frame at 3.58 MHz):
    ///   * 192 visible scanlines, each ~228 T-states. Snapshot the
    ///     per-scanline-mutable registers at the start of each, then
    ///     run the CPU; if R0[4] (IE2) is set and the active line
    ///     matches R19, fire a line interrupt so the game's handler
    ///     gets a chance to update R5/R6/R11/R23 before the *next*
    ///     line is captured.
    ///   * VBlank — raise the frame-IRQ flag and run the rest of the
    ///     frame's CPU budget in one go (no line-interrupt logic
    ///     required outside of the visible area).
    fn step_frame(&mut self) {
        self.clock.0 = Wrapping(0);
        // Clear S2 bit 6 (VR) at the start of a new frame — software that
        // polls VR to detect the VBlank edge needs to see it fall here so
        // the next VBlank's rising edge is observable.
        self.bus.vdp.clear_vblank_flag();
        // Keep the VDP's beam-phase counter (drives S2's HR bit) in step
        // with the T-state clock we just reset.
        self.bus.vdp.reset_scanline_phase();

        // Cycle-accurate(-ish) line interrupts: we step the CPU one
        // instruction at a time and re-check the current scanline
        // BETWEEN instructions. Line IRQs and snapshots fire at the
        // exact instruction boundary nearest the target T-state — at
        // most ~25 T-states late (the longest Z80 instruction), versus
        // the ~200-state slop the previous per-scanline-batched approach
        // produced. Per-instruction `execute_next` is a few % slower than
        // `execute_with_limit` but still well under the frame budget.
        let visible_end_clock: i32 = SCANLINE_TSTATES * VISIBLE_LINES as i32;
        let mut last_line: i32 = -1;
        let mut vblank_fired = false;

        while self.clock.0.0 < FRAME_TSTATES {
            let current_line = self.clock.0.0 / SCANLINE_TSTATES;

            // Walk last_line up to current_line (capped at the last
            // visible line), snapshotting and firing line interrupts at
            // each boundary we cross. The `while` covers the case where
            // a single long instruction crosses a scanline boundary —
            // we still process the snapshot/IRQ for the new line.
            let target = current_line.min(VISIBLE_LINES as i32 - 1);
            while last_line < target {
                last_line += 1;
                self.bus.vdp.snapshot_scanline(last_line as usize);
                // Per fMSX (MSX.c line-coincidence): FH (S1 bit 0) is set on
                // EVERY coincidence — only the IRQ is gated on IE1. Games
                // without IE1 poll S1 for the match (Space Manbow does this
                // for its scroll split); gating FH on IE1 starved that poll
                // loop forever. Off-coincidence with IE1 disabled, FH drops
                // again immediately, so the poll sees a one-scanline pulse.
                if self.bus.vdp.line_irq_target(last_line as u8) {
                    self.bus.vdp.fire_line_irq();
                } else if self.bus.vdp.regs[0] & 0x10 == 0 {
                    self.bus.vdp.clear_line_irq_flag();
                }
            }

            // VBlank fires once, at the first instruction boundary after
            // the end of the visible scan-out. The frame IRQ enables
            // bit 5 of R1; the handler runs naturally in the remaining
            // CPU time.
            if !vblank_fired && self.clock.0.0 >= visible_end_clock {
                self.bus.vdp.start_vblank();
                vblank_fired = true;
            }

            // MSX_PCTRACE=2: per-instruction ring buffer + reboot detector.
            // (=1 keeps the cheaper one-sample-per-frame mode below.)
            static PCTRACE_RING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let ring_on = *PCTRACE_RING.get_or_init(|| {
                std::env::var_os("MSX_PCTRACE").is_some_and(|v| v == *"2")
            });
            if ring_on {
                let pc = self.cpu.get_pc();
                self.pc_ring[self.pc_ring_idx] = pc;
                self.pc_ring_idx = (self.pc_ring_idx + 1) % self.pc_ring.len();
                if pc < 0x10 && self.pc_prev >= 0x100 {
                    let mut trail: Vec<String> = Vec::with_capacity(self.pc_ring.len());
                    for i in 0..self.pc_ring.len() {
                        let idx = (self.pc_ring_idx + i) % self.pc_ring.len();
                        trail.push(format!("{:04X}", self.pc_ring[idx]));
                    }
                    eprintln!("[reboot] PC {:04X} -> {:04X}; trail: {}",
                              self.pc_prev, pc, trail.join(" "));
                }
                self.pc_prev = pc;
            }

            let before = self.clock.0.0;
            let _ = self.cpu.execute_next(
                &mut self.bus,
                &mut self.clock,
                Option::<fn(z80emu::CpuDebug)>::None,
            );
            // Advance the VDP command-engine busy timer by the T-states this
            // instruction consumed, so a polled CE bit clears at the right
            // time relative to the beam (see Vdp::tick / command_duration).
            let dt = self.clock.0.0 - before;
            if dt > 0 {
                self.bus.vdp.tick(dt);
            }
        }

        // Defensive: if the frame budget was so short we never reached
        // visible_end_clock (won't happen with our FRAME_TSTATES, but
        // belt-and-suspenders), still raise VBlank so the game's
        // interrupt handler runs at least once per call.
        if !vblank_fired {
            self.bus.vdp.start_vblank();
        }

        // Hang-hunting aid: MSX_PCTRACE=1 prints one PC sample per frame.
        // A wedged game shows up as the same handful of addresses repeating;
        // map those against the ROM/mapper banks to find the poll loop.
        if std::env::var_os("MSX_PCTRACE").is_some() {
            eprintln!("[pc] {:04X}", self.cpu.get_pc());
        }
    }

    /// Pace MSX emulation against wall-clock time. Uses a fractional-frame
    /// accumulator so the long-run frame rate is exactly `MSX_HZ` regardless
    /// of how often this is called — a 60 Hz monitor produces 1 frame per
    /// call, a 30 Hz monitor 2 frames per call, and a 16 ms-then-1 ms
    /// invocation pattern still nets out to 60 frames per second.
    ///
    /// Returns how many MSX frames were emulated this call. The caller uses
    /// this to present exactly one surface frame per emulated frame: when the
    /// host refresh runs faster than 60 Hz (or `about_to_wait` fires several
    /// times per emulated frame), a zero-frame call must NOT trigger a redraw
    /// — re-presenting the same VRAM at an irregular phase relative to the
    /// emulation is what turns a game's steady 30 Hz sprite-flicker (Vampire
    /// Killer et al.) into visible strobing. Showing each emulated frame once
    /// keeps the flicker cadence regular, matching real hardware.
    fn step_to_realtime(&mut self) -> u32 {
        let now = Instant::now();
        let elapsed = now
            .duration_since(self.last_step)
            .as_secs_f64()
            .min(MAX_CATCHUP_SECS);
        self.last_step = now;

        self.msx_frame_accumulator += elapsed * MSX_HZ;

        // Step AT MOST one emulated frame per call. The render pass that
        // follows presents whatever VRAM this frame produced; if we stepped
        // two frames here the first frame's VRAM would be overwritten before
        // it was ever drawn, silently dropping it. For software
        // sprite-multiplexing games (Vampire Killer) every frame is a distinct
        // phase of the sprite rotation, so a dropped frame is a dropped phase
        // — exactly the flicker we were chasing. Windows' coarse WaitUntil
        // timer makes `elapsed` jitter between ~16 ms and ~31 ms, which under
        // the old while-loop produced an irregular 1-then-2 frame cadence and
        // thus an irregular drop pattern.
        //
        // Keep the fractional remainder in the accumulator so the long-run
        // rate stays exactly MSX_HZ, but clamp it so a long stall (alt-tab,
        // breakpoint) can't bank dozens of frames and then fast-forward.
        let mut frames_stepped = 0;
        if self.msx_frame_accumulator >= 1.0 {
            self.step_frame();
            self.msx_frame_accumulator -= 1.0;
            frames_stepped += 1;
        }
        if self.msx_frame_accumulator > 1.0 {
            self.msx_frame_accumulator = 1.0;
        }
        frames_stepped
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(
            &format!("[msx_rs] resize event: {}×{}", new_size.width, new_size.height).into(),
        );

        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // On web, keep the canvas drawing buffer in sync with the surface
            // size — otherwise wgpu's swapchain texture and the canvas would
            // disagree on dimensions and the browser would stretch the result.
            #[cfg(target_arch = "wasm32")]
            {
                use winit::platform::web::WindowExtWebSys;
                if let Some(canvas) = self.window.canvas() {
                    canvas.set_width(new_size.width);
                    canvas.set_height(new_size.height);
                }
            }
        }
    }

    fn render(&mut self) -> Result<(), RenderError> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => return Err(RenderError::Timeout),
            wgpu::CurrentSurfaceTexture::Outdated => return Err(RenderError::Outdated),
            wgpu::CurrentSurfaceTexture::Lost => return Err(RenderError::Lost),
            wgpu::CurrentSurfaceTexture::Validation => return Err(RenderError::Validation),
        };

        // The VDP renders at its own native canvas size (320×240) into the
        // intermediate texture; the post-process pass then upscales that to
        // whatever the surface is.
        self.bus
            .vdp
            .upload(&self.queue, (vdp::CANVAS_W, vdp::CANVAS_H));
        let backdrop = self.bus.vdp.backdrop_rgba();
        self.post
            .upload(&self.queue, (self.config.width, self.config.height), backdrop);

        // Same backdrop colour used for both passes:
        //   * Pass 1 clears the VDP's own border area — saves the shader from
        //     drawing the border explicitly when it falls outside the active
        //     256×192 region. (The VDP shader still paints it; harmless.)
        //   * Pass 2 clears the surface letterbox so the window frame matches
        //     the in-canvas border seamlessly.
        let clear_color = wgpu::Color {
            r: backdrop[0] as f64,
            g: backdrop[1] as f64,
            b: backdrop[2] as f64,
            a: backdrop[3] as f64,
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Pass 1: VDP → 320×240 intermediate texture.
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("VDP pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.post.intermediate_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            self.bus.vdp.draw(&mut render_pass);
        }

        // Pass 2: post-process → surface. Letterbox area is filled by the
        // clear (and the shader returns backdrop for fragments outside the
        // integer-scaled viewport, in case the surface is sub-pixel exotic).
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            self.post.draw(&mut render_pass, self.shader_mode);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.window.pre_present_notify();
        surface_texture.present();

        Ok(())
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
    /// Current modifier state (Alt, Cmd/Meta, Ctrl, Shift). Tracked
    /// continuously via `WindowEvent::ModifiersChanged` so we can recognise
    /// chord shortcuts (e.g. Alt+Enter for fullscreen) in key handlers.
    modifiers: ModifiersState,
    /// On the web `State::new` can't be awaited synchronously — we spawn it
    /// via `wasm_bindgen_futures::spawn_local` and the future writes the
    /// finished State into this slot. `about_to_wait` checks it on every
    /// loop iteration and moves it into `self.state` when it appears.
    #[cfg(target_arch = "wasm32")]
    pending_state: std::rc::Rc<std::cell::RefCell<Option<State>>>,
    /// Inbox for cartridge ROMs picked or dropped through the browser DOM.
    /// JS-side handlers write here from a closure; `about_to_wait` drains
    /// the slot and calls `State::load_cartridge`. `Rc<RefCell>` is fine
    /// because winit's web event loop is single-threaded.
    #[cfg(target_arch = "wasm32")]
    pending_rom: std::rc::Rc<std::cell::RefCell<Option<Vec<u8>>>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
        let mut attributes = Window::default_attributes()
            .with_title("MSX.rs - Emulator")
            .with_inner_size(winit::dpi::PhysicalSize::new(640, 480));

        // On the web, ask winit to create a canvas and append it to the
        // document body. We could also pass our own canvas via `with_canvas`
        // if the host page wanted to place it somewhere specific.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            attributes = attributes.with_append(true);
        }

        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        // Winit's `with_inner_size` on web only sets a *requested* size; the
        // canvas DOM element's `width`/`height` attributes (which is what
        // determines wgpu's drawing buffer size) stays at the browser default
        // until we set them explicitly. Read the viewport dimensions from JS
        // so the drawing buffer matches the displayed canvas exactly.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            if let Some(canvas) = window.canvas() {
                let web_window = web_sys::window().expect("no JS window");
                let w = web_window.inner_width().ok()
                    .and_then(|v| v.as_f64()).unwrap_or(640.0) as u32;
                let h = web_window.inner_height().ok()
                    .and_then(|v| v.as_f64()).unwrap_or(480.0) as u32;
                web_sys::console::log_1(
                    &format!("[msx_rs] initial canvas size: {}×{}", w, h).into(),
                );
                canvas.set_width(w);
                canvas.set_height(h);

                // Winit slaps inline `style.width`/`style.height` on the canvas
                // (in CSS pixels matching the drawing-buffer size). Inline
                // styles beat external stylesheets, so our `width: 100%` in
                // index.html does nothing. Force the canvas to viewport size
                // with inline styles of our own — and pin it via `position:
                // fixed` so the browser doesn't reflow it back to natural size.
                let style = canvas.style();
                let _ = style.set_property("position", "fixed");
                let _ = style.set_property("top", "0");
                let _ = style.set_property("left", "0");
                let _ = style.set_property("width", "100vw");
                let _ = style.set_property("height", "100vh");
                let _ = style.set_property("image-rendering", "pixelated");
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let rom = load_cartridge_rom();
            self.state = Some(pollster::block_on(State::new(window, rom)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let slot = std::rc::Rc::clone(&self.pending_state);
            wasm_bindgen_futures::spawn_local(async move {
                let rom = load_cartridge_rom().await;
                let state = State::new(window, rom).await;
                *slot.borrow_mut() = Some(state);
            });
            install_rom_inputs(std::rc::Rc::clone(&self.pending_rom));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Any keystroke counts as a user gesture for the browser's
                // audio autoplay policy — try to resume the AudioContext.
                // No-op on native and after the first successful resume.
                if event.state == ElementState::Pressed {
                    state.audio.resume();
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    // Fullscreen toggle: Alt+Enter (Windows/Linux) or
                    // Cmd+Enter (Mac). Both nicely outside the MSX matrix
                    // and not eaten by the browser or macOS media keys.
                    let fullscreen_modifier =
                        self.modifiers.alt_key() || self.modifiers.super_key();
                    if fullscreen_modifier
                        && code == KeyCode::Enter
                        && event.state == ElementState::Pressed
                    {
                        let target = if state.window.fullscreen().is_some() {
                            None
                        } else {
                            Some(Fullscreen::Borderless(None))
                        };
                        state.window.set_fullscreen(target);
                    } else if self.modifiers.alt_key()
                        && code == KeyCode::KeyS
                        && event.state == ElementState::Pressed
                    {
                        // Shader toggle. Only the Alt modifier — not super —
                        // because Cmd+S is "save" in every browser and would
                        // pop the download dialog on the web build.
                        state.shader_mode = state.shader_mode.toggle();
                        #[cfg(target_arch = "wasm32")]
                        web_sys::console::log_1(
                            &format!("[msx_rs] shader: {}", state.shader_mode.label()).into(),
                        );
                        #[cfg(not(target_arch = "wasm32"))]
                        eprintln!("shader: {}", state.shader_mode.label());
                    } else if let Some((row, col)) = map_key(code) {
                        let pressed = event.state == ElementState::Pressed;
                        state.bus.ppi.set_key(row, col, pressed);
                    }
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::DroppedFile(path) => {
                // Native drag-and-drop: winit hands us a filesystem path.
                // Read it synchronously — these are user-initiated, so the
                // tiny stall is fine and avoids an executor here.
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        eprintln!("dropped cartridge: {} ({} bytes)", path.display(), bytes.len());
                        state.load_cartridge(Some(bytes));
                    }
                    Err(e) => eprintln!("failed to read dropped file {}: {}", path.display(), e),
                }
            }
            WindowEvent::RedrawRequested => match state.render() {
                Ok(_) => {}
                Err(RenderError::Timeout) => {}
                Err(RenderError::Outdated) | Err(RenderError::Lost) => state.resize(state.size),
                Err(RenderError::Validation) => event_loop.exit(),
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // On the web, the async `State::new` finishes after some round-trips
        // through the browser's event loop. Check if it landed in the slot
        // and promote it to live state.
        #[cfg(target_arch = "wasm32")]
        if self.state.is_none() {
            if let Some(state) = self.pending_state.borrow_mut().take() {
                self.state = Some(state);
            }
        }

        let Some(state) = self.state.as_mut() else { return };

        // Drain any cartridge ROM dropped or picked through the browser DOM
        // before we step the next frame — the swap resets the CPU and clock,
        // so doing it mid-frame would leave half-executed state behind.
        #[cfg(target_arch = "wasm32")]
        if let Some(rom) = self.pending_rom.borrow_mut().take() {
            state.load_cartridge(Some(rom));
        }

        // Drive emulation here rather than from `render()` so the audio thread
        // keeps getting fresh PSG/SCC state even when the window is hidden or
        // occluded — `render()` doesn't fire in those cases, but `about_to_wait`
        // does. `step_to_realtime` itself is rate-limited by wall clock, so
        // calling it on every loop iteration is fine.
        //
        // Only request a redraw when at least one new MSX frame was emulated.
        // Frame-locking presentation to emulation this way stops the renderer
        // from re-presenting unchanged VRAM at the host refresh rate, which
        // (on >60 Hz displays especially) sampled the emulation at an
        // irregular phase and made per-frame sprite flicker strobe. With this
        // gate each emulated frame is shown exactly once; vsync simply holds
        // the last frame when the host refresh outruns 60 Hz.
        let frames_stepped = state.step_to_realtime();
        if frames_stepped > 0 {
            state.window.request_redraw();
        }

        // Schedule the next wake-up one MSX frame from now. When the window
        // is visible, vsync paces us anyway; when hidden, this keeps the loop
        // from busy-spinning while still feeding the audio thread.
        let next = Instant::now() + Duration::from_secs_f64(1.0 / MSX_HZ);
        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        log::init_from_environment();
        let event_loop = EventLoop::new().unwrap();
        let mut app = App::default();
        event_loop.run_app(&mut app).unwrap();
    }
}

/// WebAssembly entry point. Trunk wires this up via `wasm-bindgen(start)`.
/// We use `spawn_app` instead of `run_app` because the browser's main thread
/// can't block on the event loop — it has to return so JS can keep ticking.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn web_main() {
    console_error_panic_hook::set_once();
    log::init_from_environment();
    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().unwrap();
    let app = App::default();
    event_loop.spawn_app(app);
}

// --- Web ROM input: drag-and-drop -------------------------------------------
//
// `dragover` is intercepted (and preventDefault'd — that's what tells the
// browser "yes, a drop here is OK") and the `drop` handler reads the first
// file from the DataTransfer. Bytes land in `pending_rom`, which
// `about_to_wait` drains once per loop tick.
//
// Reads are async (`Blob::array_buffer()` returns a Promise) so they run via
// `spawn_local` and the bytes appear on the slot once the browser finishes —
// typically a few ms for ROM-sized files.

#[cfg(target_arch = "wasm32")]
fn install_rom_inputs(slot: std::rc::Rc<std::cell::RefCell<Option<Vec<u8>>>>) {
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;

    let win = web_sys::window().expect("no JS window");
    let doc = win.document().expect("no document");
    let body = doc.body().expect("no body");

    // dragover: signal the drop is allowed. Without preventDefault here the
    // browser's default handler kicks in and the subsequent drop never fires.
    let dragover = Closure::wrap(Box::new(|event: web_sys::DragEvent| {
        event.prevent_default();
    }) as Box<dyn FnMut(_)>);
    body.add_event_listener_with_callback("dragover", dragover.as_ref().unchecked_ref())
        .expect("addEventListener dragover");
    dragover.forget();

    // drop: read the first file in the DataTransfer.
    let drop_slot = std::rc::Rc::clone(&slot);
    let drop_cb = Closure::wrap(Box::new(move |event: web_sys::DragEvent| {
        event.prevent_default();
        if let Some(dt) = event.data_transfer() {
            if let Some(files) = dt.files() {
                if let Some(file) = files.get(0) {
                    read_file_into_slot(file, std::rc::Rc::clone(&drop_slot));
                }
            }
        }
    }) as Box<dyn FnMut(_)>);
    body.add_event_listener_with_callback("drop", drop_cb.as_ref().unchecked_ref())
        .expect("addEventListener drop");
    drop_cb.forget();
}

/// Read a `File` as bytes and drop the result into `slot`. Async because
/// `Blob::array_buffer()` returns a Promise; we hop on `spawn_local`.
#[cfg(target_arch = "wasm32")]
fn read_file_into_slot(file: web_sys::File, slot: std::rc::Rc<std::cell::RefCell<Option<Vec<u8>>>>) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let name = file.name();
    wasm_bindgen_futures::spawn_local(async move {
        match JsFuture::from(file.array_buffer()).await {
            Ok(value) => {
                if let Ok(ab) = value.dyn_into::<js_sys::ArrayBuffer>() {
                    let bytes = js_sys::Uint8Array::new(&ab).to_vec();
                    web_sys::console::log_1(
                        &format!("[msx_rs] loaded ROM: {} ({} bytes)", name, bytes.len()).into(),
                    );
                    *slot.borrow_mut() = Some(bytes);
                } else {
                    web_sys::console::error_1(&"file.array_buffer() returned non-ArrayBuffer".into());
                }
            }
            Err(e) => web_sys::console::error_1(&e),
        }
    });
}
