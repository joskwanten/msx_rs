//! egui overlay: a menu bar (File / Machine / Video) drawn on top of the
//! emulator frame. Toggle with F9 (Esc closes it).
//!
//! The UI is immediate-mode and must not borrow the emulator's state while it
//! runs, so it works off a cheap [`UiSnapshot`] (copied each frame) and emits a
//! list of [`UiAction`] commands. `main.rs` applies those after the frame is
//! drawn — keeping borrows clean and making the menu logic independent of the
//! emulator internals.
//!
//! Three wgpu objects live here: the egui [`Context`], the `egui_winit` event
//! intake (`State`), and the `egui_wgpu` [`Renderer`]. Render order in
//! `State::render`: VDP pass → post pass → (if visible) this egui pass, all onto
//! the same surface view with `LoadOp::Load` so the UI layers over the game.

use egui::{Context, FullOutput, ViewportId};
use winit::window::Window;

use crate::post::ShaderMode;
use crate::slot::CartridgeMapper;

/// Read-only view of emulator state the menu needs. Copied each frame so the UI
/// closure never borrows `State`.
pub struct UiSnapshot {
    pub shader: ShaderMode,
    pub paused: bool,
    pub crt_blur: f32,
    pub mapper: Option<CartridgeMapper>,
    pub fullscreen: bool,
}

/// A command the menu emits; `State::apply_ui_action` carries it out after the
/// frame is painted.
pub enum UiAction {
    SetShader(ShaderMode),
    SetCrtBlur(f32),
    ToggleFullscreen,
    Reset,
    SetPaused(bool),
    SetForcedMapper(Option<CartridgeMapper>),
    OpenRom,
    /// File ▸ Quit — native only (the web build has no window to close).
    #[cfg(not(target_arch = "wasm32"))]
    Quit,
}

/// Z80 register file at a moment in time, copied for the CPU debug panel.
#[derive(Clone, Copy, Default)]
pub struct CpuRegs {
    pub pc: u16,
    pub sp: u16,
    pub af: u16,
    pub bc: u16,
    pub de: u16,
    pub hl: u16,
    pub ix: u16,
    pub iy: u16,
    pub i: u8,
    pub r: u8,
    pub im: u8,
    pub iff1: bool,
    pub iff2: bool,
    pub halt: bool,
}

/// Borrowed read-only view of emulator internals for the debug windows. Built
/// fresh each frame in `State::render`; nothing here is mutated by the UI.
pub struct DebugView<'a> {
    pub cpu: CpuRegs,
    pub pc_ring: &'a [u16],
    pub pc_ring_idx: usize,
    /// VDP control registers R0–R63 (`pub regs` on the VDP).
    pub vdp_regs: &'a [u8; 64],
    /// VDP status registers S0–S9.
    pub vdp_status: &'a [u8; 10],
    /// 16-entry palette in shader RGBA (`f32`) form.
    pub palette: &'a [[f32; 4]; 16],
    pub vram: &'a [u8],
    pub is_pal: bool,
    pub fps: f32,
}

/// Which debug windows are open. Toggled from the Debug menu; persisted in `Gui`.
#[derive(Default)]
struct DebugPanels {
    cpu: bool,
    vdp: bool,
    palette: bool,
    sprites: bool,
    vram: bool,
}

/// All four post shaders, in cycle order — for the Video menu's radio list.
const SHADERS: [ShaderMode; 4] = [
    ShaderMode::Sharp,
    ShaderMode::Crt,
    ShaderMode::Pixely,
    ShaderMode::Hq4x,
];

/// Mapper-override choices: "auto" plus the canonical name per mapper.
const MAPPERS: [(&str, Option<CartridgeMapper>); 6] = [
    ("auto-detect", None),
    ("plain", Some(CartridgeMapper::Plain)),
    ("konami", Some(CartridgeMapper::KonamiBasic)),
    ("konami-scc", Some(CartridgeMapper::KonamiSCC)),
    ("ascii8", Some(CartridgeMapper::Ascii8)),
    ("ascii16", Some(CartridgeMapper::Ascii16)),
];

fn mapper_label(m: Option<CartridgeMapper>) -> &'static str {
    MAPPERS
        .iter()
        .find(|(_, v)| *v == m)
        .map(|(name, _)| *name)
        .unwrap_or("auto-detect")
}

pub struct Gui {
    ctx: Context,
    egui_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    /// Menu bar shown? Toggled by F9 / Esc in `main.rs`.
    pub visible: bool,
    /// Which debug windows are open.
    panels: DebugPanels,
}

impl Gui {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &Window,
    ) -> Self {
        let ctx = Context::default();
        let egui_state = egui_winit::State::new(
            ctx.clone(),
            ViewportId::ROOT,
            window,
            None, // native pixels-per-point: let winit report it
            None, // theme: follow system
            None, // max texture side: query from device defaults
        );
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );
        Self {
            ctx,
            egui_state,
            renderer,
            visible: false,
            panels: DebugPanels::default(),
        }
    }

    pub fn egui_ctx(&self) -> &Context {
        &self.ctx
    }

    /// Feed a window event to egui. The caller forwards keystrokes to the MSX
    /// only when `consumed` is false, and repaints when `repaint` is set.
    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.egui_state.on_window_event(window, event)
    }

    /// Build the menu (+ any open debug windows) for this frame and collect the
    /// actions the user triggered.
    pub fn run(
        &mut self,
        window: &Window,
        snap: &UiSnapshot,
        dbg: &DebugView,
    ) -> (Vec<UiAction>, FullOutput) {
        let raw_input = self.egui_state.take_egui_input(window);
        let mut actions = Vec::new();
        // Borrow the panel flags out here so the closure can mutate them while
        // `self.ctx` is borrowed by `run_ui` (disjoint fields).
        let panels = &mut self.panels;
        let mut out = self.ctx.run_ui(raw_input, |ui| {
            build_menu(ui, snap, dbg, panels, &mut actions);
        });
        // Apply cursor/clipboard side effects, then hand the rest to paint.
        let platform_output = std::mem::take(&mut out.platform_output);
        self.egui_state.handle_platform_output(window, platform_output);
        (actions, out)
    }

    /// Tessellate and record the egui draw pass into `encoder`, layered over the
    /// existing surface contents. Returns any user-callback command buffers that
    /// must be submitted *before* `encoder` (empty for our pure-widget UI).
    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        screen: &egui_wgpu::ScreenDescriptor,
        full_output: FullOutput,
    ) -> Vec<wgpu::CommandBuffer> {
        let paint_jobs = self
            .ctx
            .tessellate(full_output.shapes, screen.pixels_per_point);

        for (id, delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        let user_cmds = self
            .renderer
            .update_buffers(device, queue, encoder, &paint_jobs, screen);

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Load: keep the post-processed frame underneath.
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &paint_jobs, screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        user_cmds
    }
}

/// The actual menu layout. Pushes a `UiAction` for every control the user hits.
/// `ui` is the central area egui hands us; we carve a top strip off it for the
/// menu bar, then float any open debug windows over the game.
fn build_menu(
    ui: &mut egui::Ui,
    snap: &UiSnapshot,
    dbg: &DebugView,
    panels: &mut DebugPanels,
    actions: &mut Vec<UiAction>,
) {
    egui::Panel::top("menu_bar").show_inside(ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open ROM…").clicked() {
                    actions.push(UiAction::OpenRom);
                    ui.close();
                }
                ui.add_enabled(false, egui::Button::new("Open Disk…"))
                    .on_disabled_hover_text("Runtime disk-swap komt in Phase 2");
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        actions.push(UiAction::Quit);
                        ui.close();
                    }
                }
            });

            ui.menu_button("Machine", |ui| {
                if ui.button("Reset").clicked() {
                    actions.push(UiAction::Reset);
                    ui.close();
                }
                let mut paused = snap.paused;
                if ui.checkbox(&mut paused, "Pause").changed() {
                    actions.push(UiAction::SetPaused(paused));
                }
                ui.separator();
                ui.label("Mapper (next load)");
                egui::ComboBox::from_id_salt("mapper")
                    .selected_text(mapper_label(snap.mapper))
                    .show_ui(ui, |ui| {
                        for (name, value) in MAPPERS {
                            if ui
                                .selectable_label(snap.mapper == value, name)
                                .clicked()
                            {
                                actions.push(UiAction::SetForcedMapper(value));
                            }
                        }
                    });
            });

            ui.menu_button("Video", |ui| {
                ui.label("Shader");
                for shader in SHADERS {
                    if ui
                        .selectable_label(snap.shader == shader, shader.label())
                        .clicked()
                    {
                        actions.push(UiAction::SetShader(shader));
                    }
                }
                if snap.shader == ShaderMode::Crt {
                    ui.separator();
                    let mut blur = snap.crt_blur;
                    if ui
                        .add(egui::Slider::new(&mut blur, 0.0..=1.5).text("CRT blur"))
                        .changed()
                    {
                        actions.push(UiAction::SetCrtBlur(blur));
                    }
                }
                ui.separator();
                let mut fullscreen = snap.fullscreen;
                if ui.checkbox(&mut fullscreen, "Fullscreen").changed() {
                    actions.push(UiAction::ToggleFullscreen);
                }
            });

            ui.menu_button("Debug", |ui| {
                ui.checkbox(&mut panels.cpu, "CPU registers");
                ui.checkbox(&mut panels.vdp, "VDP registers");
                ui.checkbox(&mut panels.palette, "Palette");
                ui.checkbox(&mut panels.sprites, "Sprites");
                ui.checkbox(&mut panels.vram, "VRAM");
            });

            // Live FPS on the right edge of the bar.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("{:.0} fps", dbg.fps));
            });
        });
    });

    // Floating debug windows, each toggled from the Debug menu.
    let ctx = ui.ctx().clone();
    egui::Window::new("CPU")
        .open(&mut panels.cpu)
        .resizable(false)
        .show(&ctx, |ui| cpu_panel(ui, dbg));
    egui::Window::new("VDP registers")
        .open(&mut panels.vdp)
        .default_width(320.0)
        .show(&ctx, |ui| vdp_panel(ui, dbg));
    egui::Window::new("Palette")
        .open(&mut panels.palette)
        .resizable(false)
        .show(&ctx, |ui| palette_panel(ui, dbg));
    egui::Window::new("Sprites")
        .open(&mut panels.sprites)
        .default_width(300.0)
        .show(&ctx, |ui| sprite_panel(ui, dbg));
    egui::Window::new("VRAM")
        .open(&mut panels.vram)
        .default_width(560.0)
        .show(&ctx, |ui| vram_panel(ui, dbg));
}

/// CPU register file + the recent-PC ring buffer.
fn cpu_panel(ui: &mut egui::Ui, dbg: &DebugView) {
    let c = &dbg.cpu;
    egui::Grid::new("cpu_regs").striped(true).show(ui, |ui| {
        ui.monospace("AF");
        ui.monospace(format!("{:04X}", c.af));
        ui.monospace("PC");
        ui.monospace(format!("{:04X}", c.pc));
        ui.end_row();
        ui.monospace("BC");
        ui.monospace(format!("{:04X}", c.bc));
        ui.monospace("SP");
        ui.monospace(format!("{:04X}", c.sp));
        ui.end_row();
        ui.monospace("DE");
        ui.monospace(format!("{:04X}", c.de));
        ui.monospace("IX");
        ui.monospace(format!("{:04X}", c.ix));
        ui.end_row();
        ui.monospace("HL");
        ui.monospace(format!("{:04X}", c.hl));
        ui.monospace("IY");
        ui.monospace(format!("{:04X}", c.iy));
        ui.end_row();
    });
    ui.separator();
    // Decode the flag bits from F (low byte of AF): S Z - H - P/V N C.
    let f = c.af as u8;
    let flag = |bit: u8, name: &str| {
        if f & bit != 0 {
            name.to_string()
        } else {
            name.to_ascii_lowercase()
        }
    };
    ui.monospace(format!(
        "flags {} {} {} {} {}",
        flag(0x80, "S"),
        flag(0x40, "Z"),
        flag(0x10, "H"),
        flag(0x04, "P"),
        flag(0x01, "C"),
    ));
    ui.monospace(format!(
        "I={:02X} R={:02X}  IM{}  IFF{}{}  {}",
        c.i,
        c.r,
        c.im,
        c.iff1 as u8,
        c.iff2 as u8,
        if c.halt { "HALT" } else { "" },
    ));

    ui.separator();
    ui.label("Recent PC (oldest → newest)");
    // Walk the ring from the entry after the write cursor (oldest) around to it.
    let n = dbg.pc_ring.len();
    let mut line = String::new();
    for k in 0..n {
        let pc = dbg.pc_ring[(dbg.pc_ring_idx + k) % n];
        line.push_str(&format!("{pc:04X} "));
        if k % 8 == 7 {
            ui.monospace(&line);
            line.clear();
        }
    }
    if !line.is_empty() {
        ui.monospace(&line);
    }
}

/// Decoded screen mode from the M1–M5 bits spread across R0 and R1.
fn screen_mode(regs: &[u8; 64]) -> &'static str {
    let m1 = (regs[1] >> 4) & 1;
    let m2 = (regs[1] >> 3) & 1;
    let m3 = (regs[0] >> 1) & 1;
    let m4 = (regs[0] >> 2) & 1;
    let m5 = (regs[0] >> 3) & 1;
    match (m5, m4, m3, m2, m1) {
        (0, 0, 0, 0, 0) => "GRAPHIC 1 (SCREEN 1)",
        (0, 0, 0, 0, 1) => "TEXT 1 (SCREEN 0)",
        (0, 0, 0, 1, 0) => "MULTICOLOUR (SCREEN 3)",
        (0, 0, 1, 0, 0) => "GRAPHIC 2 (SCREEN 2)",
        (0, 1, 0, 0, 0) => "GRAPHIC 3 (SCREEN 4)",
        (0, 1, 1, 0, 0) => "GRAPHIC 4 (SCREEN 5)",
        (1, 0, 0, 0, 0) => "GRAPHIC 5 (SCREEN 6)",
        (1, 0, 1, 0, 0) => "GRAPHIC 6 (SCREEN 7)",
        (1, 1, 1, 0, 0) => "GRAPHIC 7 (SCREEN 8)",
        (0, 0, 0, 1, 1) => "TEXT 2 (SCREEN 0:80)",
        _ => "unknown / mixed",
    }
}

/// All VDP control + status registers, plus a few decoded fields.
fn vdp_panel(ui: &mut egui::Ui, dbg: &DebugView) {
    let r = dbg.vdp_regs;
    ui.monospace(format!("Mode: {}", screen_mode(r)));
    ui.monospace(format!(
        "{}  display {}  sprites {}",
        if dbg.is_pal { "PAL/50Hz" } else { "NTSC/60Hz" },
        if r[1] & 0x40 != 0 { "on" } else { "off" },
        if r[8] & 0x02 != 0 { "off" } else { "on" },
    ));
    // Table base addresses (in VRAM), derived the standard way.
    ui.monospace(format!(
        "name {:04X}  colour {:04X}  pattern {:04X}",
        (r[2] as usize & 0x7F) << 10,
        ((r[3] as usize) | ((r[10] as usize) << 8)) << 6,
        (r[4] as usize & 0x3F) << 11,
    ));
    ui.monospace(format!(
        "sprite-attr {:04X}  sprite-patt {:04X}",
        ((r[5] as usize & 0x7F) | ((r[11] as usize & 0x03) << 8)) << 7,
        (r[6] as usize & 0x3F) << 11,
    ));
    ui.separator();

    egui::Grid::new("vdp_regs").striped(true).show(ui, |ui| {
        for row in 0..6 {
            for col in 0..4 {
                let i = row * 4 + col;
                if i <= 23 {
                    ui.monospace(format!("R{i:<2}={:02X}", r[i]));
                }
            }
            ui.end_row();
        }
    });
    ui.separator();
    ui.label("Status S0–S9");
    let mut line = String::new();
    for (i, s) in dbg.vdp_status.iter().enumerate() {
        line.push_str(&format!("S{i}={s:02X} "));
        if i % 5 == 4 {
            ui.monospace(&line);
            line.clear();
        }
    }
    if !line.is_empty() {
        ui.monospace(&line);
    }
}

/// The 16-entry palette as colour swatches with their RGB values.
fn palette_panel(ui: &mut egui::Ui, dbg: &DebugView) {
    egui::Grid::new("palette").striped(true).show(ui, |ui| {
        for (i, c) in dbg.palette.iter().enumerate() {
            let rgb = [
                (c[0] * 255.0).round() as u8,
                (c[1] * 255.0).round() as u8,
                (c[2] * 255.0).round() as u8,
            ];
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(20.0, 16.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 2.0, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
            ui.monospace(format!("{i:>2}: {:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2]));
            if i % 2 == 1 {
                ui.end_row();
            }
        }
    });
}

/// The 32 sprite-attribute-table entries (screen-mode-dependent layout).
fn sprite_panel(ui: &mut egui::Ui, dbg: &DebugView) {
    let r = dbg.vdp_regs;
    let sat = ((r[5] as usize & 0x7F) | ((r[11] as usize & 0x03) << 8)) << 7;
    let large = r[1] & 0x02 != 0; // 16×16 sprites
    let mag = r[1] & 0x01 != 0; // magnified
    ui.monospace(format!(
        "SAT @ {sat:04X}   size {}   mag {}",
        if large { "16×16" } else { "8×8" },
        if mag { "2×" } else { "1×" },
    ));
    ui.separator();
    egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
        egui::Grid::new("sprites").striped(true).num_columns(5).show(ui, |ui| {
            ui.monospace("#");
            ui.monospace("Y");
            ui.monospace("X");
            ui.monospace("patt");
            ui.monospace("colr");
            ui.end_row();
            for i in 0..32usize {
                let base = sat + i * 4;
                if base + 3 >= dbg.vram.len() {
                    break;
                }
                let y = dbg.vram[base];
                // Y == 0xD0 ends the visible sprite list (TMS / V9938 modes 1–3).
                let terminated = y == 0xD0;
                ui.monospace(format!("{i:>2}"));
                ui.monospace(format!("{y:02X}"));
                ui.monospace(format!("{:02X}", dbg.vram[base + 1]));
                ui.monospace(format!("{:02X}", dbg.vram[base + 2]));
                ui.monospace(format!("{:02X}", dbg.vram[base + 3]));
                ui.end_row();
                if terminated {
                    break;
                }
            }
        });
    });
}

/// A scrollable hex view of VRAM. The slider picks the start address.
fn vram_panel(ui: &mut egui::Ui, dbg: &DebugView) {
    // egui is immediate-mode; keep the scroll address in egui's temp memory so
    // it persists across frames without a field on `Gui`.
    let id = egui::Id::new("vram_addr");
    let mut addr = ui.ctx().data(|d| d.get_temp::<usize>(id).unwrap_or(0));
    let max = dbg.vram.len().saturating_sub(16 * 16);
    ui.add(egui::Slider::new(&mut addr, 0..=max).hexadecimal(4, false, true).text("addr"));
    addr &= !0xF; // align to 16
    ui.ctx().data_mut(|d| d.insert_temp(id, addr));

    ui.separator();
    egui::ScrollArea::vertical().max_height(280.0).show(ui, |ui| {
        for row in 0..16 {
            let line_addr = addr + row * 16;
            if line_addr >= dbg.vram.len() {
                break;
            }
            let mut hex = format!("{line_addr:05X}: ");
            let mut ascii = String::new();
            for col in 0..16 {
                let a = line_addr + col;
                if a < dbg.vram.len() {
                    let b = dbg.vram[a];
                    hex.push_str(&format!("{b:02X} "));
                    ascii.push(if (0x20..0x7F).contains(&b) { b as char } else { '.' });
                }
            }
            ui.monospace(format!("{hex}  {ascii}"));
        }
    });
}
