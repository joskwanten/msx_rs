// TMS9918A host-side state + GPU resources, extended with V9938 register
// storage. Scanline / mode-decode logic lives in vdp.wgsl; the CPU side
// writes VRAM and the TMS9918 register set (R0-R7), then uploads them
// once per frame. V9938 register storage (R8-R23, R32-R46), extended
// status (S1-S9), and the 9-bit palette live here but don't yet affect
// rendering — they collect MSX2 writes so software doesn't get garbage
// back, and so the command engine in Phase 2 has somewhere to plug in.

use crate::mlog;

// 128 KiB — V9938 maximum. The shader still only reads the first 16 KiB
// for MSX1 modes, but the command engine targets the full address space
// via R2 (display page) and R14 (extended VRAM address bit). Keeping the
// buffer at 128 KiB also means writes by the engine can't accidentally
// wrap into the visible page; software that aims a copy at 0x18000 gets
// a real 0x18000, not 0x18000 & 0x3FFF.
pub const VRAM_SIZE: usize = 128 * 1024;

/// MSX overscan canvas size — the 256×192 active area plus a 32-pixel side
/// border and 24-pixel top/bottom border filled with the backdrop colour.
/// The VDP renders into a 320×240 offscreen texture of this size, which the
/// post-process pass then upscales + letterboxes to the surface.
pub const CANVAS_W: u32 = 320;
pub const CANVAS_H: u32 = 240;

// TMS9918 fixed palette — the user's TypeScript reference, ported from
// 0xRRGGBBAA u32 literals to normalized RGBA floats. Index 0 is transparent.
// Values are in *sRGB* color space (matching how the bytes appear on a
// monitor); we convert them to linear once at startup before uploading,
// because the wgpu surface does the linear → sRGB gamma curve for us.
const PALETTE_SRGB: [[f32; 4]; 16] = [
    [0.0000, 0.0000, 0.0000, 0.0], //  0 transparent
    [0.0000, 0.0000, 0.0000, 1.0], //  1 black                 #000000
    [0.1294, 0.7843, 0.2588, 1.0], //  2 medium green          #21C842
    [0.3686, 0.8627, 0.4706, 1.0], //  3 light green           #5EDC78
    [0.3294, 0.3333, 0.9294, 1.0], //  4 dark blue             #5455ED
    [0.4902, 0.4627, 0.9882, 1.0], //  5 light blue            #7D76FC
    [0.8314, 0.3216, 0.3020, 1.0], //  6 dark red              #D4524D
    [0.2588, 0.9216, 0.9608, 1.0], //  7 cyan                  #42EBF5
    [0.9882, 0.3333, 0.3294, 1.0], //  8 medium red            #FC5554
    [1.0000, 0.4745, 0.4706, 1.0], //  9 light red             #FF7978
    [0.8314, 0.7569, 0.3294, 1.0], // 10 dark yellow           #D4C154
    [0.9020, 0.8078, 0.5020, 1.0], // 11 light yellow          #E6CE80
    [0.1294, 0.6902, 0.2314, 1.0], // 12 dark green            #21B03B
    [0.7882, 0.3569, 0.7294, 1.0], // 13 magenta               #C95BBA
    [0.8000, 0.8000, 0.8000, 1.0], // 14 gray                  #CCCCCC
    [1.0000, 1.0000, 1.0000, 1.0], // 15 white                 #FFFFFF
];

/// Convert one sRGB-encoded channel value to linear light. Standard formula
/// from the sRGB IEC 61966-2-1 spec. wgpu's `BGRA8UnormSrgb` surfaces apply
/// the inverse transform on write, so our shader output needs to be in
/// linear space for the final pixel colors to match the sRGB bytes we
/// originally specified.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Palette ready for the shader, computed once at first access.
///
/// On *native* macOS the surface format ends up being `Bgra8UnormSrgb`, so
/// wgpu applies the linear → sRGB gamma curve for us; the shader needs to
/// hand it linear values, hence the conversion.
///
/// On *web* (WebGPU) the surface format is `Bgra8Unorm` — no automatic gamma
/// curve. The browser treats the canvas's color space as sRGB by default,
/// which means the framebuffer values it reads back ARE the displayed sRGB
/// values. So we want to feed the shader the *raw* sRGB palette there.
static PALETTE: std::sync::LazyLock<[[f32; 4]; 16]> = std::sync::LazyLock::new(|| {
    // `mut` is needed on native (the iter_mut loop below); on wasm32 the
    // loop is cfg'd out and `out` is never mutated. Silence the unused-mut
    // warning specifically for that target instead of restructuring.
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut out = PALETTE_SRGB;
    #[cfg(not(target_arch = "wasm32"))]
    {
        for entry in out.iter_mut() {
            entry[0] = srgb_to_linear(entry[0]);
            entry[1] = srgb_to_linear(entry[1]);
            entry[2] = srgb_to_linear(entry[2]);
        }
    }
    out
});

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    framebuffer_size: [f32; 2],
    _pad: [u32; 2],
    // R0-R23 in 6 vec4<u32> chunks. R8-R23 are V9938-only control regs
    // (mode bits, palette pointer, command engine setup), used by the
    // wider-mask V9938 shading paths. Command-engine regs R32-R46 stay on
    // the CPU side because the engine runs there.
    regs: [[u32; 4]; 6],
    // Per-scanline snapshots of R5/R6/R11/R23, packed:
    //   bits  0..7  = R5
    //   bits  8..15 = R6
    //   bits 16..23 = R11
    //   bits 24..31 = R23
    // 256 entries × 1 u32, 4 scanlines per vec4 → 64 vec4 = 1 KiB.
    scanline_regs: [[u32; 4]; 64],
    // Second per-scanline array, packed:
    //   bits  0..7  = R2  (display page selector)
    //   bits  8..15 = R0  (mode bits M3/M4/M5 + IE2)
    //   bits 16..23 = R1  (mode bits M1/M2 + display enable)
    //   bits 24..31 = R7  (backdrop colour for border cycling)
    // Same per-vec4 layout (4 scanlines per vec4). Lets the shader's
    // mode dispatch swap rendering paths mid-frame (KV2's score area
    // is G1 text below a G4 playfield; Vampire Killer does the same).
    scanline_regs2: [[u32; 4]; 64],
    // Third per-scanline array — table-base registers that vary when
    // software flips between bitmap and tile modes in different bands:
    //   bits  0..7  = R3  (colour table base, low byte)
    //   bits  8..15 = R4  (pattern generator table base)
    //   bits 16..23 = R10 (colour table extension, G3+ high address bits)
    //   bits 24..31 = reserved
    scanline_regs3: [[u32; 4]; 64],
    palette: [[f32; 4]; 16],
}

pub struct Vdp {
    pub vram: Box<[u8; VRAM_SIZE]>,

    /// All VDP registers. R0-R7 are the TMS9918 set (still drive rendering
    /// today). R8-R23 are V9938 control registers — stored but unused by
    /// the current shader. R32-R46 are command-engine registers; storage
    /// only in Phase 1, the state machine that consumes them lands in
    /// Phase 2. Reserved slots (R24-R31, R47-R63) stay zero.
    pub regs: [u8; 64],

    // Port-protocol state. The VDP talks to the CPU through ports 0x98
    // (data) and 0x99 (control / status). Writes to 0x99 come in pairs and
    // need a one-byte latch; reads from VRAM happen via an auto-incrementing
    // 14-bit address pointer (V9938 extends to 17 bits via R14; for now we
    // stay at 14 bits because VRAM is still 16 KiB).
    vram_address: u16,

    /// S0-S9 — V9938 status registers, indexed directly. S0 keeps the
    /// TMS9918 clear-on-read semantics for VBLANK / sprite-5th / sprite-
    /// collision / 5th-sprite-number. S1-S9 are read-only snapshots of
    /// state that other subsystems update (line interrupts, command
    /// engine, palette pointer, etc.); zero until those features land.
    status: [u8; 10],

    latched_data: u8,
    has_latched_data: bool,

    /// V9938 16-entry palette, already in the vec4<f32> RGBA format the
    /// shader consumes. Initialised from the fixed [`PALETTE`] table at
    /// boot (and on cartridge swap reset) so MSX1 software gets the same
    /// look it always had. MSX2 software replaces individual entries by
    /// writing pairs of bytes to port 0x9A — the conversion from the
    /// 3-bit-per-channel hardware format to vec4 happens in
    /// `write_palette`.
    palette: [[f32; 4]; 16],

    /// Half-finished palette write — the first byte of a 0x9A pair is
    /// buffered here until the second byte arrives.
    palette_pending: Option<u8>,

    /// Per-scanline snapshots of the four registers MSX2 software most
    /// commonly rewrites from a line-interrupt handler: R5 (sprite attr
    /// table base), R6 (sprite pattern table base), R11 (SAT high bits),
    /// and R23 (vertical scroll). Captured at the start of each visible
    /// scanline in [`crate::main::State::step_frame`]; the shader reads
    /// the entry matching its `py` so split-screen-scroll and per-band
    /// sprite tables render correctly.
    pub scanline_snap: Box<[LineSnapshot; 256]>,

    /// Active CPU-driven command-engine transfer, if any. LMMC/HMMC stream
    /// pixels FROM the CPU into VRAM (via writes to R44); LMCM streams
    /// pixels FROM VRAM to the CPU (via reads of status register S7).
    /// Persists across many instructions — each R44 write / S7 read
    /// transfers exactly one pixel or byte.
    cpu_xfer: CpuXfer,
    /// Per-pixel transfer counters: X advances on each pixel, wraps to 0
    /// and increments Y when it reaches NX. When Y reaches NY the transfer
    /// completes, TR/CE clear, and `cpu_xfer` returns to `None`.
    cpu_xfer_x: u32,
    cpu_xfer_y: u32,

    /// Set when a line interrupt has fired and the CPU hasn't yet
    /// acknowledged by reading status register S1. Combined with R0[4]
    /// (IE2) by `is_irq_pending` so the CPU's IRQ line goes high.
    line_irq_pending: bool,

    vram_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

/// One scanline's worth of the registers that vary mid-frame on V9938 —
/// captured at the start of each visible scanline so the shader can
/// reproduce split-screen scrolls and per-band sprite tables. See the
/// `Vdp.scanline_snap` field for the full story.
///
/// Currently tracked:
///   r0  — mode bits M3/M4/M5 + IE2 (mode-switch per band)
///   r1  — mode bits M1/M2 + display enable + IE1
///   r2  — name/pattern table base (page selector in G4: bits 6:5)
///   r3  — colour table base (low byte)
///   r4  — pattern generator table base
///   r5  — sprite attribute table base
///   r6  — sprite pattern table base
///   r7  — backdrop colour (border colour cycling)
///   r10 — colour table base extension (G3+, address bits 16:14)
///   r11 — SAT high bits (extended VRAM addressing)
///   r23 — vertical scroll
#[derive(Copy, Clone, Default)]
pub struct LineSnapshot {
    pub r0: u8,
    pub r1: u8,
    pub r2: u8,
    pub r3: u8,
    pub r4: u8,
    pub r5: u8,
    pub r6: u8,
    pub r7: u8,
    pub r10: u8,
    pub r11: u8,
    pub r23: u8,
}

/// Active CPU-streamed command-engine transfer. The V9938 has three of
/// these; LMMC and HMMC pump CPU → VRAM (data arrives via R44 writes),
/// LMCM pumps VRAM → CPU (data is read from status register S7). Each
/// pump advances the (X, Y) counters in the parent `Vdp` and clears
/// itself when the rectangle is full.
#[derive(Copy, Clone, PartialEq)]
pub enum CpuXfer {
    /// No active CPU transfer; R44 / S7 behave as plain registers.
    None,
    /// Logical Move CPU → VRAM. Each R44 write supplies one *pixel*
    /// (low bits of `value`, masked to mode bpp). Applies the logic
    /// operation that was attached to the command's R46 byte.
    Lmmc { logic_op: u8 },
    /// High-speed Move CPU → VRAM. Each R44 write supplies one *byte*
    /// (multiple pixels in 4 bpp / 2 bpp modes) written verbatim to
    /// VRAM — no logic op, no per-pixel work.
    Hmmc,
    /// Logical Move VRAM → CPU. Each S7 read returns one pixel and
    /// auto-advances the source pointer.
    Lmcm,
}

impl Default for CpuXfer {
    fn default() -> Self {
        CpuXfer::None
    }
}

/// Convert a 3-bit-per-channel V9938 palette colour to the same vec4<f32>
/// RGBA space the shader uses for [`PALETTE`]. The native build feeds the
/// shader linear values (the surface is sRGB), the web build feeds raw
/// sRGB (surface is Unorm) — same target-conditional path the fixed
/// palette already follows.
fn v9938_to_palette_entry(r: u8, g: u8, b: u8) -> [f32; 4] {
    let r = (r & 0x07) as f32 / 7.0;
    let g = (g & 0x07) as f32 / 7.0;
    let b = (b & 0x07) as f32 / 7.0;
    #[cfg(not(target_arch = "wasm32"))]
    {
        [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0]
    }
    #[cfg(target_arch = "wasm32")]
    {
        [r, g, b, 1.0]
    }
}

impl Vdp {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let vram_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VDP VRAM"),
            size: VRAM_SIZE as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VDP Uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VDP BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VDP BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vram_buf.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VDP shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vdp.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VDP PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VDP pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            vram: Box::new([0u8; VRAM_SIZE]),
            regs: [0u8; 64],
            vram_address: 0,
            status: [0u8; 10],
            latched_data: 0,
            has_latched_data: false,
            palette: *PALETTE,
            palette_pending: None,
            scanline_snap: Box::new([LineSnapshot::default(); 256]),
            line_irq_pending: false,
            cpu_xfer: CpuXfer::None,
            cpu_xfer_x: 0,
            cpu_xfer_y: 0,
            vram_buf,
            uniform_buf,
            bind_group,
            pipeline,
        }
    }

    pub fn upload(&self, queue: &wgpu::Queue, framebuffer_size: (u32, u32)) {
        queue.write_buffer(&self.vram_buf, 0, &self.vram[..]);

        // Pack R0-R23 into 6 vec4<u32> chunks. The shader needs R10/R11
        // for G3 base-address extensions and R14 (extended VRAM bank) for
        // sprite-attribute lookup, so we send the whole control-register
        // block in one upload.
        let mut regs_packed = [[0u32; 4]; 6];
        for (i, &b) in self.regs[..24].iter().enumerate() {
            regs_packed[i / 4][i % 4] = b as u32;
        }

        // Pack the 256 per-scanline snapshots. First array holds R5/R6/
        // R11/R23 (4 bytes per scanline, 4 scanlines per vec4). Second
        // array holds R2 (1 byte per scanline) with the other 3 bytes
        // reserved.
        let mut scanline_packed = [[0u32; 4]; 64];
        let mut scanline_packed2 = [[0u32; 4]; 64];
        let mut scanline_packed3 = [[0u32; 4]; 64];
        for (line, snap) in self.scanline_snap.iter().enumerate() {
            let packed = (snap.r5 as u32)
                | ((snap.r6 as u32) << 8)
                | ((snap.r11 as u32) << 16)
                | ((snap.r23 as u32) << 24);
            scanline_packed[line / 4][line % 4] = packed;
            let packed2 = (snap.r2 as u32)
                | ((snap.r0 as u32) << 8)
                | ((snap.r1 as u32) << 16)
                | ((snap.r7 as u32) << 24);
            scanline_packed2[line / 4][line % 4] = packed2;
            let packed3 = (snap.r3 as u32)
                | ((snap.r4 as u32) << 8)
                | ((snap.r10 as u32) << 16);
            scanline_packed3[line / 4][line % 4] = packed3;
        }

        let uniforms = Uniforms {
            framebuffer_size: [framebuffer_size.0 as f32, framebuffer_size.1 as f32],
            _pad: [0; 2],
            regs: regs_packed,
            scanline_regs: scanline_packed,
            scanline_regs2: scanline_packed2,
            scanline_regs3: scanline_packed3,
            palette: self.palette,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn draw(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Raise the VBLANK flag in the status register. Called once per emulated
    /// frame to simulate the end-of-frame interrupt. The CPU will see this via
    /// `is_irq_pending` and jump to its interrupt handler if GINT is enabled.
    ///
    /// We piggy-back the sprite-status computation here: real hardware sets
    /// the 5S and Collision bits during the visible scan, and they become
    /// "final" by the time VBLANK fires. The CPU's VBLANK ISR then reads
    /// the full status byte in one go (which also clears it).
    pub fn start_vblank(&mut self) {
        self.update_sprite_status();
        self.status[0] |= 0x80;
        // S2 bit 6 (VR) = "vertical retrace in progress". Beam-racing
        // V9938 code polls this to detect the start of VBlank rather
        // than (or alongside) the IRQ. Cleared at the top of the next
        // frame's visible scan-out by `clear_vblank_flag`.
        self.status[2] |= 0x40;
    }

    /// Clear S2 bit 6 (VR), called at the start of a new frame's visible
    /// scan-out so beam-racing software sees the right "vertical
    /// retrace" edge.
    pub fn clear_vblank_flag(&mut self) {
        self.status[2] &= !0x40;
    }

    /// Compute the per-frame sprite-related status bits:
    ///   bit 6 (5S): set when ≥5 sprites occupy the same scanline. The first
    ///               such encounter (lowest Y, then lowest sprite index)
    ///               additionally writes its sprite index into bits 4..0.
    ///   bit 5 (C):  set when two sprite pixels collide at the same screen
    ///               position. No index recorded — just a boolean.
    ///
    /// Color-0 sprites participate in the scanline count (real hardware
    /// doesn't know they're invisible until after pattern lookup) but they
    /// don't contribute pixels to collision detection.
    fn update_sprite_status(&mut self) {
        let sat_base = ((self.regs[5] & 0x7F) as usize) << 7;
        let sg_base = ((self.regs[6] & 0x07) as usize) << 11;
        let r1 = self.regs[1];
        let size16 = r1 & 0x02 != 0;
        let mag = r1 & 0x01 != 0;
        let sprite_size: i32 = (if size16 { 16 } else { 8 }) << (if mag { 1 } else { 0 });

        // Gather visible sprites (Y, X, pattern, color, original index).
        // Y=0xD0 terminates the list — entries past it are suppressed.
        let mut sprites: Vec<(u8, i32, i32, u8, u8)> = Vec::with_capacity(32);
        for s in 0..32u8 {
            let entry = sat_base + (s as usize) * 4;
            let y_raw = self.vram[entry];
            if y_raw == 0xD0 {
                break;
            }
            let sy = if y_raw > 238 {
                (y_raw as i32) - 255
            } else {
                (y_raw as i32) + 1
            };
            let x_raw = self.vram[entry + 1];
            let pat = self.vram[entry + 2];
            let cbyte = self.vram[entry + 3];
            let color = cbyte & 0x0F;
            let mut sx = x_raw as i32;
            if cbyte & 0x80 != 0 {
                sx -= 32;
            }
            sprites.push((s, sx, sy, pat, color));
        }

        // 5th-sprite flag — first encounter (lowest scanline, then lowest
        // sprite index) wins and gets to write its index into bits 4..0.
        // Status bits 5 (collision) and 7 (VBLANK) must be preserved.
        for line in 0..192i32 {
            let mut count = 0u32;
            for &(s_idx, _sx, sy, _pat, _color) in &sprites {
                let dy = line - sy;
                if dy < 0 || dy >= sprite_size {
                    continue;
                }
                count += 1;
                if count == 5 {
                    if self.status[0] & 0x40 == 0 {
                        self.status[0] = (self.status[0] & 0xA0) | 0x40 | s_idx;
                    }
                    break;
                }
            }
        }

        // Collision flag — rasterize each (color != 0) sprite into a
        // screen-sized occupancy grid. Any second hit at the same pixel
        // raises the flag. We don't break early: the cost of finishing
        // the loop is small, and other emulator state stays clean.
        let mut occupancy = vec![false; 256 * 192];
        for &(_s_idx, sx, sy, pat, color) in &sprites {
            if color == 0 {
                continue;
            }
            for dy in 0..sprite_size {
                let screen_y = sy + dy;
                if !(0..192).contains(&screen_y) {
                    continue;
                }
                for dx in 0..sprite_size {
                    let screen_x = sx + dx;
                    if !(0..256).contains(&screen_x) {
                        continue;
                    }

                    let (lx, ly) = if mag {
                        ((dx as u32) >> 1, (dy as u32) >> 1)
                    } else {
                        (dx as u32, dy as u32)
                    };

                    let byte_offset = if size16 {
                        let quad_x = lx >> 3;
                        let quad_y = ly >> 3;
                        let pidx = ((pat as u32) & 0xFC) + quad_x * 2 + quad_y;
                        pidx * 8 + (ly & 7)
                    } else {
                        (pat as u32) * 8 + ly
                    };

                    let pat_byte = self.vram[sg_base + byte_offset as usize];
                    let bit = 7u32 - (lx & 7);
                    if (pat_byte >> bit) & 1 == 0 {
                        continue;
                    }

                    let idx = (screen_y as usize) * 256 + (screen_x as usize);
                    if occupancy[idx] {
                        self.status[0] |= 0x20;
                    } else {
                        occupancy[idx] = true;
                    }
                }
            }
        }
    }

    /// Whether the VDP is asserting its IRQ line. True when VBLANK has been
    /// raised AND register R1 bit 5 (GINT — generate interrupt) is set. Reading
    /// port 0x99 clears the VBLANK flag, which is the CPU's way of acknowledging.
    pub fn is_irq_pending(&self) -> bool {
        let frame_irq = self.regs[1] & 0x20 != 0 && self.status[0] & 0x80 != 0;
        // V9938 line interrupt: enabled by R0[4] (IE2). Pending bit is
        // raised by `fire_line_irq` and cleared by the CPU reading S1.
        let line_irq = self.regs[0] & 0x10 != 0 && self.line_irq_pending;
        frame_irq || line_irq
    }

    /// Snapshot the per-scanline-mutable registers at the start of a
    /// visible scanline. Called from `step_frame` before stepping the CPU
    /// for that line, so any line-interrupt handler that runs during the
    /// line uses the snapshot from the NEXT line — matches real hardware
    /// where the IRQ fires at the end of a line and the handler sets up
    /// for the next.
    pub fn snapshot_scanline(&mut self, line: usize) {
        if line < self.scanline_snap.len() {
            self.scanline_snap[line] = LineSnapshot {
                r0: self.regs[0],
                r1: self.regs[1],
                r2: self.regs[2],
                r3: self.regs[3],
                r4: self.regs[4],
                r5: self.regs[5],
                r6: self.regs[6],
                r7: self.regs[7],
                r10: self.regs[10],
                r11: self.regs[11],
                r23: self.regs[23],
            };
        }
    }

    /// Fire a V9938 line interrupt: set FH (S1 bit 0) and latch the
    /// pending flag that `is_irq_pending` ORs with the VBLANK source.
    /// CPU acknowledges by reading S1 (see `read_status`).
    pub fn fire_line_irq(&mut self) {
        self.status[1] |= 0x01;
        self.line_irq_pending = true;
    }

    /// True when the current scanline matches R19 — i.e. this is where
    /// the line interrupt should fire. Caller checks IE2 (R0 bit 4) too
    /// before actually firing.
    pub fn line_irq_target(&self, line: u8) -> bool {
        // R19 is the line counter; matches when scanline == R19.
        self.regs[19] == line
    }

    /// Wipe VRAM and registers — used on cartridge swap so the BIOS can boot
    /// the new game on a clean slate. The GPU resources stay alive (VRAM
    /// upload re-syncs them on the next frame).
    pub fn reset(&mut self) {
        self.vram.fill(0);
        self.regs = [0u8; 64];
        self.vram_address = 0;
        self.status = [0u8; 10];
        self.latched_data = 0;
        self.has_latched_data = false;
        // Reset to TMS9918 defaults so a cartridge swap doesn't leave the
        // next game starting in a black palette. MSX2 software writes its
        // own colours via 0x9A anyway; MSX1 software gets its expected
        // look from the start.
        self.palette = *PALETTE;
        self.palette_pending = None;
        *self.scanline_snap = [LineSnapshot::default(); 256];
        self.line_irq_pending = false;
        self.cpu_xfer = CpuXfer::None;
        self.cpu_xfer_x = 0;
        self.cpu_xfer_y = 0;
    }

    /// Current backdrop colour as a 4-component RGBA value in the same space
    /// as [`PALETTE`] — linear on native, sRGB on web. Used by the host to
    /// pick clear colours so window letterboxing matches the in-canvas border
    /// seamlessly. Palette index 0 (transparent) collapses to opaque black.
    pub fn backdrop_rgba(&self) -> [f32; 4] {
        let idx = (self.regs[7] & 0x0F) as usize;
        if idx == 0 {
            // Transparent → black: a clear colour with alpha 0 would let the
            // browser's page background show through on the web build.
            #[cfg(not(target_arch = "wasm32"))]
            {
                [srgb_to_linear(0.0), srgb_to_linear(0.0), srgb_to_linear(0.0), 1.0]
            }
            #[cfg(target_arch = "wasm32")]
            {
                [0.0, 0.0, 0.0, 1.0]
            }
        } else {
            PALETTE[idx]
        }
    }

    /// Hand-crafted Screen 2 state: eight vertical colored bars in the middle of the screen,
    /// on a dark-blue backdrop. Kept around as a CPU-less rendering check.
    #[allow(dead_code)]
    pub fn load_demo(&mut self) {
        // R0 = 0x02  M3 = 1 → Screen 2
        // R1 = 0xC0  16 KiB, display enabled, no IRQ, 8×8 sprites no mag
        // R2 = 0x06  name table  = 0x1800
        // R3 = 0xFF  color table = 0x2000
        // R4 = 0x03  pattern tab = 0x0000
        // R5 = 0x36  sprite attr = 0x1B00 (unused in milestone 1)
        // R6 = 0x07  sprite pats = 0x3800 (unused in milestone 1)
        // R7 = 0x04  backdrop    = dark blue
        // Only set R0-R7; R8+ remain at their reset values (zero).
        self.regs[..8].copy_from_slice(&[0x02, 0xC0, 0x06, 0xFF, 0x03, 0x36, 0x07, 0x04]);

        self.vram.fill(0);

        const PT_BASE: usize = 0x0000;
        const NT_BASE: usize = 0x1800;
        const CT_BASE: usize = 0x2000;

        // Tiles 1..=8 in every bank: solid 8×8 (all rows = 0xFF).
        for bank in 0..3 {
            for tile in 1..=8 {
                let off = PT_BASE + bank * 256 * 8 + tile * 8;
                for r in 0..8 {
                    self.vram[off + r] = 0xFF;
                }
            }
        }

        // Name table: rows 8..15 (middle third) → 4 cols per bar, tile = col/4 + 1.
        // Other rows stay tile 0 (transparent → backdrop).
        for row in 8..16 {
            for col in 0..32 {
                self.vram[NT_BASE + row * 32 + col] = ((col / 4) + 1) as u8;
            }
        }

        // Color table for bank 1 (middle third): tile N gets bar color N as fg.
        let bank1_ct = CT_BASE + 256 * 8;
        let bar_colors: [u8; 8] = [2, 3, 5, 7, 8, 11, 13, 15];
        for (i, &fg) in bar_colors.iter().enumerate() {
            let tile = i + 1;
            for r in 0..8 {
                self.vram[bank1_ct + tile * 8 + r] = fg << 4;
            }
        }
    }
}

// --- Port I/O ---------------------------------------------------------------
//
// The TMS9918 talks to the CPU through two I/O ports:
//
//   0x98 — data port: read or write a single byte at the current VRAM address,
//          which auto-increments (and wraps within the 14-bit space).
//
//   0x99 — control / status port:
//          - Reads return the status register (bit 7 = VBLANK, bit 6 = 5th
//            sprite, bit 5 = sprite collision, bits 4-0 = sprite number),
//            then clear it.
//          - Writes come in PAIRS. The first byte is latched. The second
//            byte's two top bits select what to do:
//              0b10xx_xxxx → register write: low 3 bits = register number,
//                            data = the latched first byte.
//              0b01xx_xxxx → VRAM-write address: address = (low 6 bits << 8)
//                            | latched.
//              0b00xx_xxxx → VRAM-read address: same formula, same pointer.
//
//   Either kind of read (0x98 or 0x99) clears the latch, as does a data
//   write to 0x98 — only a write to 0x99 actually drives the latch.

impl crate::bus::Io for Vdp {
    fn in8(&mut self, port: u8) -> u8 {
        // Real hardware resets the latch on any read of either control
        // port (0x98 or 0x99). 0x9A/0x9B reads aren't defined on the
        // V9938, so we don't touch the latch there.
        match port {
            0x98 => {
                self.has_latched_data = false;
                self.read_data()
            }
            0x99 => {
                self.has_latched_data = false;
                self.read_status()
            }
            _ => 0xFF,
        }
    }

    fn out8(&mut self, port: u8, value: u8) {
        match port {
            0x98 => {
                self.has_latched_data = false;
                self.write_data(value);
            }
            0x99 => self.write_control(value),
            // V9938 only — silently ignored on MSX1 software because it
            // never writes to these ports. MSX2 software sets up the
            // palette via 0x9A and uses 0x9B for indirect register access
            // (through R17) so it can drive command-engine setups in
            // tight loops without re-selecting the register each time.
            0x9A => self.write_palette(value),
            0x9B => self.write_indirect(value),
            _ => {}
        }
    }
}

impl Vdp {
    fn read_data(&mut self) -> u8 {
        let addr = self.full_vram_addr();
        let value = self.vram[addr];
        self.advance_vram_pointer();
        value
    }

    fn read_status(&mut self) -> u8 {
        // V9938 routes 0x99 reads through the status-register selector
        // in R15 (low nibble). The TMS9918 had only one status register,
        // which lives at index 0 here — and R15 defaults to 0, so MSX1
        // software keeps getting S0 like before.
        //
        // S0 has clear-on-read semantics on the latch bits (VBLANK,
        // sprite-5th, sprite-collision); S1 has clear-on-read for FH
        // (line-interrupt flag) and also acknowledges the pending IRQ
        // so the CPU stops re-entering its handler. S2-S9 are sampled
        // state — read without side effects.
        let sel = (self.regs[15] & 0x0F) as usize;
        match sel {
            0 => std::mem::replace(&mut self.status[0], 0),
            1 => {
                let v = self.status[1];
                self.status[1] &= !0x01; // FH cleared on read
                self.line_irq_pending = false;
                v
            }
            // S7 has a side effect during LMCM: each read returns the
            // next pixel and advances the source pointer. Outside of an
            // LMCM transfer it's just a plain status register (e.g. the
            // colour latched by POINT).
            7 => self.pump_cpu_xfer_read(),
            n if n < self.status.len() => self.status[n],
            _ => 0xFF, // S10-S15 unused on V9938
        }
    }

    fn write_data(&mut self, value: u8) {
        let addr = self.full_vram_addr();
        self.vram[addr] = value;
        self.advance_vram_pointer();
    }

    /// V9938 17-bit VRAM address = R14[2:0] << 14 | vram_address[13:0].
    /// R14 defaults to 0 at reset, so TMS9918 software keeps reading and
    /// writing the first 16 KiB exactly as before (R14 = 0 → base = 0).
    /// MSX2 software writes R14 to access the higher banks (0x4000–0x1FFFF).
    fn full_vram_addr(&self) -> usize {
        let bank = (self.regs[14] & 0x07) as u32;
        let addr = (bank << 14) | (self.vram_address as u32 & 0x3FFF);
        (addr as usize) & (VRAM_SIZE - 1)
    }

    /// Increment the low 14-bit pointer, wrapping back to 0 at 0x3FFF.
    /// We do NOT auto-increment R14 on wrap — that breaks TMS9918-style
    /// software that does long sequential writes inside one 16 KiB bank
    /// and expects the address to loop back to 0 within the same bank.
    /// MSX1 software (and MSX1-compat-mode MSX2 software like Kings
    /// Valley 2's Screen 1 paths) hits this constantly. MSX2 software
    /// that wants to write across banks sets R14 explicitly before each
    /// batch — which is what the V9938 documentation specifies.
    fn advance_vram_pointer(&mut self) {
        self.vram_address = self.vram_address.wrapping_add(1) & 0x3FFF;
    }

    fn write_control(&mut self, value: u8) {
        if !self.has_latched_data {
            self.latched_data = value;
            self.has_latched_data = true;
            return;
        }

        self.has_latched_data = false;

        if value & 0x80 != 0 {
            // Register write. TMS9918 only used bits 0..2 (R0-R7); V9938
            // uses bits 0..5 (R0-R47). MSX1 software writes zero in the
            // upper 3 bits anyway, so masking 0x3F is backward-compatible.
            let register = (value & 0x3F) as usize;
            self.write_register(register, self.latched_data);
        } else {
            // VRAM address setup. Bit 6 distinguishes read-intent from
            // write-intent on real hardware, but a single pointer serves
            // both. V9938 extends the address to 17 bits via R14 — we
            // ignore that for now because VRAM is still 16 KiB.
            self.vram_address = (((value & 0x3F) as u16) << 8) | (self.latched_data as u16);
        }
    }

    /// Set one VDP register with a logging hook. Called from the two
    /// register-write paths (direct via 0x99, indirect via 0x9B → R17).
    /// Writing to R46 (the command register) triggers the command engine.
    fn write_register(&mut self, reg: usize, value: u8) {
        if reg >= self.regs.len() {
            return;
        }
        mlog!(VDP_REG, "R{} = 0x{:02X}", reg, value);
        self.regs[reg] = value;
        if reg == 46 {
            self.execute_command();
        }
        // R44 is the command-engine colour register; writing to it during
        // an active LMMC / HMMC streams one more pixel / byte through the
        // transfer pipeline.
        if reg == 44 {
            self.pump_cpu_xfer_write(value);
        }
    }

    /// Port 0x9A — V9938 palette write. Each palette entry takes two
    /// bytes; the first is buffered until the second arrives. After the
    /// pair lands, R16 (the palette pointer) auto-increments through 16
    /// entries.
    ///
    /// Byte format on real hardware:
    ///   byte 1: `0 R R R 0 B B B` — red in bits 6-4, blue in bits 2-0
    ///   byte 2: `0 0 0 0 0 G G G` — green in bits 2-0
    fn write_palette(&mut self, value: u8) {
        match self.palette_pending {
            None => {
                // First byte: latch and wait for the matching second one.
                self.palette_pending = Some(value);
            }
            Some(first) => {
                let r = (first >> 4) & 0x07;
                let b = first & 0x07;
                let g = value & 0x07;
                let ptr = (self.regs[16] & 0x0F) as usize;
                self.palette[ptr] = v9938_to_palette_entry(r, g, b);
                mlog!(VDP_PAL, "palette[{}] = R{} G{} B{}", ptr, r, g, b);
                self.regs[16] = (self.regs[16].wrapping_add(1)) & 0x0F;
                self.palette_pending = None;
            }
        }
    }

    /// Port 0x9B — V9938 indirect register write. The register number
    /// comes from R17 (bits 0-5), and bit 7 of R17 suppresses the
    /// auto-increment. Lets MSX2 software stream values into one register
    /// in a tight loop (e.g. pumping VRAM data through the command
    /// engine's color register) without re-selecting it every time.
    fn write_indirect(&mut self, value: u8) {
        let r17 = self.regs[17];
        let reg = (r17 & 0x3F) as usize;
        let auto_inc_disabled = r17 & 0x80 != 0;
        self.write_register(reg, value);
        if !auto_inc_disabled {
            // Only the low 6 bits increment; bit 7 (auto-inc-disable
            // flag) is preserved.
            self.regs[17] = (r17 & 0x80) | (r17.wrapping_add(1) & 0x3F);
        }
    }
}

// --- V9938 command engine ---------------------------------------------------
//
// The CPU sets up an operation in R32-R45 and writes the command code (plus
// logic-op nibble) to R46. Real hardware then runs the operation across many
// cycles; we run it to completion synchronously and clear the CE bit (S2[0])
// before returning, which is what most software expects anyway because
// almost all of it polls CE before issuing the next command.
//
// Pixel coordinates are interpreted relative to the current screen mode's
// bitmap page. For now `page_base` is hard-coded to 0 — software that
// double-buffers via R2 will run into this; it lands in Phase 3 alongside
// real R2 / R14 routing.

/// Layout of the bitmap page for the current screen mode. Returned by
/// `pixel_layout()` when the engine can operate; `None` for TMS9918 modes
/// and the V9938 text/sprite modes where the command engine is undefined.
#[derive(Copy, Clone)]
#[allow(dead_code)] // `height` is doc-only — command engine clamps Y by
                    // VRAM extent, not visible rows; kept here for the
                    // mode-info comment and for when sprite/cursor code
                    // needs the visible-area limit.
struct PixelLayout {
    /// Bits per pixel: 2 (G5), 4 (G4/G6), or 8 (G7).
    bpp: u32,
    /// Bytes per row in VRAM.
    pitch: u32,
    /// Pixels per row — for clamping command-engine coordinates.
    width: u32,
    /// Visible rows — informational. The command engine deliberately
    /// does NOT clamp Y to this, because V9938 software stages graphics
    /// in off-screen pages and transfers them in via LMMM/HMMM.
    height: u32,
}

impl PixelLayout {
    /// Pixels packed into one VRAM byte.
    fn pixels_per_byte(&self) -> u32 {
        8 / self.bpp
    }

    /// Mask covering one pixel's worth of bits within a byte.
    fn pixel_mask(&self) -> u8 {
        (1u8 << self.bpp) - 1
    }
}

/// Borrowed-state context the command engine works on. Everything the
/// engine reads or writes flows through these three slices: registers
/// (R0-R63 for mode + setup + command), the full VRAM buffer (128 KiB),
/// and the status register set (S0-S9 for completion / engine-busy bits).
///
/// Splitting state out of `Vdp` lets tests construct a context from local
/// arrays without needing a wgpu device — `Vdp::execute_command` is now a
/// thin wrapper that builds a `CmdCtx` from `&mut self`.
struct CmdCtx<'a> {
    regs: &'a mut [u8; 64],
    vram: &'a mut [u8],
    status: &'a mut [u8; 10],
    /// CPU-streamed transfer state — initialised by the LMMC/HMMC/LMCM
    /// command handlers and stepped by `pump_write` (called from R44
    /// writes) or `pump_read` (called from S7 reads).
    cpu_xfer: &'a mut CpuXfer,
    cpu_xfer_x: &'a mut u32,
    cpu_xfer_y: &'a mut u32,
}

impl<'a> CmdCtx<'a> {
    /// Decode the current screen mode into a command-engine pixel layout.
    /// Returns `None` for modes the command engine isn't defined on
    /// (TMS9918 modes + G3 text/sprite mode).
    fn pixel_layout(&self) -> Option<PixelLayout> {
        // V9938 mode-bit positions:
        //   M5 = R0[3], M4 = R0[2], M3 = R0[1]
        //   M2 = R1[3], M1 = R1[4]
        let m5 = (self.regs[0] >> 3) & 1;
        let m4 = (self.regs[0] >> 2) & 1;
        let m3 = (self.regs[0] >> 1) & 1;
        let m2 = (self.regs[1] >> 3) & 1;
        let m1 = (self.regs[1] >> 4) & 1;
        // Pack as MSB→LSB: M5 M4 M3 M2 M1.
        let mode = (m5 << 4) | (m4 << 3) | (m3 << 2) | (m2 << 1) | m1;
        match mode {
            0b01100 => Some(PixelLayout { bpp: 4, pitch: 128, width: 256, height: 212 }), // G4 (Screen 5)
            0b10000 => Some(PixelLayout { bpp: 2, pitch: 128, width: 512, height: 212 }), // G5 (Screen 6)
            0b10100 => Some(PixelLayout { bpp: 4, pitch: 256, width: 512, height: 212 }), // G6 (Screen 7)
            0b11100 => Some(PixelLayout { bpp: 8, pitch: 256, width: 256, height: 212 }), // G7 (Screen 8)
            _ => None,
        }
    }

    /// Build a u32 from a (lo, hi) register pair. Command engine arguments
    /// live in `R32-R45` as little-endian halves of 9/10-bit values; the
    /// higher bits beyond the field width are reserved and ignored.
    fn cmd_word(&self, lo_reg: usize) -> u32 {
        u16::from_le_bytes([self.regs[lo_reg], self.regs[lo_reg + 1]]) as u32
    }

    /// Read one pixel from VRAM at screen coordinate `(x, y)`. Out-of-
    /// bounds reads return 0, matching what unmapped VRAM would give us.
    fn read_pixel(&self, layout: &PixelLayout, x: u32, y: u32) -> u8 {
        let addr = (y * layout.pitch + x / layout.pixels_per_byte()) as usize;
        if addr >= self.vram.len() {
            return 0;
        }
        let byte = self.vram[addr];
        match layout.bpp {
            8 => byte,
            4 => {
                // High nibble is the leftmost pixel.
                let shift = (1 - (x & 1)) * 4;
                (byte >> shift) & 0x0F
            }
            2 => {
                // Four pixels per byte, leftmost in bits 7:6.
                let shift = (3 - (x & 3)) * 2;
                (byte >> shift) & 0x03
            }
            _ => 0,
        }
    }

    /// Write one pixel to VRAM at screen coordinate `(x, y)`. The pixel
    /// value is masked to the mode's bpp before writing.
    fn write_pixel(&mut self, layout: &PixelLayout, x: u32, y: u32, color: u8) {
        let addr = (y * layout.pitch + x / layout.pixels_per_byte()) as usize;
        if addr >= self.vram.len() {
            return;
        }
        let color = color & layout.pixel_mask();
        match layout.bpp {
            8 => self.vram[addr] = color,
            4 => {
                // High nibble (x even) shifts by 4; low nibble shifts by 0.
                // Mask of bits to *clear* lives at the same shifted position.
                let shift = (1 - (x & 1)) * 4;
                let mask = 0x0F_u8 << shift;
                self.vram[addr] = (self.vram[addr] & !mask) | (color << shift);
            }
            2 => {
                let shift = (3 - (x & 3)) * 2;
                let mask = 0x03_u8 << shift;
                self.vram[addr] = (self.vram[addr] & !mask) | (color << shift);
            }
            _ => {}
        }
    }

    /// Decode a command write to R46 and run it synchronously. The high
    /// nibble of R46 is the command opcode, the low nibble is the logic
    /// operation for the L-family commands.
    fn execute(&mut self) {
        // Mark engine busy — even though we complete synchronously, some
        // setup code reads CE right after a command write to confirm it
        // landed. Clearing happens at the end.
        self.status[2] |= 0x01;

        let cmd = self.regs[46] >> 4;
        let logic_op = self.regs[46] & 0x0F;

        match cmd {
            0x0 => self.cmd_stop(),
            0x4 => self.cmd_point(),
            0x5 => self.cmd_pset(logic_op),
            0x6 => self.cmd_srch(),
            0x7 => self.cmd_line(logic_op),
            0x8 => self.cmd_lmmv(logic_op),
            0x9 => self.cmd_lmmm(logic_op),
            0xA => self.cmd_lmcm(),
            0xB => self.cmd_lmmc(logic_op),
            0xF => self.cmd_hmmc(),
            0xC => self.cmd_hmmv(),
            0xD => self.cmd_hmmm(),
            0xE => self.cmd_ymmm(),
            other => {
                mlog!(VDP_CMD, "unimplemented command 0x{:X}", other);
            }
        }

        // Command done — clear CE *unless* we just set up a CPU transfer,
        // which stays "executing" until R44 streaming completes / S7
        // streaming drains. The pump methods clear CE on the last byte.
        if *self.cpu_xfer == CpuXfer::None {
            self.status[2] &= !0x01;
        }
    }

    fn cmd_stop(&mut self) {
        mlog!(VDP_CMD, "STOP");
        // Abort any active CPU transfer so the next R44 write / S7 read
        // doesn't pump into stale state. The CE-clear at the end of
        // `execute` then takes effect because cpu_xfer is back to None.
        *self.cpu_xfer = CpuXfer::None;
        *self.cpu_xfer_x = 0;
        *self.cpu_xfer_y = 0;
        // Status bits: clear TR (transfer ready). CE clears in `execute`.
        self.status[2] &= !0x80;
    }

    /// HMMV — High-speed Move VDP to VRAM. Fill a byte-aligned rectangle
    /// with the color byte. Faster than LMMV because it doesn't read-
    /// modify-write per pixel, just writes whole bytes.
    fn cmd_hmmv(&mut self) {
        let Some(layout) = self.pixel_layout() else {
            mlog!(VDP_CMD, "HMMV: no command-capable mode");
            return;
        };
        let dx = self.cmd_word(36) & 0x1FF;
        let dy = self.cmd_word(38) & 0x3FF;
        let nx = (self.cmd_word(40) & 0x3FF).max(1);
        let ny = (self.cmd_word(42) & 0x3FF).max(1);
        let clr = self.regs[44];
        let arg = self.regs[45];
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };

        let ppb = layout.pixels_per_byte();
        // Convert pixel-X to byte-X. HMMV is byte-granular.
        let dx_byte = (dx as i32) / (ppb as i32);
        let nx_bytes = nx.div_ceil(ppb) as i32;

        mlog!(VDP_CMD, "HMMV dst=({},{}) {}x{} clr=0x{:02X} arg=0x{:02X}",
              dx, dy, nx, ny, clr, arg);

        // Y is bound by the VRAM extent, NOT by layout.height — V9938
        // software composes graphics in off-screen pages (Y > visible
        // height) and then transfers them to the displayed page via
        // LMMM/HMMM. Same for the read/write commands below.
        for iy in 0..(ny as i32) {
            let y = dy as i32 + iy * diy;
            if y < 0 {
                continue;
            }
            for ix in 0..nx_bytes {
                let bx = dx_byte + ix * dix;
                if bx < 0 || bx as u32 >= layout.pitch {
                    continue;
                }
                let addr = (y as u32 * layout.pitch + bx as u32) as usize;
                if addr < self.vram.len() {
                    self.vram[addr] = clr;
                }
            }
        }
    }

    /// HMMM — High-speed Move VRAM to VRAM. Byte-aligned copy.
    fn cmd_hmmm(&mut self) {
        let Some(layout) = self.pixel_layout() else { return };
        let sx = self.cmd_word(32) & 0x1FF;
        let sy = self.cmd_word(34) & 0x3FF;
        let dx = self.cmd_word(36) & 0x1FF;
        let dy = self.cmd_word(38) & 0x3FF;
        let nx = (self.cmd_word(40) & 0x3FF).max(1);
        let ny = (self.cmd_word(42) & 0x3FF).max(1);
        let arg = self.regs[45];
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };

        let ppb = layout.pixels_per_byte();
        let sx_byte = (sx as i32) / (ppb as i32);
        let dx_byte = (dx as i32) / (ppb as i32);
        let nx_bytes = nx.div_ceil(ppb) as i32;

        mlog!(VDP_CMD, "HMMM src=({},{}) dst=({},{}) {}x{}",
              sx, sy, dx, dy, nx, ny);

        for iy in 0..(ny as i32) {
            let sy_now = sy as i32 + iy * diy;
            let dy_now = dy as i32 + iy * diy;
            if sy_now < 0 || dy_now < 0 { continue; }
            // No layout.height clamp — source/destination may live in
            // off-screen pages. The inner byte-address bounds check below
            // handles VRAM overflow.
            for ix in 0..nx_bytes {
                let sbx = sx_byte + ix * dix;
                let dbx = dx_byte + ix * dix;
                if sbx < 0 || dbx < 0 { continue; }
                if sbx as u32 >= layout.pitch || dbx as u32 >= layout.pitch {
                    continue;
                }
                let src_addr = (sy_now as u32 * layout.pitch + sbx as u32) as usize;
                let dst_addr = (dy_now as u32 * layout.pitch + dbx as u32) as usize;
                if src_addr < self.vram.len() && dst_addr < self.vram.len() {
                    self.vram[dst_addr] = self.vram[src_addr];
                }
            }
        }
    }

    /// LMMV — Logical Move VDP to VRAM. Fill a rectangle pixel-by-pixel,
    /// applying a logic op (and optional transparent skip) per pixel.
    /// Slower than HMMV but respects pixel boundaries and lets games do
    /// "draw a colored shape but skip transparent pixels" in one go.
    fn cmd_lmmv(&mut self, logic_op: u8) {
        let Some(layout) = self.pixel_layout() else { return };
        let dx = self.cmd_word(36) & 0x1FF;
        let dy = self.cmd_word(38) & 0x3FF;
        let nx = (self.cmd_word(40) & 0x3FF).max(1);
        let ny = (self.cmd_word(42) & 0x3FF).max(1);
        let clr = self.regs[44] & layout.pixel_mask();
        let arg = self.regs[45];
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };

        mlog!(VDP_CMD, "LMMV dst=({},{}) {}x{} clr={} op=0x{:X}",
              dx, dy, nx, ny, clr, logic_op);

        for iy in 0..(ny as i32) {
            let y = dy as i32 + iy * diy;
            if y < 0 { continue; }
            for ix in 0..(nx as i32) {
                let x = dx as i32 + ix * dix;
                if x < 0 || x as u32 >= layout.width { continue; }
                let dst = self.read_pixel(&layout, x as u32, y as u32);
                let new = apply_logic_op(clr, dst, logic_op, layout.pixel_mask());
                self.write_pixel(&layout, x as u32, y as u32, new);
            }
        }
    }

    /// LMMM — Logical Move VRAM to VRAM. Pixel-by-pixel copy with logic
    /// op applied between source pixel and destination pixel.
    fn cmd_lmmm(&mut self, logic_op: u8) {
        let Some(layout) = self.pixel_layout() else { return };
        let sx = self.cmd_word(32) & 0x1FF;
        let sy = self.cmd_word(34) & 0x3FF;
        let dx = self.cmd_word(36) & 0x1FF;
        let dy = self.cmd_word(38) & 0x3FF;
        let nx = (self.cmd_word(40) & 0x3FF).max(1);
        let ny = (self.cmd_word(42) & 0x3FF).max(1);
        let arg = self.regs[45];
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };

        mlog!(VDP_CMD, "LMMM src=({},{}) dst=({},{}) {}x{} op=0x{:X}",
              sx, sy, dx, dy, nx, ny, logic_op);

        for iy in 0..(ny as i32) {
            let sy_now = sy as i32 + iy * diy;
            let dy_now = dy as i32 + iy * diy;
            if sy_now < 0 || dy_now < 0 { continue; }
            // No layout.height clamp — see HMMV / HMMM. Off-screen pages
            // are routinely used as source for sprite/tile composition.
            for ix in 0..(nx as i32) {
                let sx_now = sx as i32 + ix * dix;
                let dx_now = dx as i32 + ix * dix;
                if sx_now < 0 || dx_now < 0 { continue; }
                if sx_now as u32 >= layout.width || dx_now as u32 >= layout.width { continue; }
                let src = self.read_pixel(&layout, sx_now as u32, sy_now as u32);
                let dst = self.read_pixel(&layout, dx_now as u32, dy_now as u32);
                let new = apply_logic_op(src, dst, logic_op, layout.pixel_mask());
                self.write_pixel(&layout, dx_now as u32, dy_now as u32, new);
            }
        }
    }

    /// PSET — set a single pixel at (DX, DY) to CLR, applying logic op.
    fn cmd_pset(&mut self, logic_op: u8) {
        let Some(layout) = self.pixel_layout() else { return };
        let dx = self.cmd_word(36) & 0x1FF;
        let dy = self.cmd_word(38) & 0x3FF;
        let clr = self.regs[44] & layout.pixel_mask();
        if dx as u32 >= layout.width { return; }
        // No DY clamp — write_pixel's internal byte-address bound check
        // catches truly-out-of-VRAM cases; off-screen rows are valid
        // destinations for staging.
        let dst = self.read_pixel(&layout, dx as u32, dy as u32);
        let new = apply_logic_op(clr, dst, logic_op, layout.pixel_mask());
        self.write_pixel(&layout, dx as u32, dy as u32, new);
        mlog!(VDP_CMD, "PSET ({},{}) = {} op=0x{:X}", dx, dy, clr, logic_op);
    }

    /// POINT — read pixel at (SX, SY), return value in S7.
    fn cmd_point(&mut self) {
        let Some(layout) = self.pixel_layout() else { return };
        let sx = self.cmd_word(32) & 0x1FF;
        let sy = self.cmd_word(34) & 0x3FF;
        if sx as u32 >= layout.width {
            self.status[7] = 0;
            return;
        }
        // No SY clamp — see PSET.
        let value = self.read_pixel(&layout, sx as u32, sy as u32);
        self.status[7] = value;
        mlog!(VDP_CMD, "POINT ({},{}) = {}", sx, sy, value);
    }

    /// LINE — Bresenham-style line from (DX, DY) along the major axis,
    /// applying a logic op per pixel.
    ///
    /// Register layout (per V9938 manual):
    ///   DX, DY  → start point (R36..R39)
    ///   NX      → major-axis length (R40/R41), # pixels along major
    ///   NY      → minor-axis counter (R42/R43), # of minor-axis steps
    ///   CLR     → pixel color (R44)
    ///   ARG     → direction + axis flags (R45)
    ///                bit 0 = MAJ (0 = X is major, 1 = Y is major)
    ///                bit 2 = DIX (X direction: 0 = +, 1 = −)
    ///                bit 3 = DIY (Y direction: 0 = +, 1 = −)
    ///
    /// Classic accumulator: every step along the major axis adds NY to an
    /// accumulator; when it exceeds NX the minor axis steps and the
    /// accumulator wraps. Produces the standard MSX2 line.
    fn cmd_line(&mut self, logic_op: u8) {
        let Some(layout) = self.pixel_layout() else { return };
        let dx = (self.cmd_word(36) & 0x1FF) as i32;
        let dy = (self.cmd_word(38) & 0x3FF) as i32;
        let nx = (self.cmd_word(40) & 0x3FF) as i32;
        let ny = (self.cmd_word(42) & 0x3FF) as i32;
        let clr = self.regs[44] & layout.pixel_mask();
        let arg = self.regs[45];
        let maj_is_y = arg & 0x01 != 0;
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };

        mlog!(VDP_CMD, "LINE ({},{}) maj={} nx={} ny={} clr={} dix={} diy={}",
              dx, dy, if maj_is_y { "Y" } else { "X" }, nx, ny, clr, dix, diy);

        let mut x = dx;
        let mut y = dy;
        let mut acc: i32 = 0;
        // NX is "steps along major axis" — endpoint inclusive.
        let steps = nx.max(1);
        for _ in 0..=steps {
            // Y is unclamped beyond the visible page — LINE may target
            // off-screen rows for staging. write_pixel handles VRAM bounds.
            if x >= 0 && y >= 0 && (x as u32) < layout.width {
                let dst = self.read_pixel(&layout, x as u32, y as u32);
                let new = apply_logic_op(clr, dst, logic_op, layout.pixel_mask());
                self.write_pixel(&layout, x as u32, y as u32, new);
            }
            // Step along major axis every iteration.
            if maj_is_y {
                y += diy;
            } else {
                x += dix;
            }
            // Accumulator decides if we step the minor axis this iteration.
            acc += ny;
            if acc >= steps {
                acc -= steps;
                if maj_is_y {
                    x += dix;
                } else {
                    y += diy;
                }
            }
        }
    }

    /// SRCH — Search along row SY starting at SX for the first pixel that
    /// either matches or doesn't match CLR (depending on the EQ flag),
    /// moving in the DIX direction. Outcome:
    ///   - BD bit (S2[4]) is set when the border is hit before a match
    ///   - On match, S8/S9 = X coordinate where the match was found
    fn cmd_srch(&mut self) {
        let Some(layout) = self.pixel_layout() else { return };
        let sx = (self.cmd_word(32) & 0x1FF) as i32;
        let sy = (self.cmd_word(34) & 0x3FF) as i32;
        let clr = self.regs[44] & layout.pixel_mask();
        let arg = self.regs[45];
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let eq = arg & 0x02 != 0; // 0 = look for !=, 1 = look for ==

        mlog!(VDP_CMD, "SRCH from ({},{}) dix={} clr={} eq={}",
              sx, sy, dix, clr, eq);

        // Clear previous match state (BD) before we start.
        self.status[2] &= !0x10;

        let mut x = sx;
        loop {
            if x < 0 || (x as u32) >= layout.width {
                // Hit the border without finding a match.
                self.status[2] |= 0x10; // BD
                mlog!(VDP_CMD, "SRCH: border at X={}", x);
                break;
            }
            let pixel = self.read_pixel(&layout, x as u32, sy as u32);
            let matches = if eq { pixel == clr } else { pixel != clr };
            if matches {
                let xu = x as u32;
                self.status[8] = xu as u8;
                self.status[9] = ((xu >> 8) & 0x01) as u8;
                mlog!(VDP_CMD, "SRCH found at X={}", xu);
                break;
            }
            x += dix;
        }
    }

    /// YMMM — Move VRAM bytes vertically. Used for fast scrolling. Copies
    /// the entire row from (DX..DX+vram-edge-of-row, SY) to (DX..,DY),
    /// for NY rows. Only DY direction matters; DIX flag is unused per spec.
    ///
    /// We honour DIY: positive scrolls down, negative scrolls up.
    fn cmd_ymmm(&mut self) {
        let Some(layout) = self.pixel_layout() else { return };
        let sy = (self.cmd_word(34) & 0x3FF) as i32;
        let dx = (self.cmd_word(36) & 0x1FF) as i32;
        let dy = (self.cmd_word(38) & 0x3FF) as i32;
        let ny = (self.cmd_word(42) & 0x3FF).max(1) as i32;
        let arg = self.regs[45];
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };

        mlog!(VDP_CMD, "YMMM sy={} dx={} dy={} ny={} diy={}", sy, dx, dy, ny, diy);

        let ppb = layout.pixels_per_byte();
        let dx_byte = (dx as u32) / ppb;
        let row_remaining = layout.pitch.saturating_sub(dx_byte);

        for iy in 0..ny {
            let sy_now = sy + iy * diy;
            let dy_now = dy + iy * diy;
            if sy_now < 0 || dy_now < 0 { continue; }
            // Y is unclamped — YMMM commonly scrolls off-screen rows.
            let src_off = (sy_now as u32 * layout.pitch + dx_byte) as usize;
            let dst_off = (dy_now as u32 * layout.pitch + dx_byte) as usize;
            for b in 0..row_remaining as usize {
                if src_off + b >= self.vram.len() || dst_off + b >= self.vram.len() {
                    break;
                }
                self.vram[dst_off + b] = self.vram[src_off + b];
            }
        }
    }

    /// LMMC — Logical Move CPU → VRAM. The command itself only sets up
    /// the destination rectangle and the active transfer state; the
    /// actual pixels arrive one-at-a-time via subsequent CPU writes to
    /// R44, each pumped through `pump_write`.
    fn cmd_lmmc(&mut self, logic_op: u8) {
        mlog!(VDP_CMD, "LMMC start logic_op=0x{:X}", logic_op);
        *self.cpu_xfer = CpuXfer::Lmmc { logic_op };
        *self.cpu_xfer_x = 0;
        *self.cpu_xfer_y = 0;
        // TR = transfer ready (CPU may write the first pixel).
        // CE = command executing (stays set until the rectangle is full).
        self.status[2] |= 0x81;
    }

    /// HMMC — High-speed Move CPU → VRAM. Like LMMC but byte-granular:
    /// each CPU byte is written directly to VRAM (no pixel masking, no
    /// logic op). Used to stream pre-packed bitmap data fast.
    fn cmd_hmmc(&mut self) {
        mlog!(VDP_CMD, "HMMC start");
        *self.cpu_xfer = CpuXfer::Hmmc;
        *self.cpu_xfer_x = 0;
        *self.cpu_xfer_y = 0;
        self.status[2] |= 0x81;
    }

    /// LMCM — Logical Move VRAM → CPU. CPU drains pixels by reading
    /// status register S7; each read returns the current pixel and
    /// advances the source pointer.
    fn cmd_lmcm(&mut self) {
        mlog!(VDP_CMD, "LMCM start");
        *self.cpu_xfer = CpuXfer::Lmcm;
        *self.cpu_xfer_x = 0;
        *self.cpu_xfer_y = 0;
        // Pre-load the first pixel into S7 so the CPU's first S7 read
        // returns valid data before any pump.
        self.preload_lmcm_pixel();
        self.status[2] |= 0x81;
    }

    /// Compute the next LMCM source pixel and stash it in S7.
    fn preload_lmcm_pixel(&mut self) {
        let Some(layout) = self.pixel_layout() else {
            return;
        };
        let sx = (self.cmd_word(32) & 0x1FF) as i32;
        let sy = (self.cmd_word(34) & 0x3FF) as i32;
        let arg = self.regs[45];
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };
        let x = sx + (*self.cpu_xfer_x as i32) * dix;
        let y = sy + (*self.cpu_xfer_y as i32) * diy;
        if x >= 0 && y >= 0 && (x as u32) < layout.width {
            self.status[7] = self.read_pixel(&layout, x as u32, y as u32);
        } else {
            self.status[7] = 0;
        }
    }

    /// Advance one step of an active CPU → VRAM transfer. Called from
    /// `Vdp::write_register` whenever the CPU writes to R44 *and* a
    /// transfer is active. Handles the pixel-level work for LMMC and the
    /// byte-level work for HMMC; auto-clears TR/CE on the final write.
    fn pump_write(&mut self, value: u8) {
        let kind = *self.cpu_xfer;
        let Some(layout) = self.pixel_layout() else {
            return;
        };
        let dx = (self.cmd_word(36) & 0x1FF) as i32;
        let dy = (self.cmd_word(38) & 0x3FF) as i32;
        let nx = (self.cmd_word(40) & 0x3FF).max(1);
        let ny = (self.cmd_word(42) & 0x3FF).max(1);
        let arg = self.regs[45];
        let dix: i32 = if arg & 0x04 != 0 { -1 } else { 1 };
        let diy: i32 = if arg & 0x08 != 0 { -1 } else { 1 };

        match kind {
            CpuXfer::Lmmc { logic_op } => {
                // One pixel per write: low bits of `value` masked to bpp.
                let x = dx + (*self.cpu_xfer_x as i32) * dix;
                let y = dy + (*self.cpu_xfer_y as i32) * diy;
                if x >= 0 && y >= 0 && (x as u32) < layout.width {
                    let src = value & layout.pixel_mask();
                    let dst = self.read_pixel(&layout, x as u32, y as u32);
                    let new = apply_logic_op(src, dst, logic_op, layout.pixel_mask());
                    self.write_pixel(&layout, x as u32, y as u32, new);
                }
                *self.cpu_xfer_x += 1;
                if *self.cpu_xfer_x >= nx {
                    *self.cpu_xfer_x = 0;
                    *self.cpu_xfer_y += 1;
                }
            }
            CpuXfer::Hmmc => {
                // One byte per write — `nx` is in pixels so we advance
                // by `pixels_per_byte` per iteration.
                let ppb = layout.pixels_per_byte();
                let dx_byte = (dx as u32) / ppb;
                let bx = dx_byte as i32 + (*self.cpu_xfer_x as i32) * dix;
                let y = dy + (*self.cpu_xfer_y as i32) * diy;
                if bx >= 0 && y >= 0 && (bx as u32) < layout.pitch {
                    let addr = (y as u32 * layout.pitch + bx as u32) as usize;
                    if addr < self.vram.len() {
                        self.vram[addr] = value;
                    }
                }
                *self.cpu_xfer_x += 1;
                // HMMC advances by one byte-stride per write, so the
                // row's worth of bytes is nx / pixels_per_byte.
                if *self.cpu_xfer_x >= nx.div_ceil(ppb) {
                    *self.cpu_xfer_x = 0;
                    *self.cpu_xfer_y += 1;
                }
            }
            _ => return,
        }

        if *self.cpu_xfer_y >= ny {
            // Rectangle filled — transfer done.
            mlog!(VDP_CMD, "CPU xfer write complete");
            *self.cpu_xfer = CpuXfer::None;
            *self.cpu_xfer_x = 0;
            *self.cpu_xfer_y = 0;
            self.status[2] &= !0x81; // Clear TR and CE.
        }
    }

    /// Advance one step of an active LMCM transfer. Returns the pixel
    /// currently in S7 (= the one the CPU just sees), then advances the
    /// source pointer and preloads the next pixel for the *next* read.
    fn pump_read(&mut self) -> u8 {
        let nx = (self.cmd_word(40) & 0x3FF).max(1);
        let ny = (self.cmd_word(42) & 0x3FF).max(1);

        let pixel = self.status[7];

        *self.cpu_xfer_x += 1;
        if *self.cpu_xfer_x >= nx {
            *self.cpu_xfer_x = 0;
            *self.cpu_xfer_y += 1;
        }

        if *self.cpu_xfer_y >= ny {
            mlog!(VDP_CMD, "CPU xfer read complete");
            *self.cpu_xfer = CpuXfer::None;
            *self.cpu_xfer_x = 0;
            *self.cpu_xfer_y = 0;
            self.status[2] &= !0x81;
        } else {
            // Preload the next pixel so the next S7 read sees fresh data.
            self.preload_lmcm_pixel();
        }

        pixel
    }
}

impl Vdp {
    /// Build a `CmdCtx` over `self`'s state and run the command currently
    /// staged in R46. Triggered by `write_register` when R46 is written.
    fn execute_command(&mut self) {
        let mut ctx = CmdCtx {
            regs: &mut self.regs,
            vram: &mut self.vram[..],
            status: &mut self.status,
            cpu_xfer: &mut self.cpu_xfer,
            cpu_xfer_x: &mut self.cpu_xfer_x,
            cpu_xfer_y: &mut self.cpu_xfer_y,
        };
        ctx.execute();
    }

    /// Called when the CPU writes to R44. If a CPU → VRAM transfer
    /// (LMMC / HMMC) is active, this advances the transfer by one pixel
    /// (LMMC) or one byte (HMMC); otherwise it's a no-op and R44 keeps
    /// the value just written.
    fn pump_cpu_xfer_write(&mut self, value: u8) {
        if self.cpu_xfer == CpuXfer::None {
            return;
        }
        let mut ctx = CmdCtx {
            regs: &mut self.regs,
            vram: &mut self.vram[..],
            status: &mut self.status,
            cpu_xfer: &mut self.cpu_xfer,
            cpu_xfer_x: &mut self.cpu_xfer_x,
            cpu_xfer_y: &mut self.cpu_xfer_y,
        };
        ctx.pump_write(value);
    }

    /// Called when the CPU reads S7 with `R15 = 7`. If a VRAM → CPU
    /// transfer (LMCM) is active, this returns the next pixel and
    /// advances the source pointer. Otherwise S7 returns whatever was
    /// last latched (e.g. by POINT).
    fn pump_cpu_xfer_read(&mut self) -> u8 {
        if self.cpu_xfer != CpuXfer::Lmcm {
            return self.status[7];
        }
        let mut ctx = CmdCtx {
            regs: &mut self.regs,
            vram: &mut self.vram[..],
            status: &mut self.status,
            cpu_xfer: &mut self.cpu_xfer,
            cpu_xfer_x: &mut self.cpu_xfer_x,
            cpu_xfer_y: &mut self.cpu_xfer_y,
        };
        ctx.pump_read()
    }
}

/// Apply a V9938 logic op between a source pixel and the existing
/// destination pixel.
///
/// Logic-op encoding in R46[3:0]:
///   bit 3 = T (transparent skip — when set, source == 0 leaves dst alone)
///   bits 0..2 = base op: IMP(0), AND(1), OR(2), XOR(3), NOT(4)
fn apply_logic_op(src: u8, dst: u8, op: u8, pixel_mask: u8) -> u8 {
    if (op & 0x08) != 0 && src == 0 {
        // Transparent variant: source-zero is a "no-op" pixel.
        return dst;
    }
    let result = match op & 0x07 {
        0 => src,                   // IMP — straight write
        1 => src & dst,             // AND
        2 => src | dst,             // OR
        3 => src ^ dst,             // XOR
        4 => !src,                  // NOT
        _ => src,                   // 5-7 reserved → behave like IMP
    };
    result & pixel_mask
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a borrowed-state command context backed by local arrays —
    /// no Vdp, no wgpu device. The VRAM is a `Vec` because 128 KiB is too
    /// big to put on the stack.
    fn ctx_g4_inputs() -> ([u8; 64], Vec<u8>, [u8; 10]) {
        let mut regs = [0u8; 64];
        // G4 (Screen 5): M5=0 M4=1 M3=1 M2=0 M1=0
        //   R0[3:1] = M5 M4 M3 = 0 1 1 → 0x06
        //   R1[4:3] = M1 M2 = 0 0 → 0x00
        regs[0] = 0x06;
        regs[1] = 0x00;
        let vram = vec![0u8; VRAM_SIZE];
        let status = [0u8; 10];
        (regs, vram, status)
    }

    /// Convenience for shaping a test: build the context, configure it
    /// inside a closure, then assert against `regs`, `vram`, `status`.
    fn with_g4(setup: impl FnOnce(&mut CmdCtx)) -> ([u8; 64], Vec<u8>, [u8; 10]) {
        let (mut regs, mut vram, mut status) = ctx_g4_inputs();
        let mut cpu_xfer = CpuXfer::None;
        let mut cpu_xfer_x = 0u32;
        let mut cpu_xfer_y = 0u32;
        {
            let mut ctx = CmdCtx {
                regs: &mut regs,
                vram: &mut vram,
                status: &mut status,
                cpu_xfer: &mut cpu_xfer,
                cpu_xfer_x: &mut cpu_xfer_x,
                cpu_xfer_y: &mut cpu_xfer_y,
            };
            setup(&mut ctx);
        }
        (regs, vram, status)
    }

    #[test]
    fn pixel_layout_classifies_screen5() {
        let (regs, mut vram, mut status) = ctx_g4_inputs();
        let mut cpu_xfer = CpuXfer::None;
        let mut cpu_xfer_x = 0u32;
        let mut cpu_xfer_y = 0u32;
        let ctx = CmdCtx {
            regs: &mut regs.clone(),
            vram: &mut vram,
            status: &mut status,
            cpu_xfer: &mut cpu_xfer,
            cpu_xfer_x: &mut cpu_xfer_x,
            cpu_xfer_y: &mut cpu_xfer_y,
        };
        let layout = ctx.pixel_layout().expect("G4 must classify");
        assert_eq!(layout.bpp, 4);
        assert_eq!(layout.pitch, 128);
        assert_eq!(layout.width, 256);
    }

    #[test]
    fn hmmv_fills_rectangle() {
        let (_regs, vram, status) = with_g4(|ctx| {
            // Fill a 4×2 pixel rectangle at (0, 0) with byte 0xAB.
            // G4 packs 2 pixels per byte, so 4 pixels = 2 bytes wide.
            ctx.regs[40] = 4; ctx.regs[41] = 0;  // NX
            ctx.regs[42] = 2; ctx.regs[43] = 0;  // NY
            ctx.regs[44] = 0xAB;                  // CLR
            ctx.regs[46] = 0xC0;                  // HMMV
            ctx.execute();
        });
        assert_eq!(vram[0], 0xAB);
        assert_eq!(vram[1], 0xAB);
        assert_eq!(vram[128], 0xAB);
        assert_eq!(vram[129], 0xAB);
        // CE cleared on completion.
        assert_eq!(status[2] & 0x01, 0);
    }

    #[test]
    fn lmmv_fills_pixels_with_imp_op() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            ctx.regs[36] = 1; ctx.regs[37] = 0;  // DX = 1
            ctx.regs[40] = 3; ctx.regs[41] = 0;  // NX = 3
            ctx.regs[42] = 1; ctx.regs[43] = 0;  // NY = 1
            ctx.regs[44] = 5;                     // CLR
            ctx.regs[46] = 0x80;                  // LMMV / IMP
            ctx.execute();
        });
        // Layout to read pixels back through.
        let mut regs2 = [0u8; 64]; regs2[0] = 0x06; // G4 mode bits
        let mut status2 = [0u8; 10];
        let mut v2 = vram.clone();
        let mut cpu_xfer = CpuXfer::None;
        let mut cpu_xfer_x = 0u32;
        let mut cpu_xfer_y = 0u32;
        let ctx = CmdCtx {
            regs: &mut regs2,
            vram: &mut v2,
            status: &mut status2,
            cpu_xfer: &mut cpu_xfer,
            cpu_xfer_x: &mut cpu_xfer_x,
            cpu_xfer_y: &mut cpu_xfer_y,
        };
        let layout = ctx.pixel_layout().unwrap();
        assert_eq!(ctx.read_pixel(&layout, 0, 0), 0);
        assert_eq!(ctx.read_pixel(&layout, 1, 0), 5);
        assert_eq!(ctx.read_pixel(&layout, 2, 0), 5);
        assert_eq!(ctx.read_pixel(&layout, 3, 0), 5);
        assert_eq!(ctx.read_pixel(&layout, 4, 0), 0);
    }

    #[test]
    fn lmmv_transparent_skips_color_zero() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            let layout = ctx.pixel_layout().unwrap();
            ctx.write_pixel(&layout, 0, 0, 7);
            // LMMV with CLR=0 and TIMP (op 0x08) → should leave pixel alone.
            ctx.regs[40] = 1; ctx.regs[41] = 0;
            ctx.regs[42] = 1; ctx.regs[43] = 0;
            ctx.regs[44] = 0;
            ctx.regs[46] = 0x88;
            ctx.execute();
        });
        // Read back without execute: just byte arithmetic.
        assert_eq!(vram[0] >> 4 & 0x0F, 7);
    }

    #[test]
    fn lmmv_xor_inverts_existing_pixels() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            let layout = ctx.pixel_layout().unwrap();
            ctx.write_pixel(&layout, 0, 0, 0x05);
            ctx.write_pixel(&layout, 1, 0, 0x0A);
            ctx.regs[40] = 2; ctx.regs[41] = 0;
            ctx.regs[42] = 1; ctx.regs[43] = 0;
            ctx.regs[44] = 0x0F;
            ctx.regs[46] = 0x83;  // LMMV / XOR
            ctx.execute();
        });
        // vram[0]: high nibble = pixel 0 XOR 0xF, low = pixel 1 XOR 0xF.
        assert_eq!(vram[0] >> 4 & 0x0F, 0x05 ^ 0x0F);
        assert_eq!(vram[0] & 0x0F,      0x0A ^ 0x0F);
    }

    #[test]
    fn hmmm_copies_byte_aligned_rect() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            // Seed source row 0.
            ctx.vram[0] = 0x11;
            ctx.vram[1] = 0x22;
            ctx.vram[2] = 0x33;
            ctx.vram[3] = 0x44;
            ctx.regs[38] = 1; ctx.regs[39] = 0;  // DY = 1
            ctx.regs[40] = 8; ctx.regs[41] = 0;  // NX = 8 pixels = 4 bytes
            ctx.regs[42] = 1; ctx.regs[43] = 0;  // NY = 1
            ctx.regs[46] = 0xD0;                  // HMMM
            ctx.execute();
        });
        assert_eq!(vram[128], 0x11);
        assert_eq!(vram[129], 0x22);
        assert_eq!(vram[130], 0x33);
        assert_eq!(vram[131], 0x44);
    }

    #[test]
    fn pset_writes_single_pixel() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            ctx.regs[36] = 5; ctx.regs[37] = 0;
            ctx.regs[38] = 7; ctx.regs[39] = 0;
            ctx.regs[44] = 9;
            ctx.regs[46] = 0x50;
            ctx.execute();
        });
        // (5, 7) in G4: byte at 7*128 + 2 = 898; pixel-5 → odd → low nibble.
        assert_eq!(vram[898] & 0x0F, 9);
    }

    #[test]
    fn point_reads_pixel_into_s7() {
        let (_regs, _vram, status) = with_g4(|ctx| {
            let layout = ctx.pixel_layout().unwrap();
            ctx.write_pixel(&layout, 10, 4, 0xC);
            ctx.regs[32] = 10; ctx.regs[33] = 0;
            ctx.regs[34] = 4;  ctx.regs[35] = 0;
            ctx.regs[46] = 0x40;
            ctx.execute();
        });
        assert_eq!(status[7], 0xC);
    }

    #[test]
    fn line_horizontal() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            // Horizontal line: maj=X, NX=10, NY=0, +X, color 3.
            ctx.regs[40] = 10; ctx.regs[41] = 0;
            ctx.regs[42] = 0;  ctx.regs[43] = 0;
            ctx.regs[44] = 3;
            ctx.regs[45] = 0; // X major, +X, +Y
            ctx.regs[46] = 0x70;
            ctx.execute();
        });
        // First 11 pixels in row 0 should be color 3 (10 steps → endpoints
        // inclusive = 11). Check via byte arithmetic.
        for x in 0..=10u32 {
            let byte = vram[(x / 2) as usize];
            let nib = if x & 1 == 0 { byte >> 4 } else { byte & 0x0F };
            assert_eq!(nib, 3, "pixel ({},0) expected 3, got {}", x, nib);
        }
        // Pixel 11 should NOT be set. Odd x → low nibble of byte 5.
        let byte = vram[5];
        let nib = byte & 0x0F;
        assert_eq!(nib, 0, "pixel 11 should still be 0");
    }

    #[test]
    fn line_vertical() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            // Vertical line: maj=Y, NX=5 (steps), NY=0 (no minor), +Y.
            // Starts at (0, 0), ends at (0, 5).
            ctx.regs[40] = 5; ctx.regs[41] = 0;
            ctx.regs[42] = 0; ctx.regs[43] = 0;
            ctx.regs[44] = 7;
            ctx.regs[45] = 0x01; // Y major
            ctx.regs[46] = 0x70;
            ctx.execute();
        });
        // Rows 0..=5, X=0 → high nibble of byte (y*128).
        for y in 0..=5u32 {
            let byte = vram[(y * 128) as usize];
            assert_eq!(byte >> 4, 7, "pixel (0,{}) expected 7", y);
        }
    }

    #[test]
    fn srch_finds_matching_color() {
        let (_regs, _vram, status) = with_g4(|ctx| {
            // Seed pixel (5, 0) with color 9, leave others 0.
            let layout = ctx.pixel_layout().unwrap();
            ctx.write_pixel(&layout, 5, 0, 9);
            // SRCH from (0, 0) for color 9 (EQ = 1), +X.
            ctx.regs[32] = 0;  ctx.regs[33] = 0;  // SX
            ctx.regs[34] = 0;  ctx.regs[35] = 0;  // SY
            ctx.regs[44] = 9;                      // CLR
            ctx.regs[45] = 0x02;                   // EQ
            ctx.regs[46] = 0x60;
            ctx.execute();
        });
        // S8/S9 should report X = 5; BD bit should be clear.
        assert_eq!(status[8], 5);
        assert_eq!(status[9], 0);
        assert_eq!(status[2] & 0x10, 0, "BD must be clear on match");
    }

    #[test]
    fn srch_sets_bd_on_border_hit() {
        let (_regs, _vram, status) = with_g4(|ctx| {
            // No matching pixel exists; search for color 5 (EQ).
            ctx.regs[32] = 0; ctx.regs[33] = 0;
            ctx.regs[34] = 0; ctx.regs[35] = 0;
            ctx.regs[44] = 5;
            ctx.regs[45] = 0x02;
            ctx.regs[46] = 0x60;
            ctx.execute();
        });
        // Should hit the right border → BD set.
        assert_ne!(status[2] & 0x10, 0, "BD must be set when no match found");
    }

    #[test]
    fn ymmm_scrolls_row_to_next() {
        let (_regs, vram, _status) = with_g4(|ctx| {
            // Seed row 0 with a recognisable pattern.
            for x in 0..16usize {
                ctx.vram[x] = 0xA0 + x as u8;
            }
            // YMMM copies the entire row from (DX, SY) to (DX, DY).
            ctx.regs[34] = 0; ctx.regs[35] = 0;  // SY = 0
            ctx.regs[36] = 0; ctx.regs[37] = 0;  // DX = 0
            ctx.regs[38] = 1; ctx.regs[39] = 0;  // DY = 1
            ctx.regs[42] = 1; ctx.regs[43] = 0;  // NY = 1
            ctx.regs[45] = 0;                     // +Y
            ctx.regs[46] = 0xE0;                  // YMMM
            ctx.execute();
        });
        // Row 1 (starts at byte 128) should now mirror row 0.
        for x in 0..16usize {
            assert_eq!(vram[128 + x], 0xA0 + x as u8, "byte {} mismatch", x);
        }
    }

    #[test]
    fn lmmc_starts_transfer_and_sets_tr_ce() {
        // LMMC sets up a CPU → VRAM transfer and waits for R44 writes.
        // CE + TR should be set after the command; the transfer is
        // pumped externally by `Vdp::pump_cpu_xfer_write` which we
        // don't exercise in the borrowed-state harness.
        let (_regs, _vram, status) = with_g4(|ctx| {
            ctx.regs[40] = 4; ctx.regs[41] = 0;  // NX = 4 pixels
            ctx.regs[42] = 1; ctx.regs[43] = 0;  // NY = 1 row
            ctx.regs[46] = 0xB0;                  // LMMC, IMP
            ctx.execute();
        });
        // Both TR (S2 bit 7) and CE (S2 bit 0) must be set so the CPU
        // knows it can start streaming R44 writes.
        assert_ne!(status[2] & 0x80, 0, "TR must be set");
        assert_ne!(status[2] & 0x01, 0, "CE must be set");
    }

    #[test]
    fn hmmc_starts_transfer_and_sets_tr_ce() {
        let (_regs, _vram, status) = with_g4(|ctx| {
            ctx.regs[40] = 8; ctx.regs[41] = 0;
            ctx.regs[42] = 1; ctx.regs[43] = 0;
            ctx.regs[46] = 0xF0;                  // HMMC
            ctx.execute();
        });
        assert_ne!(status[2] & 0x80, 0, "TR must be set");
        assert_ne!(status[2] & 0x01, 0, "CE must be set");
    }

    #[test]
    fn lmcm_starts_transfer_and_preloads_s7() {
        let (_regs, vram, status) = with_g4(|ctx| {
            // Seed a pixel for LMCM to pick up.
            let layout = ctx.pixel_layout().unwrap();
            ctx.write_pixel(&layout, 0, 0, 0xD);
            ctx.regs[40] = 4; ctx.regs[41] = 0;
            ctx.regs[42] = 1; ctx.regs[43] = 0;
            ctx.regs[46] = 0xA0;                  // LMCM
            ctx.execute();
        });
        // S7 should already have the first pixel ready.
        assert_eq!(status[7], 0xD, "first LMCM pixel should be preloaded into S7");
        assert_ne!(status[2] & 0x80, 0, "TR must be set");
        assert_ne!(status[2] & 0x01, 0, "CE must be set");
        // VRAM is the same (LMCM doesn't modify VRAM, only reads from it).
        let _ = vram;
    }

    #[test]
    fn stop_aborts_active_cpu_transfer() {
        let (_regs, _vram, status) = with_g4(|ctx| {
            ctx.regs[40] = 4; ctx.regs[41] = 0;
            ctx.regs[42] = 1; ctx.regs[43] = 0;
            ctx.regs[46] = 0xB0;                  // LMMC
            ctx.execute();
            // ... CPU never writes R44, then game issues STOP.
            ctx.regs[46] = 0x00;                  // STOP
            ctx.execute();
        });
        // STOP clears both TR and CE.
        assert_eq!(status[2] & 0x81, 0, "STOP must clear TR + CE");
    }
}
