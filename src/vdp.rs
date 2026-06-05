// TMS9918A host-side state + GPU resources.
// All scanline / mode-decode logic lives in vdp.wgsl; the CPU side just
// writes VRAM and 8 registers, then uploads them once per frame.

pub const VRAM_SIZE: usize = 16 * 1024;

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
    regs: [[u32; 4]; 2],
    palette: [[f32; 4]; 16],
}

pub struct Vdp {
    pub vram: Box<[u8; VRAM_SIZE]>,
    pub regs: [u8; 8],

    // Port-protocol state. The TMS9918 talks to the CPU through ports 0x98
    // (data) and 0x99 (control / status). Writes to 0x99 come in pairs and
    // need a one-byte latch; reads from VRAM happen via an auto-incrementing
    // 14-bit address pointer.
    vram_address: u16,
    vdp_status: u8,
    latched_data: u8,
    has_latched_data: bool,

    vram_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
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
            regs: [0u8; 8],
            vram_address: 0,
            vdp_status: 0,
            latched_data: 0,
            has_latched_data: false,
            vram_buf,
            uniform_buf,
            bind_group,
            pipeline,
        }
    }

    pub fn upload(&self, queue: &wgpu::Queue, framebuffer_size: (u32, u32)) {
        queue.write_buffer(&self.vram_buf, 0, &self.vram[..]);

        let mut regs_packed = [[0u32; 4]; 2];
        for (i, &b) in self.regs.iter().enumerate() {
            regs_packed[i / 4][i % 4] = b as u32;
        }
        let uniforms = Uniforms {
            framebuffer_size: [framebuffer_size.0 as f32, framebuffer_size.1 as f32],
            _pad: [0; 2],
            regs: regs_packed,
            palette: *PALETTE,
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
        self.vdp_status |= 0x80;
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
                    if self.vdp_status & 0x40 == 0 {
                        self.vdp_status = (self.vdp_status & 0xA0) | 0x40 | s_idx;
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
                        self.vdp_status |= 0x20;
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
        self.regs[1] & 0x20 != 0 && self.vdp_status & 0x80 != 0
    }

    /// Wipe VRAM and registers — used on cartridge swap so the BIOS can boot
    /// the new game on a clean slate. The GPU resources stay alive (VRAM
    /// upload re-syncs them on the next frame).
    pub fn reset(&mut self) {
        self.vram.fill(0);
        self.regs = [0u8; 8];
        self.vram_address = 0;
        self.vdp_status = 0;
        self.latched_data = 0;
        self.has_latched_data = false;
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
        self.regs = [0x02, 0xC0, 0x06, 0xFF, 0x03, 0x36, 0x07, 0x04];

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
        // Real hardware resets the latch on any read of either port.
        self.has_latched_data = false;
        match port {
            0x98 => self.read_data(),
            0x99 => self.read_status(),
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
            _ => {}
        }
    }
}

impl Vdp {
    fn read_data(&mut self) -> u8 {
        let value = self.vram[(self.vram_address & 0x3FFF) as usize];
        self.vram_address = self.vram_address.wrapping_add(1) & 0x3FFF;
        value
    }

    fn read_status(&mut self) -> u8 {
        // Returns the previous status and resets it in one move.
        std::mem::replace(&mut self.vdp_status, 0)
    }

    fn write_data(&mut self, value: u8) {
        self.vram[(self.vram_address & 0x3FFF) as usize] = value;
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
            // Register write: bits 0..2 of the second byte pick R0..R7,
            // payload is the latched first byte.
            let register = (value & 0x07) as usize;
            self.regs[register] = self.latched_data;
        } else {
            // VRAM address setup. Bit 6 distinguishes read-intent from
            // write-intent on real hardware, but a single 14-bit pointer
            // serves both — so we drop the distinction.
            self.vram_address = (((value & 0x3F) as u16) << 8) | (self.latched_data as u16);
        }
    }
}
