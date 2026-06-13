//! Post-process pipeline: takes the 320×240 intermediate texture the VDP
//! renders into, and upscales/letterboxes it onto the surface — either with
//! pixel-perfect nearest sampling (`ShaderMode::Sharp`) or with a CRT-ish
//! blur + scanlines + vignette (`ShaderMode::Crt`).
//!
//! Two pipelines share one bind-group layout. The bind groups differ only in
//! which sampler they hold (nearest vs linear), so switching modes between
//! frames is just a swap of the bind group and the pipeline — no buffer
//! reupload.

use crate::vdp::{CANVAS_H, CANVAS_W};

/// Which post-process shader to run this frame. Toggle at runtime via the
/// host's Alt+S shortcut; persisted as a `?shader=…` URL parameter on web.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShaderMode {
    /// Integer-scale nearest-neighbour upscale. The default — crisp MSX
    /// pixels, no smoothing.
    Sharp,
    /// Soft blur + scanlines + gentle vignette. Mimics a low-end CRT.
    Crt,
    /// EPX / Scale2x edge-aware upscale — diagonals get anti-aliased per the
    /// 5-tap palette-equality rule, everything else stays as crisp as Sharp.
    Pixely,
    /// hq4x — Maxim Stepin's LUT-based 4× upscale. Smoother than Pixely: a
    /// YUV-threshold edge test over the full 3×3 neighbourhood drives a baked
    /// 256-pattern lookup table (`hq4x_lut.png`) of blend weights.
    Hq4x,
}

impl ShaderMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sharp" => Some(ShaderMode::Sharp),
            "crt" => Some(ShaderMode::Crt),
            "pixely" | "scale2x" | "epx" => Some(ShaderMode::Pixely),
            "hq4x" | "hqx" => Some(ShaderMode::Hq4x),
            _ => None,
        }
    }

    pub fn toggle(self) -> Self {
        // Cycle: Sharp → Crt → Pixely → Hq4x → Sharp. Alt+S walks through them.
        match self {
            ShaderMode::Sharp => ShaderMode::Crt,
            ShaderMode::Crt => ShaderMode::Pixely,
            ShaderMode::Pixely => ShaderMode::Hq4x,
            ShaderMode::Hq4x => ShaderMode::Sharp,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ShaderMode::Sharp => "sharp",
            ShaderMode::Crt => "crt",
            ShaderMode::Pixely => "pixely",
            ShaderMode::Hq4x => "hq4x",
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PostUniforms {
    output_size: [f32; 2],
    /// CRT blur tap-distance in native-texel units. Tuned per platform to
    /// compensate for the linear-space-vs-sRGB-space blending difference
    /// between native (Bgra8UnormSrgb surface → gamma-correct blur → looks
    /// blurrier) and web (Bgra8Unorm surface → display-space blur → looks
    /// crisper). Unused by Sharp / Pixely but lives in the shared uniform
    /// block so the layout matches across all three shaders.
    crt_blur: f32,
    /// Padding to align `backdrop` to a 16-byte boundary. Unused by the
    /// shaders.
    _pad: f32,
    backdrop: [f32; 4],
}

/// Per-platform CRT blur radius. Native sees a tighter tap because its blur
/// happens in linear space, which physically bleeds bright colours more than
/// the gamma-incorrect sRGB blur on web; tighter tap brings the perceived
/// softness back in line with the web look.
#[cfg(target_arch = "wasm32")]
const CRT_BLUR: f32 = 0.60;
#[cfg(not(target_arch = "wasm32"))]
const CRT_BLUR: f32 = 0.36;

pub struct Post {
    /// 320×240 texture the VDP renders into. `intermediate_view()` exposes
    /// it as a render target for the VDP pass.
    intermediate_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    bind_group_nearest: wgpu::BindGroup,
    bind_group_linear: wgpu::BindGroup,
    pipeline_sharp: wgpu::RenderPipeline,
    pipeline_crt: wgpu::RenderPipeline,
    pipeline_pixely: wgpu::RenderPipeline,
    pipeline_hq4x: wgpu::RenderPipeline,
}

impl Post {
    /// Build the intermediate texture, samplers, bind groups, and both
    /// pipelines. `target_format` is the surface format — used for both the
    /// intermediate (so the colour-space behaviour matches end-to-end) and
    /// the final pipeline's colour target.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        // Intermediate render target. Same format as the surface so colour
        // space stays consistent across both passes (linear on native sRGB
        // surfaces, raw sRGB-encoded on web's Bgra8Unorm).
        let intermediate = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post intermediate"),
            size: wgpu::Extent3d {
                width: CANVAS_W,
                height: CANVAS_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let intermediate_view = intermediate.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post sampler nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post sampler linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // hq4x weight lookup table: a 256×256 RGBA8 image baked from Stepin's
        // 256-pattern ruleset. Embedded in the binary and decoded once here.
        // Stored as non-sRGB Rgba8Unorm so the sampler returns the raw 0..1
        // blend weights (not gamma-decoded colours), and read with NEAREST.
        let lut_view = load_hq4x_lut(device, queue);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post uniforms"),
            size: std::mem::size_of::<PostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // hq4x weight LUT. Only the hq4x shader reads this; the other
                // three declare no @binding(3) and ignore it. Present in the
                // shared layout so all four pipelines keep one bind group.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Both upscale bind groups sample the intermediate the VDP rendered.
        let bind_group_nearest = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post BG nearest"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler_nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
            ],
        });
        let bind_group_linear = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post BG linear"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline_sharp = build_pipeline(
            device,
            &pipeline_layout,
            target_format,
            "post sharp",
            include_str!("post_sharp.wgsl"),
        );
        let pipeline_crt = build_pipeline(
            device,
            &pipeline_layout,
            target_format,
            "post crt",
            include_str!("post_crt.wgsl"),
        );
        let pipeline_pixely = build_pipeline(
            device,
            &pipeline_layout,
            target_format,
            "post pixely",
            include_str!("post_pixely.wgsl"),
        );
        let pipeline_hq4x = build_pipeline(
            device,
            &pipeline_layout,
            target_format,
            "post hq4x",
            include_str!("post_hq4x.wgsl"),
        );

        Self {
            intermediate_view,
            uniform_buf,
            bind_group_nearest,
            bind_group_linear,
            pipeline_sharp,
            pipeline_crt,
            pipeline_pixely,
            pipeline_hq4x,
        }
    }

    /// View handle for the intermediate texture — what the VDP renders into.
    pub fn intermediate_view(&self) -> &wgpu::TextureView {
        &self.intermediate_view
    }

    pub fn upload(&self, queue: &wgpu::Queue, output_size: (u32, u32), backdrop: [f32; 4]) {
        let u = PostUniforms {
            output_size: [output_size.0 as f32, output_size.1 as f32],
            crt_blur: CRT_BLUR,
            _pad: 0.0,
            backdrop,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn draw(&self, render_pass: &mut wgpu::RenderPass, mode: ShaderMode) {
        match mode {
            ShaderMode::Sharp => {
                render_pass.set_pipeline(&self.pipeline_sharp);
                render_pass.set_bind_group(0, &self.bind_group_nearest, &[]);
            }
            ShaderMode::Crt => {
                render_pass.set_pipeline(&self.pipeline_crt);
                render_pass.set_bind_group(0, &self.bind_group_linear, &[]);
            }
            ShaderMode::Pixely => {
                // EPX compares pixels by exact palette colour, so we need
                // *nearest* sampling on the neighbour fetches — bilinear
                // would smear adjacent palette entries into each other and
                // every comparison would fail.
                render_pass.set_pipeline(&self.pipeline_pixely);
                render_pass.set_bind_group(0, &self.bind_group_nearest, &[]);
            }
            ShaderMode::Hq4x => {
                // Like Pixely, hq4x classifies edges on exact palette colours,
                // so the source must be sampled *nearest*. The LUT (bound in
                // the same group) is read nearest too.
                render_pass.set_pipeline(&self.pipeline_hq4x);
                render_pass.set_bind_group(0, &self.bind_group_nearest, &[]);
            }
        }
        render_pass.draw(0..3, 0..1);
    }
}

/// Decode the embedded hq4x weight LUT and upload it as a 256×256 Rgba8Unorm
/// texture. The PNG is verified non-interlaced RGBA8 at build time, so any
/// decode failure here is a corrupted binary, not a runtime input — hence the
/// `expect`s rather than a fallible return.
fn load_hq4x_lut(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    const LUT_PNG: &[u8] = include_bytes!("hq4x_lut.png");

    let mut reader = png::Decoder::new(LUT_PNG)
        .read_info()
        .expect("hq4x LUT: PNG header");
    let mut rgba = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut rgba).expect("hq4x LUT: PNG decode");
    assert!(
        info.color_type == png::ColorType::Rgba && info.bit_depth == png::BitDepth::Eight,
        "hq4x LUT must be 8-bit RGBA",
    );

    let size = wgpu::Extent3d {
        width: info.width,
        height: info.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hq4x LUT"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba[..(info.width * info.height * 4) as usize],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(info.width * 4),
            rows_per_image: Some(info.height),
        },
        size,
    );

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    target_format: wgpu::TextureFormat,
    label: &str,
    wgsl: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
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
    })
}
