// Sharp post-process — integer-scale upscale of the 320×240 intermediate
// texture into the surface, with backdrop-coloured letterboxing.
//
// The intermediate is sampled with a *nearest* sampler so MSX pixels stay
// crisp. We also snap the UV to the texel centre before sampling, which
// guarantees no row/column drift even when the integer scale puts a fragment
// exactly on a texel boundary.

struct Uniforms {
    output_size: vec2<f32>,
    crt_blur: f32,        // unused here; present so the layout matches the CRT shader
    _pad: f32,
    backdrop: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Single fullscreen triangle: covers the whole NDC range plus overscan.
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

const NATIVE_W: f32 = 320.0;
const NATIVE_H: f32 = 240.0;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let native = vec2<f32>(NATIVE_W, NATIVE_H);
    let fb = u.output_size;

    // Largest integer scale that still fits in the surface — same formula as
    // the old in-shader letterbox. scale = 1 if the surface is smaller than
    // native, which lets BIOS init still draw something even on tiny windows.
    let scale = max(1.0, floor(min(fb.x / native.x, fb.y / native.y)));
    let viewport = native * scale;
    let offset = (fb - viewport) * 0.5;
    let local = (in.pos.xy - offset) / scale;

    // Outside the viewport → letterbox, paint with the backdrop so the border
    // blends with the MSX-internal one (the VDP also paints its 32×24 border
    // in the backdrop colour). Title screens that recolour R7 get the full
    // window matched in one go.
    if (local.x < 0.0 || local.x >= native.x || local.y < 0.0 || local.y >= native.y) {
        return u.backdrop;
    }

    // Snap to texel centre — robust even when the GPU rounds the fragment
    // position to a texel boundary at an integer scale.
    let uv = (floor(local) + 0.5) / native;
    // textureSampleLevel instead of textureSample: the latter needs implicit
    // derivatives, which WGSL only allows from uniform control flow — and our
    // letterbox `if` makes the flow non-uniform from the validator's POV.
    // We have no mipmaps anyway, so LOD 0 is what we want.
    return textureSampleLevel(src, samp, uv, 0.0);
}
