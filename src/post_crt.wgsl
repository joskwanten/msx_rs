// CRT-ish post-process. Same integer-scale viewport as the sharp shader,
// but:
//
//   * The intermediate is sampled with a *linear* sampler so neighbouring
//     MSX pixels bleed into each other slightly.
//   * A small 5-tap cross-shaped gaussian blur softens the result further —
//     enough to feel like a CRT but not so much that text becomes unreadable.
//   * Horizontal scanlines: each native pixel row is brighter at its centre
//     and dims toward the top/bottom edge. Visible when the integer scale is
//     ≥ 3, which is the common case on modern monitors.
//   * A gentle radial vignette darkens the corners.
//
// The blending is technically wrong on the web build (the intermediate stores
// sRGB-encoded floats and we're averaging them as if they were linear), but
// the MSX palette is saturated enough that the artefacts don't show — and
// the look is what we're after, not photometric accuracy.

struct Uniforms {
    output_size: vec2<f32>,
    /// Tap-distance for the 5-tap cross blur, in native-texel units. Set
    /// per platform by the Rust side: tighter on native (~0.42) where the
    /// surface is sRGB and the blur happens in linear space (gamma-correct,
    /// bleeds brights more), wider on web (~0.60) where the surface is
    /// Unorm and the blur runs in display space.
    crt_blur: f32,
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
    let scale = max(1.0, floor(min(fb.x / native.x, fb.y / native.y)));
    let viewport = native * scale;
    let offset = (fb - viewport) * 0.5;
    let local = (in.pos.xy - offset) / scale;

    let in_viewport = local.x >= 0.0 && local.x < native.x
                   && local.y >= 0.0 && local.y < native.y;

    // Base colour: blurred MSX-canvas sample inside the viewport, the
    // backdrop everywhere else. The host-window letterbox still gets the
    // scanlines + vignette treatment below — feels more "the whole screen is
    // a CRT" instead of "a CRT image inside a flat black frame".
    //
    // textureSampleLevel instead of textureSample: WGSL only permits implicit
    // derivatives from uniform control flow, and the viewport branch is
    // non-uniform. Explicit LOD 0 sidesteps the rule and is what we want
    // anyway (we have no mipmaps).
    var rgba: vec4<f32>;
    if (in_viewport) {
        let uv = local / native;
        let texel = vec2<f32>(1.0 / native.x, 1.0 / native.y);
        // 5-tap cross blur. Tap separation set by the host (`u.crt_blur`) so
        // we can tune it per platform — see PostUniforms in post.rs.
        let off = texel * u.crt_blur;
        rgba = textureSampleLevel(src, samp, uv, 0.0) * 0.40;
        rgba = rgba + textureSampleLevel(src, samp, uv + vec2<f32>( off.x, 0.0), 0.0) * 0.15;
        rgba = rgba + textureSampleLevel(src, samp, uv + vec2<f32>(-off.x, 0.0), 0.0) * 0.15;
        rgba = rgba + textureSampleLevel(src, samp, uv + vec2<f32>(0.0,  off.y), 0.0) * 0.15;
        rgba = rgba + textureSampleLevel(src, samp, uv + vec2<f32>(0.0, -off.y), 0.0) * 0.15;
    } else {
        rgba = u.backdrop;
    }

    // Scanlines apply across the entire surface. `local.y` is continuous in
    // native-pixel space (extends past the viewport into negative / >native
    // values for the letterbox), and `fract()` keeps the phase aligned —
    // there's no seam where the scanline pattern hits the canvas border.
    //
    // Triangle wave 1 − 2·|phase − 0.5| peaks at 1.0 in the middle of each
    // row and falls to 0 at the boundaries; scaled by 0.22 it gives a subtle
    // horizontal striping (about 40% less intense than the first pass).
    let line_phase = fract(local.y);
    let line_intensity = 1.0 - 0.22 * (1.0 - (1.0 - 2.0 * abs(line_phase - 0.5)));
    rgba = rgba * line_intensity;

    // Vignette: radial darkening toward the corners — uses the host window
    // centre (`in.pos.xy`-based), so the falloff is symmetric across the full
    // surface rather than the inner viewport only.
    let centre = u.output_size * 0.5;
    let to_centre = (in.pos.xy - centre) / centre;
    let vignette = 1.0 - 0.12 * dot(to_centre, to_centre);
    rgba = rgba * vignette;

    rgba.a = 1.0;
    return rgba;
}
