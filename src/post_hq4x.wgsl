// hq4x post-process — faithful GPU port of Maxim Stepin's hq4x.
//
// Based on the LUT formulation by Cameron Zemek and Jules Blok
// (CrossVR/hqx-shader, LGPL-2.1). Where the `pixely` shader (EPX/Scale2x)
// decides edges by *exact* colour equality on a 5-tap cross, hq4x:
//
//   1. classifies the full 3×3 neighbourhood with a YUV-threshold "diff", so
//      near-but-not-identical colours can still register as an edge;
//   2. packs the 8 centre-vs-neighbour comparisons (an 8-bit "pattern") plus a
//      4-bit "cross" of edges *between* the cardinal neighbours, plus the 4×4
//      sub-pixel cell, into a single lookup;
//   3. reads four blend weights from a 256×256 lookup table and mixes four
//      candidate colours (centre + the diagonal / horizontal / vertical
//      neighbour toward this sub-pixel).
//
// The LUT is Stepin's hand-tuned ruleset baked into `hq4x_lut.png` — there is
// no compact closed form, which is exactly why hqx needs the table. It is
// uploaded as a second texture at @binding(3), added to the shared bind-group
// layout; the other three shaders simply ignore that binding.
//
// Both the source and the LUT are read with NEAREST sampling: the source so
// neighbour fetches land on exact palette colours (bilinear would smear them
// and every `diff` would misfire), the LUT so each weight texel is read
// verbatim. The host binds the nearest bind group for this mode.

struct Uniforms {
    output_size: vec2<f32>,
    crt_blur: f32,   // unused here; present so the shared uniform layout matches
    _pad: f32,
    backdrop: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var lut: texture_2d<f32>;

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
const SCALE: f32 = 4.0;

// BT.601 luma/chroma. Stepin's `diff` thresholds are expressed against these.
fn rgb2yuv(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>( 0.299,  0.587,  0.114)),
        dot(c, vec3<f32>(-0.169, -0.331,  0.500)),
        dot(c, vec3<f32>( 0.500, -0.419, -0.081)),
    );
}

// Per-channel "these are different colours" thresholds (the hqx constants,
// scaled to 0..1). The reference adds a constant offset to both YUV values
// before subtracting; it cancels, so we omit it.
const YUV_THRESHOLD: vec3<f32> = vec3<f32>(48.0 / 255.0, 7.0 / 255.0, 6.0 / 255.0);

fn diff(a: vec3<f32>, b: vec3<f32>) -> bool {
    return any(abs(a - b) > YUV_THRESHOLD);
}

fn b2f(v: bool) -> f32 {
    return select(0.0, 1.0, v);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let native = vec2<f32>(NATIVE_W, NATIVE_H);
    let fb = u.output_size;

    // Integer-scale letterbox, identical to the other post shaders.
    let scale = max(1.0, floor(min(fb.x / native.x, fb.y / native.y)));
    let viewport = native * scale;
    let offset = (fb - viewport) * 0.5;
    let local = (in.pos.xy - offset) / scale;

    if (local.x < 0.0 || local.x >= native.x || local.y < 0.0 || local.y >= native.y) {
        return u.backdrop;
    }

    let texel = vec2<f32>(1.0 / native.x, 1.0 / native.y);
    let src_px = floor(local);
    let fp = local - src_px;                          // sub-pixel position, 0..1
    let centre_uv = (src_px + vec2<f32>(0.5, 0.5)) * texel;

    // `quad` points toward the corner this sub-pixel sits in; the candidate
    // neighbours are taken in that direction (sign(fp - 0.5) ∈ {-1, +1}).
    let quad = sign(fp - vec2<f32>(0.5, 0.5));

    // Four candidate colours (RGB — these get blended), all NEAREST.
    let p1 = textureSampleLevel(src, samp, centre_uv, 0.0).rgb;                                    // centre
    let p2 = textureSampleLevel(src, samp, centre_uv + texel * quad, 0.0).rgb;                     // diagonal
    let p3 = textureSampleLevel(src, samp, centre_uv + vec2<f32>(texel.x, 0.0) * quad, 0.0).rgb;   // horizontal
    let p4 = textureSampleLevel(src, samp, centre_uv + vec2<f32>(0.0, texel.y) * quad, 0.0).rgb;   // vertical

    // 3×3 neighbourhood in YUV for the edge classification.
    //   w1 w2 w3
    //   w4 w5 w6
    //   w7 w8 w9
    let w1 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>(-texel.x, -texel.y), 0.0).rgb);
    let w2 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>( 0.0,     -texel.y), 0.0).rgb);
    let w3 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>( texel.x, -texel.y), 0.0).rgb);
    let w4 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>(-texel.x,  0.0),     0.0).rgb);
    let w5 = rgb2yuv(p1);
    let w6 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>( texel.x,  0.0),     0.0).rgb);
    let w7 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>(-texel.x,  texel.y), 0.0).rgb);
    let w8 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>( 0.0,      texel.y), 0.0).rgb);
    let w9 = rgb2yuv(textureSampleLevel(src, samp, centre_uv + vec2<f32>( texel.x,  texel.y), 0.0).rgb);

    // 8-bit edge pattern: centre (w5) vs each neighbour. Bit order matches the
    // LUT's x axis.
    let pattern =
        b2f(diff(w5, w1)) *   1.0 +
        b2f(diff(w5, w2)) *   2.0 +
        b2f(diff(w5, w3)) *   4.0 +
        b2f(diff(w5, w4)) *   8.0 +
        b2f(diff(w5, w6)) *  16.0 +
        b2f(diff(w5, w7)) *  32.0 +
        b2f(diff(w5, w8)) *  64.0 +
        b2f(diff(w5, w9)) * 128.0;

    // 4-bit "cross": edges between the cardinal neighbours.
    let cross =
        b2f(diff(w4, w2)) * 1.0 +
        b2f(diff(w2, w6)) * 2.0 +
        b2f(diff(w8, w4)) * 4.0 +
        b2f(diff(w6, w8)) * 8.0;

    // LUT coordinate. x = edge pattern (0..255). y = cross (0..15) selecting a
    // 16-row block, plus the 4×4 sub-pixel cell (fp·SCALE) within it.
    let sub = floor(fp * SCALE);                      // 0..3 in each axis
    let index = vec2<f32>(
        pattern,
        cross * (SCALE * SCALE) + sub.y * SCALE + sub.x,
    );

    let step = vec2<f32>(1.0 / 256.0, 1.0 / (16.0 * SCALE * SCALE));
    let lut_uv = index * step + step * 0.5;           // hit the texel centre
    let weights = textureSampleLevel(lut, samp, lut_uv, 0.0);

    let sum = dot(weights, vec4<f32>(1.0, 1.0, 1.0, 1.0));
    let res = (p1 * weights.x + p2 * weights.y + p3 * weights.z + p4 * weights.w) / sum;

    return vec4<f32>(res, 1.0);
}
