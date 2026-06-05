// Pixely post-process — EPX / Scale2x upscaler.
//
// EPX (also called Scale2x in its 2× form) is a 5-tap, palette-based edge
// scaler. For each output fragment we look at the *source* pixel it falls in
// (P) plus its four cardinal neighbours:
//
//        A
//      C P B
//        D
//
// Then we split P into four quadrants and decide per quadrant:
//
//   ┌────┬────┐
//   │ TL │ TR │   TL gets A if  C==A && C!=D && A!=B
//   ├────┼────┤   TR gets B if  A==B && A!=C && B!=D
//   │ BL │ BR │   BL gets C if  D==C && D!=B && C!=A
//   └────┴────┘   BR gets D if  B==D && B!=A && D!=C
//
// Anywhere the rules don't fire, the quadrant keeps the centre colour P.
// Net effect: anti-aliased diagonals on chunky pixel art without the blur
// that bilinear or a Gaussian post would add. Works especially well on MSX1
// content (Mode 1/2 graphics, sprite contours) where the palette is small
// and pixels are large.
//
// The rule only looks at exact colour equality. Because every pixel in the
// intermediate texture came out of the same 16-entry palette, equal source
// colours produce *exactly* equal RGBA values — no floating-point fuzz to
// worry about. (We compare on RGB only to be safe against alpha-channel
// quirks in future renderers.)

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
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

const NATIVE_W: f32 = 320.0;
const NATIVE_H: f32 = 240.0;

// Two pixels considered the "same" when their RGB delta is tiny. Squared
// distance + a small threshold avoids the brittleness of strict `==` on
// floats and still tells palette-equal pixels apart from any other pair.
fn same_colour(a: vec4<f32>, b: vec4<f32>) -> bool {
    let d = a.rgb - b.rgb;
    return dot(d, d) < 0.0001;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let native = vec2<f32>(NATIVE_W, NATIVE_H);
    let fb = u.output_size;

    let scale = max(1.0, floor(min(fb.x / native.x, fb.y / native.y)));
    let viewport = native * scale;
    let offset = (fb - viewport) * 0.5;
    let local = (in.pos.xy - offset) / scale;

    if (local.x < 0.0 || local.x >= native.x || local.y < 0.0 || local.y >= native.y) {
        return u.backdrop;
    }

    // Source pixel that this fragment sits in, and the fragment's position
    // *within* that pixel (0..1 in each axis). The latter selects the EPX
    // quadrant.
    let src_px = floor(local);
    let frac = local - src_px;

    let texel = vec2<f32>(1.0 / native.x, 1.0 / native.y);
    let centre_uv = (src_px + vec2<f32>(0.5, 0.5)) * texel;

    // Five samples: centre + four cardinal neighbours. textureSampleLevel
    // (not textureSample) because the letterbox `if` above makes the control
    // flow non-uniform from the WGSL validator's POV, and only the explicit-
    // LOD variants are allowed there. We have no mipmaps, LOD 0 is what we
    // want anyway.
    let p = textureSampleLevel(src, samp, centre_uv, 0.0);
    let a = textureSampleLevel(src, samp, centre_uv + vec2<f32>(0.0, -texel.y), 0.0);
    let c = textureSampleLevel(src, samp, centre_uv + vec2<f32>(-texel.x, 0.0), 0.0);
    let b = textureSampleLevel(src, samp, centre_uv + vec2<f32>( texel.x, 0.0), 0.0);
    let d = textureSampleLevel(src, samp, centre_uv + vec2<f32>(0.0,  texel.y), 0.0);

    var out = p;
    let top = frac.y < 0.5;
    let left = frac.x < 0.5;

    if (top && left) {
        if (same_colour(c, a) && !same_colour(c, d) && !same_colour(a, b)) {
            out = a;
        }
    } else if (top && !left) {
        if (same_colour(a, b) && !same_colour(a, c) && !same_colour(b, d)) {
            out = b;
        }
    } else if (!top && left) {
        if (same_colour(d, c) && !same_colour(d, b) && !same_colour(c, a)) {
            out = c;
        }
    } else {
        if (same_colour(b, d) && !same_colour(b, a) && !same_colour(d, c)) {
            out = d;
        }
    }

    return out;
}
