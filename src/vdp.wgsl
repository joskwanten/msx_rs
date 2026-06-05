// TMS9918A pixel shader — supports Graphic 1 (Screen 1), Graphic 2 (Screen 2),
// and Text (Screen 0). Multicolor (Screen 3) is detected but falls back to
// backdrop for now.
//
// Mode is selected by three register bits (M1, M2, M3). Beware the naming:
// per the TMS9918 datasheet, **M3 lives in R0 bit 1**, not R1. Easy to swap
// because R0/R1 are adjacent and the bit-position numbers are close.
//
//   M1 (R1 bit 4)  M2 (R1 bit 3)  M3 (R0 bit 1)   Mode
//   ─────────────  ─────────────  ─────────────   ──────────────────
//        0              0              0          Graphic 1 (Screen 1)
//        1              0              0          Text       (Screen 0)
//        0              1              0          Multicolor (Screen 3)
//        0              0              1          Graphic 2  (Screen 2)
//
// Per-mode VRAM layouts:
//
//   Graphic 1 (Screen 1):
//     NT  base = (R2 & 0x0F) << 10   — 32×24 tile indices
//     PT  base = (R4 & 0x07) << 11   — 256 patterns × 8 bytes
//     CT  base = R7's <... no, R3 << 6 — 32 bytes (1 byte per group of 8 patterns)
//
//   Graphic 2 (Screen 2):
//     NT  base = (R2 & 0x0F) << 10   — 32×24 tile indices
//     PT  base = (R4 & 0x04) << 11   — 3 banks × 256 patterns × 8 bytes
//     CT  base = (R3 & 0x80) << 6    — 6 KiB, one byte per pixel-row
//
//   Text (Screen 0):
//     NT  base = (R2 & 0x0F) << 10   — 40×24 character indices
//     PT  base = (R4 & 0x07) << 11   — 256 patterns × 8 bytes (only 6 bits/row)
//     fg = R7 >> 4, bg = R7 & 0x0F (whole screen, no per-tile colors)
//     8-pixel border left + right (text area = pixels 8..247)
//
// In Graphic 1/2: color 0 in a tile means "transparent" → backdrop (R7 & 0x0F).
// R1 bit 6 = display enabled; when clear the whole screen shows backdrop.

struct Uniforms {
    framebuffer_size: vec2<f32>,
    _pad: vec2<u32>,
    regs: array<vec4<u32>, 2>,      // 8 bytes of registers, one per lane
    palette: array<vec4<f32>, 16>,  // TMS9918 fixed palette (index 0 = transparent)
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> vram: array<u32>;

fn vram_byte(addr: u32) -> u32 {
    let word = vram[addr >> 2u];
    let shift = (addr & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn reg(i: u32) -> u32 {
    let v = u.regs[i >> 2u];
    switch i & 3u {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        case 2u: { return v.z; }
        default: { return v.w; }
    }
}

fn backdrop() -> u32 {
    return reg(7u) & 0x0Fu;
}

// In Graphic 1/2, fg/bg = 0 means transparent — fall through to backdrop.
fn apply_transparency(color: u32) -> u32 {
    return select(color, backdrop(), color == 0u);
}

// ─── Sprites ───────────────────────────────────────────────────────────────
//
// TMS9918 has 32 sprites in the Sprite Attribute Table (SAT):
//
//   Byte 0  Y position. 0xD0 (208) terminates the table — every sprite past
//           that entry is suppressed regardless of state. Y = 0 means the
//           sprite displays starting at scan line 1 (the +1 offset is real
//           hardware behaviour). Values 239..255 wrap as negative — used to
//           park sprites just above the screen.
//   Byte 1  X position. Negative offsets via the Early Clock bit below.
//   Byte 2  Pattern index (masked to 4-aligned in 16×16 mode).
//   Byte 3  bit 7 = Early Clock (shifts X by -32), bits 0..3 = color.
//
// R1 bit 1 selects 8×8 vs 16×16. R1 bit 0 turns on 2× magnification. Color
// 0 means the sprite is fully transparent (pattern bit ignored).
//
// Returns the sprite color hit at (px, py), or 0xFF for "no sprite here".
// Iteration is 0..31 with the *last* hit winning, matching the TypeScript
// reference. Real TMS9918 gives sprite 0 priority instead — for our games
// the difference only shows up when sprites overlap.

fn sample_sprite(px: u32, py: u32) -> u32 {
    let sat_base = (reg(5u) & 0x7Fu) << 7u;
    let sg_base  = (reg(6u) & 0x07u) << 11u;

    let r1 = reg(1u);
    let size16 = (r1 & 0x02u) != 0u;     // SI: 16×16 sprites
    let mag    = (r1 & 0x01u) != 0u;     // MAG: 2× magnification

    // Screen bounding box, after magnification.
    var box_size: u32 = 8u;
    if (size16) { box_size = 16u; }
    if (mag)    { box_size = box_size * 2u; }

    var hit: u32 = 0xFFu;

    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let sat_addr = sat_base + s * 4u;
        let y_raw = vram_byte(sat_addr);

        // 0xD0 is the end-of-list sentinel.
        if (y_raw == 0xD0u) { break; }

        let cbyte = vram_byte(sat_addr + 3u);
        let color = cbyte & 0x0Fu;
        if (color == 0u) { continue; }    // sprite transparent

        // Y placement: +1 normally, but 239..255 wrap negative.
        var sy: i32;
        if (y_raw > 238u) {
            sy = i32(y_raw) - 255;
        } else {
            sy = i32(y_raw) + 1;
        }

        let x_raw = vram_byte(sat_addr + 1u);
        var sx: i32 = i32(x_raw);
        if ((cbyte & 0x80u) != 0u) {       // Early Clock
            sx = sx - 32;
        }

        let dx = i32(px) - sx;
        let dy = i32(py) - sy;
        if (dx < 0 || dy < 0 || dx >= i32(box_size) || dy >= i32(box_size)) {
            continue;
        }

        // Pattern-local coordinates after demagnification.
        var lx = u32(dx);
        var ly = u32(dy);
        if (mag) {
            lx = lx >> 1u;
            ly = ly >> 1u;
        }

        let pat = vram_byte(sat_addr + 2u);

        var byte_offset: u32;
        if (size16) {
            // 16×16 sprite spans 4 patterns aligned at index & 0xFC, ordered
            // TL, BL, TR, BR (vertical strips). Pick the right quadrant.
            let quad_x = lx >> 3u;
            let quad_y = ly >> 3u;
            let pattern_idx = (pat & 0xFCu) + quad_x * 2u + quad_y;
            byte_offset = pattern_idx * 8u + (ly & 7u);
        } else {
            byte_offset = pat * 8u + ly;
        }

        let pat_byte = vram_byte(sg_base + byte_offset);
        let bit = 7u - (lx & 7u);
        if (((pat_byte >> bit) & 1u) == 0u) { continue; }

        hit = color;   // overwrite — last hit wins
    }

    return hit;
}

// ─── Graphic 1 (Screen 1) ──────────────────────────────────────────────────
//
// 32×24 tiles of 8×8 pixels. Pattern table is a flat 256-entry array (no
// banking). Color table is 32 bytes: one (fg<<4 | bg) byte per *group* of 8
// consecutive tile indices. That's why MSX1 BASIC text on Screen 1 has only
// 32 / 8 = 32 color choices across the alphabet.
fn shade_graphic1(px: u32, py: u32) -> u32 {
    let nt_base = (reg(2u) & 0x0Fu) << 10u;
    let pt_base = (reg(4u) & 0x07u) << 11u;
    let ct_base = reg(3u) << 6u;

    let tile_x = px >> 3u;
    let tile_y = py >> 3u;
    let sub_x  = px & 7u;
    let sub_y  = py & 7u;

    let tile_num = vram_byte(nt_base + tile_y * 32u + tile_x);
    let pat = vram_byte(pt_base + (tile_num << 3u) + sub_y);
    let col = vram_byte(ct_base + (tile_num >> 3u));
    let is_fg = ((pat >> (7u - sub_x)) & 1u) == 1u;

    let bg = apply_transparency(select(col & 0x0Fu, (col >> 4u) & 0x0Fu, is_fg));

    let sprite = sample_sprite(px, py);
    if (sprite < 16u) { return sprite; }
    return bg;
}

// ─── Graphic 2 (Screen 2) ──────────────────────────────────────────────────
//
// Same 32×24 tile grid, but pattern + color tables are split into three
// 256-entry "banks", one per vertical third of the screen. Each color byte
// covers one *row* of one tile — so per-pixel-row coloring is possible.
fn shade_graphic2(px: u32, py: u32) -> u32 {
    let nt_base = (reg(2u) & 0x0Fu) << 10u;
    let pt_base = (reg(4u) & 0x04u) << 11u;
    let ct_base = (reg(3u) & 0x80u) << 6u;

    let tile_x = px >> 3u;
    let tile_y = py >> 3u;
    let sub_x  = px & 7u;
    let sub_y  = py & 7u;
    let third  = tile_y >> 3u;        // 0=top, 1=mid, 2=bottom

    let tile_num = vram_byte(nt_base + tile_y * 32u + tile_x);
    let bank_off = ((third << 8u) | tile_num) << 3u;

    let pat = vram_byte(pt_base + bank_off + sub_y);
    let col = vram_byte(ct_base + bank_off + sub_y);
    let is_fg = ((pat >> (7u - sub_x)) & 1u) == 1u;

    let bg = apply_transparency(select(col & 0x0Fu, (col >> 4u) & 0x0Fu, is_fg));

    let sprite = sample_sprite(px, py);
    if (sprite < 16u) { return sprite; }
    return bg;
}

// ─── Text (Screen 0) ───────────────────────────────────────────────────────
//
// 40 columns × 24 rows. Each character is 6 pixels wide (not 8). There are
// 8-pixel borders on the left and right showing the backdrop, so the active
// text area is pixels 8..247 — exactly 240 = 40 × 6. There is no color table;
// the whole screen uses the foreground (R7 high nibble) and background (R7
// low nibble) colors.
fn shade_graphic0(px: u32, py: u32) -> u32 {
    let r7 = reg(7u);
    let fg = (r7 >> 4u) & 0x0Fu;
    let bg = r7 & 0x0Fu;

    // Side borders show the backdrop (which on Screen 0 equals bg).
    if (px < 8u || px >= 248u) {
        return bg;
    }

    let nt_base = (reg(2u) & 0x0Fu) << 10u;
    let pt_base = (reg(4u) & 0x07u) << 11u;

    let text_x = px - 8u;
    let char_x = text_x / 6u;
    let char_y = py >> 3u;
    let sub_x  = text_x - char_x * 6u;   // 0..5
    let sub_y  = py & 7u;

    let tile_num = vram_byte(nt_base + char_y * 40u + char_x);
    let pat = vram_byte(pt_base + (tile_num << 3u) + sub_y);
    let is_fg = ((pat >> (7u - sub_x)) & 1u) == 1u;

    return select(bg, fg, is_fg);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Single fullscreen triangle covering [-1,1]^2 and overscan.
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // The MSX canvas mimics the CRT's overscan area: a 320×240 region with the
    // 256×192 active display centered inside, leaving a 32-pixel side border
    // and 24-pixel top/bottom border filled with the backdrop color. This is
    // the look real MSX games rely on — title screens often paint the border a
    // contrasting color via R7.
    let native = vec2<f32>(320.0, 240.0);
    let fb = u.framebuffer_size;
    let scale = max(1.0, floor(min(fb.x / native.x, fb.y / native.y)));
    let viewport = native * scale;
    let offset = (fb - viewport) * 0.5;
    let local = (in.pos.xy - offset) / scale;

    if (local.x < 0.0 || local.x >= native.x || local.y < 0.0 || local.y >= native.y) {
        // Outside the MSX canvas: paint the backdrop color too, so the
        // letterbox blends seamlessly with the border when the window is
        // larger than an integer multiple of 320×240.
        return u.palette[backdrop()];
    }

    let canvas_x = u32(local.x);
    let canvas_y = u32(local.y);

    // R1 bit 6 = display enable. When clear, the whole canvas (border AND
    // active area) is backdrop — BIOS uses this briefly during init.
    let display_on = (reg(1u) & 0x40u) != 0u;
    if (!display_on) {
        return u.palette[backdrop()];
    }

    // Active 256×192 area sits at offset (32, 24) within the 320×240 canvas.
    let in_active = canvas_x >= 32u && canvas_x < 288u
                 && canvas_y >= 24u && canvas_y < 216u;
    if (!in_active) {
        return u.palette[backdrop()];
    }

    let px = canvas_x - 32u;
    let py = canvas_y - 24u;

    // Mode dispatch — see header comment for the M1/M2/M3 truth table.
    // M3 = R0 bit 1, M2 = R1 bit 3 (per TMS9918 datasheet).
    let m1 = (reg(1u) >> 4u) & 1u;
    let m2 = (reg(1u) >> 3u) & 1u;
    let m3 = (reg(0u) >> 1u) & 1u;

    var color_idx: u32;
    if (m3 == 1u) {
        color_idx = shade_graphic2(px, py);
    } else if (m1 == 1u) {
        color_idx = shade_graphic0(px, py);
    } else if (m2 == 1u) {
        color_idx = backdrop();        // Multicolor not yet implemented
    } else {
        color_idx = shade_graphic1(px, py);
    }

    return u.palette[color_idx];
}
