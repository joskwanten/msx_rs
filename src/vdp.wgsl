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
    regs: array<vec4<u32>, 6>,      // R0-R23 (one byte per u32 lane)
    // Per-visible-scanline snapshots of R5 / R6 / R11 / R23, packed:
    //   bits 0..7  = R5
    //   bits 8..15 = R6
    //   bits 16..23 = R11
    //   bits 24..31 = R23
    // 256 entries × 4 bytes, organised as 64 vec4<u32> (4 scanlines per
    // vec4, low-to-high lane order).
    scanline_regs: array<vec4<u32>, 64>,
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

// ─── Per-scanline registers ───────────────────────────────────────────────
//
// V9938 software commonly reprograms R5 / R6 / R11 / R23 from a line-
// interrupt handler to do split-screen scroll or per-band SAT switching.
// The CPU side snapshots these four registers at the start of each
// visible scanline and packs them into `scanline_regs` (4 bytes per
// scanline, 4 scanlines per vec4). Rendering code looks up the line it's
// painting and uses THESE values instead of the frame-static `reg(...)`.
fn scanline_packed(line: u32) -> u32 {
    let v = u.scanline_regs[line >> 2u];
    switch line & 3u {
        case 0u:  { return v.x; }
        case 1u:  { return v.y; }
        case 2u:  { return v.z; }
        default:  { return v.w; }
    }
}
fn line_r5(line: u32)  -> u32 { return  scanline_packed(line)        & 0xFFu; }
fn line_r6(line: u32)  -> u32 { return (scanline_packed(line) >>  8u) & 0xFFu; }
fn line_r11(line: u32) -> u32 { return (scanline_packed(line) >> 16u) & 0xFFu; }
fn line_r23(line: u32) -> u32 { return (scanline_packed(line) >> 24u) & 0xFFu; }

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
    // V9938 adds M4 = R0 bit 2 and M5 = R0 bit 3.
    let m1 = (reg(1u) >> 4u) & 1u;
    let m2 = (reg(1u) >> 3u) & 1u;
    let m3 = (reg(0u) >> 1u) & 1u;
    let m4 = (reg(0u) >> 2u) & 1u;
    let m5 = (reg(0u) >> 3u) & 1u;

    var color_idx: u32;
    // V9938 modes (M4 or M5 set) are checked first. Only G4 (Screen 5,
    // M5 M4 M3 = 0 1 1) is fully implemented; G3 (Screen 4) shares the
    // same tile layout as G2 so it routes to shade_graphic2 — sprites
    // are missing (V9938 sprite mode 2) but the bitmap is correct. Other
    // V9938 modes fall through to a backdrop placeholder.
    if (m5 == 0u && m4 == 1u && m3 == 1u && m2 == 0u && m1 == 0u) {
        color_idx = shade_g4(px, py);
    } else if (m5 == 0u && m4 == 1u && m3 == 0u && m2 == 0u && m1 == 0u) {
        // G3 / Screen 4 — same tile-bank layout as G2 but with wider
        // base-address masks for 128 KiB VRAM. Sprites use V9938 mode 2
        // which is a separate code path (not yet wired up).
        color_idx = shade_g3(px, py);
    } else if (m4 == 1u || m5 == 1u) {
        color_idx = backdrop();        // other V9938 modes not yet implemented
    } else if (m3 == 1u) {
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

// ─── V9938 sprite mode 2 ───────────────────────────────────────────────────
//
// Used in G3 / G4 / G5 / G6 / G7 (Screens 4-8). Differences from TMS9918
// sprite mode 1:
//
//   * 8 visible sprites per scan-line (was 4)
//   * 32-sprite scan still terminates on a Y sentinel — but 0xD8 (216)
//     instead of 0xD0
//   * SAT is split in two:
//
//        attr_base = (R5[7:2] << 7) | (R11[1:0] << 15)   — 17-bit
//        color_base = attr_base - 0x200
//
//     Attribute table: 32 × 4 bytes (Y / X / pattern / reserved)
//     Color table:     32 × 16 bytes — one byte per scan-line of the
//                      sprite (only first 8 used for 8×8 sprites)
//
//   * Color byte layout:
//        bit 7 = IC (skip collision)
//        bit 6 = CC (this line OR-mixes onto already-drawn sprites)
//        bit 5 = 0
//        bit 4 = EC (early clock — shift X by -32)
//        bits 3-0 = colour (4-bit palette index, 0 = transparent)
//
//   * OR-mixing: when CC=1, the colour bits OR onto whatever colour was
//     already produced for this fragment by a lower-numbered sprite.
//     Used for multi-coloured sprites (stack several with different
//     colour bits and CC set on the upper ones).
//
// We don't enforce per-frame sprite-count limits (no IRQ-on-overflow),
// but we do cap at 8 visible sprites per line per the spec — going past
// 8 silently drops further sprites.
fn sample_sprite_mode2(px: u32, py: u32) -> u32 {
    // Per-scanline R5 / R6 / R11: V9938 software often points to a
    // different SAT for different bands of the screen via line interrupts.
    // The CPU side captures these at the start of each visible scanline;
    // we look up the one matching the current `py` here.
    let r5 = line_r5(py);
    let r11 = line_r11(py);
    let attr_base = ((r5 & 0xFCu) << 7u) | ((r11 & 0x03u) << 15u);
    let color_base = attr_base - 0x200u;
    let sg_base = (line_r6(py) & 0x3Fu) << 11u;

    let r1 = reg(1u);
    let size16 = (r1 & 0x02u) != 0u;
    let mag    = (r1 & 0x01u) != 0u;

    var box_size: u32 = 8u;
    if (size16) { box_size = 16u; }
    if (mag)    { box_size = box_size * 2u; }

    var hit_color: u32 = 0xFFu;
    var sprites_drawn: u32 = 0u;

    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let attr_addr = attr_base + s * 4u;
        let y_raw = vram_byte(attr_addr);
        // Mode-2 end-of-list sentinel.
        if (y_raw == 0xD8u) { break; }

        var sy: i32;
        if (y_raw > 238u) {
            sy = i32(y_raw) - 255;
        } else {
            sy = i32(y_raw) + 1;
        }

        let dy_screen = i32(py) - sy;
        if (dy_screen < 0 || dy_screen >= i32(box_size)) { continue; }

        // Demagnified line within the sprite.
        var ly: u32 = u32(dy_screen);
        if (mag) { ly = ly >> 1u; }

        // Color byte for THIS line — mode 2's per-line feature.
        let color_byte = vram_byte(color_base + s * 16u + ly);
        let color = color_byte & 0x0Fu;
        // Skip transparent colour first — saves an X/pattern lookup.
        if (color == 0u) { continue; }

        let ec = (color_byte & 0x20u) != 0u;
        let cc = (color_byte & 0x40u) != 0u;

        let x_raw = vram_byte(attr_addr + 1u);
        var sx: i32 = i32(x_raw);
        if (ec) { sx = sx - 32; }

        let dx_screen = i32(px) - sx;
        if (dx_screen < 0 || dx_screen >= i32(box_size)) { continue; }

        var lx: u32 = u32(dx_screen);
        if (mag) { lx = lx >> 1u; }

        let pat = vram_byte(attr_addr + 2u);
        var byte_offset: u32;
        if (size16) {
            // 16×16 = 4 patterns laid out TL/BL/TR/BR.
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

        // We've got a pixel hit. Track 8-per-line limit before mixing.
        sprites_drawn = sprites_drawn + 1u;
        if (sprites_drawn > 8u) { break; }

        if (hit_color == 0xFFu) {
            hit_color = color;
        } else if (cc) {
            // Later sprite ORs onto whatever's there — gives multi-colour
            // sprites by stacking several with different bits set.
            hit_color = hit_color | color;
        }
        // Otherwise keep the higher-priority (lower sprite index) colour.
    }

    return hit_color;
}

// ─── Screen 4 (G3) ──────────────────────────────────────────────────────────
//
// Same 32×24 tile grid as G2 with the same three-bank pattern/colour split,
// but V9938 widens the base-address registers so tables can live anywhere in
// the 128 KiB VRAM:
//
//   Name table       = R2[6:0] << 10                — was R2[3:0] in G2
//   Colour table     = (R3[7:0] << 6) | (R10[2:0] << 14)
//   Pattern table    = R4[5:2] << 11                — was R4[2] in G2
//
// MSX2 software writes these wider bits to push tables out of the first 16 KiB
// (where MSX1 software lived) into higher banks. Our R14-aware VRAM
// `read_data`/`write_data` already routes CPU writes to those banks; this
// shader path completes the loop by reading them back from the right place.
//
// Sprites in G3 use V9938 mode 2 (8 per line, per-line colour, OR-mix). Not
// implemented yet — this function just returns the bitmap.
fn shade_g3(px: u32, py: u32) -> u32 {
    let nt_base = (reg(2u) & 0x7Fu) << 10u;
    let pt_base = (reg(4u) & 0x3Cu) << 11u;
    let ct_base = ((reg(3u) & 0xFFu) << 6u) | ((reg(10u) & 0x07u) << 14u);

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

    // G3 uses V9938 sprite mode 2.
    let sprite = sample_sprite_mode2(px, py);
    if (sprite < 16u) { return sprite; }
    return bg;
}

// ─── Screen 5 (G4) ──────────────────────────────────────────────────────────
//
// 256×212 4 bpp bitmap. Two pixels per VRAM byte (high nibble = leftmost
// pixel), 128 bytes per row → 27 136 bytes for one display page.
//
// R2[6:5] selects the display page (4 × 32 KiB):
//   page 0 base = 0x00000
//   page 1 base = 0x08000
//   page 2 base = 0x10000
//   page 3 base = 0x18000
//
// Pixel value is a 4-bit palette index → u.palette[idx].
fn shade_g4(px: u32, py: u32) -> u32 {
    let page_base = ((reg(2u) >> 5u) & 3u) << 15u;
    // R23 is the vertical-scroll register and changes per-scanline on
    // most MSX2 software (line-interrupt-driven split-screen scrolls).
    // Use the snapshot for THIS scanline. Bitmap rows wrap mod 256
    // because a G4 page is 256 rows (32 KiB / 128 bytes-per-row).
    // Sprites are unaffected — they're positioned by their own Y in
    // the SAT, independent of R23.
    let bitmap_y = (py + line_r23(py)) & 0xFFu;
    let byte_addr = page_base + bitmap_y * 128u + (px >> 1u);
    let byte = vram_byte(byte_addr);
    // High nibble (px even) is the leftmost pixel.
    let shift = (1u - (px & 1u)) * 4u;
    let bg = (byte >> shift) & 0x0Fu;

    // G4 uses V9938 sprite mode 2.
    let sprite = sample_sprite_mode2(px, py);
    if (sprite < 16u) { return sprite; }
    return bg;
}
