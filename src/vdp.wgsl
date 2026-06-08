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
    // Second per-scanline array — R2 (display page selector) lives in
    // byte 0; other lanes reserved for future per-scanline regs.
    scanline_regs2: array<vec4<u32>, 64>,
    // Third per-scanline array — colour/pattern table bases:
    //   bits 0..7   = R3  (colour table base, low byte)
    //   bits 8..15  = R4  (pattern generator table base)
    //   bits 16..23 = R10 (colour table extension, G3+)
    scanline_regs3: array<vec4<u32>, 64>,
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

// Second per-scanline pack — currently only R2 in byte 0.
fn scanline_packed2(line: u32) -> u32 {
    let v = u.scanline_regs2[line >> 2u];
    switch line & 3u {
        case 0u:  { return v.x; }
        case 1u:  { return v.y; }
        case 2u:  { return v.z; }
        default:  { return v.w; }
    }
}
fn line_r2(line: u32) -> u32 { return  scanline_packed2(line)        & 0xFFu; }
fn line_r0(line: u32) -> u32 { return (scanline_packed2(line) >>  8u) & 0xFFu; }
fn line_r1(line: u32) -> u32 { return (scanline_packed2(line) >> 16u) & 0xFFu; }
fn line_r7(line: u32) -> u32 { return (scanline_packed2(line) >> 24u) & 0xFFu; }

// Third per-scanline pack — table base registers.
fn scanline_packed3(line: u32) -> u32 {
    let v = u.scanline_regs3[line >> 2u];
    switch line & 3u {
        case 0u:  { return v.x; }
        case 1u:  { return v.y; }
        case 2u:  { return v.z; }
        default:  { return v.w; }
    }
}
fn line_r3(line: u32)  -> u32 { return  scanline_packed3(line)        & 0xFFu; }
fn line_r4(line: u32)  -> u32 { return (scanline_packed3(line) >>  8u) & 0xFFu; }
fn line_r10(line: u32) -> u32 { return (scanline_packed3(line) >> 16u) & 0xFFu; }

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
    // R8 bit 1 (SPD) = sprite display disable. When set, sprites must
    // not render. Per V9938 spec §2.2 and fMSX MSX.h `SpritesOFF` macro.
    if ((reg(8u) & 0x02u) != 0u) { return 0xFFu; }

    // Per V9938 spec §1.2 sprite mode 1 (SCREEN 1/2/3 - G1/G2/MC):
    //   R#5  : |A14|A13|A12|A11|A10|A9|A8|A7|  — all 8 bits used
    //   R#11 : | 0 | 0 | 0 | 0 | 0 | 0 |A16|A15|
    //   R#6  : | 0 | 0 |A16|A15|A14|A13|A12|A11| — 6 bits (mask 0x3F)
    // fMSX MSX.c MSK[] table confirms: SCR 1/2/3 use R5 & 0xFF.
    // Previously we masked R5 with 0x7F (clearing A14) and used only 3
    // bits of R6, limiting MSX2 sprite mode 1 to the bottom 16 KiB.
    let sat_base = ((reg(5u) & 0xFFu) << 7u) | ((reg(11u) & 0x03u) << 15u);
    let sg_base  = (reg(6u) & 0x3Fu) << 11u;

    let r1 = reg(1u);
    let size16 = (r1 & 0x02u) != 0u;     // SI: 16×16 sprites
    let mag    = (r1 & 0x01u) != 0u;     // MAG: 2× magnification

    // OH = on-screen size; IH = pattern size (used for negative-Y wrap
    // threshold per fMSX Common.h Sprites()).
    var box_size: u32 = 8u;
    if (size16) { box_size = 16u; }
    if (mag)    { box_size = box_size * 2u; }
    var inner_h: u32 = 8u;
    if (size16) { inner_h = 16u; }

    // Apply per-scanline R23 (Vertical Scroll). Sprites scroll with the
    // background per V9938 hardware — verified in fMSX Common.h Sprites()
    // line 118 (`Y += VScroll;`).
    let vscroll = line_r23(py);

    var hit: u32 = 0xFFu;

    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let sat_addr = sat_base + s * 4u;
        let y_raw = vram_byte(sat_addr);

        // 0xD0 = 208 is the end-of-list sentinel for sprite mode 1.
        if (y_raw == 0xD0u) { break; }

        let cbyte = vram_byte(sat_addr + 3u);
        let color = cbyte & 0x0Fu;
        if (color == 0u) { continue; }    // sprite transparent

        // K = (y_raw - VScroll) & 0xFF; wrap negative when K > 256 - IH.
        let k_byte = (y_raw - vscroll) & 0xFFu;
        var sy: i32;
        if (k_byte > 256u - inner_h) {
            sy = i32(k_byte) + 1 - 256;
        } else {
            sy = i32(k_byte) + 1;
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

        // First hit wins — lowest sprite index = highest priority per
        // TMS9918/V9938 spec. Previously we overwrote on every hit so
        // sprite #31 always won, which was backwards from the spec.
        if (hit == 0xFFu) {
            hit = color;
        }
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
    // Per fMSX MSK[1] (SCR 1 = G1): R2 mask 0x7F, R3 mask 0xFF, R4 mask
    // 0x3F. The wider masks vs pure TMS9918 (0x0F/0xFF/0x07) let MSX2
    // software put tables anywhere in the 128 KiB V9938 VRAM while
    // running in TMS9918-compat mode 1. Pure MSX1 games don't notice
    // because they write 0 to the upper bits.
    let nt_base = (line_r2(py) & 0x7Fu) << 10u;
    let pt_base = (line_r4(py) & 0x3Fu) << 11u;
    let ct_base = line_r3(py) << 6u;

    // VScroll applied before tile-coord derivation, per fMSX RefreshLine1.
    let py_s = (py + line_r23(py)) & 0xFFu;
    let tile_x = px >> 3u;
    let tile_y = py_s >> 3u;
    let sub_x  = px & 7u;
    let sub_y  = py_s & 7u;

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
    // Per fMSX MSK[2] (SCR 2 = G2): R2 mask 0x7F, R3 mask 0x80, R4 mask
    // 0x3C. The wider R2/R4 masks (vs pure TMS9918 0x0F/0x04) support
    // V9938 G2 with extended VRAM addressing. R3 mask 0x80 is the
    // "only A13 settable, bits 0..6 forced 1" semantics matching real
    // hardware — same idea as shade_g3.
    let nt_base = (line_r2(py) & 0x7Fu) << 10u;
    let pt_base = (line_r4(py) & 0x3Cu) << 11u;
    let ct_base = (line_r3(py) & 0x80u) << 6u;

    // VScroll applied before tile-coord derivation, per fMSX RefreshLine2.
    let py_s = (py + line_r23(py)) & 0xFFu;
    let tile_x = px >> 3u;
    let tile_y = py_s >> 3u;
    let sub_x  = px & 7u;
    let sub_y  = py_s & 7u;
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

    let nt_base = (line_r2(py) & 0x0Fu) << 10u;
    let pt_base = (line_r4(py) & 0x07u) << 11u;

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

    // First the canvas-area gate (border vs active region). The R1 bit 6
    // (BL) check moves PER-SCANLINE below — same scheme as fMSX
    // RefreshLineN, which calls `ClearLine(P, XPal[BGColor])` per line
    // when ScreenON is clear at that time. Frame-global BL was wrong for
    // games like Quarth that toggle BL via line-IRQ split: previously we
    // either rendered the whole frame (using last-write BL) or blanked
    // the whole frame, never the mid-frame split.
    //
    // Visible-area height depends on R9 bit 7 (LN): clear = 192 lines
    // (TMS9918-compatible, top-aligned at y=24), set = 212 lines
    // (V9938 MSX2 mode used by games with a status bar — KV2, Vampire
    // Killer, Metal Gear, etc.). Both modes keep canvas_y=24 as the top
    // of the active area; 212-mode extends 20 rows further down so the
    // score / UI band below the playfield is no longer clipped.
    let lines_212 = (reg(9u) & 0x80u) != 0u;
    let active_bottom: u32 = select(216u, 236u, lines_212);

    // Vertical gate first. Top/bottom borders sit OUTSIDE the per-scanline
    // snapshot range (0..active_height), so they have no line-specific R7 —
    // paint them with the frame-global backdrop.
    if (canvas_y < 24u || canvas_y >= active_bottom) {
        return u.palette[backdrop()];
    }

    // Inside the active vertical band: a per-scanline R7 snapshot exists.
    // R7's low nibble is the backdrop/border colour, which games like
    // Quarth split PER BAND via line-IRQ writes. Using line_r7(py) instead
    // of the frame-global R7 makes the side borders (and the blanked-band
    // fallback below) follow whatever colour the game set for THIS line —
    // previously the last-written R7 bled across the whole frame, painting
    // the side borders grey in bands where they should match the playfield.
    let py = canvas_y - 24u;
    let line_backdrop = line_r7(py) & 0x0Fu;

    // Horizontal gate: left/right side borders follow the per-scanline
    // backdrop.
    if (canvas_x < 32u || canvas_x >= 288u) {
        return u.palette[line_backdrop];
    }

    let px = canvas_x - 32u;

    // Per-scanline BL check. Quarth toggles BL=0 via a line-IRQ split to
    // give the CPU more VRAM bandwidth for its YMMM scroll commands —
    // matching this means the blanked portion shows backdrop, not the
    // garbage VRAM data that page 1's "scratch" rows would expose.
    if ((line_r1(py) & 0x40u) == 0u) {
        return u.palette[line_backdrop];
    }

    // Mode dispatch — see header comment for the M1/M2/M3 truth table.
    // V9938 software (KV2, Vampire Killer, Quarth, ...) switches mode
    // mid-frame via line-interrupt handlers: a G4 bitmap playfield with
    // a G1 text status bar at the bottom, for example. The per-line R0
    // and R1 snapshot lets the shader run a DIFFERENT shading path on
    // each scanline.
    let r0_line = line_r0(py);
    let r1_line = line_r1(py);
    let m1 = (r1_line >> 4u) & 1u;
    let m2 = (r1_line >> 3u) & 1u;
    let m3 = (r0_line >> 1u) & 1u;
    let m4 = (r0_line >> 2u) & 1u;
    let m5 = (r0_line >> 3u) & 1u;

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
//   * Colour byte layout (verified against openMSX SpriteChecker.cc and
//     WebMSX VDP.js; previous version had EC at the wrong bit position
//     and CC semantics inverted, which is why multi-colour sprites
//     flickered and CC overlays sometimes rendered solo):
//        bit 7 = EC (early clock — shift X by -32)
//        bit 6 = CC (this sprite ONLY contributes if a lower-index
//                     CC=0 sprite has already hit this pixel; OR-mixes
//                     its colour onto that base)
//        bit 5 = IC (individual collision disable — collision detection
//                     skipped for this sprite line; we don't model
//                     collisions yet, so this bit is currently ignored)
//        bit 4 = 0  (reserved)
//        bits 3-0 = colour (4-bit palette index, 0 = transparent)
//
//   * CC semantics: a sprite with CC=1 is INVISIBLE unless there's a
//     prior hit at the same pixel from a sprite with CC=0 (lower index).
//     The CC=1 sprite's colour then ORs onto the base. Multi-colour
//     character sprites work by drawing a "base" sprite (CC=0) covering
//     the full silhouette, then adding CC=1 "highlight" sprites on top
//     for inner detail.
//
// 8-per-line cap: per real V9938, at most 8 sprites whose Y range covers
// the current scanline get processed — regardless of whether they
// actually have an opaque pixel at the X being drawn. Excess sprites
// are silently skipped (we don't yet update the 9S status flag).
fn sample_sprite_mode2(px: u32, py: u32) -> u32 {
    // R8 bit 1 (SPD) = sprite display disable. Per V9938 spec §2.2 and
    // fMSX MSX.h `SpritesOFF` macro: when set, sprites are hidden.
    if ((reg(8u) & 0x02u) != 0u) { return 0xFFu; }

    // Per-scanline R5 / R6 / R11: V9938 software often points to a
    // different SAT for different bands of the screen via line interrupts.
    // The CPU side captures these at the start of each visible scanline;
    // we look up the one matching the current `py` here.
    let r5 = line_r5(py);
    let r11 = line_r11(py);
    // V9938 spec §2.4 SAT base address layout:
    //   R#5  : |A14|A13|A12|A11|A10|A9 | A8| A7|
    //   R#11 : | 0 | 0 | 0 | 0 | 0 | 0 |A16|A15|
    // NB: The figure on page 93 annotates R5[1:0] as "always set to 1",
    // but the worked G4 example on page 40 puts SAT at 0x7A00 which is
    // only reachable WITHOUT that forcing. The example reflects how the
    // chip actually behaves; the figure annotation is misleading (we
    // tested forcing and it makes mode-2 sprites disappear entirely
    // because games write R5 expecting the raw value to be honoured).
    let attr_base = ((r5 & 0xFCu) << 7u) | ((r11 & 0x03u) << 15u);
    let color_base = attr_base - 0x200u;
    let sg_base = (line_r6(py) & 0x3Fu) << 11u;

    let r1 = reg(1u);
    let size16 = (r1 & 0x02u) != 0u;
    let mag    = (r1 & 0x01u) != 0u;

    // Inner height (IH) = unmagnified pattern size; outer height (OH) =
    // on-screen size. fMSX (Common.h ColorSprites): OH uses both SI and
    // MAG bits, IH uses only SI. The Y-wrap threshold uses IH, not OH.
    var box_size: u32 = 8u;          // OH — on-screen sprite height
    if (size16) { box_size = 16u; }
    if (mag)    { box_size = box_size * 2u; }
    var inner_h: u32 = 8u;           // IH — pattern data height
    if (size16) { inner_h = 16u; }

    // Per-scanline R23 (Vertical Scroll). Sprites scroll with the
    // background per V9938 hardware (verified via fMSX ColorSprites and
    // openMSX SpriteChecker::checkSprites2 — both subtract VScroll from
    // the raw sprite Y attribute). Without this our sprites appear at
    // fixed Y when the game intends them to track the scrolled playfield
    // — main characters in scroll-heavy games disappear when VScroll is
    // non-zero and game logic doesn't compensate.
    let vscroll = line_r23(py);

    // ── Pass 1: mark the first 8 Y-overlapping sprites (lowest index),
    //    per the V9938 8-per-line cap. Counting includes transparent and
    //    off-X sprites and stops at the Y=0xD8 sentinel — matching fMSX
    //    ColorSprites' counting loop. `marked` is a 32-bit set of the
    //    sprites that will actually be drawn.
    var marked: u32 = 0u;
    var count: u32 = 0u;
    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let y_raw = vram_byte(attr_base + s * 4u);
        if (y_raw == 0xD8u) { break; }
        let k_byte = (y_raw - vscroll) & 0xFFu;
        var sy: i32;
        if (k_byte > 256u - inner_h) {
            sy = i32(k_byte) + 1 - 256;
        } else {
            sy = i32(k_byte) + 1;
        }
        let dy = i32(py) - sy;
        if (dy < 0 || dy >= i32(box_size)) { continue; }
        count = count + 1u;
        if (count > 8u) { break; }   // 9th+ sprite on this line is dropped
        marked = marked | (1u << s);
    }

    // ── Pass 2: draw the marked sprites HIGH→LOW so the lowest index wins
    //    on overwrite (V9938/TMS9918 priority). The Color-Combination chain
    //    decides OR-combine vs overwrite per fMSX ColorSprites' `OrThem`:
    //    a sprite OR-combines iff the NEXT-HIGHER marked sprite had CC=1.
    //    `or_them` is advanced for EVERY marked sprite — including
    //    transparent-colour ones — so the CC chain propagates exactly as
    //    on hardware. (The previous code gated CC sprites behind "a lower
    //    sprite shares the scanline", which made multi-colour sprites —
    //    which Quarth uses pervasively — drop pixels or vanish.)
    //
    //    Colour byte bits (V9938 §2.6): 7=EC (shift X −32), 6=CC, 5=IC
    //    (collision-disable, unused), 3..0 = colour (0 = transparent).
    var result: u32 = 0xFFu;     // 0xFF = no sprite pixel here
    var or_them: u32 = 0u;       // bit 0x20 = prev (higher) sprite's CC
    for (var si: i32 = 31; si >= 0; si = si - 1) {
        let s = u32(si);
        if ((marked & (1u << s)) == 0u) { continue; }

        let attr_addr = attr_base + s * 4u;
        let y_raw = vram_byte(attr_addr);
        let k_byte = (y_raw - vscroll) & 0xFFu;
        var sy: i32;
        if (k_byte > 256u - inner_h) {
            sy = i32(k_byte) + 1 - 256;
        } else {
            sy = i32(k_byte) + 1;
        }
        var ly: u32 = u32(i32(py) - sy);
        if (mag) { ly = ly >> 1u; }

        // Per-line colour byte; advance the CC chain before opacity tests.
        let color_byte = vram_byte(color_base + s * 16u + ly);
        or_them = or_them | (color_byte & 0x40u);
        let or_mode = (or_them & 0x20u) != 0u;

        let color = color_byte & 0x0Fu;
        if (color != 0u) {
            let ec = (color_byte & 0x80u) != 0u;
            var sx: i32 = i32(vram_byte(attr_addr + 1u));
            if (ec) { sx = sx - 32; }
            let dx = i32(px) - sx;
            if (dx >= 0 && dx < i32(box_size)) {
                var lx: u32 = u32(dx);
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
                if (((pat_byte >> bit) & 1u) != 0u) {
                    // Opaque pixel: OR-combine into the chain, or overwrite.
                    if (or_mode && result != 0xFFu) {
                        result = result | color;
                    } else {
                        result = color;
                    }
                }
            }
        }
        or_them = or_them >> 1u;
    }

    return result;
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
    // Per fMSX MSK[4] (SCR 4 = G3 mode) and V9938 spec page 36:
    //   R#2: |A16|...|A10|  mask 0x7F (NT base, A16..A10)
    //   R#3: |A13|1|1|1|1|1|1|1|  mask 0x80 (CT base, only A13 settable;
    //        bits 0..6 are forced to 1 internally — write any value).
    //   R#4: |A16|A15|A14|A13|1|1| mask 0x3C (PT base, A16..A13).
    //   R#10: bits 0..2 contribute A14..A16 of CT base.
    //
    // Previously we masked R3 with 0xFF (treating bits 0..6 as address
    // bits) which placed the colour table in a different location than
    // fMSX. With the wrong CT base, tile colours read as garbage for
    // games that write non-zero values in R3 bits 0..6.
    let nt_base = (line_r2(py) & 0x7Fu) << 10u;
    let pt_base = (line_r4(py) & 0x3Cu) << 11u;
    let ct_base = ((line_r3(py) & 0x80u) << 6u) | ((line_r10(py) & 0x07u) << 14u);

    // Apply per-scanline R23 (Vertical Scroll) before deriving tile
    // coordinates. fMSX Common.h RefreshLine4: `Y += VScroll;` is done
    // before the tilemap index calc. Byte-wrap at 256 (real V9938
    // behaviour — beyond 192 the tile fetch wraps into the unused upper
    // VRAM region but we just read raw VRAM there).
    let py_s = (py + line_r23(py)) & 0xFFu;
    let tile_x = px >> 3u;
    let tile_y = py_s >> 3u;
    let sub_x  = px & 7u;
    let sub_y  = py_s & 7u;
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
    // Per-scanline R2: Quarth (and other MSX2 software) flips the display
    // page mid-frame via line-interrupt-driven R2 writes — top band on
    // page 0, bottom on page 1, etc. Without the per-line lookup we'd
    // render every line from the LAST-written R2, missing whichever
    // band's page was active earlier.
    let page_base = ((line_r2(py) >> 5u) & 3u) << 15u;
    // R23 likewise changes per-scanline for split-screen scrolls. Wrap
    // mod 256 because a G4 page is 256 rows (32 KiB / 128 bytes/row).
    // Sprites DO follow R23 too — see sample_sprite_mode2 for the
    // VScroll application that matches fMSX/openMSX semantics. (Earlier
    // versions of this comment incorrectly claimed sprites were R23-
    // independent; in reality V9938 sprites scroll with the background.)
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
