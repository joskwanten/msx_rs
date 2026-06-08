# msx_rs

An MSX1 / MSX2 emulator written in Rust + [wgpu](https://wgpu.rs/), with the
TMS9918 and V9938 VDPs rendered entirely in a fragment shader. Runs natively
and in the browser via WebAssembly.

This started as a learning project to understand the MSX architecture
end-to-end (Z80, slots, VDP, PSG, mappers, command engine, line interrupts)
by building one. It is not trying to compete with the excellent
[openMSX](https://openmsx.org/) on accuracy — but it does play a fair
chunk of the MSX1 and MSX2 Konami catalogue.

## Try it

Drop a `.rom` file anywhere on the window. The emulator detects the mapper
(Plain ROM, Konami MegaROM, Konami SCC) and boots. With no cartridge it
starts in C-BIOS BASIC.

## Highlights

- **TMS9918 + V9938 VDP in a pixel shader.** The active area, sprites,
  backdrop and letterbox are all drawn by `vdp.wgsl` reading VRAM uploaded
  as a texture. No per-scanline CPU rasterisation; the GPU does the work.
- **MSX2 features**: V9938 16-colour palette via port 0x9A, Screen 4 (G3)
  tile mode with V9938 sprite mode 2 (8 sprites per line, OR-mixed
  colours), Screen 5 (G4) 4 bpp bitmap, 212-line display mode (R9 bit 7),
  128 KiB VRAM with R14 extended addressing.
- **Per-scanline register snapshots** — the CPU side captures R0-R7, R10,
  R11 and R23 at the start of each visible scanline. The shader uses the
  values that were active on each scanline, so split-screen tricks land
  on the right pixel rows: mode-switching mid-frame (G4 playfield + G1
  status bar — KV2 / Usas / Vampire Killer), per-band palette swaps,
  per-line scroll offsets, etc.
- **Line interrupts** (R19 + IE2 + S1 FH bit) fire at the precise
  instruction boundary closest to the matching scanline. Cycle-accurate
  enough for most MSX2 software; not perfect for the tightest beam-
  racing (Quarth).
- **V9938 command engine**: STOP, POINT, PSET, LMMV (logical fill), LMMM
  (logical copy), HMMV (fast fill), HMMM (fast copy), LINE, SRCH, YMMM.
  Logic ops with transparent-skip (TIMP, TAND, etc.) and OR/AND/XOR/NOT
  applied per pixel. CPU transfers (LMMC, LMCM, HMMC) are stubbed.
- **Three post-process modes** (cycle with **Alt+S**):
  - `sharp` — pixel-perfect nearest upscale at integer scale.
  - `crt` — gentle blur, scanlines, and vignette across the whole surface.
  - `pixely` — EPX / Scale2x edge-aware upscale; diagonals get anti-
    aliased, flat colour stays crisp.
- **Audio.** AY-3-8910 PSG (via the [`psg`](https://crates.io/crates/psg)
  crate, configured as `ChipType::AY`) plus a hand-ported Konami SCC.
  Native uses [`cpal`](https://crates.io/crates/cpal); web uses a
  ScriptProcessorNode.
- **Konami mappers.** Plain ROM, Konami MegaROM (Knightmare / Penguin
  Adventure style), and Konami MegaROM + SCC (Salamander / Nemesis 2/3).
  Mapper auto-detected by scanning the cartridge for bank-select
  `LD (nnnn), A` instructions.
- **WASM build.** Drag-and-drop ROM loading on the page; fullscreen via
  **Alt/Cmd + Enter**.

## Build

You'll need the C-BIOS ROMs in `assets/` before building — they're
referenced via `include_bytes!`. Download them from
[cbios.sourceforge.net](https://cbios.sourceforge.net/) and place:

```
assets/cbios_main_msx2.rom    (32 KiB — MSX2 main BIOS, V9938 init)
assets/cbios_sub.rom          (16 KiB — sub-ROM, SCREEN 4-8 helpers)
assets/cbios_basic.rom        (16 KiB — BASIC interpreter, cartridge form)
```

The emulator maps the main BIOS into slot 0, the sub-ROM into subslot 3-1,
and the BASIC cartridge into slot 2. Slot 1 is the user cartridge socket
— a game ROM goes there at runtime via drag-and-drop. With slot 1 empty
the BIOS scan reaches slot 2 and boots BASIC.

### Native

```sh
cargo run --release [-- [--shader MODE] [path/to/cartridge.rom]]
```

Examples:

```sh
cargo run --release                                        # boots into BASIC
cargo run --release -- my-game.rom                         # loads a cartridge
cargo run --release -- --shader crt SALAMAND.ROM           # with CRT shader
```

### Web

```sh
trunk serve --release
# or for a static build:
trunk build --release
```

`trunk serve --port 9000` if 8080 is taken.

URL parameters:

- `?rom=path/to/file.rom` — fetches a cartridge at start. Subject to CORS,
  so usually only same-origin files work.
- `?shader=sharp|crt|pixely` — initial shader mode.

## Controls

| Input | Action |
|---|---|
| MSX keyboard keys | Mapped 1:1 from the host keyboard where layouts overlap. |
| **Alt/Cmd + Enter** | Toggle fullscreen. |
| **Alt + S** | Cycle shader: sharp → crt → pixely → sharp. |
| Drag-and-drop a `.rom` | Hot-swap the cartridge. Resets the CPU, VDP, and audio. |

## Logging

Category-gated logging via the `MSX_LOG` environment variable (native) or
`?log=` URL parameter (web). Useful when chasing a game-specific issue:

```sh
MSX_LOG=vdp_reg,vdp_pal cargo run --release -- game.rom 2>/tmp/game.log
```

Categories: `vdp_reg`, `vdp_pal`, `vdp_cmd`, `vdp_sprite`, `psg`, `scc`,
`slot`, `bus`, `all`. Off by default — no overhead when not enabled.

## Architecture (brief)

```
src/
  main.rs        Window, event loop, per-instruction CPU stepping with
                 per-scanline register snapshot and line-IRQ trigger.
  bus.rs         System bus, slot layout (BIOS / cart / sub-ROM / RAM),
                 V9938 control ports.
  slot.rs        Primary slots + subslot expansion. Konami mapper variants
                 and mapper auto-detection.
  vdp.rs        TMS9918 + V9938 register state, command engine, palette,
                 per-scanline snapshot buffer, GPU pipeline + bind groups.
  vdp.wgsl      The actual rasteriser. TMS9918 modes (G0/G1/G2) + sprite
                 mode 1, V9938 G3/G4 + sprite mode 2, V9938 palette, per-
                 scanline R0-R23 lookups, 192/212-line modes.
  ppi.rs        8255 PPI — keyboard matrix + row selection.
  psg / scc.rs  AY-3-8910 (via crate) and a hand-written Konami SCC.
  audio.rs      cpal native + ScriptProcessorNode web.
  post.rs       Final upscale pass. Three shader-mode pipelines.
  post_*.wgsl   Sharp / CRT / Pixely fragment shaders.
  log.rs        Category-gated logging, see "Logging" above.
```

CPU: [`z80emu`](https://crates.io/crates/z80emu) crate, driven one Z80
instruction at a time. Between instructions the host checks whether the
clock has crossed a scanline boundary; if so it snapshots the per-line
VDP registers and (if R19 matches and IE2 is set) fires a line interrupt.
VBlank fires once per frame at the end of the visible scan.

## Tested

Games confirmed playable as of writing:

- **Salamander** (Konami, MSX1, SCC) — full playthrough
- **Knightmare** (Konami, MSX1, MegaROM)
- **Penguin Adventure** (Konami, MSX1, MegaROM)
- **Kings Valley 2** (Konami, MSX2) — full playthrough, score bar visible
- **Usas** (Konami, MSX2) — playable, with some residual glitches
- **Quarth** (Konami, MSX2) — title screen and game work; gameplay has
  visual artifacts from beam-racing R5/R23 tricks we don't fully emulate

## What works / what doesn't

| Status | Feature |
|---|---|
| ✅ | TMS9918 modes: G0 (Screen 0), G1 (Screen 1), G2 (Screen 2) |
| ✅ | V9938 modes: G3 (Screen 4), G4 (Screen 5) |
| ✅ | V9938 sprite mode 2 (8 per line, per-line colour, OR-mix, EC) |
| ✅ | TMS9918 sprite mode 1 (4 per line, 5th-sprite flag, collision) |
| ✅ | V9938 16-colour palette via port 0x9A |
| ✅ | 192-line and 212-line display modes (R9 bit 7) |
| ✅ | Per-scanline split-screen — mode / palette / scroll / SAT per band |
| ✅ | Line interrupts (R19 + IE2 + S1 FH) |
| ✅ | Command engine: STOP/POINT/PSET/LMMV/LMMM/HMMV/HMMM/LINE/SRCH/YMMM |
| ✅ | 128 KiB VRAM with R14 extended addressing |
| ✅ | AY-3-8910 PSG audio |
| ✅ | Konami SCC audio |
| ✅ | Plain ROM + Konami MegaROM + Konami SCC mappers |
| ✅ | Keyboard input via PPI matrix |
| ✅ | Browser build with drag-and-drop |
| ⚠️ | Quarth — visual glitches from sub-instruction beam-racing timing |
| ❌ | V9938 modes G5/G6/G7 (Screen 6/7/8) |
| ❌ | CPU-transfer commands LMMC / LMCM / HMMC (stubbed; rare in Konami) |
| ❌ | MSX2 RAM mapper (>64 KiB; ports 0xFC-0xFF) |
| ❌ | FM-PAC / OPLL / MSX-MUSIC |
| ❌ | V9958 graphics (MSX2+) |
| ❌ | ASCII8 / ASCII16 / R-Type / Game Master 2 mappers |
| ❌ | Disk drive, MSX-DOS, tape |
| ❌ | Save states |
| ❌ | Sub-instruction-precise IRQ timing |

## Known issues

- **WASM bundle is larger than necessary (~2.5 MB instead of ~700 KB).**
  `index.html` currently uses `data-wasm-opt="0"` to disable `wasm-opt`. At
  `-Oz` it strips wasm-bindgen's reference-types externref table, which
  then breaks `__wbindgen_init_externref_table` at startup with
  `RangeError: failed to grow table by 4`. Re-enable once Trunk exposes a
  way to pass `--enable-reference-types --enable-bulk-memory` to `wasm-opt`
  inline, or move the build to a custom `Trunk.toml` that invokes `wasm-opt`
  with those flags explicitly.

- **Quarth scroll-band drift.** Quarth races the beam with very tight
  timing — writing R5/R23 at exact T-state positions per scanline. Our
  per-instruction stepping is precise to ~25 T-states (the longest Z80
  instruction), which is enough for most MSX2 games but not for Quarth.
  The play field renders but the side columns scroll inconsistently and
  the ship's position can drift. KV2 and Usas, which use line interrupts
  for mode-switching but don't race the beam at sub-instruction
  resolution, are unaffected.

## Credits

- **C-BIOS** — used as the MSX2 BIOS and BASIC implementation. Open source,
  no Microsoft/ASCII code, distributed under its own license; see
  [cbios.sourceforge.net](https://cbios.sourceforge.net/).
- **[z80emu](https://crates.io/crates/z80emu)** — Z80 CPU core by royaltm.
- **[psg](https://crates.io/crates/psg)** — AY-3-8910 emulator, a Rust port
  of Ayumi.
- **[wgpu](https://wgpu.rs/) / [winit](https://github.com/rust-windowing/winit)
  / [cpal](https://crates.io/crates/cpal) / [trunk](https://trunkrs.dev/)**
  — for everything that's not the MSX itself.
- The **openMSX** project and the **MSX Resource Center** for the docs and
  test ROMs that made the VDP, slot, and command engine behaviour
  debuggable from the outside.

## License

[MIT](LICENSE). The C-BIOS ROMs in `assets/` are not part of this project's
copyright — they are licensed under their own terms by the C-BIOS authors.
