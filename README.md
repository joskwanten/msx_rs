# msx_rs

An MSX1 emulator written in Rust + [wgpu](https://wgpu.rs/), with the TMS9918
VDP rendered entirely in a fragment shader. Runs natively and in the browser
via WebAssembly.

This is a learning project — the goal was to understand the MSX architecture
end-to-end (Z80, slots, VDP, PSG, mappers) by building one, not to compete
with the excellent [openMSX](https://openmsx.org/). Expect it to play your
favourite Konami cartridge from 1986 reasonably well; don't expect MSX2
graphics, FM audio, save states, or accurate sub-instruction timing.

## Try it

Drop a `.rom` file anywhere on the window. The emulator detects the mapper
(Plain ROM, Konami MegaROM, Konami SCC) and boots. With no cartridge it
starts in C-BIOS BASIC.

## Highlights

- **TMS9918 VDP in a pixel shader.** The active area, sprites, backdrop and
  letterbox are all drawn by `vdp.wgsl` reading VRAM uploaded as a texture.
  No per-scanline CPU rasterisation.
- **Three post-process modes** (cycle with **Alt+S**):
  - `sharp` — pixel-perfect nearest upscale with integer scaling.
  - `crt` — gentle blur, scanlines, and vignette across the whole surface.
  - `pixely` — EPX / Scale2x edge-aware upscale; diagonals get anti-aliased,
    flat colour stays crisp.
- **Audio.** AY-3-8910 PSG (via the [`psg`](https://crates.io/crates/psg)
  crate, configured as `ChipType::AY`) plus a hand-ported Konami SCC.
  Native uses [`cpal`](https://crates.io/crates/cpal); web uses a
  ScriptProcessorNode.
- **Konami mappers.** Plain ROM, Konami MegaROM (Knightmare / Penguin
  Adventure style), and Konami MegaROM + SCC (Salamander / Nemesis 2/3).
  Mapper auto-detected by scanning for the bank-select `LD (nnnn), A`
  instructions in the cartridge.
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

## Architecture (brief)

```
src/
  main.rs        Window, event loop, frame pacing, drag-and-drop wiring.
  bus.rs         System bus: Memory + Io traits, z80emu adapters, PSG ports.
  slot.rs        Primary slots + subslot expansion. Konami mapper variants.
  vdp.rs         TMS9918 register state, VRAM, GPU pipeline + bind groups.
  vdp.wgsl       The actual rasteriser. Modes 0/1/2 + sprites.
  ppi.rs         8255 PPI — keyboard matrix + row selection.
  psg / scc.rs   AY-3-8910 (via crate) and a hand-written SCC.
  audio.rs       cpal native + ScriptProcessorNode web.
  post.rs        Final upscale pass. Three shader-mode pipelines.
  post_*.wgsl    Sharp / CRT / Pixely fragment shaders.
```

CPU: [`z80emu`](https://crates.io/crates/z80emu) crate, driven one MSX frame
at a time. VBLANK is asserted at the start of each frame; the game's IRQ
handler then runs near the top of the T-state budget.

## What works / what doesn't

| Status | Feature |
|---|---|
| ✅ | TMS9918 Mode 0 (text), Mode 1 (graphics 1), Mode 2 (graphics 2) |
| ✅ | Sprites with per-line collision (mode 1 sprite size + magnify) |
| ✅ | AY-3-8910 PSG audio |
| ✅ | Konami SCC audio |
| ✅ | Plain ROM + Konami MegaROM + Konami SCC mappers |
| ✅ | Keyboard input via PPI matrix |
| ✅ | Browser build with drag-and-drop |
| ❌ | MSX2 / V9938 / V9958 graphics |
| ❌ | FM-PAC / OPLL / MSX-MUSIC |
| ❌ | ASCII8 / ASCII16 / R-Type / Game Master 2 mappers |
| ❌ | Disk drive, MSX-DOS, tape |
| ❌ | Save states |
| ❌ | Sub-instruction timing accuracy |

## Known issues

- **WASM bundle is larger than necessary (~2.5 MB instead of ~700 KB).**
  `index.html` currently uses `data-wasm-opt="0"` to disable `wasm-opt`. At
  `-Oz` it strips wasm-bindgen's reference-types externref table, which
  then breaks `__wbindgen_init_externref_table` at startup with
  `RangeError: failed to grow table by 4`. Re-enable once Trunk exposes a
  way to pass `--enable-reference-types --enable-bulk-memory` to `wasm-opt`
  inline, or move the build to a custom `Trunk.toml` that invokes `wasm-opt`
  with those flags explicitly.

## Credits

- **C-BIOS** — used as the MSX1 BIOS and BASIC implementation. Open source,
  no Microsoft/ASCII code, distributed under its own license; see
  [cbios.sourceforge.net](https://cbios.sourceforge.net/).
- **[z80emu](https://crates.io/crates/z80emu)** — Z80 CPU core by royaltm.
- **[psg](https://crates.io/crates/psg)** — AY-3-8910 emulator, a Rust port
  of Ayumi.
- **[wgpu](https://wgpu.rs/) / [winit](https://github.com/rust-windowing/winit)
  / [cpal](https://crates.io/crates/cpal) / [trunk](https://trunkrs.dev/)**
  — for everything that's not the MSX itself.
- The **openMSX** project and the **MSX Resource Center** for the docs and
  test ROMs that made the VDP and slot behaviour debuggable from the outside.

## License

[MIT](LICENSE). The C-BIOS ROMs in `assets/` are not part of this project's
copyright — they are licensed under their own terms by the C-BIOS authors.
