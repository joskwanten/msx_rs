# Assets

This directory holds the ROM files that `src/bus.rs` embeds at build time
via `include_bytes!`. They are **not** part of `msx_rs`'s MIT license —
they ship under the C-BIOS authors' own terms (see below).

## Files

| File | Size | Maps to | Purpose |
|---|---|---|---|
| `cbios_main_msx2.rom` | 32 KiB | Slot 0, `0x0000-0x7FFF` | C-BIOS MSX2 Main — BIOS routines, V9938 init, MSX2 BASIC bootstrap. Backward-compatible with MSX1 software (V9938 supports TMS9918 modes). |
| `cbios_sub.rom`       | 16 KiB | Slot 3-1, `0x0000-0x3FFF` | C-BIOS Sub-ROM — SCREEN 4-8 helpers (line drawing, palette, paging, BLOAD/BSAVE for graphics) and other V9938-specific routines. The main BIOS pages it in via inter-slot calls when needed. |
| `cbios_basic.rom`     | 16 KiB | Slot 2, `0x4000-0x7FFF` | C-BIOS BASIC interpreter, formatted as a cartridge ("AB" header at offset 0, entry point 0x4010). |

The BIOS scans primary slots 1 → 2 → 3 for "AB" cartridge headers. With a
game in slot 1 (drag-and-drop), the scan stops there and the game boots.
Without a game, slot 2 wins and the BASIC prompt appears.

The MSX1-only main BIOS (`cbios_main_msx1.rom`) is no longer used — MSX2
main is a strict superset (V9938 supports all TMS9918 modes).

## Source

Both files come from the C-BIOS project — a clean-room open-source
re-implementation of the MSX BIOS and BASIC, written without using any
Microsoft or ASCII code so it can be distributed freely.

- Project: <https://cbios.sourceforge.net/>
- GitHub mirror: <https://github.com/joyrex2001/cbios>

## License

C-BIOS is released under the BSD-2-Clause license. From the C-BIOS
distribution's `Readme.txt`:

> Copyright (c) 2002-2005 BouKiCHi.
> Copyright (c) 2003 Reikan.
> Copyright (c) 2004-2005, 2007 Maarten ter Huurne.
> Copyright (c) 2004 Albert Beevendorp.
> Copyright (c) 2004 Manuel Bilderbeek.
> Copyright (c) 2004-2006 Joost Yervante Damad.
> Copyright (c) 2004-2005 Jussi Pitkänen.
> Copyright (c) 2006-2007 Eric Boon.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions
> are met:
>
>   1. Redistributions of source code must retain the above copyright
>      notice, this list of conditions and the following disclaimer.
>   2. Redistributions in binary form must reproduce the above
>      copyright notice, this list of conditions and the following
>      disclaimer in the documentation and/or materials provided with
>      the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE AUTHORS "AS IS" AND ANY EXPRESS OR
> IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY
> DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
> GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
> OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
> EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Replacing with the original BIOS

If you have a real MSX BIOS dump and prefer the Microsoft/ASCII variant
(jingle, logo, original BASIC behaviour), you can drop it in alongside
these files and adjust the `include_bytes!` paths in `src/bus.rs`. Note
that the Microsoft MSX BIOS is **not** freely redistributable — keep it
out of any commits / public deploys.
