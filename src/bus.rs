//! MSX system bus + memory/IO abstractions.
//!
//! Two layers live here:
//!
//! 1. **Domain traits** [`Memory`] and [`Io`] — the MSX-side abstraction.
//!    Clean `read8`/`write8` and `in8`/`out8` signatures, no CPU-specific
//!    baggage. Slots, ROMs, cartridges, and the bus itself implement these.
//!
//! 2. **z80emu adapters** — thin bridges that translate the CPU's
//!    `Timestamp` / break-cause API into calls on the domain traits.
//!
//! For now [`Bus`] is just a flat 64 KiB RAM — enough to run hand-written
//! Z80 code and watch I/O happen. Slots, subslots, BIOS, VDP and cartridges
//! come later, layered on top of the same two traits.

#![allow(dead_code)] // WIP module — warnings will fade once main.rs wires it up.

use crate::ppi::Ppi;
use crate::scc::Scc;
use crate::slot::{
    detect_mapper, CartridgeMapper, KonamiMegaRomCartridge, KonamiMegaRomSccCartridge,
    MappedRamSlot, RomSlot, Slot, Slots, SubslottedSlot,
};
use crate::vdp::Vdp;
use psg::PSG;
use std::num::NonZeroU16;
use std::sync::{Arc, Mutex};

/// Build the slot contents for a cartridge: detect the mapper and wrap the
/// ROM in the matching `Slot` variant, or return `Slot::Empty` when no ROM
/// is supplied. With slot 1 empty the BIOS scan continues to slot 2 where
/// C-BIOS BASIC is mounted, so boot falls through to a BASIC prompt. Used
/// by both `Bus::new` and `Bus::swap_cartridge`.
fn build_cartridge_slot(rom: Option<Vec<u8>>, scc: Arc<Mutex<Scc>>) -> Slot {
    match rom {
        None => Slot::Empty,
        Some(bytes) => match detect_mapper(&bytes) {
            CartridgeMapper::Plain => Slot::Rom(RomSlot::new(bytes.into_boxed_slice(), 0x4000)),
            CartridgeMapper::KonamiBasic => {
                Slot::KonamiMegaRomCartridge(KonamiMegaRomCartridge::new(bytes.into_boxed_slice()))
            }
            CartridgeMapper::KonamiSCC => Slot::KonamiMegaRomSccCartridge(
                KonamiMegaRomSccCartridge::new(bytes.into_boxed_slice(), Some(scc)),
            ),
        },
    }
}

/// C-BIOS MSX2 Main ROM — 32 KiB, embedded at compile time. BIOS routines
/// including V9938 init, IO, ISR, and MSX2 BASIC bootstrap. Mounted at slot
/// 0, mapped at 0x0000-0x7FFF. Backward-compatible with MSX1 software:
/// TMS9918 modes remain available because the V9938 implements them as
/// subset modes. Open source, no Microsoft/ASCII code.
/// See <https://cbios.sourceforge.net/>.
const CBIOS_MAIN: &[u8] = include_bytes!("../assets/cbios_main_msx2.rom");

/// C-BIOS Sub-ROM — 16 KiB, embedded at compile time. Contains the MSX2
/// SCREEN 4-8 helpers (line drawing, palette, SET PAGE, BLOAD/BSAVE for
/// graphics, etc.) and a few extension routines the main BIOS calls into
/// via inter-slot calls. Mounted in subslot 3-1, mapped at 0x0000 within
/// that subslot — the main BIOS pages it in via standard slot switching.
const CBIOS_SUB: &[u8] = include_bytes!("../assets/cbios_sub.rom");

/// C-BIOS BASIC interpreter — 16 KiB cartridge ROM with the standard MSX
/// "AB" cartridge header at offset 0 and entry point 0x4010. We slot it
/// into the second cartridge socket (slot 2). The BIOS scans slots 1 →
/// 2 → 3 for "AB" headers: a game in slot 1 wins, otherwise BASIC fires.
const CBIOS_BASIC: &[u8] = include_bytes!("../assets/cbios_basic.rom");

/// MSX-level memory abstraction. Mirrors your TypeScript
/// `Memory { uread8, uwrite8 }` — minus the unsigned hint, since Rust's
/// `u8` is already unsigned.
pub trait Memory {
    fn read8(&self, addr: u16) -> u8;
    fn write8(&mut self, addr: u16, value: u8);
}

/// MSX-level I/O port abstraction. Named after the Z80 mnemonics (`IN` / `OUT`)
/// — also conveniently avoids a method-name collision with [`Memory`].
///
/// Both methods take `&mut self`: many ports have read-side effects (the VDP
/// status register clears the 0x99 latch on read, for instance).
pub trait Io {
    fn in8(&mut self, port: u8) -> u8;
    fn out8(&mut self, port: u8, value: u8);
}

/// The MSX system bus.
///
/// Owns the peripherals and the slot map. PSG lives behind an Arc because
/// the audio thread also needs access; the SCC follows the same pattern
/// but is owned by the cartridge.
pub struct Bus {
    pub slots: Slots,
    pub vdp: Vdp,
    pub ppi: Ppi,
    psg: Arc<Mutex<PSG>>,
    /// Last value written to port 0xA0 — selects which of the PSG's 14
    /// registers the next 0xA1 write or 0xA2 read targets.
    psg_reg_select: u8,
}

impl Bus {
    pub fn new(
        vdp: Vdp,
        psg: Arc<Mutex<PSG>>,
        scc: Arc<Mutex<Scc>>,
        cartridge_rom: Option<Vec<u8>>,
    ) -> Self {
        // MSX2 slot layout (C-BIOS-style):
        //   Slot 0:    C-BIOS MSX2 Main — 32 KiB BIOS at 0x0000.
        //   Slot 1:    external cartridge socket — game ROM when provided,
        //              empty otherwise. Drag-and-drop swaps this slot at runtime.
        //   Slot 2:    C-BIOS BASIC — 16 KiB cartridge-style ROM at 0x4000.
        //              Slot-scan order means a game in slot 1 wins; without
        //              one, BASIC fires.
        //   Slot 3:    expanded —
        //              3-0: empty
        //              3-1: C-BIOS Sub-ROM (display/SCREEN 4-8 helpers, 16 KiB
        //                   at 0x0000). The main BIOS pages it in via inter-
        //                   slot calls (CALSLT) when it needs the V9938-specific
        //                   routines.
        //              3-2: empty
        //              3-3: V9938 RAM mapper — 256 KiB pool addressed through
        //                   four 16 KiB banks (ports 0xFC-0xFF). MSX2 BIOS
        //                   uses 64 KiB linear at boot via the mapper's
        //                   default 3/2/1/0 setup; games that need more
        //                   reprogramme the banks.
        let bios = RomSlot::new(Box::from(CBIOS_MAIN), 0x0000);
        let basic = RomSlot::new(Box::from(CBIOS_BASIC), 0x4000);
        let sub_rom = RomSlot::new(Box::from(CBIOS_SUB), 0x0000);

        let slot3 = SubslottedSlot::new([
            Slot::Empty,                   // 3-0
            Slot::Rom(sub_rom),            // 3-1 ← C-BIOS Sub-ROM
            Slot::Empty,                   // 3-2
            Slot::MappedRam(MappedRamSlot::new()),  // 3-3 ← V9938 RAM mapper (256 KiB)
        ]);

        let cartridge_slot = build_cartridge_slot(cartridge_rom, scc);

        let slots = Slots::new([
            Slot::Rom(bios),
            cartridge_slot,
            Slot::Rom(basic),
            Slot::Subslotted(Box::new(slot3)),
        ]);
        Self {
            slots,
            vdp,
            ppi: Ppi::new(),
            psg,
            psg_reg_select: 0,
        }
    }

    /// Hot-swap the cartridge in primary slot 1. Caller is responsible for
    /// resetting CPU / VDP / audio around this — see `State::load_cartridge`
    /// in main.rs. The slot register is reset to 0 here so the BIOS init
    /// path starts mapping the BIOS at page 0, same as cold boot.
    pub fn swap_cartridge(&mut self, rom: Option<Vec<u8>>, scc: Arc<Mutex<Scc>>) {
        let cartridge_slot = build_cartridge_slot(rom, scc);
        self.slots.set_slot(1, cartridge_slot);
        self.slots.slot_register = 0;
        self.ppi.release_all();
        self.psg_reg_select = 0;
        // Silence the PSG. There's no `reset()` on the crate, but writing 0
        // to all 14 registers zeroes the channel volumes (regs 8/9/10), which
        // kills any tone or noise that was still ringing.
        {
            let mut psg = self.psg.lock().unwrap();
            for r in 0..14u8 {
                psg.set_register(r, 0);
            }
        }
    }
}

impl Memory for Bus {
    fn read8(&self, addr: u16) -> u8 {
        self.slots.read8(addr)
    }

    fn write8(&mut self, addr: u16, value: u8) {
        self.slots.write8(addr, value);
    }
}

impl Io for Bus {
    fn in8(&mut self, port: u8) -> u8 {
        // Pattern 3 — hardcoded match. When the device count climbs further,
        // factor out into a routing table (pattern 2). Unmapped ports return
        // open-bus 0xFF.
        match port {
            0x98 | 0x99 => self.vdp.in8(port),
            0xA2 => 0xFF, // PSG port-B read (not wired up yet)
            0xA8 => self.slots.slot_register,
            0xA9 => self.ppi.read_row(), // keyboard row state
            // V9938 RAM mapper: bank-select register read-back. Returns
            // the bank index ORed with 0xF0 so software detects this is
            // a 16-bank (= 256 KiB) mapper.
            0xFC..=0xFF => self
                .slots
                .mapper()
                .map(|m| m.get_bank((port - 0xFC) as usize))
                .unwrap_or(0xFF),
            _ => 0xFF,
        }
    }

    fn out8(&mut self, port: u8, value: u8) {
        match port {
            0x98 | 0x99 => self.vdp.out8(port, value),
            // V9938 palette write (0x9A) and indirect register write (0x9B).
            // MSX1 software never touches these; the VDP itself ignores
            // them silently if registers aren't wired up yet, so it's safe
            // to route them unconditionally regardless of machine type.
            0x9A | 0x9B => self.vdp.out8(port, value),
            // PSG register select — store the index for the next 0xA1 write.
            // Only the low 4 bits select a real register (0..13); higher
            // values address ports A and B of the PSG (keyboard scan, etc.).
            0xA0 => self.psg_reg_select = value,
            // PSG data write — push into the chip's selected register.
            0xA1 => self
                .psg
                .lock()
                .unwrap()
                .set_register(self.psg_reg_select, value),
            0xA8 => self.slots.slot_register = value,
            // PPI port C: low nibble = keyboard row select, high nibble
            // would drive CAPS LED / kana indicator (ignored here).
            0xAA => self.ppi.write_port_c(value),
            // PPI control register — software sets the 8255 mode at boot
            // (port A out, B in, C lo out / hi out). We don't model the
            // 8255 modes, so the value is irrelevant.
            0xAB => {}
            // V9938 RAM mapper bank selectors — one per CPU page. Writing
            // a value selects which 16 KiB bank appears in that page;
            // page 0 = port 0xFC, page 1 = 0xFD, etc.
            0xFC..=0xFF => {
                if let Some(m) = self.slots.mapper_mut() {
                    m.set_bank((port - 0xFC) as usize, value);
                }
            }
            _ => {} // unmapped writes silently dropped
        }
    }
}

// --- z80emu adapter layer ---------------------------------------------------

impl z80emu::Memory for Bus {
    type Timestamp = i32;

    // The canonical non-mut read. z80emu's default impls of `read_mem` and
    // `read_mem16` both route through here, so overriding only this one
    // covers 8-bit data reads AND 16-bit immediate operands (JP nn, LD HL,nn,
    // etc.) in one shot.
    fn read_debug(&self, addr: u16) -> u8 {
        self.read8(addr)
    }

    fn read_opcode(&mut self, pc: u16, _ir: u16, _ts: i32) -> u8 {
        self.read8(pc)
    }

    fn write_mem(&mut self, addr: u16, value: u8, _ts: i32) {
        self.write8(addr, value);
    }
}

impl z80emu::Io for Bus {
    type Timestamp = i32;
    type WrIoBreak = ();
    type RetiBreak = ();

    fn read_io(&mut self, port: u16, _ts: i32) -> (u8, Option<NonZeroU16>) {
        (self.in8((port & 0xFF) as u8), None)
    }

    fn write_io(
        &mut self,
        port: u16,
        data: u8,
        _ts: i32,
    ) -> (Option<()>, Option<NonZeroU16>) {
        self.out8((port & 0xFF) as u8, data);
        (None, None)
    }

    /// Called by z80emu before each instruction fetch. Returns true when an
    /// interrupt is pending; CPU then enters the IRQ acknowledge cycle (which
    /// in IM 1 just jumps to 0x0038).
    fn is_irq(&mut self, _ts: i32) -> bool {
        self.vdp.is_irq_pending()
    }
}
