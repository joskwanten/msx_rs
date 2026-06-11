//! MSX primary slot selection.
//!
//! The Z80's 64 KiB address space is split into four 16 KiB pages, and each
//! page can be served by one of four primary slots. The *slot register*
//! (exposed on I/O port 0xA8) encodes the choice: two bits per page,
//! low two bits = page 0.
//!
//! ```text
//!   slot_register: SSPP_NNMM
//!                  ││ ││ │└─ page 0 slot
//!                  ││ │└─── page 1 slot
//!                  │└─────── page 2 slot
//!                  └──────── page 3 slot
//! ```
//!
//! For now this models primary slots only. Subslot expansion (the `FFFFh`
//! write/read protocol on expanded slots) lands as a future [`Slot`] variant.

#![allow(dead_code)] // WIP module — some variants (Rom) wait for BIOS data.

use crate::bus::Memory;
use crate::scc::Scc;
use std::sync::{Arc, Mutex};

/// One slot's contents. New slot types extend this enum — that's the price
/// (and benefit) of static dispatch: the compiler reminds you everywhere a
/// new variant needs handling.
pub enum Slot {
    /// Nothing plugged in. Reads return 0xFF (open bus), writes ignored.
    Empty,
    /// Read-only memory: BIOS, BASIC, cartridge ROM.
    Rom(RomSlot),
    /// 64 KiB of RAM, addressable across all pages.
    Ram(RamSlot),
    /// V9938 RAM mapper: 256 KiB pool, four bank-select registers (one per
    /// CPU page) driven by I/O ports 0xFC-0xFF. Lets MSX2 software see
    /// more than 64 KiB of RAM by paging 16 KiB banks into each address-
    /// space quarter. Required by heavier MSX2 titles (Aleste, SD
    /// Snatcher, etc.).
    MappedRam(MappedRamSlot),
    /// An expanded slot: 4 subslots selected via the FFFFh protocol.
    /// Boxed because the variant transitively contains `[Slot; 4]` — otherwise
    /// the enum would have infinite size.
    Subslotted(Box<SubslottedSlot>),
    /// Konami SCC mega-ROM (Salamander, Nemesis 2/3, F1-Spirit, Snake's Revenge).
    KonamiMegaRomSccCartridge(KonamiMegaRomSccCartridge),
    /// Konami "basic" mega-ROM, no SCC (Penguin Adventure, Knightmare, Goonies).
    KonamiMegaRomCartridge(KonamiMegaRomCartridge),
    /// ASCII 8 KiB mega-ROM (Ys, Aleste, most Falcom/Compile titles).
    Ascii8Cartridge(Ascii8Cartridge),
    /// ASCII 16 KiB mega-ROM (Hydlide, many ASCII/HAL/Sony titles).
    Ascii16Cartridge(Ascii16Cartridge),
}

impl Memory for Slot {
    fn read8(&self, addr: u16) -> u8 {
        match self {
            Slot::Empty => 0xFF,
            Slot::Rom(r) => r.read8(addr),
            Slot::Ram(r) => r.read8(addr),
            Slot::MappedRam(r) => r.read8(addr),
            Slot::Subslotted(s) => s.read8(addr),
            Slot::KonamiMegaRomSccCartridge(c) => c.read8(addr),
            Slot::KonamiMegaRomCartridge(c) => c.read8(addr),
            Slot::Ascii8Cartridge(c) => c.read8(addr),
            Slot::Ascii16Cartridge(c) => c.read8(addr),
        }
    }

    fn write8(&mut self, addr: u16, value: u8) {
        match self {
            Slot::Empty => {}
            Slot::Rom(_) => {} // ROM: writes ignored
            Slot::Ram(r) => r.write8(addr, value),
            Slot::MappedRam(r) => r.write8(addr, value),
            Slot::Subslotted(s) => s.write8(addr, value),
            Slot::KonamiMegaRomSccCartridge(c) => c.write8(addr, value),
            Slot::KonamiMegaRomCartridge(c) => c.write8(addr, value),
            Slot::Ascii8Cartridge(c) => c.write8(addr, value),
            Slot::Ascii16Cartridge(c) => c.write8(addr, value),
        }
    }
}

/// A ROM-backed slot. Holds a blob mapped at `base`; reads outside its
/// range return 0xFF (open bus).
///
/// For VG-8020 BIOS+BASIC: `RomSlot::new(rom_bytes, 0x0000)`, where
/// `rom_bytes` is 32 KiB covering pages 0 and 1.
pub struct RomSlot {
    rom: Box<[u8]>,
    base: u16,
}

impl RomSlot {
    pub fn new(rom: Box<[u8]>, base: u16) -> Self {
        Self { rom, base }
    }
}

impl Memory for RomSlot {
    fn read8(&self, addr: u16) -> u8 {
        let offset = addr.wrapping_sub(self.base) as usize;
        self.rom.get(offset).copied().unwrap_or(0xFF)
    }

    fn write8(&mut self, _addr: u16, _value: u8) {
        // ROM — writes silently ignored.
    }
}

/// 64 KiB of RAM. On the VG-8020 this lives in slot 3 (later: subslot 3-3).
pub struct RamSlot {
    ram: Box<[u8; 0x10000]>,
}

impl RamSlot {
    pub fn new() -> Self {
        Self {
            ram: Box::new([0u8; 0x10000]),
        }
    }
}

impl Default for RamSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for RamSlot {
    fn read8(&self, addr: u16) -> u8 {
        self.ram[addr as usize]
    }

    fn write8(&mut self, addr: u16, value: u8) {
        self.ram[addr as usize] = value;
    }
}

/// V9938 memory mapper: 256 KiB of RAM addressed through four 16 KiB
/// banks (one per CPU address-space page). Software selects which bank
/// occupies each page by writing the bank index to I/O ports 0xFC..0xFF:
///
///   port 0xFC ← bank for page 0 (0x0000-0x3FFF)
///   port 0xFD ← bank for page 1 (0x4000-0x7FFF)
///   port 0xFE ← bank for page 2 (0x8000-0xBFFF)
///   port 0xFF ← bank for page 3 (0xC000-0xFFFF)
///
/// Reading those ports returns the last-written bank index ORed with
/// `0xF0` — the high nibble being "open bus / not implemented" tells
/// software that this is a 16-bank (= 256 KiB) mapper, the standard
/// for mid-range MSX2 machines.
///
/// Real hardware MSX2 RAM mappers come in 64/128/256/512/1024 KiB
/// flavours; 256 KiB is the sweet spot for Konami-era cartridges and
/// keeps the on-disk size of the emulator reasonable.
const MAPPER_BANKS: usize = 16;
const MAPPER_RAM_SIZE: usize = MAPPER_BANKS * 0x4000;

pub struct MappedRamSlot {
    ram: Box<[u8; MAPPER_RAM_SIZE]>,
    /// Bank-select register per CPU page. Low 4 bits select the 16 KiB
    /// bank (0..15); the upper 4 bits are open bus (returned as 1s when
    /// the CPU reads the port back).
    banks: [u8; 4],
}

impl MappedRamSlot {
    pub fn new() -> Self {
        Self {
            ram: Box::new([0u8; MAPPER_RAM_SIZE]),
            // Default: linear identity mapping — page 0 → bank 3,
            // page 1 → bank 2, page 2 → bank 1, page 3 → bank 0. This
            // mirrors what the C-BIOS init sets up so software that
            // probes the mapper before configuring it gets a sensible
            // 64 KiB view of RAM.
            banks: [3, 2, 1, 0],
        }
    }

    /// Update the bank register for one CPU page (0..3). Low 4 bits go
    /// to the bank index; the upper 4 are stored as-is but masked off
    /// on reads — software relies on that mask for mapper-size probing.
    pub fn set_bank(&mut self, page: usize, value: u8) {
        if page < 4 {
            self.banks[page] = value & 0x0F;
        }
    }

    /// Read back a bank register. The high nibble is set to indicate
    /// "this is a 16-bank mapper" — software writes 0xFF to a port and
    /// reads back 0x0F to confirm only 4 bank-select bits are wired.
    pub fn get_bank(&self, page: usize) -> u8 {
        if page < 4 {
            self.banks[page] | 0xF0
        } else {
            0xFF
        }
    }
}

impl Default for MappedRamSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for MappedRamSlot {
    fn read8(&self, addr: u16) -> u8 {
        let page = (addr >> 14) as usize;
        let bank = self.banks[page] as usize;
        let offset = (bank << 14) | (addr as usize & 0x3FFF);
        self.ram[offset]
    }

    fn write8(&mut self, addr: u16, value: u8) {
        let page = (addr >> 14) as usize;
        let bank = self.banks[page] as usize;
        let offset = (bank << 14) | (addr as usize & 0x3FFF);
        self.ram[offset] = value;
    }
}

/// Primary slot map. Owns the slot register and the four slot contents.
pub struct Slots {
    pub slot_register: u8,
    slots: [Slot; 4],
}

impl Slots {
    pub fn new(slots: [Slot; 4]) -> Self {
        Self {
            slot_register: 0,
            slots,
        }
    }

    /// Replace the contents of one primary slot. Used at runtime when the
    /// user drops a new cartridge into a running emulator — bus-level state
    /// (slot register etc.) is left alone; the CPU reset that follows the
    /// swap will redo slot selection through the BIOS init.
    pub fn set_slot(&mut self, idx: usize, slot: Slot) {
        self.slots[idx] = slot;
    }

    /// Find the (first) RAM mapper in the slot tree. Used by the bus to
    /// route I/O ports 0xFC-0xFF to the mapper's bank-select registers.
    /// Walks subslots one level deep — sufficient for our layout, where
    /// the mapper lives in subslot 3-3.
    pub fn mapper_mut(&mut self) -> Option<&mut MappedRamSlot> {
        for slot in self.slots.iter_mut() {
            match slot {
                Slot::MappedRam(m) => return Some(m),
                Slot::Subslotted(s) => {
                    for sub in s.subslots.iter_mut() {
                        if let Slot::MappedRam(m) = sub {
                            return Some(m);
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Read-only variant of `mapper_mut` for I/O reads of 0xFC-0xFF.
    pub fn mapper(&self) -> Option<&MappedRamSlot> {
        for slot in self.slots.iter() {
            match slot {
                Slot::MappedRam(m) => return Some(m),
                Slot::Subslotted(s) => {
                    for sub in s.subslots.iter() {
                        if let Slot::MappedRam(m) = sub {
                            return Some(m);
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Decode which slot is currently mapped to the page containing `addr`.
    ///
    /// Returning a `usize` (rather than `&Slot`) lets the caller borrow
    /// `&mut self.slots[idx]` afterwards without a lifetime conflict — small
    /// but recurring borrow-checker dance in Rust emulator code.
    fn selected_index(&self, addr: u16) -> usize {
        let page = (addr >> 14) as usize; // 0..3
        ((self.slot_register >> (page * 2)) & 0b11) as usize
    }
}

impl Memory for Slots {
    fn read8(&self, addr: u16) -> u8 {
        self.slots[self.selected_index(addr)].read8(addr)
    }

    fn write8(&mut self, addr: u16, value: u8) {
        let idx = self.selected_index(addr);
        // MSX_LOG=slot: trace writes into the cartridge socket's ROM
        // window — on mega-ROMs these are exactly the bank-switch writes,
        // which reveals a dump's real mapper protocol when the detection
        // heuristic comes up short (indirect switchers like Aleste).
        if idx == 1 && (0x4000..=0xBFFF).contains(&addr) {
            crate::mlog!(SLOT, "cart wr {:04X} = {:02X}", addr, value);
        }
        self.slots[idx].write8(addr, value);
    }
}

/// An expanded slot's four subslots, addressed via the FFFFh protocol:
///
/// - **Write to 0xFFFF** sets the subslot register. The byte is *not* also
///   written to the underlying memory at 0xFFFF — that's a real-hardware
///   detail, the expander circuit intercepts.
/// - **Read from 0xFFFF** returns the *complement* of the subslot register.
///   Software uses this to detect whether a slot is expanded: write a known
///   value, read it back, check if it came back inverted.
///
/// Every other address routes through the subslot bits the same way the
/// primary slot register does (two bits per 16 KiB page).
pub struct SubslottedSlot {
    pub subslot_register: u8,
    pub subslots: [Slot; 4],
}

impl SubslottedSlot {
    pub fn new(subslots: [Slot; 4]) -> Self {
        Self {
            subslot_register: 0,
            subslots,
        }
    }

    fn selected_index(&self, addr: u16) -> usize {
        let page = (addr >> 14) as usize;
        ((self.subslot_register >> (page * 2)) & 0b11) as usize
    }
}

impl Memory for SubslottedSlot {
    fn read8(&self, addr: u16) -> u8 {
        if addr == 0xFFFF {
            !self.subslot_register
        } else {
            self.subslots[self.selected_index(addr)].read8(addr)
        }
    }

    fn write8(&mut self, addr: u16, value: u8) {
        if addr == 0xFFFF {
            self.subslot_register = value;
        } else {
            let idx = self.selected_index(addr);
            self.subslots[idx].write8(addr, value);
        }
    }
}

/// Konami-SCC mega-ROM cartridge mapper.
///
/// The cartridge occupies pages 1 and 2 of the Z80 address space
/// (0x4000-0xBFFF), divided into four 8 KiB *regions*. Each region has its
/// own bank register pointing into the ROM. Bank switching is triggered by
/// writes to one specific 2 KiB window per region:
///
/// | Region        | Address range   | Bank-switch window |
/// |---------------|-----------------|--------------------|
/// | 0x4000-0x5FFF | bank 1          | 0x5000-0x57FF      |
/// | 0x6000-0x7FFF | bank 2          | 0x7000-0x77FF      |
/// | 0x8000-0x9FFF | bank 3          | 0x9000-0x97FF      |
/// | 0xA000-0xBFFF | bank 4          | 0xB000-0xB7FF      |
///
/// Writes anywhere else in the cartridge area are silently ignored — that's
/// crucial: if we banked on every write (the naive approach), random
/// stores from the game's runtime would corrupt the bank registers and
/// the game would crash mid-init.
///
/// Special case: when the bank for region 0x8000-0x9FFF is set to `0x3F`,
/// the address window 0x9800-0x9FFF turns into SCC sound-chip registers.
/// Writes there do *not* change the bank — they would drive the audio chip.
/// (We acknowledge them but drop the data; sound emulation is future work.)
///
/// Bank values are masked to 6 bits — Konami-SCC hardware addresses up to
/// 64 banks (512 KiB).
pub struct KonamiMegaRomSccCartridge {
    rom: Box<[u8]>,
    /// One bank register per 8 KiB address-space region. Initial state
    /// presents banks 0..3 across the four cartridge regions, exposing the
    /// first 32 KiB of ROM linearly at 0x4000-0xBFFF — that's where the
    /// MSX "AB" cartridge header lives at 0x4000.
    selected_pages: [u8; 8],
    /// SCC sound chip. Shared with the audio thread; writes to the SCC
    /// register window forward into it. `None` means no audio output
    /// configured (e.g. during tests).
    scc: Option<Arc<Mutex<Scc>>>,
}

const KONAMI_PAGE_SIZE: usize = 0x2000;
const KONAMI_SCC_BANK: u8 = 0x3F;

impl KonamiMegaRomSccCartridge {
    pub fn new(rom: Box<[u8]>, scc: Option<Arc<Mutex<Scc>>>) -> Self {
        Self {
            rom,
            selected_pages: [0, 0, 0, 1, 2, 3, 0, 0],
            scc,
        }
    }

    fn region(addr: u16) -> usize {
        (addr >> 13) as usize // 0..7
    }
}

impl Memory for KonamiMegaRomSccCartridge {
    fn read8(&self, addr: u16) -> u8 {
        let region = Self::region(addr);
        let bank = self.selected_pages[region] as usize;
        let offset = bank * KONAMI_PAGE_SIZE + (addr as usize & 0x1FFF);
        self.rom.get(offset).copied().unwrap_or(0xFF)
    }

    fn write8(&mut self, addr: u16, value: u8) {
        // SCC audio register window — bank 3 (region 4) at 0x9800-0x9FFF when
        // that region's bank is the magic 0x3F. Forward to the audio chip
        // (only the low 256 of the 2 KiB window contain real registers; the
        // rest mirrors) and leave the bank register alone.
        if (0x9800..=0x9FFF).contains(&addr) && self.selected_pages[4] == KONAMI_SCC_BANK {
            if let Some(scc) = &self.scc {
                let reg = (addr & 0xFF) as u8;
                scc.lock().unwrap().write_reg(reg, value);
            }
            return;
        }

        // Bank switching only on writes to the 2 KiB select window of each
        // cartridge region. Anything else is ignored — that's what makes the
        // mapper "precise" instead of "rough".
        match addr {
            0x5000..=0x57FF => self.selected_pages[2] = value & 0x3F,
            0x7000..=0x77FF => self.selected_pages[3] = value & 0x3F,
            0x9000..=0x97FF => self.selected_pages[4] = value & 0x3F,
            0xB000..=0xB7FF => self.selected_pages[5] = value & 0x3F,
            _ => {}
        }
    }
}

/// Konami "standard" mega-ROM mapper — same 8 KiB region layout as the SCC
/// variant but with different bank-select windows and no audio chip. Region
/// 2 (0x4000-0x5FFF) is fixed to bank 0; the other three regions are switched
/// by writing anywhere in their 8 KiB range:
///
/// | Region        | Switch window  |
/// |---------------|----------------|
/// | 0x4000-0x5FFF | (fixed bank 0) |
/// | 0x6000-0x7FFF | 0x6000-0x7FFF  |
/// | 0x8000-0x9FFF | 0x8000-0x9FFF  |
/// | 0xA000-0xBFFF | 0xA000-0xBFFF  |
///
/// Games: Goonies, Penguin Adventure, Knightmare, Yie Ar Kung-Fu, and most
/// pre-1987 Konami 128 KiB cartridges.
pub struct KonamiMegaRomCartridge {
    rom: Box<[u8]>,
    selected_pages: [u8; 8],
}

impl KonamiMegaRomCartridge {
    pub fn new(rom: Box<[u8]>) -> Self {
        Self {
            rom,
            selected_pages: [0, 0, 0, 1, 2, 3, 0, 0],
        }
    }
}

impl Memory for KonamiMegaRomCartridge {
    fn read8(&self, addr: u16) -> u8 {
        let region = (addr >> 13) as usize;
        let bank = self.selected_pages[region] as usize;
        let offset = bank * KONAMI_PAGE_SIZE + (addr as usize & 0x1FFF);
        self.rom.get(offset).copied().unwrap_or(0xFF)
    }

    fn write8(&mut self, addr: u16, value: u8) {
        match addr {
            0x6000..=0x7FFF => self.selected_pages[3] = value & 0x3F,
            0x8000..=0x9FFF => self.selected_pages[4] = value & 0x3F,
            0xA000..=0xBFFF => self.selected_pages[5] = value & 0x3F,
            _ => {}
        }
    }
}

/// ASCII 8 KiB mega-ROM mapper. Four 8 KiB regions at 0x4000-0xBFFF, all
/// switchable; the bank-select registers live in 2 KiB windows packed into
/// 0x6000-0x7FFF (fMSX MapROM, MAP_ASCII8: region = `(addr & 0x1800) >> 11`):
///
/// | Region        | Switch window  |
/// |---------------|----------------|
/// | 0x4000-0x5FFF | 0x6000-0x67FF  |
/// | 0x6000-0x7FFF | 0x6800-0x6FFF  |
/// | 0x8000-0x9FFF | 0x7000-0x77FF  |
/// | 0xA000-0xBFFF | 0x7800-0x7FFF  |
///
/// All bank registers reset to 0, so the cartridge boots showing the first
/// 8 KiB (with the "AB" header) in every region. Games: most Falcom/Compile
/// /T&E mega-ROMs (Ys, Dragon Slayer, Aleste, ...). The SRAM variant
/// (Xanadu et al.) is not modelled yet — bank values are masked to ROM size.
pub struct Ascii8Cartridge {
    rom: Box<[u8]>,
    /// Bank per 8 KiB region of the full address space (only 2..=5 used).
    selected_pages: [u8; 8],
    /// Region maps the cartridge's SRAM instead of a ROM bank. Selected by
    /// writing a bank value with the bit just above the ROM's bank count
    /// set — fMSX MapROM: `if (V & (ROMMask+1))` (Hydlide 3, Xanadu).
    sram_mapped: [bool; 8],
    /// 8 KiB battery-backed SRAM, shared by all regions that map it.
    /// In-memory only for now — survives a session, not an exit.
    sram: Box<[u8; KONAMI_PAGE_SIZE]>,
    /// Power-of-two mask over the ROM's 8 KiB bank count.
    bank_mask: u8,
}

impl Ascii8Cartridge {
    pub fn new(rom: Box<[u8]>) -> Self {
        let banks = (rom.len() / KONAMI_PAGE_SIZE).max(1);
        let bank_mask = (banks.next_power_of_two() - 1).min(0xFF) as u8;
        Self {
            rom,
            selected_pages: [0; 8],
            sram_mapped: [false; 8],
            sram: Box::new([0; KONAMI_PAGE_SIZE]),
            bank_mask,
        }
    }
}

impl Memory for Ascii8Cartridge {
    fn read8(&self, addr: u16) -> u8 {
        let region = (addr >> 13) as usize;
        if self.sram_mapped[region] {
            return self.sram[addr as usize & 0x1FFF];
        }
        let bank = self.selected_pages[region] as usize;
        let offset = bank * KONAMI_PAGE_SIZE + (addr as usize & 0x1FFF);
        self.rom.get(offset).copied().unwrap_or(0xFF)
    }

    fn write8(&mut self, addr: u16, value: u8) {
        if (0x6000..=0x7FFF).contains(&addr) {
            // Window index 0..3 → address-space regions 2..5. The select
            // windows always win over an SRAM mapping at 0x6000-0x7FFF —
            // that's why carts put their SRAM window at 0x8000-0xBFFF.
            let window = ((addr & 0x1800) >> 11) as usize;
            let sram_bit = self.bank_mask as u16 + 1;
            self.sram_mapped[window + 2] = (value as u16 & sram_bit) != 0;
            self.selected_pages[window + 2] = value & self.bank_mask;
            return;
        }
        // Data writes land in SRAM when the target region maps it.
        if (0x4000..=0xBFFF).contains(&addr) {
            let region = (addr >> 13) as usize;
            if self.sram_mapped[region] {
                self.sram[addr as usize & 0x1FFF] = value;
            }
        }
    }
}

/// ASCII 16 KiB mega-ROM mapper. Two 16 KiB regions:
///
/// | Region        | Switch window  |
/// |---------------|----------------|
/// | 0x4000-0x7FFF | 0x6000-0x67FF  |
/// | 0x8000-0xBFFF | 0x7000-0x77FF  |
///
/// Both bank registers reset to 0. Games: Hydlide, R-Type's relatives,
/// many ASCII/HAL/Sony mega-ROMs. (Androgynus writes its bank to 0x77FF —
/// inside the canonical window, so it works; the 2 KiB SRAM variant of
/// Hydlide 2 is not modelled yet.)
pub struct Ascii16Cartridge {
    rom: Box<[u8]>,
    /// Bank per 16 KiB region: [0x4000-0x7FFF, 0x8000-0xBFFF].
    selected_pages: [u8; 2],
    /// Region maps the cartridge's SRAM — same select rule as ASCII8
    /// (bank value with the over-ROM bit set; Hydlide 2's 2 KiB SRAM).
    sram_mapped: [bool; 2],
    /// 2 KiB battery-backed SRAM, mirrored across the 16 KiB region.
    /// In-memory only for now.
    sram: Box<[u8; 0x800]>,
    /// Power-of-two mask over the ROM's 16 KiB bank count.
    bank_mask: u8,
}

const ASCII16_PAGE_SIZE: usize = 0x4000;

impl Ascii16Cartridge {
    pub fn new(rom: Box<[u8]>) -> Self {
        let banks = (rom.len() / ASCII16_PAGE_SIZE).max(1);
        let bank_mask = (banks.next_power_of_two() - 1).min(0xFF) as u8;
        Self {
            rom,
            selected_pages: [0; 2],
            sram_mapped: [false; 2],
            sram: Box::new([0; 0x800]),
            bank_mask,
        }
    }
}

impl Memory for Ascii16Cartridge {
    fn read8(&self, addr: u16) -> u8 {
        if !(0x4000..=0xBFFF).contains(&addr) {
            return 0xFF;
        }
        let region = ((addr - 0x4000) >> 14) as usize;
        if self.sram_mapped[region] {
            return self.sram[addr as usize & 0x7FF];
        }
        let bank = self.selected_pages[region] as usize;
        let offset = bank * ASCII16_PAGE_SIZE + (addr as usize & 0x3FFF);
        self.rom.get(offset).copied().unwrap_or(0xFF)
    }

    fn write8(&mut self, addr: u16, value: u8) {
        match addr {
            0x6000..=0x67FF | 0x7000..=0x77FF => {
                let region = ((addr >> 12) & 1) as usize;
                let sram_bit = self.bank_mask as u16 + 1;
                self.sram_mapped[region] = (value as u16 & sram_bit) != 0;
                self.selected_pages[region] = value & self.bank_mask;
            }
            0x4000..=0xBFFF => {
                let region = ((addr - 0x4000) >> 14) as usize;
                if self.sram_mapped[region] {
                    self.sram[addr as usize & 0x7FF] = value;
                }
            }
            _ => {}
        }
    }
}

/// Cartridge mapper type, returned by [`detect_mapper`].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CartridgeMapper {
    /// Plain ROM (≤ 32 KiB), mapped linearly at 0x4000.
    Plain,
    /// Konami basic mega-ROM — bank-switch on writes to 0x6000/0x8000/0xA000.
    KonamiBasic,
    /// Konami SCC mega-ROM — bank-switch on writes to 0x5000/0x7000/0x9000/0xB000.
    KonamiSCC,
    /// ASCII 8 KiB mapper — bank-switch windows packed into 0x6000-0x7FFF.
    Ascii8,
    /// ASCII 16 KiB mapper — bank-switch at 0x6000-0x67FF / 0x7000-0x77FF.
    Ascii16,
}

impl CartridgeMapper {
    /// Accepted names for the `--mapper` / `?mapper=` override, paired with
    /// the mapper they select. One row per mapper — extend this table when
    /// a new mapper lands and the CLI/web override picks it up for free.
    /// First name per mapper is the canonical one shown in error messages.
    pub const NAMES: &'static [(&'static str, CartridgeMapper)] = &[
        ("plain", CartridgeMapper::Plain),
        ("konami", CartridgeMapper::KonamiBasic),
        ("konami-scc", CartridgeMapper::KonamiSCC),
        ("scc", CartridgeMapper::KonamiSCC),
        ("ascii8", CartridgeMapper::Ascii8),
        ("ascii16", CartridgeMapper::Ascii16),
    ];

    /// Parse a user-supplied mapper name (case-insensitive). `None` for
    /// unknown names — the caller decides whether to warn or fall back to
    /// auto-detection.
    pub fn parse(name: &str) -> Option<Self> {
        let name = name.to_ascii_lowercase();
        Self::NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, m)| *m)
    }

    /// The valid override names, for help/error text.
    pub fn name_list() -> String {
        Self::NAMES
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Pick a mapper for the given ROM. Plain ROMs are detected by size; for
/// larger ROMs we scan the code for `LD (nn), A` instructions targeting
/// each mapper family's bank-select addresses and pick the dominant one.
///
/// Ported from fMSX's GuessROM (MSX.c). The subtlety is the OVERLAPPING
/// addresses: 0x6000 is a select address for Konami-basic, ASCII8 *and*
/// ASCII16, and 0x7000 for Konami-SCC, ASCII8 and ASCII16 — each hit
/// counts for every family it could belong to, and the per-family totals
/// decide. The starting biases mirror fMSX: Konami-basic is the fallback
/// (+1) and ASCII16 is preferred over ASCII8 (-1 on ASCII8), because an
/// ASCII16 ROM's 0x6000/0x7000 writes also count for ASCII8 — only real
/// 0x6800/0x7800 hits should swing it to ASCII8.
pub fn detect_mapper(rom: &[u8]) -> CartridgeMapper {
    if rom.len() <= 32 * 1024 {
        return CartridgeMapper::Plain;
    }

    // Counts start at fMSX's biases, kept as i32 so ASCII8 can sit at -1
    // relative to ASCII16 (fMSX inits all to 1, bumps its generic default,
    // and decrements ASCII8).
    let mut kon: i32 = 1; // KonamiBasic — also the tie fallback
    let mut scc: i32 = 0;
    let mut a8: i32 = -1;
    let mut a16: i32 = 0;
    for window in rom.windows(3) {
        // 0x32 = LD (nn), A — the canonical bank-switch instruction.
        if window[0] != 0x32 {
            continue;
        }
        match u16::from_le_bytes([window[1], window[2]]) {
            0x5000 | 0x9000 | 0xB000 => scc += 1,
            0x4000 | 0x8000 | 0xA000 => kon += 1,
            0x6800 | 0x7800 => a8 += 1,
            0x6000 => {
                kon += 1;
                a8 += 1;
                a16 += 1;
            }
            0x7000 => {
                scc += 1;
                a8 += 1;
                a16 += 1;
            }
            0x77FF => a16 += 1,
            _ => {}
        }
    }

    // Highest count wins; ties resolve in declaration order (Konami-basic
    // first, matching fMSX's index order with GEN8≈KonamiBasic in front).
    let candidates = [
        (kon, CartridgeMapper::KonamiBasic),
        (scc, CartridgeMapper::KonamiSCC),
        (a8, CartridgeMapper::Ascii8),
        (a16, CartridgeMapper::Ascii16),
    ];
    let mut best = candidates[0];
    for c in &candidates[1..] {
        if c.0 > best.0 {
            best = *c;
        }
    }
    best.1
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic 64 KiB ROM whose 8 KiB banks are filled with their own
    /// bank index, so a read instantly reveals which bank is mapped.
    fn numbered_rom(bank_size: usize, banks: usize) -> Box<[u8]> {
        let mut rom = vec![0u8; bank_size * banks];
        for (i, chunk) in rom.chunks_mut(bank_size).enumerate() {
            chunk.fill(i as u8);
        }
        rom.into_boxed_slice()
    }

    #[test]
    fn ascii8_switches_all_four_regions() {
        let mut c = Ascii8Cartridge::new(numbered_rom(0x2000, 8));
        // Reset state: every region shows bank 0.
        assert_eq!(c.read8(0x4000), 0);
        assert_eq!(c.read8(0xA000), 0);
        // Window per region: 6000→4000, 6800→6000, 7000→8000, 7800→A000.
        c.write8(0x6000, 4);
        c.write8(0x6800, 5);
        c.write8(0x7000, 6);
        c.write8(0x7800, 7);
        assert_eq!(c.read8(0x4000), 4);
        assert_eq!(c.read8(0x6000), 5);
        assert_eq!(c.read8(0x8000), 6);
        assert_eq!(c.read8(0xBFFF), 7);
        // A bank value with the over-ROM bit set maps the SRAM instead
        // (fMSX `V & (ROMMask+1)`). Writes land in SRAM and read back;
        // deselecting brings the ROM bank back with SRAM contents intact.
        c.write8(0x7000, 8); // region 0x8000-0x9FFF → SRAM
        c.write8(0x8000, 0xAB);
        assert_eq!(c.read8(0x8000), 0xAB);
        c.write8(0x7000, 6); // back to ROM bank 6
        assert_eq!(c.read8(0x8000), 6);
        c.write8(0x7000, 8);
        assert_eq!(c.read8(0x8000), 0xAB);
    }

    #[test]
    fn ascii16_sram_select_and_mirror() {
        let mut c = Ascii16Cartridge::new(numbered_rom(0x4000, 8));
        // Over-ROM bit (8 banks → bit 8) maps the 2 KiB SRAM, mirrored
        // across the 16 KiB region (Hydlide 2 style).
        c.write8(0x7000, 8);
        c.write8(0x8000, 0x5A);
        assert_eq!(c.read8(0x8000), 0x5A);
        assert_eq!(c.read8(0x8800), 0x5A); // 2 KiB mirror
        c.write8(0x7000, 2);
        assert_eq!(c.read8(0x8000), 2);
    }

    #[test]
    fn ascii16_switches_both_regions() {
        let mut c = Ascii16Cartridge::new(numbered_rom(0x4000, 8));
        assert_eq!(c.read8(0x4000), 0);
        assert_eq!(c.read8(0x8000), 0);
        c.write8(0x6000, 3);
        c.write8(0x7000, 5);
        assert_eq!(c.read8(0x4000), 3);
        assert_eq!(c.read8(0x7FFF), 3);
        assert_eq!(c.read8(0x8000), 5);
        assert_eq!(c.read8(0xBFFF), 5);
        // Androgynus-style: select register at the window's last byte.
        c.write8(0x77FF, 1);
        assert_eq!(c.read8(0x8000), 1);
        // Writes outside the select windows do NOT switch banks.
        c.write8(0x6800, 7);
        assert_eq!(c.read8(0x4000), 3);
    }

    /// Build a >32 KiB ROM stuffed with `LD (addr),A` instructions for the
    /// given switch addresses — the detector counts exactly these.
    fn rom_with_writes(addrs: &[u16]) -> Vec<u8> {
        let mut rom = vec![0u8; 64 * 1024];
        let mut pos = 0usize;
        for addr in addrs.iter().cycle().take(64) {
            rom[pos] = 0x32;
            rom[pos + 1] = (*addr & 0xFF) as u8;
            rom[pos + 2] = (*addr >> 8) as u8;
            pos += 4; // gap so windows don't overlap mid-instruction
        }
        rom
    }

    #[test]
    fn detect_konami_scc_rom() {
        let rom = rom_with_writes(&[0x5000, 0x7000, 0x9000, 0xB000]);
        assert_eq!(detect_mapper(&rom), CartridgeMapper::KonamiSCC);
    }

    #[test]
    fn detect_konami_basic_rom() {
        let rom = rom_with_writes(&[0x6000, 0x8000, 0xA000]);
        assert_eq!(detect_mapper(&rom), CartridgeMapper::KonamiBasic);
    }

    #[test]
    fn detect_ascii8_rom() {
        // ASCII8 games hit all four packed windows; the 6800/7800 writes
        // are what distinguishes them from ASCII16.
        let rom = rom_with_writes(&[0x6000, 0x6800, 0x7000, 0x7800]);
        assert_eq!(detect_mapper(&rom), CartridgeMapper::Ascii8);
    }

    #[test]
    fn detect_ascii16_rom() {
        // Only 6000/7000 traffic: counts for ASCII8 and ASCII16 alike, but
        // ASCII16's starting bias wins — mirroring fMSX's preference.
        let rom = rom_with_writes(&[0x6000, 0x7000]);
        assert_eq!(detect_mapper(&rom), CartridgeMapper::Ascii16);
    }

    #[test]
    fn detect_small_rom_as_plain() {
        let rom = vec![0u8; 16 * 1024];
        assert_eq!(detect_mapper(&rom), CartridgeMapper::Plain);
    }
}
