//! MSX keyboard side of the PPI (Intel 8255).
//!
//! The PPI sits on I/O ports 0xA8..0xAB. Port A (0xA8) is the primary slot
//! select — already handled in [`crate::bus::Bus`] alongside the slot map.
//! The remaining ports drive keyboard scanning:
//!
//!   Port B (0xA9, read)  — current row's column state (inverted: 0 = pressed)
//!   Port C (0xAA, write) — low nibble selects which row to read,
//!                          high nibble drives CAPS LED, kana, click out
//!   Port D (0xAB, write) — control register; software sets PPI mode once
//!                          at boot and never touches it again
//!
//! The keyboard itself is an 11-row × 8-column matrix. At rest every bit
//! reads as 1 (high); a pressed key pulls its column line low.

const ROWS: usize = 11;

pub struct Ppi {
    /// Inverted row state — `rows[r]` has bit `c` set when key (r, c) is
    /// *not* pressed. All 0xFF at rest.
    rows: [u8; ROWS],
    /// Low nibble of the last write to port 0xAA — selects which row port
    /// 0xA9 reads back.
    selected_row: u8,
}

impl Ppi {
    pub fn new() -> Self {
        Self {
            rows: [0xFFu8; ROWS],
            selected_row: 0,
        }
    }

    /// Mark a key as pressed or released. Out-of-range coordinates are
    /// silently ignored — host-key mapping tables can include MSX-only
    /// keys with no real counterpart and they'll just never fire.
    pub fn set_key(&mut self, row: u8, col: u8, pressed: bool) {
        if (row as usize) >= ROWS || col >= 8 {
            return;
        }
        let mask = 1u8 << col;
        if pressed {
            self.rows[row as usize] &= !mask;
        } else {
            self.rows[row as usize] |= mask;
        }
    }

    /// Read the currently selected row — returned over port 0xA9.
    pub fn read_row(&self) -> u8 {
        let idx = (self.selected_row & 0x0F) as usize;
        self.rows.get(idx).copied().unwrap_or(0xFF)
    }

    /// Write to port 0xAA. Only the low 4 bits matter to us (row select);
    /// the high nibble would drive the CAPS LED and kana indicator in real
    /// hardware, which we ignore.
    pub fn write_port_c(&mut self, value: u8) {
        self.selected_row = value & 0x0F;
    }

    /// Release every key in the matrix — used on cartridge swap, otherwise a
    /// key that was held down at the moment of the swap would stay "pressed"
    /// from the new game's POV until the host fires its keyup event (which it
    /// might not, if focus moved to the file picker in between).
    pub fn release_all(&mut self) {
        self.rows.fill(0xFF);
    }
}

impl Default for Ppi {
    fn default() -> Self {
        Self::new()
    }
}
