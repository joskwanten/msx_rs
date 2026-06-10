//! RP-5C01 real-time clock (I/O ports 0xB4/0xB5), modelled on fMSX
//! (MSX.c `RTCIn` + the 0xB4/0xB5 port handlers).
//!
//! The MSX2 BIOS stores its boot settings (screen width, colours, beep)
//! as nibbles in the chip's CMOS RAM blocks 2/3, guarded by a checksum.
//! Without a chip that *retains writes*, the Philips sub-ROM rewrites its
//! defaults and re-verifies forever — the boot never reaches screen init.
//! C-BIOS never touches the RTC, which is why this stayed unimplemented.
//!
//! Register map (per bank, 13 four-bit registers):
//!   bank 0     — live time (sec/min/hour/weekday/day/month/year nibbles);
//!                reads come from the host clock, writes are ignored.
//!   bank 1     — alarm + mode flags (stored, otherwise inert).
//!   banks 2/3  — CMOS RAM, the part the BIOS actually needs to stick.
//!   reg 13     — mode: bits 0-1 select the bank, bit 3 = timer enable.
//!   regs 14/15 — test/reset; read as 0xF, writes ignored.
//!
//! All reads return the nibble with the upper four bits high (open bus),
//! exactly like fMSX's `return(J|0xF0)`.

pub struct Rtc {
    /// Register index latched by the last port 0xB4 write (0-15).
    reg: u8,
    /// Mode register (reg 13): bank select in bits 0-1.
    mode: u8,
    /// Four banks of 13 nibble registers. Bank 0's time registers are
    /// shadowed by the host clock on read; the rest hold what was written.
    banks: [[u8; 13]; 4],
}

impl Rtc {
    pub fn new() -> Self {
        Self { reg: 0, mode: 0, banks: [[0; 13]; 4] }
    }

    /// Port 0xB4 write — select the register for the next 0xB5 access.
    pub fn select(&mut self, value: u8) {
        self.reg = value & 0x0F;
    }

    /// Port 0xB5 write — store a nibble in the selected register.
    pub fn write(&mut self, value: u8) {
        let value = value & 0x0F;
        match self.reg {
            0..=12 => {
                let bank = (self.mode & 0x03) as usize;
                self.banks[bank][self.reg as usize] = value;
            }
            13 => self.mode = value,
            _ => {} // 14/15: test/reset — ignored
        }
    }

    /// Port 0xB5 read — current register value, upper nibble high.
    pub fn read(&self) -> u8 {
        let nibble = match self.reg {
            13 => self.mode,
            14 | 15 => 0x0F,
            r => {
                let bank = (self.mode & 0x03) as usize;
                if bank == 0 {
                    host_time_nibble(r)
                } else {
                    self.banks[bank][r as usize]
                }
            }
        };
        nibble | 0xF0
    }
}

/// Bank-0 time registers from the host clock (UTC — close enough for a
/// boot screen; fMSX uses localtime). Register layout per the RP-5C01:
///   0/1 = seconds ones/tens, 2/3 = minutes, 4/5 = hours,
///   6 = weekday, 7/8 = day, 9/10 = month, 11/12 = year since 1980.
#[cfg(not(target_arch = "wasm32"))]
fn host_time_nibble(reg: u8) -> u8 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (s, mi, h) = (secs % 60, (secs / 60) % 60, (secs / 3600) % 24);
    let days = (secs / 86_400) as i64;
    let weekday = ((days + 4) % 7) as u64; // epoch day 0 = Thursday
    // Civil-date from day count (Howard Hinnant's algorithm).
    let (y, m, d) = {
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
        (y as u64, m as u64, d as u64)
    };
    let year80 = y.saturating_sub(1980);
    (match reg {
        0 => s % 10,
        1 => s / 10,
        2 => mi % 10,
        3 => mi / 10,
        4 => h % 10,
        5 => h / 10,
        6 => weekday,
        7 => d % 10,
        8 => d / 10,
        9 => m % 10,
        10 => m / 10,
        11 => year80 % 10,
        12 => (year80 / 10) % 10,
        _ => 0x0F,
    }) as u8
}

/// No reliable wall clock on bare wasm32 — boot with a fixed midnight;
/// the BIOS only needs consistent reads, not the right time.
#[cfg(target_arch = "wasm32")]
fn host_time_nibble(_reg: u8) -> u8 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Philips sub-ROM's settings dance: select CMOS bank 2, write
    /// nibbles, read them back. Retention is the whole point of the chip.
    #[test]
    fn cmos_banks_retain_writes() {
        let mut rtc = Rtc::new();
        rtc.select(13);
        rtc.write(0x02); // bank 2
        rtc.select(5);
        rtc.write(0x0A);
        assert_eq!(rtc.read(), 0xFA);
        // Bank 3 register 5 is independent of bank 2's.
        rtc.select(13);
        rtc.write(0x03);
        rtc.select(5);
        assert_eq!(rtc.read(), 0xF0);
    }

    #[test]
    fn mode_register_reads_back() {
        let mut rtc = Rtc::new();
        rtc.select(13);
        rtc.write(0x0A);
        assert_eq!(rtc.read(), 0xFA);
    }
}
