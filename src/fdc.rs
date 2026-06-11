//! WD2793 floppy disk controller + one disk drive, as wired in the
//! Philips NMS-8245 (and most Philips/Sony MSX2 machines): the FDC's
//! registers are memory-mapped into the top of the DISK.ROM page,
//! subslot 3-3 at 0x4000-0x7FFF.
//!
//! Register map (Philips interface; openMSX `PhilipsFDC` layout):
//!
//!   0x7FF8  R: status      W: command
//!   0x7FF9  R/W: track register
//!   0x7FFA  R/W: sector register
//!   0x7FFB  R/W: data register (sector bytes flow through here)
//!   0x7FFC  R/W: side select (bit 0)
//!   0x7FFD  R/W: drive select / motor
//!   0x7FFF  R: bit 7 = !DRQ, bit 6 = !INTRQ — the driver's poll target
//!
//! Commands complete instantly: the WD2793's seek/rotation delays exist
//! for mechanical reasons our flat .DSK image doesn't have. The DISK.ROM
//! driver polls 0x7FFF for DRQ and the status register for BUSY, so
//! "ready immediately" just makes those loops exit on their first pass.
//! (Same philosophy as the VDP command engine's synchronous execution.)
//!
//! .DSK images are raw sector dumps: 720 KiB = 80 tracks × 2 sides ×
//! 9 sectors × 512 bytes (360 KiB = single-sided). Writes land in the
//! in-memory image only — persisting back to the host file is a later
//! step.

const SECTOR_SIZE: usize = 512;
const SECTORS_PER_TRACK: u8 = 9;

/// One inserted floppy: the raw image plus its decoded geometry.
pub struct DiskImage {
    data: Vec<u8>,
    sides: u8,
}

impl DiskImage {
    /// Wrap a raw .DSK byte image. Geometry is derived from the file
    /// size; anything that isn't a known single/double-sided size is
    /// treated as double-sided (covers 640K/8-sector variants poorly,
    /// but those are rare on MSX).
    pub fn new(data: Vec<u8>) -> Self {
        let sides = if data.len() <= 80 * SECTORS_PER_TRACK as usize * SECTOR_SIZE {
            1
        } else {
            2
        };
        Self { data, sides }
    }

    /// Byte offset of (track, side, sector) — sector is 1-based, per the
    /// WD2793's sector register convention.
    fn offset(&self, track: u8, side: u8, sector: u8) -> Option<usize> {
        if sector == 0 || sector > SECTORS_PER_TRACK || side >= self.sides {
            return None;
        }
        let logical = (track as usize * self.sides as usize + side as usize)
            * SECTORS_PER_TRACK as usize
            + (sector as usize - 1);
        let offset = logical * SECTOR_SIZE;
        (offset + SECTOR_SIZE <= self.data.len()).then_some(offset)
    }
}

/// Transfer in progress through the data register.
enum Transfer {
    None,
    /// Reading: remaining bytes stream out of `buffer[pos..]`.
    Read { buffer: Vec<u8>, pos: usize },
    /// Writing: incoming bytes fill `buffer`; flushed to the image when
    /// full. `offset` is the image position the sector belongs at.
    Write { buffer: Vec<u8>, offset: usize },
}

pub struct Wd2793 {
    disk: Option<DiskImage>,
    /// FDC registers.
    track: u8,
    sector: u8,
    data: u8,
    status: u8,
    /// Side select latch (0x7FFC bit 0).
    side: u8,
    /// Drive select / motor latch (0x7FFD). Bit 0 selects drive B —
    /// the 8245 has one drive, so that reads as "not ready".
    drive: u8,
    /// Last command's type-I flag — the status register's bit layout
    /// differs between seek-class and transfer-class commands.
    type1_status: bool,
    transfer: Transfer,
    intrq: bool,
    /// Index-pulse toggle for type-I status bit 1: drivers count index
    /// pulses to verify the disk is spinning; flipping it on every status
    /// read is enough to keep those loops moving.
    index_flip: bool,
}

impl Wd2793 {
    pub fn new(disk: Option<DiskImage>) -> Self {
        Self {
            disk,
            track: 0,
            sector: 1,
            data: 0,
            status: 0,
            side: 0,
            drive: 0,
            type1_status: true,
            transfer: Transfer::None,
            intrq: false,
            index_flip: false,
        }
    }

    /// True when the selected drive can service commands: drive A with a
    /// disk inserted. Drive B (bit 0 of the drive latch) doesn't exist.
    fn drive_ready(&self) -> bool {
        self.disk.is_some() && (self.drive & 0x01) == 0
    }

    fn drq(&self) -> bool {
        !matches!(self.transfer, Transfer::None)
    }

    /// Memory-mapped register read (already masked to 0x7FF8-0x7FFF).
    pub fn read(&mut self, addr: u16) -> u8 {
        match addr & 0x07 {
            0 => {
                // Status read clears INTRQ (WD279x semantics).
                self.intrq = false;
                let mut st = self.status;
                if self.type1_status {
                    // Type I layout: bit7 NOT-READY, bit2 TRACK0, bit1 INDEX.
                    if !self.drive_ready() {
                        st |= 0x80;
                    }
                    if self.track == 0 {
                        st |= 0x04;
                    }
                    self.index_flip = !self.index_flip;
                    if self.index_flip {
                        st |= 0x02;
                    }
                } else {
                    // Type II/III layout: bit7 NOT-READY, bit1 DRQ.
                    if !self.drive_ready() {
                        st |= 0x80;
                    }
                    if self.drq() {
                        st |= 0x02;
                    }
                }
                st
            }
            1 => self.track,
            2 => self.sector,
            3 => self.read_data(),
            4 => self.side,
            5 => self.drive,
            // 0x7FFF: bit 7 = !DRQ, bit 6 = !INTRQ, rest high.
            _ => {
                let mut v = 0xC0 | 0x3F;
                if self.drq() {
                    v &= !0x80;
                }
                if self.intrq {
                    v &= !0x40;
                }
                v
            }
        }
    }

    /// Memory-mapped register write.
    pub fn write(&mut self, addr: u16, value: u8) {
        match addr & 0x07 {
            0 => self.command(value),
            1 => self.track = value,
            2 => self.sector = value,
            3 => self.write_data(value),
            4 => self.side = value & 0x01,
            5 => self.drive = value,
            _ => {}
        }
    }

    /// Execute a command byte. Everything completes synchronously; BUSY
    /// (bit 0) is only ever observable as already-cleared, except during
    /// an in-flight data transfer where DRQ does the pacing.
    fn command(&mut self, cmd: u8) {
        self.intrq = false;
        match cmd >> 4 {
            // Type I — RESTORE: head to track 0.
            0x0 => {
                self.track = 0;
                self.finish_type1();
            }
            // Type I — SEEK: target track arrives via the data register.
            0x1 => {
                self.track = self.data;
                self.finish_type1();
            }
            // Type I — STEP / STEP-IN / STEP-OUT (with/without update).
            // Plain STEP repeats the last direction; we approximate with
            // step-in, which is what drivers use it for in practice.
            0x2 | 0x3 | 0x4 | 0x5 => {
                self.track = self.track.saturating_add(1);
                self.finish_type1();
            }
            0x6 | 0x7 => {
                self.track = self.track.saturating_sub(1);
                self.finish_type1();
            }
            // Type II — READ SECTOR (0x8 single, 0x9 multiple; multiple
            // is unused by the MSX driver, served as single).
            0x8 | 0x9 => self.begin_read(),
            // Type II — WRITE SECTOR.
            0xA | 0xB => self.begin_write(),
            // Type III — READ ADDRESS: stream the 6-byte ID field of the
            // "next" sector header (track, side, sector, size=2, crc, crc).
            0xC => {
                self.type1_status = false;
                self.status = 0;
                let id = vec![self.track, self.side, self.sector, 2, 0, 0];
                self.transfer = Transfer::Read { buffer: id, pos: 0 };
            }
            // Type IV — FORCE INTERRUPT: abort whatever is in flight.
            0xD => {
                self.transfer = Transfer::None;
                self.status = 0;
                self.type1_status = true;
                self.intrq = true;
            }
            // Type III — READ TRACK / WRITE TRACK. Read-track is unused
            // by the MSX driver; write-track is FORMAT. Swallow the bytes
            // so a FORMAT appears to succeed without remastering the
            // image (the subsequent sector writes lay down real data).
            0xE => {
                self.type1_status = false;
                self.status = 0;
                self.intrq = true;
            }
            0xF => {
                self.type1_status = false;
                self.status = 0;
                // Accept (and discard) a track's worth of format bytes.
                self.transfer = Transfer::Write {
                    buffer: Vec::with_capacity(6250),
                    offset: usize::MAX, // sentinel: format data, not a sector
                };
            }
            _ => unreachable!(),
        }
    }

    /// Wrap up a type-I command: status reflects head position, INTRQ
    /// signals completion.
    fn finish_type1(&mut self) {
        self.type1_status = true;
        self.status = 0x20; // head loaded
        self.intrq = true;
    }

    fn begin_read(&mut self) {
        self.type1_status = false;
        let Some(offset) = self
            .disk
            .as_ref()
            .filter(|_| self.drive_ready())
            .and_then(|d| d.offset(self.track, self.side, self.sector))
        else {
            // Record not found — bit 4; INTRQ ends the command.
            self.status = 0x10;
            self.intrq = true;
            return;
        };
        let disk = self.disk.as_ref().unwrap();
        let buffer = disk.data[offset..offset + SECTOR_SIZE].to_vec();
        self.status = 0x01; // busy until the last byte is pulled
        self.transfer = Transfer::Read { buffer, pos: 0 };
    }

    fn begin_write(&mut self) {
        self.type1_status = false;
        let Some(offset) = self
            .disk
            .as_ref()
            .filter(|_| self.drive_ready())
            .and_then(|d| d.offset(self.track, self.side, self.sector))
        else {
            self.status = 0x10;
            self.intrq = true;
            return;
        };
        self.status = 0x01;
        self.transfer = Transfer::Write {
            buffer: Vec::with_capacity(SECTOR_SIZE),
            offset,
        };
    }

    /// Data-register read: next byte of an active read transfer.
    fn read_data(&mut self) -> u8 {
        if let Transfer::Read { buffer, pos } = &mut self.transfer {
            let byte = buffer.get(*pos).copied().unwrap_or(0);
            *pos += 1;
            if *pos >= buffer.len() {
                self.transfer = Transfer::None;
                self.status = 0;
                self.intrq = true;
            }
            self.data = byte;
        }
        self.data
    }

    /// Data-register write: next byte of an active write transfer.
    fn write_data(&mut self, value: u8) {
        self.data = value;
        if let Transfer::Write { buffer, offset } = &mut self.transfer {
            buffer.push(value);
            if *offset == usize::MAX {
                // Format stream: discard once a track's worth arrived.
                if buffer.len() >= 6250 {
                    self.transfer = Transfer::None;
                    self.status = 0;
                    self.intrq = true;
                }
            } else if buffer.len() >= SECTOR_SIZE {
                let offset = *offset;
                let sector: Vec<u8> = std::mem::take(buffer);
                if let Some(disk) = self.disk.as_mut() {
                    disk.data[offset..offset + SECTOR_SIZE].copy_from_slice(&sector);
                }
                self.transfer = Transfer::None;
                self.status = 0;
                self.intrq = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk_with_pattern() -> DiskImage {
        // Double-sided 720K image where every sector is filled with its
        // logical index, so reads are self-identifying.
        let mut data = vec![0u8; 80 * 2 * 9 * SECTOR_SIZE];
        for (i, chunk) in data.chunks_mut(SECTOR_SIZE).enumerate() {
            chunk.fill(i as u8);
        }
        DiskImage::new(data)
    }

    /// Drive the FDC exactly like the DISK.ROM driver does: command,
    /// then pull bytes while 0x7FFF reports DRQ.
    #[test]
    fn read_sector_streams_512_bytes() {
        let mut fdc = Wd2793::new(Some(disk_with_pattern()));
        fdc.write(0x7FF9, 2); // track 2
        fdc.write(0x7FFA, 3); // sector 3
        fdc.write(0x7FFC, 1); // side 1
        fdc.write(0x7FF8, 0x80); // READ SECTOR
        let mut bytes = Vec::new();
        while fdc.read(0x7FFF) & 0x80 == 0 {
            bytes.push(fdc.read(0x7FFB));
        }
        assert_eq!(bytes.len(), SECTOR_SIZE);
        // Logical sector: (track*2 + side)*9 + (sector-1) = (2*2+1)*9+2 = 47.
        assert!(bytes.iter().all(|&b| b == 47));
        // INTRQ raised, BUSY cleared.
        assert_eq!(fdc.read(0x7FFF) & 0x40, 0);
        assert_eq!(fdc.read(0x7FF8) & 0x01, 0);
    }

    #[test]
    fn write_sector_lands_in_image() {
        let mut fdc = Wd2793::new(Some(disk_with_pattern()));
        fdc.write(0x7FF9, 0);
        fdc.write(0x7FFA, 1);
        fdc.write(0x7FFC, 0);
        fdc.write(0x7FF8, 0xA0); // WRITE SECTOR
        for _ in 0..SECTOR_SIZE {
            fdc.write(0x7FFB, 0xAA);
        }
        // Read it back.
        fdc.write(0x7FF8, 0x80);
        let mut bytes = Vec::new();
        while fdc.read(0x7FFF) & 0x80 == 0 {
            bytes.push(fdc.read(0x7FFB));
        }
        assert!(bytes.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn missing_disk_reports_record_not_found() {
        let mut fdc = Wd2793::new(None);
        fdc.write(0x7FF8, 0x80);
        // RNF set, no DRQ.
        assert_ne!(fdc.read(0x7FF8) & 0x10, 0);
        assert_ne!(fdc.read(0x7FFF) & 0x80, 0);
    }

    #[test]
    fn restore_homes_to_track_zero() {
        let mut fdc = Wd2793::new(Some(disk_with_pattern()));
        fdc.write(0x7FF9, 40);
        fdc.write(0x7FF8, 0x00); // RESTORE
        assert_eq!(fdc.read(0x7FF9), 0);
        // Type-I status: TRACK0 set.
        assert_ne!(fdc.read(0x7FF8) & 0x04, 0);
    }
}
