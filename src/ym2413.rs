//! YM2413 (OPLL) FM sound generator — MSX-MUSIC / FM-PAC.
//!
//! Rust port of Jarek Burczynski's ym2413.c (MAME lineage) carrying
//! EkeEke's Genesis Plus GX hardware-verified fixes (2021-2022): the
//! instrument ROM extracted from a YM2413B die, EG resolution/dump-rate/
//! attack behaviour cross-checked against real chips and Nuked-OPLL.
//! Reference C source: ~/Projects/Msx/RogueDrive/Nextor/ym2413.c — the
//! port is structured 1:1 against it so the two can be diffed, and a
//! golden-master test (tests below + tools/ym2413_ref.c harness) verified
//! sample-exact output at the time of porting.
//!
//! The chip generates one sample per 72 master-clock cycles: 3.579545 MHz
//! / 72 ≈ 49716 Hz. The audio layer resamples that to the device rate.
//!
//! C-to-Rust mapping notes:
//! - The C globals (`ym2413`, `output[2]`, `LFO_AM`, `LFO_PM`) are fields
//!   on [`Ym2413`]; channel/slot pointers become (channel, slot) indices.
//! - `eg_type`/`vib`/`sus` keep their "truthy byte" C semantics as bools.
//! - The float-built lookup tables collapse to exact integers except
//!   `tl_tab`/`sin_tab`, which are built with the same f64 formulas.
//! - The Master System FM-adapter enable latch (`status`) is forced ON in
//!   `reset` — MSX has no such latch and the chip always sounds.

const FREQ_SH: u32 = 16;
const LFO_SH: u32 = 24;
const FREQ_MASK: u32 = (1 << FREQ_SH) - 1;

const ENV_BITS: u32 = 10;
const MAX_ATT_INDEX: i32 = (1 << (ENV_BITS - 3)) - 1; // 127
const MIN_ATT_INDEX: i32 = 0;

const SIN_BITS: u32 = 10;
const SIN_LEN: usize = 1 << SIN_BITS;
const SIN_MASK: usize = SIN_LEN - 1;

const TL_RES_LEN: usize = 256;
const TL_TAB_LEN: usize = 11 * 2 * TL_RES_LEN;
const ENV_QUIET: u32 = (TL_TAB_LEN as u32) >> 5;

const RATE_STEPS: usize = 16;

// Envelope generator phases.
const EG_DMP: u8 = 5;
const EG_ATT: u8 = 4;
const EG_DEC: u8 = 3;
const EG_SUS: u8 = 2;
const EG_REL: u8 = 1;
const EG_OFF: u8 = 0;

/// Key scale level table (C `ksl_tab`, dB values / 0.1875 — all exact).
#[rustfmt::skip]
const KSL_TAB: [u32; 8 * 16] = [
    /* OCT 0 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /* OCT 1 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 8, 10, 12, 14, 16,
    /* OCT 2 */ 0, 0, 0, 0, 0, 6, 10, 14, 16, 20, 22, 24, 26, 28, 30, 32,
    /* OCT 3 */ 0, 0, 0, 10, 16, 22, 26, 30, 32, 36, 38, 40, 42, 44, 46, 48,
    /* OCT 4 */ 0, 0, 16, 26, 32, 38, 42, 46, 48, 52, 54, 56, 58, 60, 62, 64,
    /* OCT 5 */ 0, 16, 32, 42, 48, 54, 58, 62, 64, 68, 70, 72, 74, 76, 78, 80,
    /* OCT 6 */ 0, 32, 48, 58, 64, 70, 74, 78, 80, 84, 86, 88, 90, 92, 94, 96,
    /* OCT 7 */ 0, 48, 64, 74, 80, 86, 90, 94, 96, 100, 102, 104, 106, 108, 110, 112,
];

/// Sustain level table: 3 dB per step in envelope units (C `sl_tab`).
const SL_TAB: [u32; 16] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
];

#[rustfmt::skip]
const EG_INC: [u8; 14 * RATE_STEPS] = [
    /* 0 */ 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
    /* 1 */ 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1,
    /* 2 */ 0,1, 1,1, 0,1, 0,1, 0,1, 1,1, 0,1, 0,1,
    /* 3 */ 0,1, 1,1, 0,1, 1,1, 0,1, 1,1, 0,1, 1,1,
    /* 4 */ 0,1, 1,1, 1,1, 1,1, 0,1, 1,1, 1,1, 1,1,
    /* 5 */ 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1,
    /* 6 */ 0,1, 0,1, 1,1, 1,1, 0,1, 0,1, 0,1, 0,1,
    /* 7 */ 0,1, 0,1, 1,1, 1,1, 0,1, 0,1, 1,1, 1,1,
    /* 8 */ 0,1, 0,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1,
    /* 9 */ 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1,
    /*10 */ 1,1, 1,1, 2,2, 2,2, 1,1, 1,1, 1,1, 1,1,
    /*11 */ 1,1, 1,1, 2,2, 2,2, 1,1, 1,1, 2,2, 2,2,
    /*12 */ 1,1, 1,1, 2,2, 2,2, 2,2, 2,2, 2,2, 2,2,
    /*13 */ 2,2, 2,2, 2,2, 2,2, 2,2, 2,2, 2,2, 2,2,
];

#[rustfmt::skip]
const EG_MUL: [u8; 17 * RATE_STEPS] = [
    /* 0 */ 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
    /* 1 */ 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1,
    /* 2 */ 0,1, 1,1, 0,1, 0,1, 0,1, 1,1, 0,1, 0,1,
    /* 3 */ 0,1, 1,1, 0,1, 1,1, 0,1, 1,1, 0,1, 1,1,
    /* 4 */ 0,1, 1,1, 1,1, 1,1, 0,1, 1,1, 1,1, 1,1,
    /* 5 */ 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1,
    /* 6 */ 1,1, 1,1, 2,2, 2,2, 1,1, 1,1, 1,1, 1,1,
    /* 7 */ 1,1, 1,1, 2,2, 2,2, 1,1, 1,1, 2,2, 2,2,
    /* 8 */ 1,1, 1,1, 2,2, 2,2, 2,2, 2,2, 2,2, 2,2,
    /* 9 */ 2,2, 2,2, 2,2, 2,2, 2,2, 2,2, 2,2, 2,2,
    /*10 */ 2,2, 2,2, 4,4, 4,4, 2,2, 2,2, 2,2, 2,2,
    /*11 */ 2,2, 2,2, 4,4, 4,4, 2,2, 2,2, 4,4, 4,4,
    /*12 */ 2,2, 2,2, 4,4, 4,4, 4,4, 4,4, 4,4, 4,4,
    /*13 */ 4,4, 4,4, 4,4, 4,4, 4,4, 4,4, 4,4, 4,4,
    /*14 */ 4,4, 4,4, 8,8, 8,8, 4,4, 4,4, 4,4, 4,4,
    /*15 */ 4,4, 4,4, 8,8, 8,8, 4,4, 4,4, 8,8, 8,8,
    /*16 */ 4,4, 4,4, 8,8, 8,8, 8,8, 8,8, 8,8, 8,8,
];

/// Envelope generator rate → `EG_INC`/`EG_MUL` row offset (×RATE_STEPS).
#[rustfmt::skip]
const EG_RATE_SELECT: [u16; 16 + 64 + 16] = {
    const fn o(a: u16) -> u16 { a * RATE_STEPS as u16 }
    [
        // 16 infinite time rates
        o(0), o(0), o(0), o(0), o(0), o(0), o(0), o(0),
        o(0), o(0), o(0), o(0), o(0), o(0), o(0), o(0),
        // rate 00
        o(0), o(0), o(0), o(0),
        // rates 01-11
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        o(1), o(2), o(3), o(4),
        // rate 12
        o(1), o(2), o(3), o(4),
        // rate 13
        o(5), o(6), o(7), o(8),
        // rate 14
        o(9), o(10), o(11), o(12),
        // rate 15
        o(13), o(13), o(13), o(13),
        // 16 dummy rates
        o(13), o(13), o(13), o(13), o(13), o(13), o(13), o(13),
        o(13), o(13), o(13), o(13), o(13), o(13), o(13), o(13),
    ]
};

#[rustfmt::skip]
const EG_RATE_SHIFT: [u8; 16 + 64 + 16] = [
    // 16 infinite time rates
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    // rate 00
    13, 13, 13, 13,
    // rates 01-11
    12, 12, 12, 12,
    11, 11, 11, 11,
    10, 10, 10, 10,
     9,  9,  9,  9,
     8,  8,  8,  8,
     7,  7,  7,  7,
     6,  6,  6,  6,
     5,  5,  5,  5,
     4,  4,  4,  4,
     3,  3,  3,  3,
     2,  2,  2,  2,
    // rate 12
     1,  1,  1,  1,
    // rates 13-15
     0,  0,  0,  0,
     0,  0,  0,  0,
     0,  0,  0,  0,
    // 16 dummy rates
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Multiple table (C `mul_tab`, values ×2 — all exact).
const MUL_TAB: [u8; 16] = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 20, 24, 24, 30, 30];

const LFO_AM_TAB_ELEMENTS: u32 = 210;

#[rustfmt::skip]
const LFO_AM_TABLE: [u8; LFO_AM_TAB_ELEMENTS as usize] = [
    0,0,0,0,0,0,0,
    1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7,
    8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11, 12,12,12,12, 13,13,13,13,
    14,14,14,14, 15,15,15,15, 16,16,16,16, 17,17,17,17, 18,18,18,18,
    19,19,19,19, 20,20,20,20, 21,21,21,21, 22,22,22,22, 23,23,23,23,
    24,24,24,24, 25,25,25,25,
    26,26,26,
    25,25,25,25, 24,24,24,24, 23,23,23,23, 22,22,22,22, 21,21,21,21,
    20,20,20,20, 19,19,19,19, 18,18,18,18, 17,17,17,17, 16,16,16,16,
    15,15,15,15, 14,14,14,14, 13,13,13,13, 12,12,12,12, 11,11,11,11,
    10,10,10,10, 9,9,9,9, 8,8,8,8, 7,7,7,7, 6,6,6,6, 5,5,5,5,
    4,4,4,4, 3,3,3,3, 2,2,2,2, 1,1,1,1,
];

#[rustfmt::skip]
const LFO_PM_TABLE: [i8; 8 * 8] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0,-1, 0, 0, 0,
    2, 1, 0,-1,-2,-1, 0, 1,
    3, 1, 0,-1,-3,-1, 0, 1,
    4, 2, 0,-2,-4,-2, 0, 2,
    5, 2, 0,-2,-5,-2, 0, 2,
    6, 3, 0,-3,-6,-3, 0, 3,
    7, 3, 0,-3,-7,-3, 0, 3,
];

/// Instrument ROM (C `table`), values extracted from a YM2413B die.
/// Rows: 0 = user, 1-15 = fixed instruments, 16-18 = rhythm.
#[rustfmt::skip]
const INSTRUMENT_ROM: [[u8; 8]; 19] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // 0 (user)
    [0x71, 0x61, 0x1e, 0x17, 0xd0, 0x78, 0x00, 0x17], // 1
    [0x13, 0x41, 0x1a, 0x0d, 0xd8, 0xf7, 0x23, 0x13], // 2
    [0x13, 0x01, 0x99, 0x00, 0xf2, 0xc4, 0x11, 0x23], // 3
    [0x31, 0x61, 0x0e, 0x07, 0xa8, 0x64, 0x70, 0x27], // 4
    [0x32, 0x21, 0x1e, 0x06, 0xe0, 0x76, 0x00, 0x28], // 5
    [0x31, 0x22, 0x16, 0x05, 0xe0, 0x71, 0x00, 0x18], // 6
    [0x21, 0x61, 0x1d, 0x07, 0x82, 0x81, 0x10, 0x07], // 7
    [0x23, 0x21, 0x2d, 0x14, 0xa2, 0x72, 0x00, 0x07], // 8
    [0x61, 0x61, 0x1b, 0x06, 0x64, 0x65, 0x10, 0x17], // 9
    [0x41, 0x61, 0x0b, 0x18, 0x85, 0xf7, 0x71, 0x07], // A
    [0x13, 0x01, 0x83, 0x11, 0xfa, 0xe4, 0x10, 0x04], // B
    [0x17, 0xc1, 0x24, 0x07, 0xf8, 0xf8, 0x22, 0x12], // C
    [0x61, 0x50, 0x0c, 0x05, 0xc2, 0xf5, 0x20, 0x42], // D
    [0x01, 0x01, 0x55, 0x03, 0xc9, 0x95, 0x03, 0x02], // E
    [0x61, 0x41, 0x89, 0x03, 0xf1, 0xe4, 0x40, 0x13], // F
    [0x01, 0x01, 0x18, 0x0f, 0xdf, 0xf8, 0x6a, 0x6d], // BD
    [0x01, 0x01, 0x00, 0x00, 0xc8, 0xd8, 0xa7, 0x48], // HH, SD
    [0x05, 0x01, 0x00, 0x00, 0xf8, 0xaa, 0x59, 0x55], // TOM, TOP CYM
];

#[derive(Clone, Copy, Default)]
struct Slot {
    ar: u32,
    dr: u32,
    rr: u32,
    ksr_shift: u8, // C `KSR`: 0 or 2
    ksl: u8,
    ksr: u8, // C `ksr`: kcode >> KSR
    mul: u8,

    phase: u32,
    freq: u32,
    fb_shift: u8,
    op1_out: [i32; 2],

    eg_type: bool,
    state: u8,
    tl: u32,
    tll: i32,
    volume: i32,
    sl: u32,

    eg_sh_dp: u8,
    eg_sel_dp: u16,
    eg_sh_ar: u8,
    eg_sel_ar: u16,
    eg_sh_dr: u8,
    eg_sel_dr: u16,
    eg_sh_rr: u8,
    eg_sel_rr: u16,
    eg_sh_rs: u8,
    eg_sel_rs: u16,

    key: u32,

    am_mask: u32,
    vib: bool,

    wavetable: usize,
}

#[derive(Clone, Copy, Default)]
struct Channel {
    slot: [Slot; 2],
    block_fnum: u32,
    fc: u32,
    ksl_base: u32,
    kcode: u8,
    sus: bool,
}

pub struct Ym2413 {
    ch: [Channel; 9],
    instvol_r: [u8; 9],

    eg_cnt: u32,
    eg_timer: u32,
    eg_timer_add: u32,
    eg_timer_overflow: u32,

    rhythm: u8,

    lfo_am_cnt: u32,
    lfo_am_inc: u32,
    lfo_pm_cnt: u32,
    lfo_pm_inc: u32,

    noise_rng: u32,
    noise_p: u32,
    noise_f: u32,

    inst_tab: [[u8; 8]; 19],
    fn_tab: [u32; 1024],

    address: u8,
    /// SMS FM-adapter output latch — forced 1 on MSX (see module docs).
    status: u8,

    // Tables built once (C statics filled by init_tables()).
    tl_tab: Box<[i32; TL_TAB_LEN]>,
    sin_tab: Box<[u32; SIN_LEN * 2]>,

    // Per-sample working state (C globals LFO_AM / LFO_PM / output[2]).
    lfo_am: u32,
    lfo_pm: i32,
    output: [i32; 2],
}

impl Ym2413 {
    pub fn new() -> Self {
        let mut tl_tab = Box::new([0i32; TL_TAB_LEN]);
        let mut sin_tab = Box::new([0u32; SIN_LEN * 2]);

        // C init_tables(): build the TL and sine tables with the chip's
        // peculiar 'decibel' fixed-point encoding. ENV_STEP = 128/1024.
        const ENV_STEP: f64 = 128.0 / ((1 << ENV_BITS) as f64);
        for x in 0..TL_RES_LEN {
            let m = ((1u32 << 16) as f64
                / 2f64.powf((x as f64 + 1.0) * (ENV_STEP / 4.0) / 8.0))
                .floor();
            let mut n = (m as i32) >> 4;
            n = if n & 1 != 0 { (n >> 1) + 1 } else { n >> 1 };
            tl_tab[x * 2] = n;
            tl_tab[x * 2 + 1] = -n;
            for i in 1..11 {
                tl_tab[x * 2 + i * 2 * TL_RES_LEN] = n >> i;
                tl_tab[x * 2 + 1 + i * 2 * TL_RES_LEN] = -(n >> i);
            }
        }
        for i in 0..SIN_LEN {
            let m = (((i * 2 + 1) as f64) * std::f64::consts::PI / SIN_LEN as f64).sin();
            let o = if m > 0.0 {
                8.0 * (1.0 / m).log2()
            } else {
                8.0 * (-1.0 / m).log2()
            } / (ENV_STEP / 4.0);
            let mut n = (2.0 * o) as i32;
            n = if n & 1 != 0 { (n >> 1) + 1 } else { n >> 1 };
            sin_tab[i] = (n * 2) as u32 + if m >= 0.0 { 0 } else { 1 };
            // Waveform 1: positive half only.
            sin_tab[SIN_LEN + i] = if i & (1 << (SIN_BITS - 1)) != 0 {
                TL_TAB_LEN as u32
            } else {
                sin_tab[i]
            };
        }

        // C OPLL_initalize(): all freqbase terms are exact at 1.0.
        let mut fn_tab = [0u32; 1024];
        for (i, e) in fn_tab.iter_mut().enumerate() {
            *e = (i as u32) << 12; // i * 64 * (1 << (FREQ_SH - 10))
        }

        let mut chip = Self {
            ch: [Channel::default(); 9],
            instvol_r: [0; 9],
            eg_cnt: 0,
            eg_timer: 0,
            eg_timer_add: 1 << 16,
            eg_timer_overflow: 1 << 16,
            rhythm: 0,
            lfo_am_cnt: 0,
            lfo_am_inc: (1 << LFO_SH) / 64,
            lfo_pm_cnt: 0,
            lfo_pm_inc: (1 << LFO_SH) / 1024,
            noise_rng: 0,
            noise_p: 0,
            noise_f: 1 << FREQ_SH,
            inst_tab: [[0; 8]; 19],
            fn_tab,
            address: 0,
            status: 0,
            tl_tab,
            sin_tab,
            lfo_am: 0,
            lfo_pm: 0,
            output: [0; 2],
        };
        chip.reset();
        chip
    }

    /// C YM2413ResetChip + the MSX deviation: output latch ON.
    pub fn reset(&mut self) {
        self.eg_timer = 0;
        self.eg_cnt = 0;
        self.noise_rng = 1;
        self.inst_tab = INSTRUMENT_ROM;

        self.write_reg(0x0F, 0);
        for r in (0x10..=0x3F).rev() {
            self.write_reg(r, 0);
        }

        for ch in self.ch.iter_mut() {
            for slot in ch.slot.iter_mut() {
                slot.wavetable = 0;
                slot.state = EG_OFF;
                slot.volume = MAX_ATT_INDEX;
            }
        }

        // MSX: no SMS enable latch — the chip always outputs.
        self.status = 1;
    }

    /// C YM2413Write: a = port (0 address / 1 data / 2 SMS enable latch).
    pub fn write(&mut self, a: u32, v: u8) {
        if a & 2 == 0 {
            if a & 1 == 0 {
                self.address = v;
            } else {
                self.write_reg(self.address, v);
            }
        } else {
            self.status = v & 0x01;
        }
    }

    // --- Per-sample machinery -----------------------------------------------

    /// C advance_lfo().
    fn advance_lfo(&mut self) {
        self.lfo_am_cnt += self.lfo_am_inc;
        if self.lfo_am_cnt >= (LFO_AM_TAB_ELEMENTS << LFO_SH) {
            self.lfo_am_cnt -= LFO_AM_TAB_ELEMENTS << LFO_SH;
        }
        self.lfo_am = (LFO_AM_TABLE[(self.lfo_am_cnt >> LFO_SH) as usize] >> 1) as u32;
        self.lfo_pm_cnt = self.lfo_pm_cnt.wrapping_add(self.lfo_pm_inc);
        self.lfo_pm = ((self.lfo_pm_cnt >> LFO_SH) & 7) as i32;
    }

    /// C advance(): envelope generator, phase generator, noise.
    fn advance(&mut self) {
        self.eg_timer += self.eg_timer_add;
        while self.eg_timer >= self.eg_timer_overflow {
            self.eg_timer -= self.eg_timer_overflow;
            self.eg_cnt += 1;

            for i in 0..9 * 2 {
                let rhythm_on = self.rhythm & 0x20 != 0;
                let eg_cnt = self.eg_cnt;
                let ch_sus = self.ch[i >> 1].sus;
                let op = &mut self.ch[i >> 1].slot[i & 1];
                match op.state {
                    EG_DMP => {
                        if (op.volume & !3) == (MAX_ATT_INDEX & !3) {
                            op.state = EG_ATT;
                            if op.ar + op.ksr as u32 >= 16 + 60 {
                                op.volume = MIN_ATT_INDEX;
                            }
                            // Carrier hit zero → reset BOTH operators' phases.
                            if i & 1 != 0 {
                                let ch = &mut self.ch[i >> 1];
                                ch.slot[0].phase = 0;
                                ch.slot[1].phase = 0;
                            }
                        } else if eg_cnt & ((1 << op.eg_sh_dp) - 1) == 0 {
                            op.volume += EG_INC[op.eg_sel_dp as usize
                                + ((eg_cnt >> op.eg_sh_dp) & 15) as usize]
                                as i32;
                        }
                    }
                    EG_ATT => {
                        if op.volume == MIN_ATT_INDEX {
                            op.state = EG_DEC;
                        } else if eg_cnt & (((1 << op.eg_sh_ar) - 1) & !3) == 0 {
                            op.volume += (!op.volume
                                * EG_MUL[op.eg_sel_ar as usize
                                    + ((eg_cnt >> op.eg_sh_ar) & 15) as usize]
                                    as i32)
                                >> 4;
                        }
                    }
                    EG_DEC => {
                        if (op.volume & !7) as u32 == op.sl {
                            op.state = EG_SUS;
                        } else if eg_cnt & ((1 << op.eg_sh_dr) - 1) == 0 {
                            op.volume += EG_INC[op.eg_sel_dr as usize
                                + ((eg_cnt >> op.eg_sh_dr) & 15) as usize]
                                as i32;
                            if (op.volume & !3) == (MAX_ATT_INDEX & !3) {
                                op.state = EG_OFF;
                            }
                        }
                    }
                    EG_SUS => {
                        if !op.eg_type {
                            // Percussive mode: sustain adds release rate.
                            if eg_cnt & ((1 << op.eg_sh_rr) - 1) == 0 {
                                op.volume += EG_INC[op.eg_sel_rr as usize
                                    + ((eg_cnt >> op.eg_sh_rr) & 15) as usize]
                                    as i32;
                                if (op.volume & !3) == (MAX_ATT_INDEX & !3) {
                                    op.state = EG_OFF;
                                }
                            }
                        }
                    }
                    EG_REL => {
                        // Modulators don't release, except rhythm slots.
                        if (i & 1 != 0) || (rhythm_on && i >= 12) {
                            let (sh, sel) = if op.eg_type {
                                if ch_sus {
                                    (op.eg_sh_rs, op.eg_sel_rs)
                                } else {
                                    (op.eg_sh_rr, op.eg_sel_rr)
                                }
                            } else {
                                (op.eg_sh_rs, op.eg_sel_rs)
                            };
                            if eg_cnt & ((1 << sh) - 1) == 0 {
                                op.volume += EG_INC
                                    [sel as usize + ((eg_cnt >> sh) & 15) as usize]
                                    as i32;
                                if (op.volume & !3) == (MAX_ATT_INDEX & !3) {
                                    op.state = EG_OFF;
                                }
                            }
                        }
                    }
                    EG_OFF => op.volume = MAX_ATT_INDEX,
                    _ => {}
                }
            }
        }

        // Phase generator.
        for i in 0..9 * 2 {
            let block_fnum_ch = self.ch[i >> 1].block_fnum;
            let lfo_pm = self.lfo_pm;
            let fn_tab = &self.fn_tab;
            let op = &mut self.ch[i >> 1].slot[i & 1];
            if op.vib {
                let fnum_lfo = 8 * ((block_fnum_ch & 0x01C0) >> 6);
                let mut block_fnum = block_fnum_ch * 2;
                let offset =
                    LFO_PM_TABLE[(lfo_pm + fnum_lfo as i32) as usize] as i32;
                if offset != 0 {
                    block_fnum = block_fnum.wrapping_add_signed(offset);
                    let block = (block_fnum & 0x1C00) >> 10;
                    op.phase = op.phase.wrapping_add(
                        (fn_tab[(block_fnum & 0x03FF) as usize] >> (7 - block))
                            * op.mul as u32,
                    );
                } else {
                    op.phase = op.phase.wrapping_add(op.freq);
                }
            } else {
                op.phase = op.phase.wrapping_add(op.freq);
            }
        }

        // Noise: 23-bit shift register, one shift per sample at this rate.
        self.noise_p += self.noise_f;
        let mut shifts = self.noise_p >> FREQ_SH;
        self.noise_p &= FREQ_MASK;
        while shifts > 0 {
            if self.noise_rng & 1 != 0 {
                self.noise_rng ^= 0x800302;
            }
            self.noise_rng >>= 1;
            shifts -= 1;
        }
    }

    /// C op_calc: phase-modulated operator output (pm scaled <<17).
    fn op_calc(&self, phase: u32, env: u32, pm: i32, wave_tab: usize) -> i32 {
        let idx = ((phase & !FREQ_MASK) as i32).wrapping_add(pm << 17) >> FREQ_SH;
        let p = (env << 5) as usize
            + self.sin_tab[wave_tab + (idx as usize & SIN_MASK)] as usize;
        if p >= TL_TAB_LEN {
            return 0;
        }
        self.tl_tab[p]
    }

    /// C op_calc1: feedback variant (pm used raw).
    fn op_calc1(&self, phase: u32, env: u32, pm: i32, wave_tab: usize) -> i32 {
        let idx = ((phase & !FREQ_MASK) as i32).wrapping_add(pm) >> FREQ_SH;
        let p = (env << 5) as usize
            + self.sin_tab[wave_tab + (idx as usize & SIN_MASK)] as usize;
        if p >= TL_TAB_LEN {
            return 0;
        }
        self.tl_tab[p]
    }

    /// C volume_calc macro.
    fn volume_calc(&self, ch: usize, slot: usize) -> u32 {
        let op = &self.ch[ch].slot[slot];
        if op.state != EG_OFF {
            (op.tll + op.volume + (self.lfo_am & op.am_mask) as i32) as u32
        } else {
            ENV_QUIET
        }
    }

    /// C chan_calc: one melody channel into output[0].
    fn chan_calc(&mut self, ch: usize) {
        // SLOT 1 (modulator with feedback).
        let env = self.volume_calc(ch, 0);
        let slot0 = &self.ch[ch].slot[0];
        let mut out = slot0.op1_out[0] + slot0.op1_out[1];
        let new_op1_out0 = slot0.op1_out[1];
        let phase_modulation = new_op1_out0;
        let (phase0, fb_shift, wave0) = (slot0.phase, slot0.fb_shift, slot0.wavetable);

        let mut new_op1_out1 = 0;
        if env < ENV_QUIET {
            if fb_shift == 0 {
                out = 0;
            }
            new_op1_out1 = self.op_calc1(phase0, env, out << fb_shift, wave0);
        }
        {
            let slot0 = &mut self.ch[ch].slot[0];
            slot0.op1_out[0] = new_op1_out0;
            slot0.op1_out[1] = new_op1_out1;
        }

        // SLOT 2 (carrier).
        let env = self.volume_calc(ch, 1);
        if env < ENV_QUIET {
            let slot1 = &self.ch[ch].slot[1];
            self.output[0] +=
                self.op_calc(slot1.phase, env, phase_modulation, slot1.wavetable);
        }
    }

    /// C rhythm_calc: bass drum + the four phase-trickery percussions
    /// into output[1]. `noise` is bit 0 of the noise shift register.
    fn rhythm_calc(&mut self, noise: u32) {
        // Bass drum — channel 6, regular 2-operator FM.
        let env = self.volume_calc(6, 0);
        let slot0 = &self.ch[6].slot[0];
        let mut out = slot0.op1_out[0] + slot0.op1_out[1];
        let new_op1_out0 = slot0.op1_out[1];
        let phase_modulation = new_op1_out0;
        let (phase0, fb_shift, wave0) = (slot0.phase, slot0.fb_shift, slot0.wavetable);

        let mut new_op1_out1 = 0;
        if env < ENV_QUIET {
            if fb_shift == 0 {
                out = 0;
            }
            new_op1_out1 = self.op_calc1(phase0, env, out << fb_shift, wave0);
        }
        {
            let slot0 = &mut self.ch[6].slot[0];
            slot0.op1_out[0] = new_op1_out0;
            slot0.op1_out[1] = new_op1_out1;
        }

        let env = self.volume_calc(6, 1);
        if env < ENV_QUIET {
            let slot1 = &self.ch[6].slot[1];
            self.output[1] +=
                self.op_calc(slot1.phase, env, phase_modulation, slot1.wavetable);
        }

        // High hat.
        let env = self.volume_calc(7, 0);
        if env < ENV_QUIET {
            let ph71 = self.ch[7].slot[0].phase >> FREQ_SH;
            let bit7 = (ph71 >> 7) & 1;
            let bit3 = (ph71 >> 3) & 1;
            let bit2 = (ph71 >> 2) & 1;
            let res1 = (bit2 ^ bit7) | bit3;
            let mut phase: u32 = if res1 != 0 { 0x200 | (0xD0 >> 2) } else { 0xD0 };

            let ph82 = self.ch[8].slot[1].phase >> FREQ_SH;
            let res2 = ((ph82 >> 3) & 1) | ((ph82 >> 5) & 1);
            if res2 != 0 {
                phase = 0x200 | (0xD0 >> 2);
            }
            if phase & 0x200 != 0 {
                if noise != 0 {
                    phase = 0x200 | 0xD0;
                }
            } else if noise != 0 {
                phase = 0xD0 >> 2;
            }
            let wave = self.ch[7].slot[0].wavetable;
            self.output[1] += self.op_calc(phase << FREQ_SH, env, 0, wave);
        }

        // Snare drum.
        let env = self.volume_calc(7, 1);
        if env < ENV_QUIET {
            let bit8 = ((self.ch[7].slot[0].phase >> FREQ_SH) >> 8) & 1;
            let mut phase: u32 = if bit8 != 0 { 0x200 } else { 0x100 };
            if noise != 0 {
                phase ^= 0x100;
            }
            let wave = self.ch[7].slot[1].wavetable;
            self.output[1] += self.op_calc(phase << FREQ_SH, env, 0, wave);
        }

        // Tom tom.
        let env = self.volume_calc(8, 0);
        if env < ENV_QUIET {
            let slot = &self.ch[8].slot[0];
            self.output[1] += self.op_calc(slot.phase, env, 0, slot.wavetable);
        }

        // Top cymbal.
        let env = self.volume_calc(8, 1);
        if env < ENV_QUIET {
            let ph71 = self.ch[7].slot[0].phase >> FREQ_SH;
            let bit7 = (ph71 >> 7) & 1;
            let bit3 = (ph71 >> 3) & 1;
            let bit2 = (ph71 >> 2) & 1;
            let res1 = (bit2 ^ bit7) | bit3;
            let mut phase: u32 = if res1 != 0 { 0x300 } else { 0x100 };

            let ph82 = self.ch[8].slot[1].phase >> FREQ_SH;
            let res2 = ((ph82 >> 3) & 1) | ((ph82 >> 5) & 1);
            if res2 != 0 {
                phase = 0x300;
            }
            let wave = self.ch[8].slot[1].wavetable;
            self.output[1] += self.op_calc(phase << FREQ_SH, env, 0, wave);
        }
    }

    // --- Register plumbing ---------------------------------------------------

    /// C KEY_ON.
    fn key_on(&mut self, ch: usize, slot: usize, key_set: u32) {
        let op = &mut self.ch[ch].slot[slot];
        if op.key == 0 {
            op.state = EG_DMP;
        }
        op.key |= key_set;
    }

    /// C KEY_OFF.
    fn key_off(&mut self, ch: usize, slot: usize, key_clr: u32) {
        let op = &mut self.ch[ch].slot[slot];
        if op.key != 0 {
            op.key &= key_clr;
            if op.key == 0 {
                op.state = if (op.volume & !3) == (MAX_ATT_INDEX & !3) {
                    EG_OFF
                } else {
                    EG_REL
                };
            }
        }
    }

    /// C CALC_FCSLOT.
    fn calc_fcslot(&mut self, ch: usize, slot: usize) {
        let (fc, kcode, sus) = {
            let c = &self.ch[ch];
            (c.fc, c.kcode, c.sus)
        };
        let op = &mut self.ch[ch].slot[slot];
        op.freq = fc * op.mul as u32;
        let ksr = kcode >> op.ksr_shift;

        if op.ksr != ksr {
            op.ksr = ksr;
            if op.ar + op.ksr as u32 >= 16 + 60 {
                op.eg_sh_ar = 13;
                op.eg_sel_ar = 0;
            } else if op.ar + op.ksr as u32 >= 16 + 48 {
                op.eg_sh_ar = 0;
                op.eg_sel_ar = EG_RATE_SELECT[(op.ar + op.ksr as u32) as usize]
                    + (4 * RATE_STEPS) as u16;
            } else {
                op.eg_sh_ar = EG_RATE_SHIFT[(op.ar + op.ksr as u32) as usize];
                op.eg_sel_ar = EG_RATE_SELECT[(op.ar + op.ksr as u32) as usize];
            }
            op.eg_sh_dr = EG_RATE_SHIFT[(op.dr + op.ksr as u32) as usize];
            op.eg_sel_dr = EG_RATE_SELECT[(op.dr + op.ksr as u32) as usize];
            op.eg_sh_rr = EG_RATE_SHIFT[(op.rr + op.ksr as u32) as usize];
            op.eg_sel_rr = EG_RATE_SELECT[(op.rr + op.ksr as u32) as usize];
        }

        let slot_rs: u32 = if sus { 16 + (5 << 2) } else { 16 + (7 << 2) };
        op.eg_sh_rs = EG_RATE_SHIFT[(slot_rs + op.ksr as u32) as usize];
        op.eg_sel_rs = EG_RATE_SELECT[(slot_rs + op.ksr as u32) as usize];

        let slot_dp: u32 = 16 + (12 << 2);
        op.eg_sh_dp = EG_RATE_SHIFT[(slot_dp + op.ksr as u32) as usize];
        op.eg_sel_dp = EG_RATE_SELECT[(slot_dp + op.ksr as u32) as usize];
    }

    /// C set_mul — `slot` is the global slot index (channel*2 + 0/1).
    fn set_mul(&mut self, slot: usize, v: u8) {
        let ch = slot / 2;
        {
            let op = &mut self.ch[ch].slot[slot & 1];
            op.mul = MUL_TAB[(v & 0x0F) as usize];
            op.ksr_shift = if v & 0x10 != 0 { 0 } else { 2 };
            op.eg_type = v & 0x20 != 0;
            op.vib = v & 0x40 != 0;
            op.am_mask = if v & 0x80 != 0 { !0 } else { 0 };
        }
        self.calc_fcslot(ch, slot & 1);
    }

    /// C set_ksl_tl.
    fn set_ksl_tl(&mut self, chan: usize, v: u8) {
        let ksl_base = self.ch[chan].ksl_base;
        let op = &mut self.ch[chan].slot[0];
        let ksl = v >> 6;
        op.ksl = if ksl != 0 { 3 - ksl } else { 31 };
        op.tl = ((v & 0x3F) as u32) << (ENV_BITS - 2 - 7);
        op.tll = op.tl as i32 + (ksl_base >> op.ksl) as i32;
    }

    /// C set_ksl_wave_fb.
    fn set_ksl_wave_fb(&mut self, chan: usize, v: u8) {
        let ksl_base = self.ch[chan].ksl_base;
        {
            let op = &mut self.ch[chan].slot[0];
            op.wavetable = (((v & 0x08) >> 3) as usize) * SIN_LEN;
            op.fb_shift = if v & 7 != 0 { (v & 7) + 8 } else { 0 };
        }
        let op = &mut self.ch[chan].slot[1];
        op.wavetable = (((v & 0x10) >> 4) as usize) * SIN_LEN;
        let ksl = v >> 6;
        op.ksl = if ksl != 0 { 3 - ksl } else { 31 };
        op.tll = op.tl as i32 + (ksl_base >> op.ksl) as i32;
    }

    /// C set_ar_dr.
    fn set_ar_dr(&mut self, slot: usize, v: u8) {
        let ch = slot / 2;
        let op = &mut self.ch[ch].slot[slot & 1];
        op.ar = if v >> 4 != 0 { 16 + (((v >> 4) as u32) << 2) } else { 0 };

        if op.ar + op.ksr as u32 >= 16 + 60 {
            op.eg_sh_ar = 13;
            op.eg_sel_ar = 0;
        } else if op.ar + op.ksr as u32 >= 16 + 48 {
            op.eg_sh_ar = 0;
            op.eg_sel_ar =
                EG_RATE_SELECT[(op.ar + op.ksr as u32) as usize] + (4 * RATE_STEPS) as u16;
        } else {
            op.eg_sh_ar = EG_RATE_SHIFT[(op.ar + op.ksr as u32) as usize];
            op.eg_sel_ar = EG_RATE_SELECT[(op.ar + op.ksr as u32) as usize];
        }

        op.dr = if v & 0x0F != 0 { 16 + (((v & 0x0F) as u32) << 2) } else { 0 };
        op.eg_sh_dr = EG_RATE_SHIFT[(op.dr + op.ksr as u32) as usize];
        op.eg_sel_dr = EG_RATE_SELECT[(op.dr + op.ksr as u32) as usize];
    }

    /// C set_sl_rr.
    fn set_sl_rr(&mut self, slot: usize, v: u8) {
        let ch = slot / 2;
        let op = &mut self.ch[ch].slot[slot & 1];
        op.sl = SL_TAB[(v >> 4) as usize];
        op.rr = if v & 0x0F != 0 { 16 + (((v & 0x0F) as u32) << 2) } else { 0 };
        op.eg_sh_rr = EG_RATE_SHIFT[(op.rr + op.ksr as u32) as usize];
        op.eg_sel_rr = EG_RATE_SELECT[(op.rr + op.ksr as u32) as usize];
    }

    /// C load_instrument.
    fn load_instrument(&mut self, chan: usize, slot: usize, inst: [u8; 8]) {
        self.set_mul(slot, inst[0]);
        self.set_mul(slot + 1, inst[1]);
        self.set_ksl_tl(chan, inst[2]);
        self.set_ksl_wave_fb(chan, inst[3]);
        self.set_ar_dr(slot, inst[4]);
        self.set_ar_dr(slot + 1, inst[5]);
        self.set_sl_rr(slot, inst[6]);
        self.set_sl_rr(slot + 1, inst[7]);
    }

    /// C update_instrument_zero: re-apply one byte of the user instrument
    /// to every channel currently playing instrument 0.
    fn update_instrument_zero(&mut self, r: u8) {
        let inst = self.inst_tab[0];
        let chan_max = if self.rhythm & 0x20 != 0 { 6 } else { 9 };
        for chan in 0..chan_max {
            if self.instvol_r[chan] & 0xF0 != 0 {
                continue;
            }
            match r & 7 {
                0 => self.set_mul(chan * 2, inst[0]),
                1 => self.set_mul(chan * 2 + 1, inst[1]),
                2 => self.set_ksl_tl(chan, inst[2]),
                3 => self.set_ksl_wave_fb(chan, inst[3]),
                4 => self.set_ar_dr(chan * 2, inst[4]),
                5 => self.set_ar_dr(chan * 2 + 1, inst[5]),
                6 => self.set_sl_rr(chan * 2, inst[6]),
                _ => self.set_sl_rr(chan * 2 + 1, inst[7]),
            }
        }
    }

    /// C OPLLWriteReg.
    pub fn write_reg(&mut self, r: u8, v: u8) {
        match r & 0xF0 {
            0x00 => match r & 0x0F {
                0x00..=0x07 => {
                    self.inst_tab[0][(r & 7) as usize] = v;
                    self.update_instrument_zero(r);
                }
                0x0E => {
                    if v & 0x20 != 0 {
                        if self.rhythm & 0x20 == 0 {
                            // Rhythm OFF → ON: load drum instruments.
                            self.load_instrument(6, 12, self.inst_tab[16]);
                            self.load_instrument(7, 14, self.inst_tab[17]);
                            {
                                let tl = (((self.instvol_r[7] >> 4) as u32) << 2)
                                    << (ENV_BITS - 2 - 7);
                                let ksl_base = self.ch[7].ksl_base;
                                let op = &mut self.ch[7].slot[0]; // HH
                                op.tl = tl;
                                op.tll = op.tl as i32 + (ksl_base >> op.ksl) as i32;
                            }
                            self.load_instrument(8, 16, self.inst_tab[18]);
                            {
                                let tl = (((self.instvol_r[8] >> 4) as u32) << 2)
                                    << (ENV_BITS - 2 - 7);
                                let ksl_base = self.ch[8].ksl_base;
                                let op = &mut self.ch[8].slot[0]; // TOM
                                op.tl = tl;
                                op.tll = op.tl as i32 + (ksl_base >> op.ksl) as i32;
                            }
                        }
                        // Drum key on/off bits.
                        if v & 0x10 != 0 {
                            self.key_on(6, 0, 2);
                            self.key_on(6, 1, 2);
                        } else {
                            self.key_off(6, 0, !2);
                            self.key_off(6, 1, !2);
                        }
                        if v & 0x01 != 0 { self.key_on(7, 0, 2) } else { self.key_off(7, 0, !2) }
                        if v & 0x08 != 0 { self.key_on(7, 1, 2) } else { self.key_off(7, 1, !2) }
                        if v & 0x04 != 0 { self.key_on(8, 0, 2) } else { self.key_off(8, 0, !2) }
                        if v & 0x02 != 0 { self.key_on(8, 1, 2) } else { self.key_off(8, 1, !2) }
                    } else {
                        // Rhythm ON → OFF: restore melody instruments.
                        if self.rhythm & 0x20 != 0 {
                            self.load_instrument(
                                6, 12, self.inst_tab[(self.instvol_r[6] >> 4) as usize],
                            );
                            self.load_instrument(
                                7, 14, self.inst_tab[(self.instvol_r[7] >> 4) as usize],
                            );
                            self.load_instrument(
                                8, 16, self.inst_tab[(self.instvol_r[8] >> 4) as usize],
                            );
                        }
                        self.key_off(6, 0, !2);
                        self.key_off(6, 1, !2);
                        self.key_off(7, 0, !2);
                        self.key_off(7, 1, !2);
                        self.key_off(8, 0, !2);
                        self.key_off(8, 1, !2);
                    }
                    self.rhythm = v & 0x3F;
                }
                _ => {}
            },

            0x10 | 0x20 => {
                let mut chan = (r & 0x0F) as usize;
                if chan >= 9 {
                    chan -= 9; // verified on real YM2413
                }
                let block_fnum;
                if r & 0x10 != 0 {
                    // 10-18: FNUM 0-7.
                    block_fnum = (self.ch[chan].block_fnum & 0x0F00) | v as u32;
                } else {
                    // 20-28: SUS, KEY, BLOCK, FNUM 8.
                    block_fnum =
                        (((v & 0x0F) as u32) << 8) | (self.ch[chan].block_fnum & 0xFF);
                    if v & 0x10 != 0 {
                        self.key_on(chan, 0, 1);
                        self.key_on(chan, 1, 1);
                    } else {
                        self.key_off(chan, 0, !1);
                        self.key_off(chan, 1, !1);
                    }
                    self.ch[chan].sus = v & 0x20 != 0;
                }

                if self.ch[chan].block_fnum != block_fnum {
                    {
                        let ch = &mut self.ch[chan];
                        ch.block_fnum = block_fnum;
                        ch.kcode = ((block_fnum & 0x0F00) >> 8) as u8;
                        ch.ksl_base = KSL_TAB[(block_fnum >> 5) as usize];
                        let bf2 = block_fnum * 2;
                        let block = (bf2 & 0x1C00) >> 10;
                        ch.fc = self.fn_tab[(bf2 & 0x03FF) as usize] >> (7 - block);
                        // Refresh total level in both slots.
                        let ksl_base = ch.ksl_base;
                        for s in 0..2 {
                            let op = &mut ch.slot[s];
                            op.tll = op.tl as i32 + (ksl_base >> op.ksl) as i32;
                        }
                    }
                    self.calc_fcslot(chan, 0);
                    self.calc_fcslot(chan, 1);
                }
            }

            0x30 => {
                let mut chan = (r & 0x0F) as usize;
                if chan >= 9 {
                    chan -= 9;
                }
                {
                    let ksl_base = self.ch[chan].ksl_base;
                    let op = &mut self.ch[chan].slot[1]; // carrier
                    op.tl = (((v & 0x0F) as u32) << 2) << (ENV_BITS - 2 - 7);
                    op.tll = op.tl as i32 + (ksl_base >> op.ksl) as i32;
                }

                if chan >= 6 && self.rhythm & 0x20 != 0 {
                    // Rhythm mode: channels 7/8 carry HH/TOM level in the
                    // upper nibble.
                    if chan >= 7 {
                        let ksl_base = self.ch[chan].ksl_base;
                        let op = &mut self.ch[chan].slot[0];
                        op.tl = (((v >> 4) as u32) << 2) << (ENV_BITS - 2 - 7);
                        op.tll = op.tl as i32 + (ksl_base >> op.ksl) as i32;
                    }
                } else if self.instvol_r[chan] & 0xF0 != v & 0xF0 {
                    self.instvol_r[chan] = v;
                    self.load_instrument(chan, chan * 2, self.inst_tab[(v >> 4) as usize]);
                }
            }

            _ => {}
        }
    }

    /// C YM2413Update, one sample: melody + rhythm mix, ×2 amplification.
    /// Output range is roughly ±12000 — the audio layer scales to f32.
    pub fn sample(&mut self) -> i32 {
        self.output = [0, 0];
        self.advance_lfo();

        for ch in 0..6 {
            self.chan_calc(ch);
        }
        if self.rhythm & 0x20 == 0 {
            for ch in 6..9 {
                self.chan_calc(ch);
            }
        } else {
            self.rhythm_calc(self.noise_rng & 1);
        }

        let out = (self.output[0] + self.output[1] * 2) * 2 * self.status as i32;
        self.advance();
        out
    }
}

impl Default for Ym2413 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scripted register sequence shared with the C golden-master harness
    /// (tools/ym2413_ref.c): instrument + volume on three channels, key-on
    /// with different fnum/block values, rhythm mode with all drums, a
    /// mid-stream key-off, and the user instrument.
    fn drive(chip: &mut dyn FnMut(u8, u8)) {
        // User instrument setup (registers 0-7).
        for (r, v) in [(0u8, 0x61u8), (1, 0x61), (2, 0x1E), (3, 0x17), (4, 0xF0), (5, 0x7F), (6, 0x00), (7, 0x17)] {
            chip(r, v);
        }
        // Channel 0: violin (inst 1), vol 0, fnum 0x AB block 4, key on.
        chip(0x30, 0x10);
        chip(0x10, 0xAB);
        chip(0x20, 0x18);
        // Channel 1: user instrument, vol 3, key on with sustain.
        chip(0x31, 0x03);
        chip(0x11, 0x55);
        chip(0x21, 0x3A);
        // Channel 2: piano (inst 3), softer.
        chip(0x32, 0x37);
        chip(0x12, 0xCD);
        chip(0x22, 0x12);
        // Rhythm mode on, all drums keyed.
        chip(0x16, 0x20);
        chip(0x26, 0x05);
        chip(0x17, 0x50);
        chip(0x27, 0x05);
        chip(0x18, 0xC0);
        chip(0x28, 0x01);
        chip(0x0E, 0x3F);
    }

    /// Smoke test: the scripted sequence produces sound, key-off decays it.
    #[test]
    fn produces_audio_and_decays() {
        let mut chip = Ym2413::new();
        drive(&mut |r, v| chip.write_reg(r, v));

        let mut peak: i32 = 0;
        for _ in 0..20_000 {
            peak = peak.max(chip.sample().abs());
        }
        assert!(peak > 1000, "expected audible output, peak {}", peak);

        // Key everything off; after a generous release the output dies.
        chip.write_reg(0x20, 0x08);
        chip.write_reg(0x21, 0x2A);
        chip.write_reg(0x22, 0x02);
        chip.write_reg(0x0E, 0x20);
        for _ in 0..200_000 {
            chip.sample();
        }
        let mut tail: i32 = 0;
        for _ in 0..2_000 {
            tail = tail.max(chip.sample().abs());
        }
        assert!(tail < 32, "expected near-silence after release, got {}", tail);
    }

    /// The golden-master comparison itself runs out-of-band (see
    /// tools/ym2413_ref.c); this pins the first samples' checksum so any
    /// future refactor that changes output gets flagged immediately.
    #[test]
    fn output_checksum_is_stable() {
        let mut chip = Ym2413::new();
        drive(&mut |r, v| chip.write_reg(r, v));
        let mut hash: u64 = 0xcbf29ce484222325;
        for _ in 0..50_000 {
            let s = chip.sample() as u64;
            hash ^= s;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        // Golden-master value: tools/ym2413_ref.c (the reference C build)
        // prints the same FNV-1a hash for this exact script — verified
        // bit-identical at porting time. If a refactor changes this,
        // re-run the C harness before accepting a new value.
        assert_eq!(hash, 0x3a47e445240cc2d7);
    }
}
