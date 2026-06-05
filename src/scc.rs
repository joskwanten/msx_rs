//! Konami SCC sound chip — five-channel wavetable synth on Konami-SCC carts.
//!
//! Direct port of the user's TypeScript `SCC_Processor`. The SCC is a small
//! custom chip embedded in Konami's cartridge ROM mapper (Salamander, F1
//! Spirit, Snake's Revenge, etc.). It exposes 256 register bytes mapped into
//! cartridge address 0x9800..0x98FF when bank 4 of the mapper is set to 0x3F.
//!
//! Register layout:
//!   0x00..0x1F   waveform table for channel 0   (signed 8-bit samples)
//!   0x20..0x3F   waveform table for channel 1
//!   0x40..0x5F   waveform table for channel 2
//!   0x60..0x7F   waveform table for channels 3 and 4 (shared)
//!   0x80..0x89   frequency divider (12-bit) for each channel (lo, hi×5)
//!   0x8A..0x8E   per-channel volume (low nibble)
//!   0x8F         channel-enable mask (bit per channel)
//!
//! Synthesis is straightforward: each channel has a 32-position phase
//! accumulator stepping at the configured frequency. The instantaneous
//! sample is the waveform byte at the current phase position, scaled by
//! per-channel volume, summed across all five channels.

const PSG_CLOCK: f32 = 3_579_545.0;
const WAVEFORM_LENGTH: u32 = 32;
const PHASE_BITS: u32 = 27;

pub struct Scc {
    /// Raw register bytes (0x00..0xFF). The TS code stores these as both
    /// signed (`Int8Array`) and unsigned (`Uint8Array`) views over the same
    /// buffer; we keep one `[u8; 256]` and cast per-use.
    regs: [u8; 256],
    /// Phase accumulator for each of five channels, fixed-point with
    /// `PHASE_BITS` of fractional bits. The top 5 bits index into the 32-
    /// position waveform.
    phase: [u32; 5],
    /// Phase increment per output sample.
    step: [u32; 5],
    sample_rate: f32,
}

impl Scc {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            regs: [0u8; 256],
            phase: [0u32; 5],
            step: [0u32; 5],
            sample_rate,
        }
    }

    /// Wipe all registers and channel state — used on cartridge swap so
    /// notes that were ringing in the previous game don't leak into the next.
    /// The SCC stays initialised against the same sample rate; only the
    /// dynamic state goes back to zero.
    pub fn reset(&mut self) {
        self.regs.fill(0);
        self.phase.fill(0);
        self.step.fill(0);
    }

    /// Write to one of the SCC's 256 mapped registers. Writing to the
    /// frequency-divider area (0x80..0x89) resets the channel's phase and
    /// recomputes its step — matching real-hardware behaviour where changing
    /// the tone period restarts the waveform.
    pub fn write_reg(&mut self, addr: u8, value: u8) {
        self.regs[addr as usize] = value;
        if (0x80..=0x89).contains(&addr) {
            let chan = ((addr - 0x80) / 2) as usize;
            self.phase[chan] = 0;
            self.step[chan] = self.compute_step(chan);
        }
    }

    fn compute_step(&self, chan: usize) -> u32 {
        // 12-bit tone period split across two consecutive registers.
        let lo = self.regs[0x80 + 2 * chan] as u32;
        let hi = (self.regs[0x80 + 2 * chan + 1] as u32) & 0x0F;
        let period = lo | (hi << 8);

        let freq = PSG_CLOCK / (WAVEFORM_LENGTH as f32 * (period as f32 + 1.0));
        ((WAVEFORM_LENGTH as f32 * freq / self.sample_rate) * (1u32 << PHASE_BITS) as f32) as u32
    }

    /// Generate one mono sample at the configured sample rate. Returns a
    /// value in roughly [-1.0, 1.0]. Designed to be called once per sample
    /// from the audio callback.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn next_sample(&mut self) -> f32 {
        let mut mix: i32 = 0;
        for chan in 0..5 {
            self.phase[chan] = self.phase[chan].wrapping_add(self.step[chan]);
            let pos = (self.phase[chan] >> PHASE_BITS) as usize; // 0..31

            // Channels 3 and 4 share waveform table 3 (offset 0x60..0x7F).
            let wave_chan = if chan > 3 { 3 } else { chan };
            let wave = self.regs[(wave_chan << 5) + pos] as i8 as i32;
            let vol = self.volume(chan) as i32;
            mix += wave * vol;
        }

        // Normalize: peak amplitude = 128 (sample) × 15 (volume) × 5 (channels) = 9600.
        mix as f32 / 9600.0
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn volume(&self, chan: usize) -> u8 {
        if self.regs[0x8F] & (1 << chan) != 0 {
            self.regs[0x8A + chan] & 0x0F
        } else {
            0
        }
    }
}
