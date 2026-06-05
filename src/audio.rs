//! Audio output: cpal stream that mixes the MSX's two sound chips.
//!
//! Two chips contribute audio on a real MSX1:
//!
//! - The **PSG** (AY-3-8910) lives on the main board; the Z80 talks to it
//!   via ports 0xA0/0xA1. It's a 3-channel square-wave + noise + envelope
//!   chip. We use the `psg` crate (a Rust port of Peter Sovietov's Ayumi).
//!
//! - The **SCC** sits inside Konami cartridges; the game writes to its
//!   registers through bank-switched cartridge addresses. Five wavetable
//!   channels, simpler than the PSG. Implemented in `crate::scc`.
//!
//! Threading model on native is the standard audio one: each chip lives
//! behind an `Arc<Mutex<_>>`. The audio callback locks both, generates one
//! mixed sample per output frame, unlocks. Register writes from the
//! emulator (via the Bus / cartridge mapper) lock briefly too. Contention
//! is low — emulator writes are sparse, audio callback runs in short bursts.
//!
//! On WebAssembly there's no `cpal` — Web Audio integration is its own
//! project. We compile a stub that owns the chips but produces no sound,
//! so the rest of the emulator (which writes to PSG/SCC registers
//! unconditionally) keeps working.

use crate::scc::Scc;
use psg::PSG;
use std::sync::{Arc, Mutex};

/// MSX-1 PSG clock: half the Z80 clock (3.579545 MHz / 2).
///
/// Note: the `psg` crate (an Ayumi port) divides this further by 8 internally
/// to produce the chip's tone-generator rate. That matches AY-3-8910 hardware
/// (which has a /8 master divider + /2 from toggle = /16 effective per-cycle).
/// Some Ayumi callers pass the *master* clock instead — that would make the
/// envelope step rate match too. If SFX sound off-pitch, try doubling this.
const MSX_PSG_CLOCK: f64 = 1_789_772.5;

pub struct Audio {
    pub psg: Arc<Mutex<PSG>>,
    pub scc: Arc<Mutex<Scc>>,
    /// The cpal stream itself on native. Holding it keeps audio alive;
    /// dropping it stops playback. We never touch it again after construction.
    #[cfg(not(target_arch = "wasm32"))]
    _stream: cpal::Stream,
    #[cfg(target_arch = "wasm32")]
    web: WebAudio,
}

/// Browser-side audio glue: AudioContext + ScriptProcessorNode + the JS
/// closure we registered for `onaudioprocess`. All three need to stay alive
/// for sound to keep flowing — drop the Audio struct and they collapse.
#[cfg(target_arch = "wasm32")]
struct WebAudio {
    context: web_sys::AudioContext,
    _processor: web_sys::ScriptProcessorNode,
    _callback: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::AudioProcessingEvent)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Audio {
    pub fn new() -> Self {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no audio output device available");
        let supported = device
            .default_output_config()
            .expect("failed to query default audio config");

        let sample_rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        let config: cpal::StreamConfig = supported.into();

        let mut psg_inner = PSG::new(MSX_PSG_CLOCK, sample_rate)
            .expect("failed to create PSG with MSX clock");
        // MSX uses the GI AY-3-8910 (or an exact YM2149 clone in some
        // machines). Default in the crate is YM, which differs in DAC table.
        psg_inner.set_chip_type(psg::ChipType::AY);
        let psg = Arc::new(Mutex::new(psg_inner));
        let scc = Arc::new(Mutex::new(Scc::new(sample_rate as f32)));

        let psg_audio = Arc::clone(&psg);
        let scc_audio = Arc::clone(&scc);

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    let mut psg = psg_audio.lock().unwrap();
                    let mut scc = scc_audio.lock().unwrap();

                    for frame in data.chunks_mut(channels) {
                        let (psg_l, psg_r) = psg.render();
                        // Average PSG channels into a mono signal, then mix
                        // SCC's mono output on top. Volumes are eyeballed —
                        // both chips peak around 1.0, so we attenuate to keep
                        // headroom and avoid clipping when both are loud.
                        let psg_mono = ((psg_l + psg_r) * 0.5) as f32;
                        let mix = psg_mono * 0.6 + scc.next_sample() * 0.6;

                        for out in frame.iter_mut() {
                            *out = mix;
                        }
                    }
                },
                |err| eprintln!("audio stream error: {err}"),
                None,
            )
            .expect("failed to build audio output stream");

        stream.play().expect("failed to start audio stream");

        Self {
            psg,
            scc,
            _stream: stream,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Audio {
    /// Wire up Web Audio output.
    ///
    /// AudioContext gives us the host sample rate. We pump samples through a
    /// `ScriptProcessorNode` — its `onaudioprocess` callback fires on the
    /// main thread (single-threaded WASM, so no contention with the emulator
    /// loop) and asks us to fill a buffer. We mix PSG + SCC the same way
    /// cpal does on native.
    ///
    /// Browsers refuse to start audio without a user gesture, so the context
    /// often boots in the `suspended` state. Call [`Audio::resume`] from a
    /// keyboard or click handler to unlock playback.
    pub fn new() -> Self {
        use wasm_bindgen::prelude::*;
        use wasm_bindgen::JsCast;

        let context = web_sys::AudioContext::new()
            .expect("failed to create AudioContext");
        let sample_rate = context.sample_rate() as u32;

        let mut psg_inner = PSG::new(MSX_PSG_CLOCK, sample_rate)
            .expect("failed to create PSG with MSX clock");
        psg_inner.set_chip_type(psg::ChipType::AY);
        let psg = Arc::new(Mutex::new(psg_inner));
        let scc = Arc::new(Mutex::new(Scc::new(sample_rate as f32)));

        // 512 samples ≈ 10.7 ms latency at 48 kHz. Each callback snapshots
        // the PSG/SCC state once and renders the whole buffer from that
        // snapshot, so mid-buffer register changes from the emulator are lost
        // — shorter buffers preserve more of those, but underrun if the main
        // thread can't keep up. 256 was on the edge in the browser; 512 trades
        // a bit of fidelity for headroom.
        let processor = context
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(512, 0, 1)
            .expect("failed to create ScriptProcessorNode");

        let psg_cb = Arc::clone(&psg);
        let scc_cb = Arc::clone(&scc);
        let callback = Closure::wrap(Box::new(move |event: web_sys::AudioProcessingEvent| {
            let output = event.output_buffer().expect("no output buffer");
            let n = output.length() as usize;

            let mut psg = psg_cb.lock().unwrap();
            let mut scc = scc_cb.lock().unwrap();

            let mut buf = vec![0.0f32; n];
            for sample in buf.iter_mut() {
                let (psg_l, psg_r) = psg.render();
                let psg_mono = ((psg_l + psg_r) * 0.5) as f32;
                *sample = psg_mono * 0.6 + scc.next_sample() * 0.6;
            }

            let _ = output.copy_to_channel(&buf, 0);
        }) as Box<dyn FnMut(web_sys::AudioProcessingEvent)>);

        processor.set_onaudioprocess(Some(callback.as_ref().unchecked_ref()));
        processor
            .connect_with_audio_node(&context.destination())
            .expect("failed to connect ScriptProcessorNode to destination");

        Self {
            psg,
            scc,
            web: WebAudio {
                context,
                _processor: processor,
                _callback: callback,
            },
        }
    }

    /// Resume the audio context if it's been suspended by the browser's
    /// autoplay policy. Safe to call repeatedly — it's a no-op once the
    /// context is already running.
    pub fn resume(&self) {
        if self.web.context.state() != web_sys::AudioContextState::Running {
            let _ = self.web.context.resume();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Audio {
    /// No-op on native — cpal starts playing immediately, no gesture needed.
    pub fn resume(&self) {}
}
