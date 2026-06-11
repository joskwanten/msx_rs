//! Tiny category-gated logging — same API on native and WASM, near-zero
//! cost when a category is off, no third-party deps.
//!
//! # Usage at call sites
//!
//! ```ignore
//! mlog!(VDP_CMD, "LMMM start: src=({},{}) dst=({},{})", sx, sy, dx, dy);
//! mlog!(SCC,     "write reg 0x{:02X} = 0x{:02X}", reg, val);
//! ```
//!
//! Output (only when the category is enabled):
//!
//! ```text
//! [vdp_cmd] LMMM start: src=(12,5) dst=(200,100)
//! [scc]     write reg 0x82 = 0x40
//! ```
//!
//! # Turning categories on
//!
//! Native (env var):
//!
//! ```text
//! MSX_LOG=vdp_cmd,vdp_pal cargo run --release
//! MSX_LOG=all            cargo run --release           # everything
//! ```
//!
//! Web (URL query parameter):
//!
//! ```text
//! https://.../msx_rs/?log=vdp_cmd,vdp_pal
//! https://.../msx_rs/?log=all
//! ```
//!
//! Unknown names emit a one-time warning and are skipped — beats failing
//! silently when someone fat-fingers `vdp_cdm`.
//!
//! # Cost
//!
//! When the category is off, a `mlog!` call is one relaxed atomic load,
//! one bitwise AND, and one predicted-not-taken branch. The format
//! arguments are never even constructed, so logging in hot paths is safe.

use std::sync::atomic::{AtomicU32, Ordering};

/// Category bitmasks. One bit per subsystem; grow as new code starts
/// needing introspection. `ALL` is the catch-all.
#[allow(non_camel_case_types)]
pub mod cat {
    pub const VDP_CMD:    u32 = 1 << 0;
    pub const VDP_PAL:    u32 = 1 << 1;
    pub const VDP_SPRITE: u32 = 1 << 2;
    pub const VDP_REG:    u32 = 1 << 3;
    pub const PSG:        u32 = 1 << 4;
    pub const SCC:        u32 = 1 << 5;
    pub const SLOT:       u32 = 1 << 6;
    pub const BUS:        u32 = 1 << 7;
    pub const FM:         u32 = 1 << 8;
    pub const ALL:        u32 = u32::MAX;
}

static ENABLED: AtomicU32 = AtomicU32::new(0);

/// True when at least one bit of `cat` is currently enabled. The macro
/// uses this as its gate — kept `#[doc(hidden)]` because it's an
/// implementation detail. `#[allow(dead_code)]` because it stays unused
/// until the first `mlog!` call site lands; the macro expansion is the
/// only legitimate caller.
#[doc(hidden)]
#[allow(dead_code)]
pub fn is_enabled(cat: u32) -> bool {
    ENABLED.load(Ordering::Relaxed) & cat != 0
}

/// Replace the active mask. Normal usage is via `init_from_environment`;
/// this is here for tests and ad-hoc toggling.
pub fn set_enabled(mask: u32) {
    ENABLED.store(mask, Ordering::Relaxed);
}

/// Parse a comma-separated list of category names into a bitmask.
/// Whitespace around names is trimmed, empty entries are ignored,
/// unknown names emit a warning and contribute 0.
pub fn parse_mask(s: &str) -> u32 {
    let mut mask = 0u32;
    for raw in s.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let bit = match name {
            "vdp_cmd"    => cat::VDP_CMD,
            "vdp_pal"    => cat::VDP_PAL,
            "vdp_sprite" => cat::VDP_SPRITE,
            "vdp_reg"    => cat::VDP_REG,
            "psg"        => cat::PSG,
            "scc"        => cat::SCC,
            "slot"       => cat::SLOT,
            "bus"        => cat::BUS,
            "fm"         => cat::FM,
            "all"        => cat::ALL,
            other => {
                warn_unknown(other);
                0
            }
        };
        mask |= bit;
    }
    mask
}

/// Read the activation list from the host: env var `MSX_LOG` on native,
/// URL query parameter `?log=` on web. Call once at startup.
pub fn init_from_environment() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(s) = std::env::var("MSX_LOG") {
            set_enabled(parse_mask(&s));
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(win) = web_sys::window() {
            if let Ok(href) = win.location().href() {
                if let Ok(url) = web_sys::Url::new(&href) {
                    if let Some(s) = url.search_params().get("log") {
                        set_enabled(parse_mask(&s));
                    }
                }
            }
        }
    }
}

/// Emit one log line. Only ever called from `mlog!` after the gate has
/// already cleared, so no extra check here.
#[doc(hidden)]
pub fn _emit(tag: &str, args: std::fmt::Arguments) {
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("[{}] {}", tag, args);
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&format!("[{}] {}", tag, args).into());
}

#[cfg(not(target_arch = "wasm32"))]
fn warn_unknown(name: &str) {
    eprintln!("warning: unknown log category '{}'", name);
}
#[cfg(target_arch = "wasm32")]
fn warn_unknown(name: &str) {
    web_sys::console::warn_1(&format!("unknown log category '{}'", name).into());
}

/// Gated log macro. Resolves to a single relaxed atomic load + AND +
/// branch when the category is disabled (no format-args construction).
/// When enabled: format + print to stderr (native) or `console.log`
/// (web), tagged with the lowercased category name.
#[macro_export]
macro_rules! mlog {
    ($cat:ident, $($arg:tt)*) => {
        if $crate::log::is_enabled($crate::log::cat::$cat) {
            // Lowercase the constant name so the printed tag reads
            // `[vdp_cmd]` not `[VDP_CMD]`. Only allocates when the gate
            // is open; on the disabled path this code never runs.
            let tag = stringify!($cat).to_lowercase();
            $crate::log::_emit(&tag, format_args!($($arg)*));
        }
    };
}
