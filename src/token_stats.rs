//! Token 统计模块：累计记录请求和响应的 token 消耗。

use std::sync::atomic::{AtomicU64, Ordering};

/// Token counter snapshot used for per-call delta and cumulative reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenSnapshot {
    /// Input/prompt token count.
    pub prompt: u64,
    /// Output/completion token count.
    pub completion: u64,
    /// Total token count.
    pub total: u64,
}

static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static COMPLETION_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Returns current cumulative counters.
pub fn snapshot() -> TokenSnapshot {
    TokenSnapshot {
        prompt: PROMPT_TOKENS.load(Ordering::Relaxed),
        completion: COMPLETION_TOKENS.load(Ordering::Relaxed),
        total: TOTAL_TOKENS.load(Ordering::Relaxed),
    }
}

/// Adds one model call's usage to global counters.
pub fn record(prompt: u64, completion: u64, total: Option<u64>) {
    PROMPT_TOKENS.fetch_add(prompt, Ordering::Relaxed);
    COMPLETION_TOKENS.fetch_add(completion, Ordering::Relaxed);
    let to_add = total.unwrap_or(prompt.saturating_add(completion));
    TOTAL_TOKENS.fetch_add(to_add, Ordering::Relaxed);
}

/// Computes difference between two snapshots.
pub fn diff(before: TokenSnapshot, after: TokenSnapshot) -> TokenSnapshot {
    TokenSnapshot {
        prompt: after.prompt.saturating_sub(before.prompt),
        completion: after.completion.saturating_sub(before.completion),
        total: after.total.saturating_sub(before.total),
    }
}
