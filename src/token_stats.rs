use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenSnapshot {
    pub prompt: u64,
    pub completion: u64,
    pub total: u64,
}

static PROMPT_TOKENS: AtomicU64 = AtomicU64::new(0);
static COMPLETION_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_TOKENS: AtomicU64 = AtomicU64::new(0);

pub fn snapshot() -> TokenSnapshot {
    TokenSnapshot {
        prompt: PROMPT_TOKENS.load(Ordering::Relaxed),
        completion: COMPLETION_TOKENS.load(Ordering::Relaxed),
        total: TOTAL_TOKENS.load(Ordering::Relaxed),
    }
}

pub fn record(prompt: u64, completion: u64, total: Option<u64>) {
    PROMPT_TOKENS.fetch_add(prompt, Ordering::Relaxed);
    COMPLETION_TOKENS.fetch_add(completion, Ordering::Relaxed);
    let to_add = total.unwrap_or(prompt.saturating_add(completion));
    TOTAL_TOKENS.fetch_add(to_add, Ordering::Relaxed);
}

pub fn diff(before: TokenSnapshot, after: TokenSnapshot) -> TokenSnapshot {
    TokenSnapshot {
        prompt: after.prompt.saturating_sub(before.prompt),
        completion: after.completion.saturating_sub(before.completion),
        total: after.total.saturating_sub(before.total),
    }
}
