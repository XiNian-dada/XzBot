//! 轻量日志模块：提供统一前缀与调试开关封装。

use std::{
    collections::VecDeque,
    io::{self, Write},
    sync::{Mutex, OnceLock},
};

use anyhow::Error;

const MAX_RECENT_LOG_LINES: usize = 5000;
static RECENT_LOGS: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();

fn push_recent(line: String) {
    let logs = RECENT_LOGS.get_or_init(|| Mutex::new(VecDeque::new()));
    if let Ok(mut guard) = logs.lock() {
        guard.push_back(line);
        while guard.len() > MAX_RECENT_LOG_LINES {
            let _ = guard.pop_front();
        }
    }
}

/// 把一行标准输出同时写到真实控制台和内存日志缓冲。
pub fn print_stdout_line(line: String) {
    let _ = writeln!(io::stdout(), "{line}");
    push_recent(line);
}

/// 把一行标准错误同时写到真实控制台和内存日志缓冲。
pub fn print_stderr_line(line: String) {
    let _ = writeln!(io::stderr(), "{line}");
    push_recent(line);
}

/// Logs an info-level message.
pub fn info(msg: impl AsRef<str>) {
    let line = format!("[INFO] {}", msg.as_ref());
    print_stdout_line(line);
}

/// Logs a warning-level message.
pub fn warn(msg: impl AsRef<str>) {
    let line = format!("[WARN] {}", msg.as_ref());
    print_stderr_line(line);
}

/// Logs an error-level message.
pub fn error(msg: impl AsRef<str>) {
    let line = format!("[ERROR] {}", msg.as_ref());
    print_stderr_line(line);
}

/// Logs an error-level message and appends the complete anyhow error chain.
pub fn error_err(msg: impl AsRef<str>, err: &Error) {
    error(format!("{}: {}", msg.as_ref(), format_error_chain(err)));
}

/// Logs a warning-level message and appends the complete anyhow error chain.
pub fn warn_err(msg: impl AsRef<str>, err: &Error) {
    warn(format!("{}: {}", msg.as_ref(), format_error_chain(err)));
}

/// Logs a debug-level message when `enabled` is true.
pub fn debug(enabled: bool, msg: impl AsRef<str>) {
    if enabled {
        let line = format!("[DEBUG] {}", msg.as_ref());
        print_stdout_line(line);
    }
}

/// Returns at most the most recent `limit` lines kept in in-memory log buffer.
pub fn recent_lines(limit: usize) -> Vec<String> {
    let logs = RECENT_LOGS.get_or_init(|| Mutex::new(VecDeque::new()));
    let Ok(guard) = logs.lock() else {
        return Vec::new();
    };

    let take = limit.min(guard.len());
    guard
        .iter()
        .skip(guard.len().saturating_sub(take))
        .cloned()
        .collect()
}

/// Formats one `anyhow::Error` into a readable single-line chain for console/log buffer output.
pub fn format_error_chain(err: &Error) -> String {
    let mut parts = Vec::new();
    for cause in err.chain() {
        let text = cause.to_string();
        if text.trim().is_empty() {
            continue;
        }
        if parts.last() == Some(&text) {
            continue;
        }
        parts.push(text);
    }

    if parts.is_empty() {
        "unknown error".to_string()
    } else {
        parts.join(" | caused by: ")
    }
}
