//! Minimal logging helpers with leveled prefixes.

/// Logs an info-level message.
pub fn info(msg: impl AsRef<str>) {
    println!("[INFO] {}", msg.as_ref());
}

/// Logs a warning-level message.
pub fn warn(msg: impl AsRef<str>) {
    eprintln!("[WARN] {}", msg.as_ref());
}

/// Logs an error-level message.
pub fn error(msg: impl AsRef<str>) {
    eprintln!("[ERROR] {}", msg.as_ref());
}

/// Logs a debug-level message when `enabled` is true.
pub fn debug(enabled: bool, msg: impl AsRef<str>) {
    if enabled {
        println!("[DEBUG] {}", msg.as_ref());
    }
}
