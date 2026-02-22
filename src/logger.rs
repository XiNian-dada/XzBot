pub fn info(msg: impl AsRef<str>) {
    println!("[INFO] {}", msg.as_ref());
}

pub fn warn(msg: impl AsRef<str>) {
    eprintln!("[WARN] {}", msg.as_ref());
}

pub fn error(msg: impl AsRef<str>) {
    eprintln!("[ERROR] {}", msg.as_ref());
}

pub fn debug(enabled: bool, msg: impl AsRef<str>) {
    if enabled {
        println!("[DEBUG] {}", msg.as_ref());
    }
}
