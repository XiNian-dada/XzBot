use anyhow::Result;
use sysinfo::System;

pub fn get_system_info(scope: &str) -> Result<String> {
    let mut system = System::new_all();
    system.refresh_all();

    let scope = scope.trim().to_lowercase();
    let out = match scope.as_str() {
        "" | "summary" => summary(&system),
        "cpu" => cpu_details(&system),
        "memory" | "mem" | "ram" => memory_details(&system),
        "load" => load_details(),
        "uptime" => uptime_details(),
        "all" => format!(
            "{}\n\n{}\n\n{}\n\n{}",
            summary(&system),
            cpu_details(&system),
            memory_details(&system),
            load_details()
        ),
        other => format!(
            "unknown scope: {other}\nallowed scopes: summary, cpu, memory, load, uptime, all"
        ),
    };

    Ok(out)
}

fn summary(system: &System) -> String {
    let host = System::host_name().unwrap_or_else(|| "unknown".to_string());
    let os = System::name().unwrap_or_else(|| "unknown".to_string());
    let kernel = System::kernel_version().unwrap_or_else(|| "unknown".to_string());
    let uptime = format_uptime(System::uptime());
    let cpu_usage = format!("{:.1}%", system.global_cpu_usage());
    let total_memory_mb = kib_to_mb(system.total_memory());
    let used_memory_mb = kib_to_mb(system.used_memory());

    format!(
        "System Summary\nhost: {host}\nos: {os}\nkernel: {kernel}\nuptime: {uptime}\ncpu_usage: {cpu_usage}\nmemory: {used_memory_mb} MB / {total_memory_mb} MB"
    )
}

fn cpu_details(system: &System) -> String {
    let mut out = String::new();
    out.push_str("CPU Details\n");
    out.push_str(&format!("logical_cores: {}\n", system.cpus().len()));
    out.push_str(&format!(
        "global_usage: {:.1}%\n",
        system.global_cpu_usage()
    ));
    for (idx, cpu) in system.cpus().iter().enumerate().take(32) {
        out.push_str(&format!(
            "core_{idx}: {} {:.1}%\n",
            cpu.brand(),
            cpu.cpu_usage()
        ));
    }
    out.trim().to_string()
}

fn memory_details(system: &System) -> String {
    let total_memory_mb = kib_to_mb(system.total_memory());
    let used_memory_mb = kib_to_mb(system.used_memory());
    let free_memory_mb = total_memory_mb.saturating_sub(used_memory_mb);
    let total_swap_mb = kib_to_mb(system.total_swap());
    let used_swap_mb = kib_to_mb(system.used_swap());

    format!(
        "Memory Details\nram_used: {used_memory_mb} MB\nram_free: {free_memory_mb} MB\nram_total: {total_memory_mb} MB\nswap_used: {used_swap_mb} MB\nswap_total: {total_swap_mb} MB"
    )
}

fn load_details() -> String {
    let load = System::load_average();
    format!(
        "Load Average\n1m: {:.2}\n5m: {:.2}\n15m: {:.2}",
        load.one, load.five, load.fifteen
    )
}

fn uptime_details() -> String {
    format!("Uptime\n{}", format_uptime(System::uptime()))
}

fn kib_to_mb(kib: u64) -> u64 {
    kib / 1024
}

fn format_uptime(total_seconds: u64) -> String {
    let days = total_seconds / 86_400;
    let hours = (total_seconds % 86_400) / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    format!("{days}d {hours}h {minutes}m {seconds}s")
}
