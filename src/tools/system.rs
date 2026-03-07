//! 系统信息工具：提供只读的主机与进程运行状态查询。

use anyhow::Result;
use std::fs;

use sysinfo::{Disks, Networks, ProcessesToUpdate, System};

/// Returns runtime metrics for current XzBot process.
pub fn get_process_info() -> Result<String> {
    let pid = sysinfo::get_current_pid().map_err(|err| anyhow::anyhow!(err))?;
    let mut system = System::new_all();
    system.refresh_processes(ProcessesToUpdate::All, true);

    let Some(process) = system.process(pid) else {
        return Ok("Process Info\nprocess not found".to_string());
    };

    let mut mem_rss_bytes = None;
    let mut mem_vms_bytes = None;
    if let Some(proc_mem) = read_proc_self_mem_bytes() {
        mem_rss_bytes = proc_mem.rss;
        mem_vms_bytes = proc_mem.vms;
    }

    let mem_rss_bytes = mem_rss_bytes.unwrap_or_else(|| process.memory().saturating_mul(1024));
    let mem_vms_bytes =
        mem_vms_bytes.unwrap_or_else(|| process.virtual_memory().saturating_mul(1024));

    let disk = process.disk_usage();
    let run_time = process.run_time();
    let status = format!("{:?}", process.status());
    let cpu = process.cpu_usage();

    let mut out = format!(
        "Process Info\npid: {}\nname: {}\nstatus: {}\nrun_time: {}\ncpu_usage: {:.1}%\nmem_rss: {}\nmem_vms: {}\ndisk_read: {}\ndisk_write: {}",
        pid.as_u32(),
        process.name().to_string_lossy(),
        status,
        format_uptime(run_time),
        cpu,
        format_bytes(mem_rss_bytes),
        format_bytes(mem_vms_bytes),
        format_bytes(disk.total_read_bytes),
        format_bytes(disk.total_written_bytes),
    );

    if let Some(cgroup) = read_cgroup_memory() {
        out.push_str(&format!(
            "\nmem_cgroup_current: {}",
            format_bytes(cgroup.current)
        ));
        if let Some(max) = cgroup.max {
            out.push_str(&format!("\nmem_cgroup_max: {}", format_bytes(max)));
        } else {
            out.push_str("\nmem_cgroup_max: unlimited");
        }
    }

    Ok(out)
}

#[derive(Debug)]
struct ProcMemBytes {
    rss: Option<u64>,
    vms: Option<u64>,
}

/// Reads `/proc/self/status` memory fields when running on Linux.
fn read_proc_self_mem_bytes() -> Option<ProcMemBytes> {
    let content = fs::read_to_string("/proc/self/status").ok()?;
    let mut rss_kb = None;
    let mut vms_kb = None;
    for line in content.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            rss_kb = parse_kb_value(value);
        } else if let Some(value) = line.strip_prefix("VmSize:") {
            vms_kb = parse_kb_value(value);
        }
    }
    if rss_kb.is_none() && vms_kb.is_none() {
        return None;
    }
    Some(ProcMemBytes {
        rss: rss_kb.map(|v| v.saturating_mul(1024)),
        vms: vms_kb.map(|v| v.saturating_mul(1024)),
    })
}

/// Parses a Linux `kB` memory value line.
fn parse_kb_value(input: &str) -> Option<u64> {
    let parts = input.split_whitespace().collect::<Vec<_>>();
    if parts.is_empty() {
        return None;
    }
    parts[0].parse::<u64>().ok()
}

#[derive(Debug)]
struct CgroupMemory {
    current: u64,
    max: Option<u64>,
}

/// Reads cgroup memory usage/limit from v2 or v1 paths.
fn read_cgroup_memory() -> Option<CgroupMemory> {
    if let (Ok(current), Ok(max)) = (
        fs::read_to_string("/sys/fs/cgroup/memory.current"),
        fs::read_to_string("/sys/fs/cgroup/memory.max"),
    ) {
        let current = current.trim().parse::<u64>().ok()?;
        let max = match max.trim() {
            "max" => None,
            v => v.parse::<u64>().ok(),
        };
        return Some(CgroupMemory { current, max });
    }

    let current = fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())?;
    let max = fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok());
    Some(CgroupMemory { current, max })
}

/// Returns system metrics for selected scope (`summary`, `cpu`, `memory`, ...).
pub fn get_system_info(scope: &str) -> Result<String> {
    let mut system = System::new_all();
    system.refresh_all();

    let scope = scope.trim().to_lowercase();
    let out = match scope.as_str() {
        "" | "summary" => summary(&system),
        "hardware" => hardware_details(&system),
        "cpu" => cpu_details(&system),
        "memory" | "mem" | "ram" => memory_details(&system),
        "disk" | "disks" | "storage" => disk_details(),
        "network" | "net" => network_details(),
        "load" => load_details(),
        "uptime" => uptime_details(),
        "all" | "full" | "detail" | "detailed" => format!(
            "{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}",
            summary(&system),
            hardware_details(&system),
            cpu_details(&system),
            memory_details(&system),
            disk_details(),
            network_details(),
            load_details()
        ),
        other => format!(
            "unknown scope: {other}\nallowed scopes: summary, hardware, cpu, memory, disk, network, load, uptime, all/full/detail"
        ),
    };

    Ok(out)
}

/// Builds compact system summary.
fn summary(system: &System) -> String {
    let host = System::host_name().unwrap_or_else(|| "unknown".to_string());
    let os = System::name().unwrap_or_else(|| "unknown".to_string());
    let os_long = System::long_os_version().unwrap_or_else(|| "unknown".to_string());
    let kernel = System::kernel_version().unwrap_or_else(|| "unknown".to_string());
    let arch = System::cpu_arch();
    let boot_time = System::boot_time();
    let uptime = format_uptime(System::uptime());
    let cpu_usage = format!("{:.1}%", system.global_cpu_usage());
    let total_memory = format_bytes(system.total_memory());
    let used_memory = format_bytes(system.used_memory());
    let memory_pct = pct(system.used_memory(), system.total_memory());
    let main_cpu = system.cpus().first();
    let cpu_model = main_cpu.map(|c| c.brand()).unwrap_or("unknown");

    format!(
        "System Summary\nhost: {host}\nos: {os}\nos_version: {os_long}\narch: {arch}\nkernel: {kernel}\nboot_time_epoch: {boot_time}\nuptime: {uptime}\ncpu_model: {cpu_model}\ncpu_usage: {cpu_usage}\nmemory: {used_memory} / {total_memory} ({memory_pct:.1}%)"
    )
}

/// Builds static hardware information section.
fn hardware_details(system: &System) -> String {
    let main_cpu = system.cpus().first();
    let cpu_model = main_cpu.map(|c| c.brand()).unwrap_or("unknown");
    let cpu_vendor = main_cpu.map(|c| c.vendor_id()).unwrap_or("unknown");
    let cpu_freq = main_cpu.map(|c| c.frequency()).unwrap_or(0);
    let logical = system.cpus().len();
    let physical = system.physical_core_count().unwrap_or(0);
    let arch = System::cpu_arch();

    format!(
        "Hardware Details\ncpu_model: {cpu_model}\ncpu_vendor: {cpu_vendor}\ncpu_frequency_mhz: {cpu_freq}\nlogical_cores: {logical}\nphysical_cores: {physical}\narch: {arch}"
    )
}

/// Builds per-core CPU usage section.
fn cpu_details(system: &System) -> String {
    let mut out = String::new();
    out.push_str("CPU Details\n");
    if let Some(main_cpu) = system.cpus().first() {
        out.push_str(&format!("model: {}\n", main_cpu.brand()));
        out.push_str(&format!("vendor: {}\n", main_cpu.vendor_id()));
        out.push_str(&format!("frequency_mhz: {}\n", main_cpu.frequency()));
    }
    out.push_str(&format!(
        "physical_cores: {}\n",
        system.physical_core_count().unwrap_or(0)
    ));
    out.push_str(&format!("logical_cores: {}\n", system.cpus().len()));
    out.push_str(&format!(
        "global_usage: {:.1}%\n",
        system.global_cpu_usage()
    ));
    for (idx, cpu) in system.cpus().iter().enumerate().take(32) {
        out.push_str(&format!(
            "core_{idx}: name={} model={} usage={:.1}% freq_mhz={}\n",
            cpu.name(),
            cpu.brand(),
            cpu.cpu_usage(),
            cpu.frequency()
        ));
    }
    out.trim().to_string()
}

/// Builds memory/swap usage section.
fn memory_details(system: &System) -> String {
    let total_memory = system.total_memory();
    let used_memory = system.used_memory();
    let free_memory = system.free_memory();
    let available_memory = system.available_memory();
    let total_swap = system.total_swap();
    let used_swap = system.used_swap();
    let free_swap = system.free_swap();

    format!(
        "Memory Details\nram_used: {} ({:.1}%)\nram_available: {}\nram_free: {}\nram_total: {}\nswap_used: {} ({:.1}%)\nswap_free: {}\nswap_total: {}",
        format_bytes(used_memory),
        pct(used_memory, total_memory),
        format_bytes(available_memory),
        format_bytes(free_memory),
        format_bytes(total_memory),
        format_bytes(used_swap),
        pct(used_swap, total_swap),
        format_bytes(free_swap),
        format_bytes(total_swap),
    )
}

/// Builds disk usage section.
fn disk_details() -> String {
    let disks = Disks::new_with_refreshed_list();
    if disks.is_empty() {
        return "Disk Details\nno disk info".to_string();
    }

    let mut out = String::new();
    out.push_str("Disk Details\n");
    for disk in &disks {
        let name = disk.name().to_string_lossy();
        let fs = disk.file_system().to_string_lossy();
        let mount = disk.mount_point().to_string_lossy();
        let total = disk.total_space();
        let available = disk.available_space();
        let used = total.saturating_sub(available);
        out.push_str(&format!(
            "- name={} mount={} fs={} kind={:?} used={} / {} ({:.1}%)\n",
            name,
            mount,
            fs,
            disk.kind(),
            format_bytes(used),
            format_bytes(total),
            pct(used, total),
        ));
    }
    out.trim().to_string()
}

/// Builds network interfaces and traffic section.
fn network_details() -> String {
    let networks = Networks::new_with_refreshed_list();
    if networks.is_empty() {
        return "Network Details\nno network interface info".to_string();
    }

    let mut out = String::new();
    out.push_str("Network Details\n");
    for (name, data) in &networks {
        let ip = data
            .ip_networks()
            .iter()
            .map(|ipn| ipn.addr.to_string())
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&format!(
            "- if={} mac={} ip=[{}] rx_total={} tx_total={} rx_packets={} tx_packets={} rx_err={} tx_err={}\n",
            name,
            data.mac_address(),
            ip,
            format_bytes(data.total_received()),
            format_bytes(data.total_transmitted()),
            data.total_packets_received(),
            data.total_packets_transmitted(),
            data.total_errors_on_received(),
            data.total_errors_on_transmitted()
        ));
    }
    out.trim().to_string()
}

/// Builds load average section.
fn load_details() -> String {
    let load = System::load_average();
    format!(
        "Load Average\n1m: {:.2}\n5m: {:.2}\n15m: {:.2}",
        load.one, load.five, load.fifteen
    )
}

/// Builds uptime section.
fn uptime_details() -> String {
    format!(
        "Uptime\nuptime: {}\nboot_time_epoch: {}",
        format_uptime(System::uptime()),
        System::boot_time()
    )
}

/// Formats seconds to `Xd Xh Xm Xs`.
fn format_uptime(total_seconds: u64) -> String {
    let days = total_seconds / 86_400;
    let hours = (total_seconds % 86_400) / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    format!("{days}d {hours}h {minutes}m {seconds}s")
}

/// Formats bytes to human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let b = bytes as f64;
    if b >= TB {
        format!("{:.2} TB", b / TB)
    } else if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.2} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

/// Calculates percentage with zero-safe denominator.
fn pct(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64) * 100.0 / (total as f64)
    }
}
