//! 插件运行时：负责外部插件发现、拉起、通信和生命周期管理。
//!
//! XzBot 的插件设计目标更接近“托管外部进程”，而不是把所有逻辑静态编译进主程序。
//! 因此这里重点处理：
//! - 插件目录扫描与配置解析
//! - 子进程拉起、标准输入输出通信
//! - 插件失败后的清理与重载

use std::{
    collections::HashMap,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::timeout;

use crate::{
    config::Config,
    logger::{debug as log_debug, info as log_info, warn as log_warn},
    onebot::{action::ActionRequest, event::MessageEvent},
};

const DEFAULT_PLUGIN_TIMEOUT_MS: u64 = 15_000;
const MANIFEST_ARG: &str = "--manifest";

/// Plugin runtime manager that discovers binaries and dispatches slash commands.
#[derive(Debug, Clone)]
pub struct PluginManager {
    plugins: Vec<ManagedPlugin>,
    command_map: HashMap<String, usize>,
    config: Arc<Config>,
}

impl PluginManager {
    /// Loads plugin binaries from `root` and builds command index.
    pub fn load_from_dir(root: &Path, config: Arc<Config>) -> Result<Self> {
        if !root.exists() {
            fs::create_dir_all(root)
                .with_context(|| format!("failed to create plugin dir {}", root.display()))?;
        }

        let mut plugins = Vec::new();
        let mut command_map = HashMap::new();

        for entry in fs::read_dir(root)
            .with_context(|| format!("failed to read plugin dir {}", root.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path.extension() == Some(OsStr::new("toml")) {
                continue;
            }

            let file_stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("plugin")
                .to_string();
            let manifest = read_plugin_manifest(&path).unwrap_or(PluginManifestInfo {
                name: file_stem.clone(),
                commands: vec![file_stem.clone()],
                timeout_ms: None,
            });
            let plugin = ManagedPlugin::new(path.clone(), manifest, root)?;

            let idx = plugins.len();
            for command in &plugin.commands {
                command_map.insert(command.to_lowercase(), idx);
            }
            plugins.push(plugin);
        }

        Ok(Self {
            plugins,
            command_map,
            config,
        })
    }

    /// Returns currently loaded plugin count.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Returns plugin names for diagnostics/logging.
    pub fn plugin_names(&self) -> Vec<String> {
        self.plugins.iter().map(|p| p.name.clone()).collect()
    }

    /// Stops all plugin processes.
    pub async fn shutdown(&self) {
        for plugin in &self.plugins {
            plugin.shutdown(self.config.debug).await;
        }
    }

    /// Tries to route an event to a plugin command and converts plugin output to actions.
    pub async fn try_handle(&self, event: &MessageEvent) -> Result<Vec<ActionRequest>> {
        let raw_text = event.text();
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        let mentioned = raw_text.contains(&at_me);
        let normalized = raw_text.replace(&at_me, "").trim().to_string();
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let (cmd, args) = match parse_command(&normalized) {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };

        if event.message_type == "group" && self.config.group.require_at && !mentioned {
            return Ok(Vec::new());
        }

        let Some(&idx) = self.command_map.get(&cmd) else {
            return Ok(Vec::new());
        };
        let plugin = &self.plugins[idx];

        let req = PluginRequest {
            request_id: plugin.next_request_id(),
            command: cmd,
            args,
            raw_text: normalized,
            message_type: event.message_type.clone(),
            user_id: event.user_id,
            group_id: event.group_id,
            self_id: event.self_id,
            config_dir: plugin.config_dir.to_string_lossy().to_string(),
        };
        let reply = plugin.call(req, self.config.debug).await?;
        let mut actions = Vec::new();
        let mention_sender = reply
            .mention_sender
            .unwrap_or(self.config.group.mention_sender);
        if let Some(file_path) = reply.file_path.as_ref() {
            let resolved = resolve_plugin_path(file_path, &plugin.config_dir);
            if !resolved.exists() {
                log_warn(format!(
                    "plugin {} file not found: {}",
                    plugin.name,
                    resolved.display()
                ));
            } else {
                let file_name = reply.file_name.clone().or_else(|| {
                    resolved
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                });
                let action = match event.message_type.as_str() {
                    "group" => {
                        let group_id = event.group_id.unwrap_or_default();
                        ActionRequest::upload_group_file(
                            group_id,
                            resolved.to_string_lossy().to_string(),
                            file_name,
                        )
                    }
                    _ => ActionRequest::upload_private_file(
                        event.user_id,
                        resolved.to_string_lossy().to_string(),
                        file_name,
                    ),
                };
                actions.push(action);
            }
        }

        if let Some(image_ref) = reply.image_url.clone().or_else(|| reply.image_path.clone()) {
            let resolved = if reply.image_url.is_some() {
                image_ref
            } else {
                resolve_plugin_path(&image_ref, &plugin.config_dir)
                    .to_string_lossy()
                    .to_string()
            };
            if reply.image_url.is_none() && !Path::new(&resolved).exists() {
                log_warn(format!(
                    "plugin {} image not found: {}",
                    plugin.name, resolved
                ));
            } else {
                let image_cq = format!("[CQ:image,file={}]", resolved);
                let action = match event.message_type.as_str() {
                    "group" => {
                        let group_id = event.group_id.unwrap_or_default();
                        let message = if mention_sender {
                            format!("[CQ:at,qq={}] {}", event.user_id, image_cq)
                        } else {
                            image_cq
                        };
                        ActionRequest::send_group_msg(group_id, message)
                    }
                    _ => ActionRequest::send_private_msg(event.user_id, image_cq),
                };
                actions.push(action);
            }
        }

        if reply.reply.trim().is_empty() {
            return Ok(actions);
        }

        let action = match event.message_type.as_str() {
            "group" => {
                let group_id = event.group_id.unwrap_or_default();
                let message = if mention_sender {
                    format!("[CQ:at,qq={}] {}", event.user_id, reply.reply)
                } else {
                    reply.reply
                };
                ActionRequest::send_group_msg(group_id, message)
            }
            _ => ActionRequest::send_private_msg(event.user_id, reply.reply),
        };

        actions.push(action);
        Ok(actions)
    }
}

/// Metadata returned by plugin executable via `--manifest`.
#[derive(Debug, Deserialize)]
struct PluginManifestInfo {
    name: String,
    #[serde(default)]
    commands: Vec<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

/// Runtime handle for one managed plugin process.
#[derive(Debug, Clone)]
struct ManagedPlugin {
    name: String,
    path: PathBuf,
    commands: Vec<String>,
    timeout_ms: u64,
    config_dir: PathBuf,
    process: Arc<Mutex<Option<PluginProcess>>>,
    seq: Arc<AtomicU64>,
}

impl ManagedPlugin {
    /// Constructs managed plugin runtime state from manifest.
    fn new(path: PathBuf, manifest: PluginManifestInfo, root: &Path) -> Result<Self> {
        let name = manifest.name.trim().to_string();
        let commands = manifest
            .commands
            .into_iter()
            .map(|c| c.trim().trim_start_matches('/').to_string())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>();
        let commands = if commands.is_empty() {
            vec![name.clone()]
        } else {
            commands
        };
        let timeout_ms = manifest.timeout_ms.unwrap_or(DEFAULT_PLUGIN_TIMEOUT_MS);
        let config_dir = root.join(&name);
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)
                .with_context(|| format!("failed to create plugin dir {}", config_dir.display()))?;
        }

        Ok(Self {
            name,
            path,
            commands,
            timeout_ms,
            config_dir,
            process: Arc::new(Mutex::new(None)),
            seq: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Allocates monotonically increasing request id for IPC correlation.
    fn next_request_id(&self) -> String {
        format!("{}-{}", self.name, self.seq.fetch_add(1, Ordering::Relaxed))
    }

    /// Sends one IPC request to plugin process and waits for response.
    async fn call(&self, request: PluginRequest, debug: bool) -> Result<PluginResponse> {
        let mut guard = self.process.lock().await;
        let process = ensure_process(&self.path, guard.take(), debug).await?;
        let mut process = process;

        let payload =
            serde_json::to_string(&request).context("failed to serialize plugin request")?;
        let payload = format!("{payload}\n");
        process
            .stdin
            .write_all(payload.as_bytes())
            .await
            .context("failed to write plugin stdin")?;
        process
            .stdin
            .flush()
            .await
            .context("failed to flush plugin stdin")?;

        let response =
            read_response(&mut process.stdout, &request.request_id, self.timeout_ms).await;

        *guard = Some(process);
        response
    }

    /// Terminates plugin process if running.
    async fn shutdown(&self, debug: bool) {
        let mut guard = self.process.lock().await;
        let Some(mut proc) = guard.take() else {
            return;
        };
        if debug {
            log_debug(debug, format!("stopping plugin {}", self.name));
        }
        if let Err(err) = proc.child.kill().await {
            log_warn(format!("failed to kill plugin {}: {}", self.name, err));
            return;
        }
        let _ = proc.child.wait().await;
    }
}

/// Live child process handles kept across calls.
#[derive(Debug)]
struct PluginProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// Ensures plugin process is running, restarting when exited.
async fn ensure_process(
    path: &Path,
    mut existing: Option<PluginProcess>,
    debug: bool,
) -> Result<PluginProcess> {
    if let Some(mut proc) = existing.take() {
        if let Ok(Some(status)) = proc.child.try_wait() {
            if debug {
                log_debug(
                    debug,
                    format!("plugin exited status={}, restarting", status),
                );
            }
        } else {
            return Ok(proc);
        }
    }

    let mut child = Command::new(path)
        .current_dir(path.parent().unwrap_or_else(|| Path::new(".")))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn plugin {}", path.display()))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("failed to capture plugin stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to capture plugin stdout"))?;
    if let Some(stderr) = child.stderr.take() {
        spawn_plugin_log_task(
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("plugin")
                .to_string(),
            stderr,
        );
    }

    Ok(PluginProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

/// Pipes plugin stderr into bot logs for debugging.
fn spawn_plugin_log_task(name: String, stderr: ChildStderr) {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    let msg = line.trim_end();
                    if msg.is_empty() {
                        continue;
                    }
                    log_info(format!("[PLUGIN:{name}] {msg}"));
                }
                Err(err) => {
                    log_warn(format!("[PLUGIN:{name}] stderr read error: {err}"));
                    break;
                }
            }
        }
    });
}

/// Reads one response line from plugin stdout with timeout.
async fn read_response(
    stdout: &mut BufReader<ChildStdout>,
    request_id: &str,
    timeout_ms: u64,
) -> Result<PluginResponse> {
    let mut line = String::new();
    let fut = async {
        loop {
            line.clear();
            let read = stdout.read_line(&mut line).await?;
            if read == 0 {
                return Err(anyhow!("plugin closed stdout"));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(resp) = serde_json::from_str::<PluginResponse>(trimmed) {
                if resp
                    .request_id
                    .as_deref()
                    .map(|v| v == request_id)
                    .unwrap_or(true)
                {
                    return Ok(resp);
                }
                continue;
            }
            return Ok(PluginResponse {
                request_id: None,
                reply: trimmed.to_string(),
                mention_sender: None,
                file_path: None,
                file_name: None,
                image_path: None,
                image_url: None,
            });
        }
    };

    timeout(Duration::from_millis(timeout_ms), fut)
        .await
        .map_err(|_| anyhow!("plugin timeout after {} ms", timeout_ms))?
}

/// IPC request payload sent from host to plugin.
#[derive(Debug, Serialize)]
struct PluginRequest {
    request_id: String,
    command: String,
    args: String,
    raw_text: String,
    message_type: String,
    user_id: i64,
    group_id: Option<i64>,
    self_id: i64,
    config_dir: String,
}

/// IPC response payload returned by plugin.
#[derive(Debug, Deserialize)]
struct PluginResponse {
    #[serde(default)]
    request_id: Option<String>,
    reply: String,
    #[serde(default)]
    mention_sender: Option<bool>,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    image_path: Option<String>,
    #[serde(default)]
    image_url: Option<String>,
}

/// Runs plugin with `--manifest` and parses manifest JSON.
fn read_plugin_manifest(path: &Path) -> Option<PluginManifestInfo> {
    let output = std::process::Command::new(path)
        .arg(MANIFEST_ARG)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim()).ok()
}

/// Parses slash command and argument string from normalized text.
fn parse_command(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim();
    if !trimmed.starts_with('/') {
        return None;
    }
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let head = parts.next()?.trim_start_matches('/');
    if head.is_empty() {
        return None;
    }
    let args = parts.next().unwrap_or("").trim().to_string();
    Some((head.to_lowercase(), args))
}

/// Resolves plugin-relative output paths against plugin config directory.
fn resolve_plugin_path(path: &str, config_dir: &Path) -> PathBuf {
    let p = Path::new(path.trim());
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        config_dir.join(p)
    }
}
