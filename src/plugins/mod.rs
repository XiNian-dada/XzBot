use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tokio::{io::AsyncWriteExt, process::Command, time::timeout};

use crate::{
    config::Config,
    logger::debug as log_debug,
    onebot::{action::ActionRequest, event::MessageEvent},
};

const DEFAULT_PLUGIN_TIMEOUT_MS: u64 = 15_000;

#[derive(Debug, Clone)]
pub struct PluginManager {
    plugins: Vec<ExternalPlugin>,
    command_map: HashMap<String, usize>,
    config: std::sync::Arc<Config>,
}

impl PluginManager {
    pub fn load_from_dir(root: &Path, config: std::sync::Arc<Config>) -> Result<Self> {
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
            if !path.is_dir() {
                continue;
            }
            let manifest_path = path.join("plugin.toml");
            if !manifest_path.exists() {
                continue;
            }
            let manifest_str = fs::read_to_string(&manifest_path)
                .with_context(|| format!("failed to read {}", manifest_path.display()))?;
            let manifest: PluginManifest = toml::from_str(&manifest_str)
                .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
            let plugin = ExternalPlugin::from_manifest(path.clone(), manifest)?;
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

    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    pub async fn try_handle(&self, event: &MessageEvent) -> Result<Option<ActionRequest>> {
        let raw_text = event.text();
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        let mentioned = raw_text.contains(&at_me);
        let normalized = raw_text.replace(&at_me, "").trim().to_string();
        if normalized.is_empty() {
            return Ok(None);
        }

        let (cmd, args) = match parse_command(&normalized) {
            Some(v) => v,
            None => return Ok(None),
        };

        if event.message_type == "group" && self.config.group.require_at && !mentioned {
            return Ok(None);
        }

        let Some(&idx) = self.command_map.get(&cmd) else {
            return Ok(None);
        };
        let plugin = &self.plugins[idx];

        let req = PluginRequest {
            command: cmd,
            args,
            raw_text: normalized,
            message_type: event.message_type.clone(),
            user_id: event.user_id,
            group_id: event.group_id,
            self_id: event.self_id,
            config_dir: plugin.config_dir.to_string_lossy().to_string(),
        };

        let reply = plugin.run(req, self.config.debug).await?;
        if reply.reply.trim().is_empty() {
            return Ok(None);
        }

        let mention_sender = reply
            .mention_sender
            .unwrap_or(self.config.group.mention_sender);

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

        Ok(Some(action))
    }
}

#[derive(Debug, Deserialize)]
struct PluginManifest {
    name: String,
    entry: String,
    #[serde(default)]
    commands: Vec<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug, Clone)]
struct ExternalPlugin {
    name: String,
    entry: PathBuf,
    commands: Vec<String>,
    timeout_ms: u64,
    dir: PathBuf,
    config_dir: PathBuf,
}

impl ExternalPlugin {
    fn from_manifest(dir: PathBuf, manifest: PluginManifest) -> Result<Self> {
        let entry = dir.join(manifest.entry);
        if !entry.exists() {
            return Err(anyhow!("plugin entry not found: {}", entry.display()));
        }
        let commands = manifest
            .commands
            .into_iter()
            .map(|c| c.trim().trim_start_matches('/').to_string())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>();
        let config_dir = dir.join("config");
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir).with_context(|| {
                format!("failed to create plugin config dir {}", config_dir.display())
            })?;
        }
        Ok(Self {
            name: manifest.name,
            entry,
            commands,
            timeout_ms: manifest.timeout_ms.unwrap_or(DEFAULT_PLUGIN_TIMEOUT_MS),
            dir,
            config_dir,
        })
    }

    async fn run(&self, request: PluginRequest, debug: bool) -> Result<PluginResponse> {
        let payload =
            serde_json::to_vec(&request).context("failed to serialize plugin request")?;
        let mut child = Command::new(&self.entry)
            .current_dir(&self.dir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to spawn plugin {}", self.name))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&payload)
                .await
                .context("failed to write plugin stdin")?;
        }

        let output = timeout(Duration::from_millis(self.timeout_ms), child.wait_with_output())
            .await
            .map_err(|_| anyhow::anyhow!("plugin timeout after {} ms", self.timeout_ms))?
            .context("failed to read plugin output")?;

        if debug {
            log_debug(
                debug,
                format!(
                    "plugin {} exit={} stderr={}",
                    self.name,
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                ),
            );
        }

        if !output.status.success() {
            return Err(anyhow!("plugin {} exited with status {}", self.name, output.status));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            return Ok(PluginResponse {
                reply: String::new(),
                mention_sender: None,
            });
        }

        if let Ok(parsed) = serde_json::from_str::<PluginResponse>(&stdout) {
            return Ok(parsed);
        }

        Ok(PluginResponse {
            reply: stdout,
            mention_sender: None,
        })
    }
}

#[derive(Debug, serde::Serialize)]
struct PluginRequest {
    command: String,
    args: String,
    raw_text: String,
    message_type: String,
    user_id: i64,
    group_id: Option<i64>,
    self_id: i64,
    config_dir: String,
}

#[derive(Debug, serde::Deserialize)]
struct PluginResponse {
    reply: String,
    #[serde(default)]
    mention_sender: Option<bool>,
}

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
