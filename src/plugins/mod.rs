//! 插件运行时：负责外部插件发现、拉起、通信和生命周期管理。
//!
//! 当前插件系统已经从“只能处理 `/命令` 的子进程”升级为更接近宿主平台的模型：
//! - 命令型插件：继续处理 `/analyze` 这类 slash command
//! - 事件型插件：订阅 `message` / `mention` / `image` / `quote` 等事件
//! - 工具型插件：向 LLM 暴露新的 function call
//!
//! 但宿主边界仍然保持克制：
//! - 插件不能直接调用任意 OneBot action
//! - 插件只能通过声明式 `PluginActionOutput` 让宿主代发消息/图片/文件
//! - 插件看见的是“只读事件快照”，不是主程序内部状态对象

use std::{
    collections::{HashMap, HashSet},
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
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::timeout;

use crate::{
    config::Config,
    llm::message_parts::parse_user_content,
    logger::{debug as log_debug, info as log_info, warn as log_warn},
    onebot::{action::ActionRequest, event::MessageEvent},
};

const DEFAULT_PLUGIN_TIMEOUT_MS: u64 = 15_000;
const DEFAULT_PLUGIN_PRIORITY: i32 = 0;
const MANIFEST_ARG: &str = "--manifest";

/// 宿主对外暴露的插件工具定义。
///
/// 插件在 manifest 里声明这些工具后，宿主会把它们并入 LLM 的工具列表。
#[derive(Debug, Clone, Deserialize)]
pub struct PluginToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(default = "default_tool_input_schema")]
    pub input_schema: Value,
}

/// 提供给控制面板/诊断接口的插件摘要。
#[derive(Debug, Clone, Serialize)]
pub struct PluginSummary {
    pub name: String,
    pub path: String,
    pub commands: Vec<String>,
    pub subscriptions: Vec<String>,
    pub tool_names: Vec<String>,
    pub timeout_ms: u64,
    pub priority: i32,
}

/// 事件分发结果：既包含插件生成的动作，也标记是否要阻断后续 AI/插件传播。
#[derive(Debug, Default)]
pub struct PluginDispatchOutcome {
    pub actions: Vec<ActionRequest>,
    pub stop_propagation: bool,
}

/// Plugin runtime manager that discovers binaries and dispatches commands/events/tools.
#[derive(Debug, Clone)]
pub struct PluginManager {
    plugins: Vec<ManagedPlugin>,
    command_map: HashMap<String, usize>,
    tool_map: HashMap<String, usize>,
    config: Arc<Config>,
}

impl PluginManager {
    /// Loads plugin binaries from `root`, sorts them by priority, and builds command/tool indexes.
    pub fn load_from_dir(root: &Path, config: Arc<Config>) -> Result<Self> {
        if !root.exists() {
            fs::create_dir_all(root)
                .with_context(|| format!("failed to create plugin dir {}", root.display()))?;
        }

        let mut plugins = Vec::new();

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
                subscriptions: Vec::new(),
                tools: Vec::new(),
                timeout_ms: None,
                priority: None,
            });
            plugins.push(ManagedPlugin::new(path.clone(), manifest, root)?);
        }

        // 优先级越高，越先收到事件/占用命令和工具名。
        plugins.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.name.cmp(&b.name))
        });

        let mut command_map: HashMap<String, usize> = HashMap::new();
        let mut tool_map: HashMap<String, usize> = HashMap::new();

        for (idx, plugin) in plugins.iter().enumerate() {
            for command in &plugin.commands {
                let key = command.to_lowercase();
                if let Some(existing_idx) = command_map.get(&key) {
                    log_warn(format!(
                        "plugin command collision '/{}': keep plugin={}, ignore plugin={}",
                        key, plugins[*existing_idx].name, plugin.name
                    ));
                    continue;
                }
                command_map.insert(key, idx);
            }

            for tool in &plugin.tools {
                let key = tool.name.to_lowercase();
                if let Some(existing_idx) = tool_map.get(&key) {
                    log_warn(format!(
                        "plugin tool collision '{}': keep plugin={}, ignore plugin={}",
                        key, plugins[*existing_idx].name, plugin.name
                    ));
                    continue;
                }
                tool_map.insert(key, idx);
            }
        }

        Ok(Self {
            plugins,
            command_map,
            tool_map,
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

    /// Returns all active plugin-defined tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.active_tool_definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect()
    }

    /// Returns detailed plugin summaries for admin UI / diagnostics.
    pub fn summaries(&self) -> Vec<PluginSummary> {
        self.plugins
            .iter()
            .map(|plugin| PluginSummary {
                name: plugin.name.clone(),
                path: plugin.path.to_string_lossy().to_string(),
                commands: plugin.commands.clone(),
                subscriptions: {
                    let mut items = plugin.subscriptions.iter().cloned().collect::<Vec<_>>();
                    items.sort();
                    items
                },
                tool_names: plugin.tools.iter().map(|tool| tool.name.clone()).collect(),
                timeout_ms: plugin.timeout_ms,
                priority: plugin.priority,
            })
            .collect()
    }

    /// Converts plugin tools into OpenAI Chat Completions schema items.
    pub fn openai_chat_tool_schemas(&self) -> Vec<Value> {
        self.active_tool_definitions()
            .into_iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                })
            })
            .collect()
    }

    /// Converts plugin tools into OpenAI Responses schema items.
    pub fn openai_responses_tool_schemas(&self) -> Vec<Value> {
        self.active_tool_definitions()
            .into_iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                })
            })
            .collect()
    }

    /// Converts plugin tools into Anthropic tool schema items.
    pub fn anthropic_tool_schemas(&self) -> Vec<Value> {
        self.active_tool_definitions()
            .into_iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })
            })
            .collect()
    }

    /// Stops all plugin processes.
    pub async fn shutdown(&self) {
        for plugin in &self.plugins {
            plugin.shutdown(self.config.debug).await;
        }
    }

    /// Dispatches one message event to all subscribed event plugins.
    ///
    /// 事件插件是“旁路扩展能力”：它们可以观察消息、返回动作，并可显式要求阻断后续传播。
    pub async fn dispatch_event(&self, event: &MessageEvent) -> Result<PluginDispatchOutcome> {
        let ctx = PluginMessageContext::from_event(event, self.config.owner.qq);
        let mut out = PluginDispatchOutcome::default();

        for plugin in self
            .plugins
            .iter()
            .filter(|plugin| plugin.matches_event(&ctx))
        {
            let request = plugin.build_event_request(&ctx);
            let reply = plugin.call(request, self.config.debug).await?;
            let normalized = reply.normalize();
            let mention_default = self.config.group.mention_sender;
            out.actions.extend(plugin_response_to_actions(
                plugin,
                &normalized,
                event,
                mention_default,
            ));
            if normalized.stop_propagation {
                out.stop_propagation = true;
                break;
            }
        }

        Ok(out)
    }

    /// Returns true when at least one event plugin would see this message.
    ///
    /// 这个方法只做轻量预判，不触发插件进程调用。
    /// 宿主可用它决定是否应该先保守地跳过 AI 侧的阶段回复。
    pub fn has_matching_event_subscriber(&self, event: &MessageEvent) -> bool {
        let ctx = PluginMessageContext::from_event(event, self.config.owner.qq);
        self.plugins.iter().any(|plugin| plugin.matches_event(&ctx))
    }

    /// Routes one slash command to its owning plugin.
    pub async fn try_handle_command(&self, event: &MessageEvent) -> Result<Vec<ActionRequest>> {
        let ctx = PluginMessageContext::from_event(event, self.config.owner.qq);
        let Some((cmd, args)) = parse_command(&ctx.normalized_text) else {
            return Ok(Vec::new());
        };

        if event.message_type == "group" && self.config.group.require_at && !ctx.mentioned {
            return Ok(Vec::new());
        }

        let Some(&idx) = self.command_map.get(&cmd) else {
            return Ok(Vec::new());
        };
        let plugin = &self.plugins[idx];
        let req = plugin.build_command_request(&ctx, cmd, args);
        let reply = plugin.call(req, self.config.debug).await?;
        let normalized = reply.normalize();
        Ok(plugin_response_to_actions(
            plugin,
            &normalized,
            event,
            self.config.group.mention_sender,
        ))
    }

    /// Returns true when current text would be claimed by a command plugin.
    pub fn has_matching_command(&self, normalized_text: &str) -> bool {
        parse_command(normalized_text)
            .map(|(cmd, _)| self.command_map.contains_key(&cmd))
            .unwrap_or(false)
    }

    /// Executes one plugin-defined tool by name.
    pub async fn call_tool(&self, tool_name: &str, arguments: Value) -> Result<Option<String>> {
        let key = tool_name.trim().to_lowercase();
        let Some(&idx) = self.tool_map.get(&key) else {
            return Ok(None);
        };

        let plugin = &self.plugins[idx];
        let req = plugin.build_tool_request(tool_name.to_string(), arguments);
        let reply = plugin.call(req, self.config.debug).await?;
        let normalized = reply.normalize();
        let result = normalized
            .tool_result
            .clone()
            .unwrap_or_else(|| normalized.reply.trim().to_string());
        if result.is_empty() {
            return Ok(Some("plugin tool returned empty result".to_string()));
        }
        Ok(Some(result))
    }

    fn active_tool_definitions(&self) -> Vec<PluginToolDefinition> {
        let mut out = Vec::new();
        for (idx, plugin) in self.plugins.iter().enumerate() {
            for tool in &plugin.tools {
                let key = tool.name.to_lowercase();
                if self.tool_map.get(&key) == Some(&idx) {
                    out.push(tool.clone());
                }
            }
        }
        out
    }
}

/// Metadata returned by plugin executable via `--manifest`.
#[derive(Debug, Deserialize)]
struct PluginManifestInfo {
    name: String,
    #[serde(default)]
    commands: Vec<String>,
    #[serde(default)]
    subscriptions: Vec<String>,
    #[serde(default)]
    tools: Vec<PluginToolDefinition>,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    priority: Option<i32>,
}

/// Runtime handle for one managed plugin process.
#[derive(Debug, Clone)]
struct ManagedPlugin {
    name: String,
    path: PathBuf,
    commands: Vec<String>,
    subscriptions: HashSet<String>,
    tools: Vec<PluginToolDefinition>,
    timeout_ms: u64,
    priority: i32,
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
            .map(|c| c.trim().trim_start_matches('/').to_lowercase())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>();
        let commands = if commands.is_empty() {
            vec![name.clone()]
        } else {
            commands
        };
        let subscriptions = manifest
            .subscriptions
            .into_iter()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect::<HashSet<_>>();
        let tools = manifest
            .tools
            .into_iter()
            .filter_map(normalize_tool_definition)
            .collect::<Vec<_>>();
        let timeout_ms = manifest.timeout_ms.unwrap_or(DEFAULT_PLUGIN_TIMEOUT_MS);
        let priority = manifest.priority.unwrap_or(DEFAULT_PLUGIN_PRIORITY);
        let config_dir = root.join(&name);
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)
                .with_context(|| format!("failed to create plugin dir {}", config_dir.display()))?;
        }

        Ok(Self {
            name,
            path,
            commands,
            subscriptions,
            tools,
            timeout_ms,
            priority,
            config_dir,
            process: Arc::new(Mutex::new(None)),
            seq: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Returns whether the plugin subscribes to any event matched by current message context.
    fn matches_event(&self, ctx: &PluginMessageContext) -> bool {
        if self.subscriptions.is_empty() {
            return false;
        }
        if self.subscriptions.contains("*") {
            return true;
        }
        ctx.event_types
            .iter()
            .any(|event_type| self.subscriptions.contains(event_type))
    }

    /// Allocates monotonically increasing request id for IPC correlation.
    fn next_request_id(&self) -> String {
        format!("{}-{}", self.name, self.seq.fetch_add(1, Ordering::Relaxed))
    }

    /// Builds one command-type plugin request.
    fn build_command_request(
        &self,
        ctx: &PluginMessageContext,
        command: String,
        args: String,
    ) -> PluginRequest {
        PluginRequest {
            request_id: self.next_request_id(),
            kind: PluginRequestKind::Command,
            command,
            args,
            raw_text: ctx.normalized_text.clone(),
            text: ctx.text.clone(),
            message_type: ctx.message_type.clone(),
            message_id: ctx.message_id,
            user_id: ctx.user_id,
            group_id: ctx.group_id,
            self_id: ctx.self_id,
            display_name: ctx.display_name.clone(),
            mentioned: ctx.mentioned,
            is_owner: ctx.is_owner,
            event_types: ctx.event_types.clone(),
            image_urls: ctx.image_urls.clone(),
            image_files: ctx.image_files.clone(),
            reply_message_ids: ctx.reply_message_ids.clone(),
            quote_texts: ctx.quote_texts.clone(),
            forward_contexts: ctx.forward_contexts.clone(),
            tool_name: None,
            tool_arguments: None,
            config_dir: self.config_dir.to_string_lossy().to_string(),
        }
    }

    /// Builds one event-type plugin request.
    fn build_event_request(&self, ctx: &PluginMessageContext) -> PluginRequest {
        PluginRequest {
            request_id: self.next_request_id(),
            kind: PluginRequestKind::Event,
            command: String::new(),
            args: String::new(),
            raw_text: ctx.raw_text.clone(),
            text: ctx.text.clone(),
            message_type: ctx.message_type.clone(),
            message_id: ctx.message_id,
            user_id: ctx.user_id,
            group_id: ctx.group_id,
            self_id: ctx.self_id,
            display_name: ctx.display_name.clone(),
            mentioned: ctx.mentioned,
            is_owner: ctx.is_owner,
            event_types: ctx.event_types.clone(),
            image_urls: ctx.image_urls.clone(),
            image_files: ctx.image_files.clone(),
            reply_message_ids: ctx.reply_message_ids.clone(),
            quote_texts: ctx.quote_texts.clone(),
            forward_contexts: ctx.forward_contexts.clone(),
            tool_name: None,
            tool_arguments: None,
            config_dir: self.config_dir.to_string_lossy().to_string(),
        }
    }

    /// Builds one tool-type plugin request.
    fn build_tool_request(&self, tool_name: String, tool_arguments: Value) -> PluginRequest {
        PluginRequest {
            request_id: self.next_request_id(),
            kind: PluginRequestKind::Tool,
            command: tool_name.clone(),
            args: tool_arguments.to_string(),
            raw_text: String::new(),
            text: String::new(),
            message_type: String::new(),
            message_id: None,
            user_id: 0,
            group_id: None,
            self_id: 0,
            display_name: String::new(),
            mentioned: false,
            is_owner: false,
            event_types: Vec::new(),
            image_urls: Vec::new(),
            image_files: Vec::new(),
            reply_message_ids: Vec::new(),
            quote_texts: Vec::new(),
            forward_contexts: Vec::new(),
            tool_name: Some(tool_name),
            tool_arguments: Some(tool_arguments),
            config_dir: self.config_dir.to_string_lossy().to_string(),
        }
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

/// IPC request kind.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum PluginRequestKind {
    Command,
    Event,
    Tool,
}

/// IPC request payload sent from host to plugin.
///
/// 为了兼容旧插件，这里保留了 `command` / `args` / `raw_text` 等字段；
/// 新插件则应优先看 `kind`、`event_types`、`tool_name`、`tool_arguments`。
#[derive(Debug, Serialize)]
struct PluginRequest {
    request_id: String,
    kind: PluginRequestKind,
    command: String,
    args: String,
    raw_text: String,
    text: String,
    message_type: String,
    message_id: Option<i64>,
    user_id: i64,
    group_id: Option<i64>,
    self_id: i64,
    display_name: String,
    mentioned: bool,
    is_owner: bool,
    event_types: Vec<String>,
    image_urls: Vec<String>,
    image_files: Vec<String>,
    reply_message_ids: Vec<i64>,
    quote_texts: Vec<String>,
    forward_contexts: Vec<String>,
    tool_name: Option<String>,
    tool_arguments: Option<Value>,
    config_dir: String,
}

/// 插件统一响应体。
///
/// 为了兼容旧版协议，`reply/file_path/image_path` 仍保留；
/// 新版协议推荐直接使用 `actions` + `stop_propagation` + `tool_result`。
#[derive(Debug, Deserialize, Default)]
struct PluginResponse {
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
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
    #[serde(default)]
    actions: Vec<PluginActionOutput>,
    #[serde(default)]
    stop_propagation: bool,
    #[serde(default)]
    tool_result: Option<String>,
}

impl PluginResponse {
    /// 把旧字段折叠进新版 `actions`，保证老插件无需改动也能继续运行。
    fn normalize(mut self) -> Self {
        if let Some(file_path) = self.file_path.clone() {
            self.actions.push(PluginActionOutput::File {
                file_path,
                file_name: self.file_name.clone(),
            });
        }

        if self.image_path.is_some() || self.image_url.is_some() {
            self.actions.push(PluginActionOutput::Image {
                image_path: self.image_path.clone(),
                image_url: self.image_url.clone(),
                caption: None,
                mention_sender: self.mention_sender,
            });
        }

        if !self.reply.trim().is_empty() {
            self.actions.push(PluginActionOutput::Message {
                text: self.reply.clone(),
                mention_sender: self.mention_sender,
            });
        }

        self
    }
}

/// 宿主支持的声明式插件输出动作。
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PluginActionOutput {
    Message {
        text: String,
        #[serde(default)]
        mention_sender: Option<bool>,
    },
    Image {
        #[serde(default)]
        image_path: Option<String>,
        #[serde(default)]
        image_url: Option<String>,
        #[serde(default)]
        caption: Option<String>,
        #[serde(default)]
        mention_sender: Option<bool>,
    },
    File {
        file_path: String,
        #[serde(default)]
        file_name: Option<String>,
    },
}

/// 当前消息被标准化后的只读事件快照。
struct PluginMessageContext {
    raw_text: String,
    normalized_text: String,
    text: String,
    message_type: String,
    message_id: Option<i64>,
    user_id: i64,
    group_id: Option<i64>,
    self_id: i64,
    display_name: String,
    mentioned: bool,
    is_owner: bool,
    event_types: Vec<String>,
    image_urls: Vec<String>,
    image_files: Vec<String>,
    reply_message_ids: Vec<i64>,
    quote_texts: Vec<String>,
    forward_contexts: Vec<String>,
}

impl PluginMessageContext {
    /// 从 OneBot 事件构造插件快照。
    ///
    /// 这里把事件分类、@ 信息、图片引用、引用回复等一次性整理好，
    /// 让插件不必反复猜 OneBot 原始结构。
    fn from_event(event: &MessageEvent, owner_qq: i64) -> Self {
        // 插件的“触发/命令/事件分类”必须只看原始消息，
        // 否则像最近群聊这种 AI 侧的附加上下文会污染插件命中结果。
        let raw_text = event.original_text();
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        let mentioned = raw_text.contains(&at_me);
        let normalized_text = raw_text.replace(&at_me, "").trim().to_string();
        let parsed_raw = parse_user_content(&normalized_text);
        let parsed_full = parse_user_content(&event.text());
        let reply_message_ids = event.reply_message_ids();
        let is_owner = event.user_id == owner_qq;
        let message_id = extract_i64_from_value(event.message_id.as_ref());
        let quote_texts = collect_enriched_parts(&event.enriched_parts, "[引用消息]");
        let forward_contexts = collect_enriched_parts(&event.enriched_parts, "[转发消息");

        let mut event_types = vec!["message".to_string()];
        match event.message_type.as_str() {
            "group" => event_types.push("group_message".to_string()),
            "private" => event_types.push("private_message".to_string()),
            other if !other.is_empty() => event_types.push(other.to_string()),
            _ => {}
        }
        if mentioned {
            event_types.push("mention".to_string());
        }
        if !parsed_full.image_urls.is_empty() || !parsed_full.image_files.is_empty() {
            event_types.push("image".to_string());
        }
        if !reply_message_ids.is_empty() {
            event_types.push("quote".to_string());
        }
        if is_owner {
            event_types.push("owner".to_string());
        }
        if parse_command(&normalized_text).is_some() {
            event_types.push("command".to_string());
        }

        Self {
            raw_text,
            normalized_text,
            // 这里的正文只保留“当前消息本身”的可读文本；
            // 引用/转发/图片额外上下文通过独立字段继续提供给插件，避免一锅炖导致插件重复解析。
            text: parsed_raw.text,
            message_type: event.message_type.clone(),
            message_id,
            user_id: event.user_id,
            group_id: event.group_id,
            self_id: event.self_id,
            display_name: event.display_name(),
            mentioned,
            is_owner,
            event_types,
            image_urls: merge_unique_strings(parsed_raw.image_urls, parsed_full.image_urls),
            image_files: merge_unique_strings(parsed_raw.image_files, parsed_full.image_files),
            reply_message_ids,
            quote_texts,
            forward_contexts,
        }
    }
}

fn merge_unique_strings(primary: Vec<String>, secondary: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in primary.into_iter().chain(secondary.into_iter()) {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            out.push(trimmed.to_string());
        }
    }
    out
}

fn collect_enriched_parts(parts: &[String], prefix: &str) -> Vec<String> {
    parts
        .iter()
        .filter_map(|part| {
            let trimmed = part.trim();
            if trimmed.starts_with(prefix) {
                Some(trimmed.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn extract_i64_from_value(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|n| i64::try_from(n).ok()))
        .or_else(|| value.as_str().and_then(|s| s.trim().parse::<i64>().ok()))
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
                ..Default::default()
            });
        }
    };

    timeout(Duration::from_millis(timeout_ms), fut)
        .await
        .map_err(|_| anyhow!("plugin timeout after {} ms", timeout_ms))?
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

/// Converts one plugin response into concrete OneBot actions.
fn plugin_response_to_actions(
    plugin: &ManagedPlugin,
    response: &PluginResponse,
    event: &MessageEvent,
    default_mention_sender: bool,
) -> Vec<ActionRequest> {
    let mut out = Vec::new();

    for action in &response.actions {
        match action {
            PluginActionOutput::Message {
                text,
                mention_sender,
            } => {
                if text.trim().is_empty() {
                    continue;
                }
                let mention_sender = mention_sender.unwrap_or(default_mention_sender);
                out.push(build_text_action(event, text.clone(), mention_sender));
            }
            PluginActionOutput::File {
                file_path,
                file_name,
            } => {
                let resolved = resolve_plugin_path(file_path, &plugin.config_dir);
                if !resolved.exists() {
                    log_warn(format!(
                        "plugin {} file not found: {}",
                        plugin.name,
                        resolved.display()
                    ));
                    continue;
                }
                let file_name = file_name.clone().or_else(|| {
                    resolved
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                });
                out.push(match event.message_type.as_str() {
                    "group" => ActionRequest::upload_group_file(
                        event.group_id.unwrap_or_default(),
                        resolved.to_string_lossy().to_string(),
                        file_name,
                    ),
                    _ => ActionRequest::upload_private_file(
                        event.user_id,
                        resolved.to_string_lossy().to_string(),
                        file_name,
                    ),
                });
            }
            PluginActionOutput::Image {
                image_path,
                image_url,
                caption,
                mention_sender,
            } => {
                let Some(image_ref) = image_url.clone().or_else(|| image_path.clone()) else {
                    continue;
                };
                let resolved = if image_url.is_some() {
                    image_ref
                } else {
                    resolve_plugin_path(&image_ref, &plugin.config_dir)
                        .to_string_lossy()
                        .to_string()
                };
                if image_url.is_none() && !Path::new(&resolved).exists() {
                    log_warn(format!(
                        "plugin {} image not found: {}",
                        plugin.name, resolved
                    ));
                    continue;
                }
                let mention_sender = mention_sender.unwrap_or(default_mention_sender);
                let image_cq = format!("[CQ:image,file={}]", resolved);
                let body = if let Some(caption) = caption {
                    if caption.trim().is_empty() {
                        image_cq
                    } else {
                        format!("{image_cq}\n{}", caption.trim())
                    }
                } else {
                    image_cq
                };
                out.push(build_text_action(event, body, mention_sender));
            }
        }
    }

    out
}

fn build_text_action(event: &MessageEvent, body: String, mention_sender: bool) -> ActionRequest {
    match event.message_type.as_str() {
        "group" => {
            let group_id = event.group_id.unwrap_or_default();
            let message = if mention_sender {
                format!("[CQ:at,qq={}] {}", event.user_id, body)
            } else {
                body
            };
            ActionRequest::send_group_msg(group_id, message)
        }
        _ => ActionRequest::send_private_msg(event.user_id, body),
    }
}

fn normalize_tool_definition(tool: PluginToolDefinition) -> Option<PluginToolDefinition> {
    let name = tool.name.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let description = if tool.description.trim().is_empty() {
        format!("Plugin tool provided by {}", name)
    } else {
        tool.description.trim().to_string()
    };
    let input_schema = if tool.input_schema.is_null() {
        default_tool_input_schema()
    } else {
        tool.input_schema
    };
    Some(PluginToolDefinition {
        name,
        description,
        input_schema,
    })
}

fn default_tool_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {}
    })
}
