//! 运行时配置模型：负责配置加载、默认值与跨字段校验。
//!
//! 配置模块的目标不是单纯“读 TOML”，而是在启动阶段尽早发现问题。
//! 因此这里会把很多跨字段约束提前校验掉，例如：
//! - 不同 AI 提供方所需的字段是否齐全
//! - OCR / 搜索 / 代理等可选子系统的参数是否合法
//! - 路由与群策略配置是否自洽

use std::{
    collections::HashSet,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use toml::Value;

const DEFAULT_CONFIG_TOML: &str = include_str!("../config.default.toml");
const OPTIONAL_OVERRIDE_FILES: &[&str] = &[
    "owner.toml",
    "server.toml",
    "policy.toml",
    "group.toml",
    "persona.toml",
    "ai.toml",
    "search.toml",
    "network.toml",
    "web_admin.toml",
];
const STRUCTURED_OVERLAY_MAPPINGS: &[(&str, &str)] = &[
    ("persona.toml", "persona"),
    ("ai.toml", "ai"),
    ("search.toml", "search"),
    ("network.toml", "network"),
    ("web_admin.toml", "web_admin"),
];

/// 根运行时配置。
///
/// 加载顺序是：
/// 1. 读取主配置 `config/config.toml`
/// 2. 再按固定顺序读取同目录下的可选覆盖文件
/// 3. 把所有 TOML 表递归合并后反序列化成最终 `Config`
///
/// 这样可以同时满足两类用户：
/// - 普通用户只改一个主配置文件
/// - 高级用户把 AI / Persona / Network 等大块配置拆到单独文件里
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Global debug logging switch.
    #[serde(default)]
    pub debug: bool,
    /// Server listener and OneBot websocket settings.
    pub server: ServerConfig,
    /// Owner account settings.
    pub owner: OwnerConfig,
    /// Permission policy settings.
    pub policy: PolicyConfig,
    /// Group trigger and list settings.
    pub group: GroupConfig,
    /// Persona/system-prompt settings.
    pub persona: PersonaConfig,
    /// LLM backend settings.
    pub ai: AiConfig,
    /// Web search settings.
    #[serde(default)]
    pub search: SearchConfig,
    /// Outbound network/proxy settings.
    #[serde(default)]
    pub network: NetworkConfig,
    /// Web admin console settings.
    #[serde(default)]
    pub web_admin: WebAdminConfig,
}

impl Config {
    /// 从磁盘加载配置。
    ///
    /// 如果主配置不存在，会先自动生成一份“精简默认模板”。
    /// 可选覆盖文件不存在时会被直接忽略。
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        try_migrate_legacy_config(path)?;
        ensure_main_config_exists(path)?;
        ensure_optional_override_files(path.parent().unwrap_or_else(|| Path::new(".")))?;

        let mut merged = read_toml_value(path)?;
        let config_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        for file_name in OPTIONAL_OVERRIDE_FILES {
            let overlay_path = config_dir.join(file_name);
            if !overlay_path.exists() {
                continue;
            }
            let mut overlay = read_toml_value(&overlay_path)?;
            if !overlay_file_enabled(&overlay) {
                continue;
            }
            strip_overlay_control_keys(&mut overlay);
            merge_toml_value(&mut merged, overlay);
        }

        let config: Config = merged.try_into().with_context(|| {
            format!("failed to parse merged config rooted at {}", path.display())
        })?;
        config
            .validate()
            .with_context(|| format!("invalid config in {}", path.display()))?;
        Ok(config)
    }

    /// Validates cross-field constraints for runtime safety.
    fn validate(&self) -> Result<()> {
        if self.server.host.trim().is_empty() {
            bail!("server.host cannot be empty");
        }
        if self.server.ws_path.trim().is_empty() {
            bail!("server.ws_path cannot be empty");
        }
        if !self.server.ws_path.starts_with('/') {
            bail!("server.ws_path must start with '/'");
        }
        if self.server.verify_token && self.server.access_token.trim().is_empty() {
            bail!("server.access_token cannot be empty when server.verify_token = true");
        }
        if self.owner.qq <= 0 {
            bail!("owner.qq must be a positive QQ number");
        }
        if self.ai.max_tokens == 0 {
            bail!("ai.max_tokens must be greater than 0");
        }
        if self.ai.timeout_ms == 0 {
            bail!("ai.timeout_ms must be greater than 0");
        }
        if self.ai.ocr_timeout_ms == 0 {
            bail!("ai.ocr_timeout_ms must be greater than 0");
        }
        if !self.ai.temperature.is_finite() {
            bail!("ai.temperature must be a finite number");
        }
        if self.ai.model.trim().is_empty() {
            bail!("ai.model cannot be empty");
        }
        let mut seen_models = HashSet::new();
        seen_models.insert(self.ai.model.trim().to_string());
        for (idx, model) in self.ai.fallback_models.iter().enumerate() {
            if model.trim().is_empty() {
                bail!("ai.fallback_models[{idx}] cannot be empty");
            }
            if !seen_models.insert(model.trim().to_string()) {
                bail!("ai.fallback_models[{idx}] duplicates another configured model");
            }
        }
        if self.ai.base_url.trim().is_empty() {
            bail!("ai.base_url cannot be empty");
        }
        if self.ai.provider == AiProvider::AnthropicCompatible
            && self.ai.anthropic_version.trim().is_empty()
        {
            bail!("ai.anthropic_version cannot be empty for anthropic_compatible");
        }
        if self.ai.vision_mode == VisionMode::Ocr && self.ai.ocr_cmd.trim().is_empty() {
            bail!("ai.ocr_cmd cannot be empty when ai.vision_mode = \"ocr\"");
        }
        if self.ai.ocr_provider == OcrProvider::Paddle {
            if self.ai.paddle_ocr_endpoint.trim().is_empty() {
                bail!("ai.paddle_ocr_endpoint cannot be empty when ai.ocr_provider = \"paddle\"");
            }
            if !self.ai.paddle_ocr_endpoint.starts_with("http://")
                && !self.ai.paddle_ocr_endpoint.starts_with("https://")
            {
                bail!("ai.paddle_ocr_endpoint must start with http:// or https://");
            }
            if self.ai.paddle_ocr_token.trim().is_empty() {
                bail!("ai.paddle_ocr_token cannot be empty when ai.ocr_provider = \"paddle\"");
            }
            if self.ai.paddle_file_type > 1 {
                bail!("ai.paddle_file_type must be 0 (pdf) or 1 (image)");
            }
        }
        if self.search.provider == SearchProvider::Searxng {
            if self.search.searxng_url.trim().is_empty() {
                bail!("search.searxng_url cannot be empty when search.provider = \"searxng\"");
            }
            if !self.search.searxng_url.starts_with("http://")
                && !self.search.searxng_url.starts_with("https://")
            {
                bail!("search.searxng_url must start with http:// or https://");
            }
        }
        if self.network.proxy_enabled {
            let url = self.network.proxy_url.trim();
            if url.is_empty() {
                bail!("network.proxy_url cannot be empty when network.proxy_enabled = true");
            }
            if !has_valid_proxy_scheme(url) {
                bail!(
                    "network.proxy_url must start with http://, https://, socks5:// or socks5h://"
                );
            }
            if self.network.proxy_test_url.trim().is_empty() {
                bail!("network.proxy_test_url cannot be empty when proxy is enabled");
            }
        }
        if self.web_admin.enabled && self.web_admin.token.trim().is_empty() {
            bail!("web_admin.token cannot be empty when web_admin.enabled = true");
        }
        if self.group.trigger_mode == TriggerMode::Prefix && self.group.prefixes.is_empty() {
            bail!("group.prefixes cannot be empty when trigger_mode = \"prefix\"");
        }
        if self.group.trigger_mode == TriggerMode::Keyword && self.group.keywords.is_empty() {
            bail!("group.keywords cannot be empty when trigger_mode = \"keyword\"");
        }
        let mut seen_group_override = HashSet::new();
        for (idx, item) in self.persona.group_overrides.iter().enumerate() {
            if item.groups.is_empty() {
                bail!("persona.group_overrides[{idx}].groups cannot be empty");
            }
            if item.system.trim().is_empty() {
                bail!("persona.group_overrides[{idx}].system cannot be empty");
            }
            for group_id in &item.groups {
                if *group_id <= 0 {
                    bail!("persona.group_overrides[{idx}].groups contains invalid group id: {group_id}");
                }
                if !seen_group_override.insert(*group_id) {
                    bail!("persona.group_overrides has duplicated group id: {group_id}");
                }
            }
        }

        Ok(())
    }
}

/// 返回主配置和所有可选覆盖文件的稳定文件名顺序。
pub fn known_config_file_names() -> Vec<&'static str> {
    let mut files = Vec::with_capacity(1 + OPTIONAL_OVERRIDE_FILES.len());
    files.push("config.toml");
    files.extend_from_slice(OPTIONAL_OVERRIDE_FILES);
    files
}

/// 尝试把旧版“单文件配置”迁移到新的 `config/` 目录结构。
///
/// 旧版布局：
/// - `<exe_dir>/config.toml`
///
/// 新版布局：
/// - `<exe_dir>/config/config.toml`
///
/// 为了避免覆盖用户已有的新配置，这里只会在“新主配置不存在”时执行一次迁移。
///
/// 迁移策略不是简单复制，而是：
/// 1. 把旧配置中 `persona / ai / search / network` 拆到各自覆盖文件
/// 2. 这些覆盖文件会自动写入 `enabled = true`
/// 3. 主配置只保留其余顶层字段，使新版 `config.toml` 更轻量
fn try_migrate_legacy_config(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let Some(config_dir) = path.parent() else {
        return Ok(());
    };
    let Some(exe_dir) = config_dir.parent() else {
        return Ok(());
    };
    let legacy_path = exe_dir.join("config.toml");
    if !legacy_path.exists() {
        return Ok(());
    }

    fs::create_dir_all(config_dir).with_context(|| {
        format!(
            "failed to create config directory for legacy migration: {}",
            config_dir.display()
        )
    })?;

    let legacy_value = read_toml_value(&legacy_path)?;
    let mut legacy_table = match legacy_value {
        Value::Table(table) => table,
        _ => bail!(
            "legacy config root must be a TOML table: {}",
            legacy_path.display()
        ),
    };

    for (file_name, section_key) in STRUCTURED_OVERLAY_MAPPINGS {
        let Some(section_value) = legacy_table.remove(*section_key) else {
            continue;
        };
        let overlay_path = config_dir.join(file_name);
        if overlay_path.exists() {
            continue;
        }

        let mut overlay_table = toml::map::Map::new();
        overlay_table.insert("enabled".to_string(), Value::Boolean(true));
        overlay_table.insert((*section_key).to_string(), section_value);
        write_toml_value(
            &overlay_path,
            &Value::Table(overlay_table),
            Some(&format!("Migrated from legacy {}", legacy_path.display())),
        )?;
    }

    write_toml_value(
        path,
        &Value::Table(legacy_table),
        Some(&format!("Migrated from legacy {}", legacy_path.display())),
    )?;
    Ok(())
}

/// 确保主配置文件存在；不存在时创建默认模板。
fn ensure_main_config_exists(path: &Path) -> Result<()> {
    match fs::metadata(path) {
        Ok(_) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "config file not found and failed to create config directory {}",
                        parent.display()
                    )
                })?;
            }
            fs::write(path, DEFAULT_CONFIG_TOML).with_context(|| {
                format!(
                    "config file not found and failed to create default config at {}",
                    path.display()
                )
            })?;
            Ok(())
        }
        Err(err) => {
            Err(err).with_context(|| format!("failed to stat config file: {}", path.display()))
        }
    }
}

/// 确保所有可选覆盖文件都存在。
///
/// 这些文件默认都是“关闭状态”，主要作用是把可扩展点显式摆给用户看，
/// 避免用户根本不知道某一类配置可以拆出去。
fn ensure_optional_override_files(config_dir: &Path) -> Result<()> {
    fs::create_dir_all(config_dir).with_context(|| {
        format!(
            "failed to ensure config directory exists: {}",
            config_dir.display()
        )
    })?;

    for file_name in OPTIONAL_OVERRIDE_FILES {
        let path = config_dir.join(file_name);
        if path.exists() {
            continue;
        }
        fs::write(&path, optional_override_template(file_name)).with_context(|| {
            format!(
                "failed to create optional override config file: {}",
                path.display()
            )
        })?;
    }

    Ok(())
}

/// 读取一个 TOML 文件为 `toml::Value`。
///
/// 空文件会被视为空表，便于把覆盖文件当作“占位配置”存在。
fn read_toml_value(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;
    if raw.trim().is_empty() {
        return Ok(Value::Table(Default::default()));
    }
    toml::from_str(&raw).with_context(|| format!("failed to parse TOML from {}", path.display()))
}

/// 以可读的 TOML 格式写出配置文件，并可选附带一行迁移说明注释。
fn write_toml_value(path: &Path, value: &Value, comment: Option<&str>) -> Result<()> {
    let body = toml::to_string_pretty(value)
        .with_context(|| format!("failed to serialize TOML for {}", path.display()))?;
    let content = match comment {
        Some(comment) => format!("# {comment}\n\n{body}"),
        None => body,
    };
    fs::write(path, content)
        .with_context(|| format!("failed to write config file {}", path.display()))
}

/// 递归合并 TOML 值。
///
/// 规则很简单：
/// - 表对表：递归合并
/// - 其他类型：覆盖值替换原值
///
/// 这意味着：
/// - `persona.toml` 可以只覆盖 `[persona]`
/// - `ai.toml` 可以只覆盖 `[ai]` 中的一小部分字段
fn merge_toml_value(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_toml_value(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_slot, overlay_value) => {
            *base_slot = overlay_value;
        }
    }
}

/// 判断一个可选覆盖文件是否启用。
///
/// 规则：
/// - 未声明 `enabled`：视为启用，用于兼容旧写法
/// - `enabled = false`：整份文件跳过
fn overlay_file_enabled(value: &Value) -> bool {
    value
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

/// 去掉覆盖文件里的控制字段，避免它被误并入最终配置。
fn strip_overlay_control_keys(value: &mut Value) {
    if let Value::Table(table) = value {
        table.remove("enabled");
    }
}

/// 返回某个可选覆盖文件的默认模板。
///
/// 模板有两个设计原则：
/// 1. 默认 `enabled = false`，保证生成后不会改变现有行为
/// 2. 只展示“这一类配置能放什么”，不强迫用户一开始就理解全部细节
fn optional_override_template(file_name: &str) -> &'static str {
    match file_name {
        "owner.toml" => {
            r#"# 可选覆盖：主人配置
# 关闭时本文件完全忽略。
enabled = false

[owner]
# qq = 123456789
"#
        }
        "server.toml" => {
            r#"# 可选覆盖：服务监听配置
enabled = false

[server]
# host = "0.0.0.0"
# port = 3000
# ws_path = "/onebot/v11/ws"
# access_token = ""
# verify_token = false
"#
        }
        "policy.toml" => {
            r#"# 可选覆盖：权限策略
enabled = false

[policy]
# permission = "whitelist"
"#
        }
        "group.toml" => {
            r#"# 可选覆盖：群触发与白黑名单
enabled = false

[group]
# whitelist = [123456789]
# blacklist = []
# trigger_mode = "at"
# prefixes = ["/xzbot"]
# keywords = ["xzbot"]
# require_at = false
# mention_sender = true
"#
        }
        "persona.toml" => {
            r#"# 可选覆盖：人设配置
# 分群人设仍然使用 [[persona.group_overrides]]
enabled = false

[persona]
# system = """
# 这里可以放更长的人设文本
# """

#[[persona.group_overrides]]
#groups = [970199915]
#system = """
# 这个群使用单独人设
# """
"#
        }
        "ai.toml" => {
            r#"# 可选覆盖：AI 与 OCR 相关配置
enabled = false

[ai]
# provider = "openai_compatible"
# base_url = "https://api.openai.com/v1"
# api_key = ""
# model = "gpt-4.1-mini"
# fallback_models = ["gpt-4.1-nano", "gpt-4o-mini"]
# stream_chat_completions = false
# temperature = 0.7
# max_tokens = 512
# timeout_ms = 20000
# wire_api = "chat_completions"
# reasoning_effort = "low"
# disable_response_storage = false
# vision_mode = "auto"
# ocr_provider = "tesseract"
# ocr_cmd = "tesseract"
# ocr_lang = "chi_sim+eng"
# ocr_timeout_ms = 8000
"#
        }
        "search.toml" => {
            r#"# 可选覆盖：搜索配置
enabled = false

[search]
# provider = "builtin"
# searxng_url = ""
"#
        }
        "network.toml" => {
            r#"# 可选覆盖：网络与代理配置
enabled = false

[network]
# proxy_enabled = false
# proxy_url = ""
# proxy_test_url = "https://www.baidu.com"
# proxy_timeout_ms = 5000
"#
        }
        "web_admin.toml" => {
            r#"# 可选覆盖：Web 控制面板配置
enabled = false

[web_admin]
# enabled = true
# token = "change-me"
# title = "XzBot Console"
"#
        }
        _ => "enabled = false\n",
    }
}

/// Server and reverse websocket access settings.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Bind host, e.g. `0.0.0.0`.
    pub host: String,
    /// Bind port.
    pub port: u16,
    /// Websocket route path.
    pub ws_path: String,
    /// Optional access token for reverse websocket auth.
    #[serde(default)]
    pub access_token: String,
    /// Enables token verification when true.
    #[serde(default)]
    pub verify_token: bool,
}

/// Bot owner identity.
#[derive(Debug, Clone, Deserialize)]
pub struct OwnerConfig {
    /// Owner QQ account id.
    pub qq: i64,
}

/// High-level permission policy.
#[derive(Debug, Clone, Deserialize)]
pub struct PolicyConfig {
    /// Permission mode used by router.
    pub permission: PermissionMode,
}

/// Supported permission modes.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Allow everyone (still subject to blacklist).
    None,
    /// Only owner private messages are allowed.
    OwnerOnly,
    /// Private messages allowed; group messages must be in whitelist.
    Whitelist,
}

/// Group-specific routing and trigger configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct GroupConfig {
    /// Allowed group ids when permission mode is whitelist.
    #[serde(default)]
    pub whitelist: Vec<i64>,
    /// Denied group ids with highest priority.
    #[serde(default)]
    pub blacklist: Vec<i64>,
    /// Group trigger mode.
    #[serde(default = "default_trigger_mode")]
    pub trigger_mode: TriggerMode,
    /// Prefix list used by `prefix` or `mixed` mode.
    #[serde(default)]
    pub prefixes: Vec<String>,
    /// Keyword list used by `keyword` or `mixed` mode.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Requires `@bot` before trigger check when true.
    #[serde(default)]
    pub require_at: bool,
    /// Mentions sender in group reply when true.
    #[serde(default = "default_true")]
    pub mention_sender: bool,
}

/// Group trigger mode.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    /// Trigger only when bot is mentioned.
    At,
    /// Trigger by configured prefixes.
    Prefix,
    /// Trigger by configured keywords.
    Keyword,
    /// Trigger when any configured strategy matches.
    Mixed,
}

/// Persona config with optional group overrides.
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaConfig {
    /// Default system prompt.
    pub system: String,
    /// Group-specific prompt overrides.
    #[serde(default)]
    pub group_overrides: Vec<GroupPersonaOverride>,
}

impl PersonaConfig {
    /// Resolves system prompt for one group, returning matched group id when override is used.
    pub fn resolve_system_for_group(&self, group_id: Option<i64>) -> (&str, Option<i64>) {
        if let Some(group_id) = group_id {
            for item in &self.group_overrides {
                if item.groups.iter().any(|id| *id == group_id) && !item.system.trim().is_empty() {
                    return (item.system.as_str(), Some(group_id));
                }
            }
        }
        (self.system.as_str(), None)
    }
}

/// One persona override entry for specific groups.
#[derive(Debug, Clone, Deserialize)]
pub struct GroupPersonaOverride {
    /// Target group ids.
    #[serde(default)]
    pub groups: Vec<i64>,
    /// Override system prompt.
    pub system: String,
}

/// LLM provider and generation settings.
#[derive(Debug, Clone, Deserialize)]
pub struct AiConfig {
    /// Backend provider type.
    pub provider: AiProvider,
    /// Provider base URL.
    pub base_url: String,
    /// OpenAI-compatible transport API shape.
    #[serde(default = "default_openai_wire_api")]
    pub wire_api: OpenAiWireApi,
    /// API key/token.
    pub api_key: String,
    /// Model id.
    pub model: String,
    /// Additional model ids used as fallback chain.
    #[serde(default)]
    pub fallback_models: Vec<String>,
    /// Whether `/chat/completions` requests should be sent as SSE stream mode.
    ///
    /// Some third-party OpenAI-compatible gateways only return valid content when
    /// `stream=true` is enabled, even if the caller eventually wants one complete reply.
    /// In that case XzBot will request stream mode and locally fold the SSE chunks back
    /// into a normal Chat Completions-style response object.
    #[serde(default)]
    pub stream_chat_completions: bool,
    /// Reasoning effort hint for providers that support it.
    #[serde(default = "default_reasoning_effort")]
    pub reasoning_effort: String,
    /// Disables provider-side response storage when supported.
    #[serde(default)]
    pub disable_response_storage: bool,
    /// Sampling temperature.
    pub temperature: f32,
    /// Maximum output tokens.
    pub max_tokens: u32,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Anthropic API version header value.
    #[serde(default = "default_anthropic_version")]
    pub anthropic_version: String,
    /// Vision handling mode.
    #[serde(default = "default_vision_mode")]
    pub vision_mode: VisionMode,
    /// OCR provider used when OCR fallback is enabled.
    #[serde(default = "default_ocr_provider")]
    pub ocr_provider: OcrProvider,
    /// OCR command binary path (tesseract mode).
    #[serde(default = "default_ocr_cmd")]
    pub ocr_cmd: String,
    /// OCR language pack list.
    #[serde(default = "default_ocr_lang")]
    pub ocr_lang: String,
    /// OCR timeout in milliseconds.
    #[serde(default = "default_ocr_timeout_ms")]
    pub ocr_timeout_ms: u64,
    /// Paddle OCR endpoint.
    #[serde(default)]
    pub paddle_ocr_endpoint: String,
    /// Paddle OCR token.
    #[serde(default)]
    pub paddle_ocr_token: String,
    /// Paddle `fileType` argument.
    #[serde(default = "default_paddle_file_type")]
    pub paddle_file_type: u8,
    /// Paddle optional parameter.
    #[serde(default)]
    pub paddle_use_doc_orientation_classify: bool,
    /// Paddle optional parameter.
    #[serde(default)]
    pub paddle_use_doc_unwarping: bool,
    /// Paddle optional parameter.
    #[serde(default)]
    pub paddle_use_chart_recognition: bool,
    /// Whether Paddle OCR requests should use configured proxy.
    #[serde(default = "default_paddle_use_proxy")]
    pub paddle_use_proxy: bool,
}

impl AiConfig {
    /// 返回按调用顺序排列的模型链。
    ///
    /// 第一个永远是主模型 `model`，后面依次接 `fallback_models`。
    pub fn model_chain(&self) -> Vec<String> {
        let mut models = Vec::with_capacity(1 + self.fallback_models.len());
        models.push(self.model.clone());
        for model in &self.fallback_models {
            let trimmed = model.trim();
            if !trimmed.is_empty() {
                models.push(trimmed.to_string());
            }
        }
        models
    }
}

/// Web 控制面板配置。
#[derive(Debug, Clone, Deserialize)]
pub struct WebAdminConfig {
    /// Whether the dashboard routes are active.
    #[serde(default)]
    pub enabled: bool,
    /// Shared secret used by the login form.
    #[serde(default)]
    pub token: String,
    /// Title shown in the dashboard header.
    #[serde(default = "default_web_admin_title")]
    pub title: String,
}

impl Default for WebAdminConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            token: String::new(),
            title: default_web_admin_title(),
        }
    }
}

/// Search backend configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    /// Search backend provider.
    #[serde(default = "default_search_provider")]
    pub provider: SearchProvider,
    /// SearxNG endpoint base URL.
    #[serde(default)]
    pub searxng_url: String,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            provider: default_search_provider(),
            searxng_url: String::new(),
        }
    }
}

/// Supported search providers.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchProvider {
    /// Built-in search pipeline.
    Builtin,
    /// External SearxNG instance.
    Searxng,
}

impl SearchProvider {
    /// Returns stable provider string used in logs/session ids.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::Searxng => "searxng",
        }
    }
}

/// Supported LLM provider types.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AiProvider {
    /// Offline mock provider.
    Mock,
    /// OpenAI-compatible provider.
    OpenaiCompatible,
    /// Anthropic-compatible provider.
    AnthropicCompatible,
}

impl AiProvider {
    /// Returns stable provider string used in logs/session ids.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::OpenaiCompatible => "openai_compatible",
            Self::AnthropicCompatible => "anthropic_compatible",
        }
    }
}

/// OpenAI-compatible HTTP API wire format.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiWireApi {
    /// Classic `/chat/completions` API.
    ChatCompletions,
    /// New `/responses` API.
    Responses,
}

/// Vision handling mode for image inputs.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionMode {
    /// Auto-detect based on model/provider behavior.
    Auto,
    /// Force multimodal image upload.
    Multimodal,
    /// Force OCR fallback.
    Ocr,
    /// Disable image processing.
    Off,
}

/// OCR provider selection.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OcrProvider {
    /// Local tesseract command.
    Tesseract,
    /// Remote Paddle OCR API.
    Paddle,
}

/// Default helper for boolean fields that should default to true.
fn default_true() -> bool {
    true
}

/// Default group trigger mode.
fn default_trigger_mode() -> TriggerMode {
    TriggerMode::Mixed
}

/// Default Anthropic API version.
fn default_anthropic_version() -> String {
    "2023-06-01".to_string()
}

/// Default OpenAI-compatible wire API.
fn default_openai_wire_api() -> OpenAiWireApi {
    OpenAiWireApi::ChatCompletions
}

/// Default reasoning effort hint.
fn default_reasoning_effort() -> String {
    "low".to_string()
}

/// Default web admin page title.
fn default_web_admin_title() -> String {
    "XzBot Console".to_string()
}

/// Default vision mode.
fn default_vision_mode() -> VisionMode {
    VisionMode::Auto
}

/// Default OCR provider.
fn default_ocr_provider() -> OcrProvider {
    OcrProvider::Tesseract
}

/// Default OCR command.
fn default_ocr_cmd() -> String {
    "tesseract".to_string()
}

/// Default OCR languages.
fn default_ocr_lang() -> String {
    "chi_sim+eng".to_string()
}

/// Default OCR timeout.
fn default_ocr_timeout_ms() -> u64 {
    8_000
}

/// Default Paddle file type (image).
fn default_paddle_file_type() -> u8 {
    1
}

/// Default Paddle proxy usage.
fn default_paddle_use_proxy() -> bool {
    true
}

/// Default search provider.
fn default_search_provider() -> SearchProvider {
    SearchProvider::Builtin
}

/// Outbound network and proxy settings.
#[derive(Debug, Clone, Deserialize)]
pub struct NetworkConfig {
    /// Enables global outbound proxy.
    #[serde(default)]
    pub proxy_enabled: bool,
    /// Proxy URL (`http://`, `https://`, `socks5://`, `socks5h://`).
    #[serde(default)]
    pub proxy_url: String,
    /// URL used to test proxy reachability at startup.
    #[serde(default = "default_proxy_test_url")]
    pub proxy_test_url: String,
    /// Proxy test timeout in milliseconds.
    #[serde(default = "default_proxy_timeout_ms")]
    pub proxy_timeout_ms: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            proxy_enabled: false,
            proxy_url: String::new(),
            proxy_test_url: default_proxy_test_url(),
            proxy_timeout_ms: default_proxy_timeout_ms(),
        }
    }
}

/// Default proxy connectivity test URL.
fn default_proxy_test_url() -> String {
    "https://www.baidu.com".to_string()
}

/// Default proxy test timeout.
fn default_proxy_timeout_ms() -> u64 {
    5_000
}

/// Validates supported proxy URL scheme.
fn has_valid_proxy_scheme(url: &str) -> bool {
    url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("socks5://")
        || url.starts_with("socks5h://")
}
