//! Runtime configuration schema, defaults, loading and validation.

use std::{collections::HashSet, fs, io::ErrorKind, path::Path};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

const DEFAULT_CONFIG_TOML: &str = include_str!("../config.toml");

/// Root runtime configuration loaded from `config.toml`.
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
}

impl Config {
    /// Loads config from disk, auto-creates default config when missing, then validates.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = match fs::read_to_string(path) {
            Ok(content) => content,
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
                fs::read_to_string(path).with_context(|| {
                    format!(
                        "default config was created but failed to read it from {}",
                        path.display()
                    )
                })?
            }
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("failed to read config file: {}", path.display()))
            }
        };
        let config: Config = toml::from_str(&raw)
            .with_context(|| format!("failed to parse TOML from {}", path.display()))?;
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
