use std::{collections::HashSet, fs, io::ErrorKind, path::Path};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

const DEFAULT_CONFIG_TOML: &str = include_str!("../config.toml");

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub debug: bool,
    pub server: ServerConfig,
    pub owner: OwnerConfig,
    pub policy: PolicyConfig,
    pub group: GroupConfig,
    pub persona: PersonaConfig,
    pub ai: AiConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub network: NetworkConfig,
}

impl Config {
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

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub ws_path: String,
    #[serde(default)]
    pub access_token: String,
    #[serde(default)]
    pub verify_token: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OwnerConfig {
    pub qq: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PolicyConfig {
    pub permission: PermissionMode,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    None,
    OwnerOnly,
    Whitelist,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GroupConfig {
    #[serde(default)]
    pub whitelist: Vec<i64>,
    #[serde(default)]
    pub blacklist: Vec<i64>,
    #[serde(default = "default_trigger_mode")]
    pub trigger_mode: TriggerMode,
    #[serde(default)]
    pub prefixes: Vec<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub require_at: bool,
    #[serde(default = "default_true")]
    pub mention_sender: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    At,
    Prefix,
    Keyword,
    Mixed,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PersonaConfig {
    pub system: String,
    #[serde(default)]
    pub group_overrides: Vec<GroupPersonaOverride>,
}

impl PersonaConfig {
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

#[derive(Debug, Clone, Deserialize)]
pub struct GroupPersonaOverride {
    #[serde(default)]
    pub groups: Vec<i64>,
    pub system: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AiConfig {
    pub provider: AiProvider,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_ms: u64,
    #[serde(default = "default_anthropic_version")]
    pub anthropic_version: String,
    #[serde(default = "default_vision_mode")]
    pub vision_mode: VisionMode,
    #[serde(default = "default_ocr_provider")]
    pub ocr_provider: OcrProvider,
    #[serde(default = "default_ocr_cmd")]
    pub ocr_cmd: String,
    #[serde(default = "default_ocr_lang")]
    pub ocr_lang: String,
    #[serde(default = "default_ocr_timeout_ms")]
    pub ocr_timeout_ms: u64,
    #[serde(default)]
    pub paddle_ocr_endpoint: String,
    #[serde(default)]
    pub paddle_ocr_token: String,
    #[serde(default = "default_paddle_file_type")]
    pub paddle_file_type: u8,
    #[serde(default)]
    pub paddle_use_doc_orientation_classify: bool,
    #[serde(default)]
    pub paddle_use_doc_unwarping: bool,
    #[serde(default)]
    pub paddle_use_chart_recognition: bool,
    #[serde(default = "default_paddle_use_proxy")]
    pub paddle_use_proxy: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_search_provider")]
    pub provider: SearchProvider,
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchProvider {
    Builtin,
    Searxng,
}

impl SearchProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::Searxng => "searxng",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AiProvider {
    Mock,
    OpenaiCompatible,
    AnthropicCompatible,
}

impl AiProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::OpenaiCompatible => "openai_compatible",
            Self::AnthropicCompatible => "anthropic_compatible",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionMode {
    Auto,
    Multimodal,
    Ocr,
    Off,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OcrProvider {
    Tesseract,
    Paddle,
}

fn default_true() -> bool {
    true
}

fn default_trigger_mode() -> TriggerMode {
    TriggerMode::Mixed
}

fn default_anthropic_version() -> String {
    "2023-06-01".to_string()
}

fn default_vision_mode() -> VisionMode {
    VisionMode::Auto
}

fn default_ocr_provider() -> OcrProvider {
    OcrProvider::Tesseract
}

fn default_ocr_cmd() -> String {
    "tesseract".to_string()
}

fn default_ocr_lang() -> String {
    "chi_sim+eng".to_string()
}

fn default_ocr_timeout_ms() -> u64 {
    8_000
}

fn default_paddle_file_type() -> u8 {
    1
}

fn default_paddle_use_proxy() -> bool {
    true
}

fn default_search_provider() -> SearchProvider {
    SearchProvider::Builtin
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetworkConfig {
    #[serde(default)]
    pub proxy_enabled: bool,
    #[serde(default)]
    pub proxy_url: String,
    #[serde(default = "default_proxy_test_url")]
    pub proxy_test_url: String,
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

fn default_proxy_test_url() -> String {
    "https://www.baidu.com".to_string()
}

fn default_proxy_timeout_ms() -> u64 {
    5_000
}

fn has_valid_proxy_scheme(url: &str) -> bool {
    url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("socks5://")
        || url.starts_with("socks5h://")
}
