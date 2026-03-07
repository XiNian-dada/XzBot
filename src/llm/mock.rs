//! Mock 模型实现：用于本地联调，不访问真实大模型服务。

use async_trait::async_trait;

use anyhow::{Context, Result};

use crate::{
    config::{NetworkConfig, SearchConfig},
    llm::Llm,
    tools::system::get_system_info,
    tools::{
        http::build_client,
        web::{extract_urls, fetch_url, search_web},
    },
};

/// Mock provider implementation that echoes input and optionally runs tools.
pub struct MockLlm {
    client: reqwest::Client,
    debug: bool,
    search: SearchConfig,
    network: NetworkConfig,
}

impl MockLlm {
    /// Constructs mock provider with shared HTTP client.
    pub fn new(
        debug: bool,
        timeout_ms: u64,
        search: SearchConfig,
        network: NetworkConfig,
    ) -> Result<Self> {
        let client = build_client(timeout_ms, &network, false)
            .context("failed to build HTTP client for MockLlm")?;
        Ok(Self {
            client,
            debug,
            search,
            network,
        })
    }
}

#[async_trait]
impl Llm for MockLlm {
    /// Generates deterministic mock reply from latest user message.
    async fn chat(
        &self,
        _session_id: String,
        messages: Vec<(String, String)>,
    ) -> anyhow::Result<String> {
        let last_user_message = messages
            .iter()
            .rev()
            .find(|(role, _)| role == "user")
            .map(|(_, content)| content.clone())
            .unwrap_or_else(|| "Hello".to_string());

        let mut reply = format!("AI Reply: {last_user_message}");

        // Mock provider 只处理当前这条用户消息中的 URL，避免反复引用历史网页。
        let urls = extract_urls(&last_user_message);

        let mut extras = Vec::new();
        for url in urls.into_iter().take(1) {
            match fetch_url(&self.client, &url, self.debug).await {
                Ok(v) => extras.push(format!("[fetch_url]\n{v}")),
                Err(err) => extras.push(format!("[fetch_url error] {err}")),
            }
        }

        if should_search(&last_user_message) {
            match search_web(
                &self.client,
                &last_user_message,
                self.debug,
                &self.search,
                &self.network,
            )
            .await
            {
                Ok(v) => extras.push(format!("[search_web]\n{v}")),
                Err(err) => extras.push(format!("[search_web error] {err}")),
            }
        }

        if let Some(scope) = detect_system_scope(&last_user_message) {
            match get_system_info(scope) {
                Ok(v) => extras.push(format!("[get_system_info:{}]\n{}", scope, v)),
                Err(err) => extras.push(format!("[get_system_info error] {err}")),
            }
        }

        if !extras.is_empty() {
            reply.push_str("\n\n");
            reply.push_str(&extras.join("\n\n"));
        }

        Ok(reply)
    }
}

/// Lightweight heuristic for deciding whether mock mode should call search tool.
fn should_search(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("最新")
        || lower.contains("新闻")
        || lower.contains("搜索")
        || lower.contains("查一下")
        || lower.contains("what is")
        || lower.contains("search")
}

/// Detects system-info scope keywords from plain text.
fn detect_system_scope(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    if lower.contains("cpu") {
        return Some("cpu");
    }
    if lower.contains("memory") || lower.contains("内存") || lower.contains("ram") {
        return Some("memory");
    }
    if lower.contains("load") || lower.contains("负载") {
        return Some("load");
    }
    if lower.contains("uptime") || lower.contains("运行时间") || lower.contains("开机") {
        return Some("uptime");
    }
    if lower.contains("系统信息")
        || lower.contains("服务器信息")
        || lower.contains("机器信息")
        || lower.contains("system info")
    {
        return Some("summary");
    }
    None
}
