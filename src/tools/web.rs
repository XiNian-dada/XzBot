//! 网页工具入口：对外暴露搜索与抓取能力，并组织共享常量。
//!
//! 这里故意只保留：
//! - 对外稳定 API（`search_web` / `fetch_url` / `extract_urls`）
//! - 搜索与抓取子模块共用的常量、类型和缓存定义
//!
//! 复杂的搜索排序逻辑已经下沉到 `web/search.rs`，这样主入口更容易读。

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE, Engine as _};
use dashmap::DashMap;
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::Value;
use std::{
    collections::HashSet,
    sync::OnceLock,
    time::{Duration, Instant},
};

const DEFAULT_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
const BROWSER_SEC_CH_UA: &str =
    "\"Chromium\";v=\"122\", \"Not(A:Brand\";v=\"24\", \"Google Chrome\";v=\"122\"";
use crate::config::{NetworkConfig, SearchConfig, SearchProvider};

const MAX_FETCH_CHARS: usize = 5000;
const MAX_SEARCH_PREVIEW_CHARS: usize = 700;
const SEARCH_CACHE_TTL_SECS: u64 = 90;

#[derive(Debug, Clone)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
    source: &'static str,
}

#[derive(Clone)]
struct CachedSearchResult {
    body: String,
    cached_at: Instant,
}

static SEARCH_RESULT_CACHE: OnceLock<DashMap<String, CachedSearchResult>> = OnceLock::new();

mod search;

/// 对外统一的网页搜索入口。
///
/// 主模块只负责暴露稳定函数签名，真正复杂的搜索流程放在 `web/search.rs`。
pub async fn search_web(
    client: &Client,
    query: &str,
    debug: bool,
    search: &SearchConfig,
    network: &NetworkConfig,
) -> Result<String> {
    search::search_web(client, query, debug, search, network).await
}

/// 抓取网页正文，并在必要时自动回退到 reader/proxy 方案。
pub async fn fetch_url(client: &Client, url: &str, debug: bool) -> Result<String> {
    let url = normalize_url_input(url);
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("invalid url (must start with http:// or https://): {url}");
    }

    if debug {
        println!("[DEBUG] tool.fetch_url url={url}");
    }

    // First pass: fetch with browser-like headers to reduce anti-bot/error stubs.
    let response = match fetch_with_browser_profile(client, &url).await {
        Ok(resp) => resp,
        Err(err) => {
            if debug {
                println!("[DEBUG] fetch primary failed url={} err={}", url, err);
                println!("[DEBUG] fetch fallback to reader proxy url={url}");
            }
            // Fallback: use reader proxy to get rendered-readable text when direct fetch fails.
            if let Ok(v) = fetch_via_reader_proxy(client, &url, debug).await {
                return Ok(v);
            }
            return Err(err).with_context(|| format!("fetch request failed: {url}"));
        }
    };
    let status = response.status();
    let final_url = response.url().to_string();
    let headers = response.headers().clone();
    let body = response
        .text()
        .await
        .context("failed to read fetched response body")?;

    if !status.is_success() {
        if debug {
            println!(
                "[DEBUG] fetch primary non-success status={} url={} final={}",
                status, url, final_url
            );
            println!("[DEBUG] fetch fallback to reader proxy url={url}");
        }
        if let Ok(v) = fetch_via_reader_proxy(client, &url, debug).await {
            return Ok(v);
        }
        bail!("fetch endpoint returned {status}: {body}");
    }

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    if content_type.contains("text/html") || body.contains("<html") {
        let (title, text) = extract_html_text(&body)?;
        let text = truncate_text(&normalize_whitespace(&text), MAX_FETCH_CHARS);
        let title = if title.trim().is_empty() {
            "(no title)".to_string()
        } else {
            title
        };

        // Detect JS-gated/anti-bot pages and fallback to reader proxy automatically.
        if should_fallback_to_reader(&body, &text) {
            if debug {
                println!(
                    "[DEBUG] fetch primary looks blocked/js-gated url={url} final={final_url}"
                );
                println!("[DEBUG] fetch fallback to reader proxy url={url}");
            }
            if let Ok(v) = fetch_via_reader_proxy(client, &url, debug).await {
                return Ok(v);
            }
        }

        let redirect_note = redirect_note(&url, &final_url);
        return Ok(format!(
            "URL Requested: {url}\nURL Final: {final_url}\n{redirect_note}Title: {title}\nContent:\n{text}"
        ));
    }

    let text = truncate_text(&normalize_whitespace(&body), MAX_FETCH_CHARS);
    let redirect_note = redirect_note(&url, &final_url);
    Ok(format!(
        "URL Requested: {url}\nURL Final: {final_url}\n{redirect_note}Content:\n{text}"
    ))
}

async fn fetch_with_browser_profile(client: &Client, url: &str) -> Result<reqwest::Response> {
    // Use browser-like headers so sites that gate by client fingerprint are less likely to
    // return placeholder/error shells.
    client
        .get(url)
        .header("User-Agent", DEFAULT_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7")
        .header("Cache-Control", "no-cache")
        .header("Pragma", "no-cache")
        .header("Sec-CH-UA", BROWSER_SEC_CH_UA)
        .header("Sec-CH-UA-Mobile", "?0")
        .header("Sec-CH-UA-Platform", "\"macOS\"")
        .header("Sec-Fetch-Dest", "document")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Site", "none")
        .header("Sec-Fetch-User", "?1")
        .header("Upgrade-Insecure-Requests", "1")
        .send()
        .await
        .context("browser-profile request failed")
}

fn should_fallback_to_reader(body: &str, extracted_text: &str) -> bool {
    // Heuristics for JS-gated or anti-bot pages where static extraction is unreliable.
    let lower = body.to_ascii_lowercase();
    let blocked_markers = [
        "javascript is not available",
        "enable javascript",
        "please enable javascript",
        "errorcontainer",
        "please turn javascript on",
        "verify you are human",
        "are you a robot",
        "are you human",
        "checking your browser",
        "access denied",
        "attention required",
        "just a moment",
        "cloudflare",
        "bot challenge",
        "security check",
        "human verification",
        "请开启 javascript",
        "请启用 javascript",
        "需要启用 javascript",
    ];
    if blocked_markers.iter().any(|v| lower.contains(v)) {
        return true;
    }

    let text_len = extracted_text.chars().count();
    let script_count = lower.match_indices("<script").count();
    text_len < 180 && script_count >= 8
}

async fn fetch_via_reader_proxy(client: &Client, url: &str, debug: bool) -> Result<String> {
    // Reader proxy is a generic fallback for dynamic pages and anti-bot front pages.
    let candidates = build_reader_proxy_candidates(url);
    for candidate in candidates {
        let response = match client
            .get(&candidate)
            .header("User-Agent", DEFAULT_UA)
            .header("Accept", "text/plain,text/markdown;q=0.9,*/*;q=0.8")
            .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(err) => {
                if debug {
                    println!(
                        "[DEBUG] fetch.reader request failed candidate={} err={}",
                        candidate, err
                    );
                }
                continue;
            }
        };

        let status = response.status();
        let final_url = response.url().to_string();
        let body = match response.text().await {
            Ok(v) => v,
            Err(err) => {
                if debug {
                    println!(
                        "[DEBUG] fetch.reader read body failed candidate={} err={}",
                        candidate, err
                    );
                }
                continue;
            }
        };

        if !status.is_success() {
            if debug {
                println!(
                    "[DEBUG] fetch.reader non-success status={} candidate={}",
                    status, candidate
                );
            }
            continue;
        }

        let text = truncate_text(&normalize_whitespace(&body), MAX_FETCH_CHARS);
        if text.is_empty() {
            continue;
        }

        if debug {
            println!(
                "[DEBUG] fetch.reader success candidate={} final={} chars={}",
                candidate,
                final_url,
                text.chars().count()
            );
        }

        let redirect = redirect_note(url, &final_url);
        return Ok(format!(
            "URL Requested: {url}\nURL Final: {final_url}\nFetch Mode: reader_proxy\n{redirect}Content:\n{text}"
        ));
    }

    bail!("reader proxy fallback failed for url: {url}")
}

fn build_reader_proxy_candidates(url: &str) -> Vec<String> {
    // Try both direct URL and explicit http-style variant accepted by some reader gateways.
    let mut out = Vec::new();
    out.push(format!("https://r.jina.ai/{url}"));
    if let Some(with_http) = reader_http_style_url(url) {
        out.push(format!("https://r.jina.ai/{with_http}"));
    }
    out
}

fn reader_http_style_url(url: &str) -> Option<String> {
    if let Some(rest) = url.strip_prefix("https://") {
        return Some(format!("http://{rest}"));
    }
    if let Some(rest) = url.strip_prefix("http://") {
        return Some(format!("http://{rest}"));
    }
    None
}

/// Extracts and deduplicates URLs from plain text.
pub fn extract_urls(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for raw in text.split_whitespace() {
        if raw.starts_with("http://") || raw.starts_with("https://") {
            let trimmed = raw
                .trim_matches(|c: char| {
                    matches!(
                        c,
                        '"' | '\''
                            | ')'
                            | '('
                            | ']'
                            | '['
                            | '}'
                            | '{'
                            | ','
                            | ';'
                            | '!'
                            | '?'
                            | '。'
                            | '，'
                    )
                })
                .replace("&amp;", "&")
                .replace("&#38;", "&");
            if !trimmed.is_empty() && seen.insert(trimmed.clone()) {
                out.push(trimmed);
            }
        }
    }

    out
}

fn normalize_url_input(url: &str) -> String {
    url.trim().replace("&amp;", "&").replace("&#38;", "&")
}

fn redirect_note(requested: &str, final_url: &str) -> String {
    if requested == final_url {
        return String::new();
    }

    let req_host = host_of(requested);
    let final_host = host_of(final_url);
    match (req_host, final_host) {
        (Some(a), Some(b))
            if a == b || a.ends_with(&format!(".{b}")) || b.ends_with(&format!(".{a}")) =>
        {
            "Redirect: yes (same site)\n".to_string()
        }
        (Some(a), Some(b)) => format!("Redirect: yes (cross-site {a} -> {b})\n"),
        _ => "Redirect: yes\n".to_string(),
    }
}

fn host_of(url: &str) -> Option<String> {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|v| v.to_string()))
}

fn extract_html_text(html: &str) -> Result<(String, String)> {
    let doc = Html::parse_document(html);
    let body_sel =
        Selector::parse("body").map_err(|err| anyhow!("failed to parse selector body: {err}"))?;
    let title_sel =
        Selector::parse("title").map_err(|err| anyhow!("failed to parse selector title: {err}"))?;

    let title = doc
        .select(&title_sel)
        .next()
        .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
        .unwrap_or_default();
    let body_text = doc
        .select(&body_sel)
        .next()
        .map(|n| n.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();

    Ok((title, body_text))
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    text.chars().take(max_chars).collect::<String>() + "...(truncated)"
}
