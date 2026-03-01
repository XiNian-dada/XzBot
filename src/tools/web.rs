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

pub async fn search_web(
    client: &Client,
    query: &str,
    debug: bool,
    search: &SearchConfig,
    network: &NetworkConfig,
) -> Result<String> {
    let query = query.trim();
    if query.is_empty() {
        bail!("query is empty");
    }
    let cache_key = search_cache_key_with_provider(query, search);
    if let Some(cached) = load_cached_search_result(&cache_key) {
        if debug {
            println!(
                "[DEBUG] search.cache hit key={} ttl={}s",
                cache_key, SEARCH_CACHE_TTL_SECS
            );
        }
        return Ok(cached);
    }

    if debug {
        println!(
            "[DEBUG] tool.search_web query={} provider={}",
            query,
            search.provider.as_str()
        );
    }

    if search.provider == SearchProvider::Searxng {
        return search_web_searxng(client, query, debug, search, &cache_key).await;
    }

    let search_client = match build_search_session_client(network) {
        Ok(c) => c,
        Err(err) => {
            if debug {
                println!(
                    "[DEBUG] build search session client failed, fallback shared client: {err}"
                );
            }
            client.clone()
        }
    };
    warmup_bing_session(&search_client, debug).await;

    let pass1_hits = match search_bing(&search_client, query, debug).await {
        Ok(v) => v,
        Err(err) => {
            if debug {
                println!("[DEBUG] search pass1 failed query={query}: {err}");
            }
            Vec::new()
        }
    };
    if debug {
        println!("[DEBUG] search.pass1 hits={}", pass1_hits.len());
    }

    let rewritten_query = rewrite_query_for_second_pass(query, &pass1_hits);
    if debug {
        println!(
            "[DEBUG] search.pass2 rewrite query={} -> {}",
            query, rewritten_query
        );
    }

    let variants = build_query_variants(&rewritten_query);
    if debug && variants.len() > 1 {
        println!("[DEBUG] search variants: {}", variants.join(" | "));
    }
    let strict_identifier = extract_identifier_token(query)
        .filter(|token| should_enable_identifier_filter(query, token));
    if debug {
        if let Some(token) = &strict_identifier {
            println!("[DEBUG] search strict identifier filter enabled token={token}");
        }
    }
    let mut strict_identifier_unresolved = false;

    let mut bing_hits = Vec::new();
    for variant in &variants {
        let before = bing_hits.len();
        match search_bing(&search_client, variant, debug).await {
            Ok(mut found) => bing_hits.append(&mut found),
            Err(err) => {
                if debug {
                    println!("[DEBUG] search.bing failed query={variant}: {err}");
                }
            }
        }
        if debug {
            println!(
                "[DEBUG] search.bing variant_done query={} added={} total={}",
                variant,
                bing_hits.len().saturating_sub(before),
                bing_hits.len()
            );
        }
    }
    if bing_hits.is_empty() && !pass1_hits.is_empty() {
        if debug {
            println!("[DEBUG] search pass2 empty, fallback to pass1 results");
        }
        bing_hits = pass1_hits.clone();
    } else {
        bing_hits.extend(pass1_hits.clone());
    }

    if let Some(token) = strict_identifier.as_ref() {
        let mut matched = bing_hits
            .iter()
            .filter(|hit| hit_matches_identifier(hit, token))
            .count();

        if matched == 0 {
            if debug {
                println!(
                    "[DEBUG] bing strict identifier matched 0 hits, recovery by extra bing queries token={token}"
                );
            }

            let mut recovery_queries = Vec::new();
            push_query_variant(&mut recovery_queries, &format!("+{token}"));
            push_query_variant(&mut recovery_queries, &format!("+\"{token}\""));
            if token.contains('_') {
                let alt = token.replace('_', "-");
                push_query_variant(&mut recovery_queries, &format!("+{alt}"));
                push_query_variant(&mut recovery_queries, &format!("+\"{alt}\""));
            }
            if token.contains('-') {
                let alt = token.replace('-', "_");
                push_query_variant(&mut recovery_queries, &format!("+{alt}"));
                push_query_variant(&mut recovery_queries, &format!("+\"{alt}\""));
            }

            for q in recovery_queries {
                match search_bing(&search_client, &q, debug).await {
                    Ok(mut found) => bing_hits.append(&mut found),
                    Err(err) => {
                        if debug {
                            println!("[DEBUG] recovery bing failed query={q}: {err}");
                        }
                    }
                }
            }

            matched = bing_hits
                .iter()
                .filter(|hit| hit_matches_identifier(hit, token))
                .count();
            if debug {
                println!("[DEBUG] bing strict identifier recovery matches={matched}");
            }
            if matched == 0 {
                strict_identifier_unresolved = true;
            }
        }
    }

    if debug {
        println!("[DEBUG] search.raw total_hits={}", bing_hits.len());
        debug_log_hits("search.raw", query, &bing_hits, 10);
    }
    if debug {
        if let Some(token) = &strict_identifier {
            let matched = bing_hits
                .iter()
                .filter(|hit| hit_matches_identifier(hit, token))
                .count();
            println!(
                "[DEBUG] search strict identifier matches={}/{} token={}",
                matched,
                bing_hits.len(),
                token
            );
        }
    }

    let raw_hits = bing_hits.clone();
    let mut ranked = rank_search_hits(
        query,
        &variants,
        bing_hits.clone(),
        strict_identifier.as_deref(),
    );
    if ranked.is_empty() && strict_identifier.is_some() {
        if debug {
            println!(
                "[DEBUG] strict identifier filter produced 0 hits, fallback to relaxed ranking"
            );
        }
        ranked = rank_search_hits(query, &variants, bing_hits.clone(), None);
    }

    if ranked.is_empty() {
        if debug {
            println!("[DEBUG] search.ranked empty after fallback");
            debug_log_hits("search.ranked.empty.raw", query, &raw_hits, 10);
        }

        let intent_terms = extract_intent_terms(query);
        let strict_token = strict_identifier.as_deref();
        let constrained_hits = take_unique_hits_preserve_order(
            raw_hits
                .iter()
                .filter(|hit| {
                    if let Some(token) = strict_token {
                        if !hit_matches_identifier(hit, token) {
                            return false;
                        }
                    }
                    if !intent_terms.is_empty() && !hit_matches_intent_terms(hit, &intent_terms) {
                        return false;
                    }
                    true
                })
                .cloned()
                .collect(),
            5,
        );

        if strict_identifier_unresolved {
            if let Some(token) = &strict_identifier {
                let out = format!(
                    "未在 Bing 中国检索到包含标识符 `{token}` 的有效结果。\n建议：\n1) 直接附上目标主页链接\n2) 改搜 `+{token}` 或 `+\"{token}\"`\n3) 检查是否存在大小写/下划线/连字符差异"
                );
                cache_search_result(&cache_key, &out);
                return Ok(out);
            }
        }

        if constrained_hits.is_empty() {
            let out = format!(
                "未在 Bing 中国检索到与 `{query}` 明确相关的结果（当前结果相关性过低，已丢弃）。\n建议：\n1) 给更具体关键词（如地名/平台/全名）\n2) 直接提供目标链接\n3) 用引号精确检索，例如 `\"{query}\"`"
            );
            cache_search_result(&cache_key, &out);
            return Ok(out);
        }

        if debug {
            println!(
                "[DEBUG] search fallback to constrained hits={}",
                constrained_hits.len()
            );
        }

        let mut out = format!("Web 搜索结果（query: {query}，按关键词强匹配回退）:\n");
        for (idx, hit) in constrained_hits.iter().enumerate() {
            if hit.snippet.is_empty() {
                out.push_str(&format!(
                    "{}. [{}] {} - {}\n",
                    idx + 1,
                    hit.source,
                    hit.title,
                    hit.url
                ));
            } else {
                out.push_str(&format!(
                    "{}. [{}] {} - {} - {}\n",
                    idx + 1,
                    hit.source,
                    hit.title,
                    hit.url,
                    hit.snippet
                ));
            }
        }
        let out = out.trim().to_string();
        cache_search_result(&cache_key, &out);
        return Ok(out);
    }

    let top_hits: Vec<SearchHit> = ranked.into_iter().take(5).collect();
    if debug {
        let snapshot = top_hits
            .iter()
            .map(|h| format!("[{}] {}", h.source, h.url))
            .collect::<Vec<_>>()
            .join(" | ");
        println!("[DEBUG] search.top {}", snapshot);
    }

    let mut out = format!("Web 搜索结果（query: {query}）:\n");
    for (idx, hit) in top_hits.iter().enumerate() {
        if hit.snippet.is_empty() {
            out.push_str(&format!(
                "{}. [{}] {} - {}\n",
                idx + 1,
                hit.source,
                hit.title,
                hit.url
            ));
        } else {
            out.push_str(&format!(
                "{}. [{}] {} - {} - {}\n",
                idx + 1,
                hit.source,
                hit.title,
                hit.url,
                hit.snippet
            ));
        }
    }

    let mut preview_count = 0usize;
    for hit in top_hits.iter().take(3) {
        if preview_count >= 2 {
            break;
        }

        match fetch_search_preview(client, &hit.url).await {
            Ok(preview) => {
                if preview_count == 0 {
                    out.push_str("\n网页核验预览（自动抓取）:\n");
                }
                out.push_str(&format!(
                    "{}. {} - {}\n{}\n",
                    preview_count + 1,
                    hit.title,
                    hit.url,
                    preview
                ));
                preview_count += 1;
            }
            Err(err) => {
                if debug {
                    println!("[DEBUG] search.preview failed url={}: {err}", hit.url);
                }
            }
        }
    }

    if preview_count == 0 {
        out.push_str("\n建议：继续调用 fetch_url 读取前 1-2 条结果后再下结论。");
    } else {
        out.push_str("\n建议：优先基于“网页核验预览”回答，不足时再 fetch_url 深读。");
    }

    let out = out.trim().to_string();
    cache_search_result(&cache_key, &out);
    Ok(out)
}

pub async fn fetch_url(client: &Client, url: &str, debug: bool) -> Result<String> {
    let url = normalize_url_input(url);
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("invalid url (must start with http:// or https://): {url}");
    }

    if debug {
        println!("[DEBUG] tool.fetch_url url={url}");
    }

    let response = client
        .get(&url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("fetch request failed: {url}"))?;
    let status = response.status();
    let final_url = response.url().to_string();
    let headers = response.headers().clone();
    let body = response
        .text()
        .await
        .context("failed to read fetched response body")?;

    if !status.is_success() {
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

fn search_result_cache() -> &'static DashMap<String, CachedSearchResult> {
    SEARCH_RESULT_CACHE.get_or_init(DashMap::new)
}

fn search_cache_key_query(query: &str) -> String {
    let normalized = normalize_search_query(query);
    if normalized.is_empty() {
        query.trim().to_lowercase()
    } else {
        normalized.to_lowercase()
    }
}

fn search_cache_key_with_provider(query: &str, search: &SearchConfig) -> String {
    let base = search_cache_key_query(query);
    match search.provider {
        SearchProvider::Builtin => format!("builtin:{base}"),
        SearchProvider::Searxng => {
            let host = normalize_searxng_cache_key(&search.searxng_url);
            format!("searxng:{host}:{base}")
        }
    }
}

fn load_cached_search_result(cache_key: &str) -> Option<String> {
    let cache = search_result_cache();
    if let Some(entry) = cache.get(cache_key) {
        if entry.cached_at.elapsed() <= Duration::from_secs(SEARCH_CACHE_TTL_SECS) {
            return Some(entry.body.clone());
        }
    }
    cache.remove(cache_key);
    None
}

fn cache_search_result(cache_key: &str, body: &str) {
    if cache_key.is_empty() || body.is_empty() {
        return;
    }
    search_result_cache().insert(
        cache_key.to_string(),
        CachedSearchResult {
            body: body.to_string(),
            cached_at: Instant::now(),
        },
    );
}

fn build_search_session_client(network: &NetworkConfig) -> Result<Client> {
    crate::tools::http::build_client(20_000, network, true)
        .context("failed to build search session client")
}

async fn warmup_bing_session(client: &Client, debug: bool) {
    let warmup_url = "https://cn.bing.com/";
    match client
        .get(warmup_url)
        .header("User-Agent", DEFAULT_UA)
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
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
    {
        Ok(resp) => {
            if debug {
                println!(
                    "[DEBUG] search.warmup url={} status={}",
                    warmup_url,
                    resp.status()
                );
            }
        }
        Err(err) => {
            if debug {
                println!("[DEBUG] search.warmup failed url={}: {}", warmup_url, err);
            }
        }
    }
}

async fn search_web_searxng(
    client: &Client,
    query: &str,
    debug: bool,
    search: &SearchConfig,
    cache_key: &str,
) -> Result<String> {
    let normalized = normalize_search_query(query);
    let q = if normalized.is_empty() {
        query.trim()
    } else {
        normalized.as_str()
    };

    let hits = search_searxng(client, &search.searxng_url, q, debug).await?;
    if debug {
        println!("[DEBUG] search.searxng raw_hits={}", hits.len());
        debug_log_hits("search.searxng", q, &hits, 8);
    }

    if hits.is_empty() {
        let out = format!("未在 SearXNG 检索到结果。query={q}");
        cache_search_result(cache_key, &out);
        return Ok(out);
    }

    // SearXNG：不做清洗，保持原始顺序，仅去重。
    let top_hits = take_unique_hits_preserve_order(hits, 5);
    let mut out = format!("Web 搜索结果（provider: searxng, query: {q}）:\n");
    for (idx, hit) in top_hits.iter().enumerate() {
        if hit.snippet.is_empty() {
            out.push_str(&format!(
                "{}. [{}] {} - {}\n",
                idx + 1,
                hit.source,
                hit.title,
                hit.url
            ));
        } else {
            out.push_str(&format!(
                "{}. [{}] {} - {} - {}\n",
                idx + 1,
                hit.source,
                hit.title,
                hit.url,
                hit.snippet
            ));
        }
    }

    let out = out.trim().to_string();
    cache_search_result(cache_key, &out);
    Ok(out)
}

fn normalize_searxng_cache_key(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_lowercase()
}

fn rewrite_query_for_second_pass(query: &str, _pass1_hits: &[SearchHit]) -> String {
    let normalized = normalize_search_query(query);
    let base = if normalized.is_empty() {
        query.trim().to_string()
    } else {
        normalized
    };

    if is_identifier_query(&base) {
        if let Some(token) = extract_identifier_token(&base) {
            let pass1_has_identifier = _pass1_hits
                .iter()
                .any(|hit| hit_matches_identifier(hit, &token));
            if !pass1_has_identifier {
                return format!("\"{token}\"");
            }
        }
    }

    base
}

async fn search_bing(client: &Client, query: &str, debug: bool) -> Result<Vec<SearchHit>> {
    let encoded = urlencoding::encode(query);
    let mut query_params = format!("q={encoded}&setlang=zh-Hans&cc=CN");
    if prefers_english_search(query) {
        query_params.push_str("&ensearch=1");
    } else {
        query_params.push_str("&ensearch=0");
    }
    let search_url = format!("https://cn.bing.com/search?{query_params}");
    let accept_language = if prefers_english_search(query) {
        "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"
    } else {
        "zh-CN,zh;q=0.9,en;q=0.8"
    };
    let response = client
        .get(&search_url)
        .header("User-Agent", DEFAULT_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", accept_language)
        .header("Referer", "https://cn.bing.com/")
        .header("Sec-CH-UA", BROWSER_SEC_CH_UA)
        .header("Sec-CH-UA-Mobile", "?0")
        .header("Sec-CH-UA-Platform", "\"macOS\"")
        .header("Sec-Fetch-Dest", "document")
        .header("Sec-Fetch-Mode", "navigate")
        .header("Sec-Fetch-Site", "same-origin")
        .header("Sec-Fetch-User", "?1")
        .header("Upgrade-Insecure-Requests", "1")
        .send()
        .await
        .with_context(|| format!("search request failed: {search_url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read search response body")?;

    if !status.is_success() {
        bail!("search endpoint returned {status}: {body}");
    }

    let raw_hits = parse_bing_results(&body)?;
    let hits = apply_operator_constraints(query, raw_hits);
    if debug {
        println!(
            "[DEBUG] search.bing query={query} url={search_url} hits={}",
            hits.len()
        );
        debug_log_hits("search.bing", query, &hits, 8);
    }
    Ok(hits)
}

async fn search_searxng(
    client: &Client,
    base_url: &str,
    query: &str,
    debug: bool,
) -> Result<Vec<SearchHit>> {
    let json_attempt = search_searxng_json(client, base_url, query, debug).await;
    match json_attempt {
        Ok(hits) if !hits.is_empty() => return Ok(hits),
        Ok(_) => {
            if debug {
                println!("[DEBUG] search.searxng json empty, fallback html");
            }
        }
        Err(err) => {
            if debug {
                println!("[DEBUG] search.searxng json failed, fallback html: {err}");
            }
        }
    }

    let html_hits = search_searxng_html(client, base_url, query, debug).await?;
    Ok(html_hits)
}

async fn search_searxng_json(
    client: &Client,
    base_url: &str,
    query: &str,
    debug: bool,
) -> Result<Vec<SearchHit>> {
    let base = base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        bail!("searxng_url is empty");
    }
    let encoded = urlencoding::encode(query);
    let search_url =
        format!("{base}/search?q={encoded}&format=json&language=zh-CN&categories=general");

    let response = client
        .get(&search_url)
        .header("User-Agent", DEFAULT_UA)
        .header("Accept", "application/json")
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .send()
        .await
        .with_context(|| format!("searxng request failed: {search_url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read searxng response body")?;

    if !status.is_success() {
        bail!("searxng endpoint returned {status}: {body}");
    }

    let value: Value =
        serde_json::from_str(&body).context("failed to parse searxng response JSON")?;
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut out = Vec::new();
    for item in results.into_iter().take(12) {
        let title = item
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let url = item
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if title.is_empty() || url.is_empty() {
            continue;
        }
        let snippet = item
            .get("content")
            .or_else(|| item.get("snippet"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        out.push(SearchHit {
            title,
            url,
            snippet,
            source: "searxng",
        });
    }

    if debug {
        println!(
            "[DEBUG] search.searxng json query={} url={} hits={}",
            query,
            search_url,
            out.len()
        );
    }

    Ok(out)
}

async fn search_searxng_html(
    client: &Client,
    base_url: &str,
    query: &str,
    debug: bool,
) -> Result<Vec<SearchHit>> {
    let base = base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        bail!("searxng_url is empty");
    }
    let encoded = urlencoding::encode(query);
    let search_url = format!("{base}/search?q={encoded}");

    let response = client
        .get(&search_url)
        .header("User-Agent", DEFAULT_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .send()
        .await
        .with_context(|| format!("searxng html request failed: {search_url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read searxng html response body")?;

    if !status.is_success() {
        bail!("searxng html endpoint returned {status}: {body}");
    }

    let hits = parse_searxng_results(&body)?;
    if debug {
        println!(
            "[DEBUG] search.searxng html query={} url={} hits={}",
            query,
            search_url,
            hits.len()
        );
    }
    Ok(hits)
}

#[allow(dead_code)]
async fn search_domestic_fallback(
    client: &Client,
    variants: &[String],
    debug: bool,
) -> Vec<SearchHit> {
    let mut out = Vec::new();
    let fallback_variants: Vec<&String> = variants.iter().take(3).collect();

    for variant in fallback_variants {
        match search_baidu(client, variant, debug).await {
            Ok(mut hits) => out.append(&mut hits),
            Err(err) => {
                if debug {
                    println!("[DEBUG] search.baidu failed query={variant}: {err}");
                }
            }
        }
        match search_sogou(client, variant, debug).await {
            Ok(mut hits) => out.append(&mut hits),
            Err(err) => {
                if debug {
                    println!("[DEBUG] search.sogou failed query={variant}: {err}");
                }
            }
        }
        match search_so360(client, variant, debug).await {
            Ok(mut hits) => out.append(&mut hits),
            Err(err) => {
                if debug {
                    println!("[DEBUG] search.so failed query={variant}: {err}");
                }
            }
        }
    }

    out
}

#[allow(dead_code)]
async fn search_baidu(client: &Client, query: &str, debug: bool) -> Result<Vec<SearchHit>> {
    let encoded = urlencoding::encode(query);
    let search_url = format!("https://www.baidu.com/s?wd={encoded}");
    let response = client
        .get(&search_url)
        .header("User-Agent", DEFAULT_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .header("Referer", "https://www.baidu.com/")
        .send()
        .await
        .with_context(|| format!("search request failed: {search_url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read baidu search response body")?;
    if !status.is_success() {
        bail!("search endpoint returned {status}: {body}");
    }

    let hits = parse_baidu_results(&body)?;
    if debug {
        println!(
            "[DEBUG] search.baidu query={query} url={search_url} hits={}",
            hits.len()
        );
        debug_log_hits("search.baidu", query, &hits, 8);
    }
    Ok(hits)
}

#[allow(dead_code)]
async fn search_sogou(client: &Client, query: &str, debug: bool) -> Result<Vec<SearchHit>> {
    let encoded = urlencoding::encode(query);
    let search_url = format!("https://www.sogou.com/web?query={encoded}");
    let response = client
        .get(&search_url)
        .header("User-Agent", DEFAULT_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .header("Referer", "https://www.sogou.com/")
        .send()
        .await
        .with_context(|| format!("search request failed: {search_url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read sogou search response body")?;
    if !status.is_success() {
        bail!("search endpoint returned {status}: {body}");
    }

    let hits = parse_sogou_results(&body)?;
    if debug {
        println!(
            "[DEBUG] search.sogou query={query} url={search_url} hits={}",
            hits.len()
        );
        debug_log_hits("search.sogou", query, &hits, 8);
    }
    Ok(hits)
}

#[allow(dead_code)]
async fn search_so360(client: &Client, query: &str, debug: bool) -> Result<Vec<SearchHit>> {
    let encoded = urlencoding::encode(query);
    let search_url = format!("https://www.so.com/s?q={encoded}");
    let response = client
        .get(&search_url)
        .header("User-Agent", DEFAULT_UA)
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .header("Referer", "https://www.so.com/")
        .send()
        .await
        .with_context(|| format!("search request failed: {search_url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read so.com search response body")?;
    if !status.is_success() {
        bail!("search endpoint returned {status}: {body}");
    }

    let hits = parse_so360_results(&body)?;
    if debug {
        println!(
            "[DEBUG] search.so query={query} url={search_url} hits={}",
            hits.len()
        );
        debug_log_hits("search.so", query, &hits, 8);
    }
    Ok(hits)
}

#[allow(dead_code)]
async fn search_duckduckgo_html(
    client: &Client,
    query: &str,
    debug: bool,
) -> Result<Vec<SearchHit>> {
    let encoded = urlencoding::encode(query);
    let url = format!("https://duckduckgo.com/html/?q={encoded}&kl=cn-zh");

    let response = client
        .get(&url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("duckduckgo html request failed: {url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read duckduckgo html response body")?;

    if !status.is_success() {
        return Ok(Vec::new());
    }

    let hits = parse_duckduckgo_html_results(&body)?;
    if debug {
        println!("[DEBUG] search.ddg_html hits={}", hits.len());
    }
    Ok(hits)
}

#[allow(dead_code)]
async fn search_duckduckgo_api(client: &Client, query: &str) -> Result<Vec<SearchHit>> {
    let encoded = urlencoding::encode(query);
    let url = format!(
        "https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&no_redirect=1&skip_disambig=1"
    );

    let response = client
        .get(&url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("duckduckgo api request failed: {url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read duckduckgo api response body")?;

    if !status.is_success() {
        return Ok(Vec::new());
    }

    let value: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return Ok(Vec::new()),
    };

    let mut out = Vec::new();
    if let Some(abstract_text) = value.get("AbstractText").and_then(Value::as_str) {
        if !abstract_text.trim().is_empty() {
            let source = value
                .get("AbstractSource")
                .and_then(Value::as_str)
                .unwrap_or("duckduckgo_api");
            let link = value
                .get("AbstractURL")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if !link.is_empty() {
                out.push(SearchHit {
                    title: source.to_string(),
                    url: link,
                    snippet: normalize_whitespace(abstract_text),
                    source: "ddg_api",
                });
            }
        }
    }

    if let Some(related) = value.get("RelatedTopics").and_then(Value::as_array) {
        for item in related.iter().take(8) {
            let Some(text) = item.get("Text").and_then(Value::as_str) else {
                continue;
            };
            let first_url = item
                .get("FirstURL")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if first_url.is_empty() {
                continue;
            }
            out.push(SearchHit {
                title: "DuckDuckGo".to_string(),
                url: first_url.to_string(),
                snippet: normalize_whitespace(text),
                source: "ddg_api",
            });
        }
    }

    Ok(out)
}

async fn fetch_search_preview(client: &Client, url: &str) -> Result<String> {
    let url = normalize_url_input(url);
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("invalid preview url: {url}");
    }

    let response = client
        .get(&url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("preview request failed: {url}"))?;
    let status = response.status();
    let final_url = response.url().to_string();
    let headers = response.headers().clone();
    let body = response
        .text()
        .await
        .context("failed to read preview body")?;

    if !status.is_success() {
        let brief = truncate_text(&normalize_whitespace(&body), 220);
        bail!("preview returned {status}: {brief}");
    }

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let redirect = redirect_note(&url, &final_url);

    if content_type.contains("text/html") || body.contains("<html") {
        let (title, text) = extract_html_text(&body)?;
        let title = if title.trim().is_empty() {
            "(no title)".to_string()
        } else {
            title
        };
        let text = truncate_text(&normalize_whitespace(&text), MAX_SEARCH_PREVIEW_CHARS);
        return Ok(format!(
            "URL Final: {final_url}\n{redirect}Title: {title}\nContent: {text}"
        ));
    }

    let text = truncate_text(&normalize_whitespace(&body), MAX_SEARCH_PREVIEW_CHARS);
    Ok(format!("URL Final: {final_url}\n{redirect}Content: {text}"))
}

fn parse_bing_results(html: &str) -> Result<Vec<SearchHit>> {
    let doc = Html::parse_document(html);
    let item_sel = Selector::parse("li.b_algo")
        .map_err(|err| anyhow!("failed to parse selector li.b_algo: {err}"))?;
    let title_sel =
        Selector::parse("h2 a").map_err(|err| anyhow!("failed to parse selector h2 a: {err}"))?;
    let snippet_sel = Selector::parse("div.b_caption p, p")
        .map_err(|err| anyhow!("failed to parse selector caption p: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&item_sel).take(12) {
        let Some(title_node) = item.select(&title_sel).next() else {
            continue;
        };

        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let raw_href = title_node
            .value()
            .attr("href")
            .map(decode_html_entities_basic)
            .unwrap_or_default();
        let href = normalize_bing_href(&raw_href);
        if href.is_empty() || !is_valid_result_url(&href) {
            continue;
        }

        let snippet = item
            .select(&snippet_sel)
            .next()
            .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        out.push(SearchHit {
            title,
            url: href,
            snippet,
            source: "bing",
        });
    }

    Ok(out)
}

fn parse_searxng_results(html: &str) -> Result<Vec<SearchHit>> {
    let doc = Html::parse_document(html);
    let item_sel = Selector::parse("article.result, div.result, li.result")
        .map_err(|err| anyhow!("failed to parse selector searxng item: {err}"))?;
    let title_sel = Selector::parse("h3 a, h4 a, a")
        .map_err(|err| anyhow!("failed to parse selector searxng title: {err}"))?;
    let snippet_sel = Selector::parse(".result__snippet, .result-content, p")
        .map_err(|err| anyhow!("failed to parse selector searxng snippet: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&item_sel).take(12) {
        let Some(title_node) = item.select(&title_sel).next() else {
            continue;
        };
        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let raw_href = title_node
            .value()
            .attr("href")
            .map(decode_html_entities_basic)
            .unwrap_or_default();
        let href = normalize_generic_href(&raw_href);
        if href.is_empty() || !is_valid_result_url(&href) {
            continue;
        }

        let snippet = item
            .select(&snippet_sel)
            .next()
            .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        out.push(SearchHit {
            title,
            url: href,
            snippet,
            source: "searxng",
        });
    }

    Ok(out)
}

#[allow(dead_code)]
fn parse_baidu_results(html: &str) -> Result<Vec<SearchHit>> {
    let doc = Html::parse_document(html);
    let item_sel = Selector::parse("div.result, div.c-container")
        .map_err(|err| anyhow!("failed to parse selector baidu item: {err}"))?;
    let title_sel = Selector::parse("h3 a, a[data-click]")
        .map_err(|err| anyhow!("failed to parse selector baidu title: {err}"))?;
    let snippet_sel = Selector::parse(
        "div.c-abstract, div.content-right_8Zs40, div.c-span-last, div.c-font-normal, span.content-right_8Zs40",
    )
    .map_err(|err| anyhow!("failed to parse selector baidu snippet: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&item_sel).take(14) {
        let Some(title_node) = item.select(&title_sel).next() else {
            continue;
        };

        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let raw_href = title_node
            .value()
            .attr("href")
            .map(decode_html_entities_basic)
            .unwrap_or_default();
        let href = normalize_generic_href(&raw_href);
        if href.is_empty() || !is_valid_result_url(&href) {
            continue;
        }

        let snippet = item
            .select(&snippet_sel)
            .next()
            .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        out.push(SearchHit {
            title,
            url: href,
            snippet,
            source: "baidu",
        });
    }

    Ok(out)
}

#[allow(dead_code)]
fn parse_sogou_results(html: &str) -> Result<Vec<SearchHit>> {
    let doc = Html::parse_document(html);
    let item_sel = Selector::parse("div.vrwrap, div.rb, li[class*=\"vr\"]")
        .map_err(|err| anyhow!("failed to parse selector sogou item: {err}"))?;
    let title_sel = Selector::parse("h3 a, h2 a, a.vr-title")
        .map_err(|err| anyhow!("failed to parse selector sogou title: {err}"))?;
    let snippet_sel = Selector::parse("p.str_info, p.str-text-info, div.text-layout, p")
        .map_err(|err| anyhow!("failed to parse selector sogou snippet: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&item_sel).take(14) {
        let Some(title_node) = item.select(&title_sel).next() else {
            continue;
        };

        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let raw_href = title_node
            .value()
            .attr("href")
            .map(decode_html_entities_basic)
            .unwrap_or_default();
        let href = normalize_generic_href(&raw_href);
        if href.is_empty() || !is_valid_result_url(&href) {
            continue;
        }

        let snippet = item
            .select(&snippet_sel)
            .next()
            .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        out.push(SearchHit {
            title,
            url: href,
            snippet,
            source: "sogou",
        });
    }

    Ok(out)
}

#[allow(dead_code)]
fn parse_so360_results(html: &str) -> Result<Vec<SearchHit>> {
    let doc = Html::parse_document(html);
    let item_sel = Selector::parse("li.res-list, li.result, div.res-list")
        .map_err(|err| anyhow!("failed to parse selector so360 item: {err}"))?;
    let title_sel = Selector::parse("h3 a, h2 a")
        .map_err(|err| anyhow!("failed to parse selector so360 title: {err}"))?;
    let snippet_sel = Selector::parse("p.res-desc, p, div.res-desc")
        .map_err(|err| anyhow!("failed to parse selector so360 snippet: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&item_sel).take(14) {
        let Some(title_node) = item.select(&title_sel).next() else {
            continue;
        };

        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let raw_href = title_node
            .value()
            .attr("href")
            .map(decode_html_entities_basic)
            .unwrap_or_default();
        let href = normalize_generic_href(&raw_href);
        if href.is_empty() || !is_valid_result_url(&href) {
            continue;
        }

        let snippet = item
            .select(&snippet_sel)
            .next()
            .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        out.push(SearchHit {
            title,
            url: href,
            snippet,
            source: "so360",
        });
    }

    Ok(out)
}

#[allow(dead_code)]
fn parse_duckduckgo_html_results(html: &str) -> Result<Vec<SearchHit>> {
    let doc = Html::parse_document(html);
    let result_sel = Selector::parse("div.result, .web-result")
        .map_err(|err| anyhow!("failed to parse selector ddg result: {err}"))?;
    let title_sel = Selector::parse("a.result__a, h2 a, a")
        .map_err(|err| anyhow!("failed to parse selector ddg title: {err}"))?;
    let snippet_sel = Selector::parse(".result__snippet, .result__body, .result-snippet")
        .map_err(|err| anyhow!("failed to parse selector ddg snippet: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&result_sel).take(12) {
        let Some(title_node) = item.select(&title_sel).next() else {
            continue;
        };

        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let raw_href = title_node.value().attr("href").unwrap_or("");
        let href = normalize_duckduckgo_href(raw_href);
        if href.is_empty() || !is_valid_result_url(&href) {
            continue;
        }

        let snippet = item
            .select(&snippet_sel)
            .next()
            .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        out.push(SearchHit {
            title,
            url: href,
            snippet,
            source: "ddg_html",
        });
    }

    Ok(out)
}

fn apply_operator_constraints(query: &str, hits: Vec<SearchHit>) -> Vec<SearchHit> {
    let site = extract_site_constraint(query).map(|v| v.to_ascii_lowercase());
    let inurl = extract_inurl_constraint(query).map(|v| v.to_ascii_lowercase());

    if site.is_none() && inurl.is_none() {
        return hits;
    }

    hits.into_iter()
        .filter(|hit| {
            if let Some(site_host) = &site {
                let Some(host) = host_of(&hit.url) else {
                    return false;
                };
                let host = host.to_ascii_lowercase();
                if !(host == *site_host || host.ends_with(&format!(".{site_host}"))) {
                    return false;
                }
            }

            if let Some(inurl_token) = &inurl {
                if !hit.url.to_ascii_lowercase().contains(inurl_token) {
                    return false;
                }
            }

            true
        })
        .collect()
}

fn extract_site_constraint(query: &str) -> Option<String> {
    let lower = query.to_ascii_lowercase();
    let pos = lower.find("site:")?;
    let remain = &query[pos + 5..];
    let raw = remain
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| c == '"' || c == '\'' || c == ',');
    if raw.is_empty() {
        return None;
    }
    Some(raw.trim_start_matches("www.").to_string())
}

fn extract_inurl_constraint(query: &str) -> Option<String> {
    let lower = query.to_ascii_lowercase();
    let pos = lower.find("inurl:")?;
    let remain = &query[pos + 6..];
    let raw = remain
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| c == '"' || c == '\'' || c == ',');
    if raw.is_empty() {
        return None;
    }
    Some(raw.to_string())
}

fn rank_search_hits(
    original_query: &str,
    query_variants: &[String],
    hits: Vec<SearchHit>,
    strict_identifier: Option<&str>,
) -> Vec<SearchHit> {
    let terms = build_query_terms(query_variants);
    let intent_terms = extract_intent_terms(original_query);
    let core_terms = extract_core_terms(original_query);
    let core_terms_active = !core_terms.is_empty()
        && hits
            .iter()
            .any(|hit| hit_matches_core_terms(hit, &core_terms));
    if hits.is_empty() {
        return Vec::new();
    }

    let mut seen = HashSet::new();
    let mut scored = Vec::new();

    for hit in hits {
        if let Some(token) = strict_identifier {
            if !hit_matches_identifier(&hit, token) {
                continue;
            }
        }
        if core_terms_active && !hit_matches_core_terms(&hit, &core_terms) {
            continue;
        }
        if !intent_terms.is_empty() && !hit_matches_intent_terms(&hit, &intent_terms) {
            continue;
        }

        let canonical = canonical_url(&hit.url);
        if canonical.is_empty() || !seen.insert(canonical) {
            continue;
        }

        let score = score_hit(&hit, query_variants, &terms, &intent_terms);
        scored.push((score, hit));
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));

    scored
        .into_iter()
        .filter(|(score, _)| *score > 0)
        .map(|(_, hit)| hit)
        .collect()
}

fn take_unique_hits_preserve_order(hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for hit in hits {
        let canonical = canonical_url(&hit.url);
        if canonical.is_empty() || !seen.insert(canonical) {
            continue;
        }
        out.push(hit);
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn build_query_variants(query: &str) -> Vec<String> {
    let mut variants = Vec::new();

    let normalized = normalize_search_query(query);
    let base = if normalized.is_empty() {
        query.trim()
    } else {
        normalized.as_str()
    };
    push_query_variant(&mut variants, base);
    let identifier_mode = is_identifier_query(base);

    if let Some(token) = extract_identifier_token(base) {
        push_query_variant(&mut variants, &format!("\"{token}\""));

        if token.contains('_') {
            let alt = token.replace('_', "-");
            push_query_variant(&mut variants, &alt);
            push_query_variant(&mut variants, &format!("\"{alt}\""));
        }
        if token.contains('-') {
            let alt = token.replace('-', "_");
            push_query_variant(&mut variants, &alt);
            push_query_variant(&mut variants, &format!("\"{alt}\""));
        }
    }

    if !identifier_mode {
        if !contains_cjk(base) {
            push_plus_variants(&mut variants, base);
        }
    }

    if normalized != query.trim() {
        push_query_variant(&mut variants, &normalized);
        if !identifier_mode {
            if !contains_cjk(&normalized) {
                push_plus_variants(&mut variants, &normalized);
            }
        }
    }

    if query_needs_recency(query) {
        let news_variant = format!("{base} 新闻 最新");
        push_query_variant(&mut variants, &news_variant);
        if !identifier_mode {
            if !contains_cjk(&news_variant) {
                push_plus_variants(&mut variants, &news_variant);
            }
        }
    }

    variants
}

fn push_query_variant(variants: &mut Vec<String>, raw: &str) {
    let q = raw.trim();
    if q.is_empty() {
        return;
    }
    if variants.iter().any(|v| v == q) {
        return;
    }
    variants.push(q.to_string());
}

fn push_plus_variants(variants: &mut Vec<String>, query: &str) {
    let plus = plusify_query_terms(query);
    if plus.is_empty() {
        return;
    }
    push_query_variant(variants, &plus);

    if query.split_whitespace().count() == 1 {
        let quoted = format!("+\"{}\"", query.trim());
        push_query_variant(variants, &quoted);
    }
}

fn plusify_query_terms(query: &str) -> String {
    let mut out = Vec::new();
    for raw in query.split_whitespace() {
        let term = raw.trim().trim_matches(|c: char| c == '"' || c == '\'');
        if term.is_empty() {
            continue;
        }
        if term.starts_with('+') {
            out.push(term.to_string());
        } else {
            out.push(format!("+{term}"));
        }
    }
    out.join(" ")
}

fn normalize_search_query(query: &str) -> String {
    let cleaned = query.replace(
        [
            '，', ',', '。', '.', '；', ';', '：', ':', '？', '?', '！', '!',
        ],
        " ",
    );

    let mut terms = Vec::new();
    for raw in cleaned.split_whitespace() {
        let mut token = raw.trim().to_string();
        for prefix in [
            "搜下",
            "搜一下",
            "搜索",
            "查下",
            "查一下",
            "帮我搜",
            "帮我查",
            "你搜",
            "你查",
            "请搜",
            "请查",
            "看下",
            "看看",
        ] {
            if let Some(stripped) = token.strip_prefix(prefix) {
                token = stripped.trim().to_string();
            }
        }
        let token = strip_query_noise(&token);
        if token.chars().count() >= 2 {
            terms.push(token);
        }
    }

    if terms.is_empty() {
        String::new()
    } else {
        terms.join(" ")
    }
}

fn strip_query_noise(token: &str) -> String {
    let mut text = token.to_string();
    for noise in [
        "最近",
        "最新",
        "刚刚",
        "近期",
        "今天",
        "昨日",
        "发生什么",
        "发生啥",
        "是什么",
        "什么",
        "一下",
        "请问",
        "新闻",
        "动态",
        "消息",
        "情况",
        "事件",
        "吗",
        "呢",
        "吧",
    ] {
        text = text.replace(noise, " ");
    }
    normalize_whitespace(&text)
}

fn query_needs_recency(query: &str) -> bool {
    let q = query.to_lowercase();
    [
        "最近",
        "最新",
        "近期",
        "刚刚",
        "今天",
        "新闻",
        "事件",
        "发生什么",
        "发生啥",
        "recent",
        "latest",
        "news",
        "what happened",
    ]
    .iter()
    .any(|k| q.contains(k))
}

fn prefers_english_search(query: &str) -> bool {
    !contains_cjk(query) && extract_identifier_token(query).is_some()
}

fn is_identifier_query(query: &str) -> bool {
    if contains_cjk(query) {
        return false;
    }
    if query.split_whitespace().count() > 3 {
        return false;
    }
    extract_identifier_token(query).is_some()
}

fn extract_identifier_token(query: &str) -> Option<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in query.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '@') {
            current.push(ch);
            continue;
        }

        if !current.is_empty() {
            tokens.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
        .into_iter()
        .filter(|t| t.chars().count() >= 3 && t.chars().any(|c| c.is_ascii_alphabetic()))
        .max_by_key(|t| t.len())
}

fn should_enable_identifier_filter(query: &str, token: &str) -> bool {
    if contains_cjk(query) {
        return false;
    }

    let word_count = query.split_whitespace().count();
    if word_count > 2 {
        return false;
    }

    token.contains('_')
        || token.contains('-')
        || token.contains('@')
        || token.chars().any(|c| c.is_ascii_digit())
}

fn hit_matches_identifier(hit: &SearchHit, token: &str) -> bool {
    let title = hit.title.to_lowercase();
    let url = decode_html_entities_basic(&hit.url).to_lowercase();
    let title_compact = compact_ascii_alnum(&title);
    let url_compact = compact_ascii_alnum(&url);

    let token_l = token.to_lowercase();
    let mut forms = vec![token_l.clone()];
    if token_l.contains('_') {
        forms.push(token_l.replace('_', "-"));
    }
    if token_l.contains('-') {
        forms.push(token_l.replace('-', "_"));
    }
    let compact = compact_ascii_alnum(&token_l);
    if compact.len() >= 3 {
        forms.push(compact);
    }

    forms.into_iter().any(|f| {
        f.len() >= 3
            && (title.contains(&f)
                || url.contains(&f)
                || title_compact.contains(&f)
                || url_compact.contains(&f))
    })
}

fn compact_ascii_alnum(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
}

fn build_query_terms(query_variants: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for variant in query_variants {
        for token in tokenize_query_terms(variant) {
            if seen.insert(token.clone()) {
                out.push(token);
            }
        }
    }

    out
}

fn tokenize_query_terms(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    for term in query.split(|c: char| {
        c.is_whitespace()
            || c == ','
            || c == '，'
            || c == ';'
            || c == '；'
            || c == ':'
            || c == '：'
            || c == '.'
            || c == '。'
    }) {
        let t = term
            .trim()
            .trim_matches(|c: char| c == '"' || c == '\'')
            .trim_start_matches('+')
            .to_lowercase();
        let t_len = t.chars().count();
        if t_len < 2 {
            continue;
        }
        out.push(t.clone());
        if contains_cjk(&t) && t_len >= 4 {
            out.extend(cjk_bigrams(&t));
        }
    }
    out
}

fn extract_intent_terms(query: &str) -> Vec<String> {
    let normalized = normalize_search_query(query);
    let base = if normalized.is_empty() {
        query
    } else {
        normalized.as_str()
    };

    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for term in base.split_whitespace() {
        let token = term
            .trim()
            .trim_matches(|c: char| c == '"' || c == '\'')
            .trim_start_matches('+')
            .to_lowercase();
        if token.chars().count() < 2 {
            continue;
        }
        if is_generic_query_token(&token) {
            continue;
        }
        if seen.insert(token.clone()) {
            out.push(token);
        }
    }

    let cn_keywords = [
        "美食", "小吃", "餐厅", "火锅", "烧烤", "足浴", "洗脚", "按摩", "酒店", "住宿", "租房",
        "攻略", "博客", "官网", "github", "gitee", "天气", "气温", "温度", "新闻", "事件", "教程",
        "评测",
    ];
    let lower_query = query.to_lowercase();
    for kw in cn_keywords {
        if lower_query.contains(kw) && seen.insert(kw.to_string()) {
            out.push(kw.to_string());
        }
    }

    out
}

fn extract_core_terms(query: &str) -> Vec<String> {
    let normalized = normalize_search_query(query);
    let base = if normalized.is_empty() {
        query
    } else {
        normalized.as_str()
    };

    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for term in base.split_whitespace() {
        let token = term
            .trim()
            .trim_matches(|c: char| c == '"' || c == '\'')
            .trim_start_matches('+')
            .to_lowercase();
        if token.chars().count() < 2 {
            continue;
        }
        if is_generic_query_token(&token) {
            continue;
        }
        if seen.insert(token.clone()) {
            out.push(token.clone());
        }
        if contains_cjk(&token) && token.chars().count() >= 4 {
            for bg in cjk_bigrams(&token) {
                if seen.insert(bg.clone()) {
                    out.push(bg);
                }
            }
        }
    }

    out
}

fn hit_matches_core_terms(hit: &SearchHit, core_terms: &[String]) -> bool {
    if core_terms.is_empty() {
        return true;
    }
    let title = hit.title.to_lowercase();
    let url = hit.url.to_lowercase();
    let snippet = hit.snippet.to_lowercase();

    core_terms.iter().any(|term| {
        title.contains(term.as_str())
            || url.contains(term.as_str())
            || (title.is_empty() && snippet.contains(term.as_str()))
    })
}

fn hit_matches_intent_terms(hit: &SearchHit, intent_terms: &[String]) -> bool {
    if intent_terms.is_empty() {
        return true;
    }
    let title = hit.title.to_lowercase();
    let snippet = hit.snippet.to_lowercase();
    let url = hit.url.to_lowercase();
    intent_terms.iter().any(|term| {
        title.contains(term.as_str())
            || snippet.contains(term.as_str())
            || url.contains(term.as_str())
    })
}

fn is_generic_query_token(token: &str) -> bool {
    matches!(
        token,
        "附近"
            | "周边"
            | "那里"
            | "那边"
            | "有没有"
            | "有啥"
            | "有哪些"
            | "推荐"
            | "大全"
            | "完整"
            | "全部"
            | "最新"
            | "最近"
            | "新闻"
            | "事件"
            | "什么"
            | "怎么"
            | "一下"
            | "请问"
            | "帮我"
            | "搜"
            | "查"
            | "look"
            | "up"
            | "search"
    )
}

fn contains_cjk(text: &str) -> bool {
    text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

fn cjk_bigrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..(chars.len() - 1) {
        if out.len() >= 10 {
            break;
        }
        out.push(chars[i..=i + 1].iter().collect::<String>());
    }
    out
}

fn score_hit(
    hit: &SearchHit,
    query_variants: &[String],
    terms: &[String],
    intent_terms: &[String],
) -> i32 {
    let title = hit.title.to_lowercase();
    let snippet = hit.snippet.to_lowercase();
    let url = hit.url.to_lowercase();
    let mut score = 0i32;

    for variant in query_variants {
        let v = variant.trim().to_lowercase();
        if v.chars().count() < 2 {
            continue;
        }
        if title.contains(&v) {
            score += 12;
            continue;
        }
        if snippet.contains(&v) {
            score += 8;
            continue;
        }
        if url.contains(&v) {
            score += 4;
        }
    }

    for term in terms {
        if term.chars().count() < 2 {
            continue;
        }
        if title.contains(term) {
            score += 5;
            continue;
        }
        if snippet.contains(term) {
            score += 3;
            continue;
        }
        if url.contains(term) {
            score += 1;
        }
    }

    if hit.snippet.is_empty() {
        score -= 2;
    }

    if is_low_signal_url(&url) {
        score -= 2;
    }

    if !intent_terms.is_empty() {
        let intent_match_count = intent_terms
            .iter()
            .filter(|term| {
                title.contains(term.as_str())
                    || snippet.contains(term.as_str())
                    || url.contains(term.as_str())
            })
            .count();
        if intent_match_count == 0 {
            return -100;
        }
        score += (intent_match_count as i32) * 6;
    }

    score
}

fn is_low_signal_url(url: &str) -> bool {
    url.contains("zhidao.baidu.com")
        || url.contains("tieba.baidu.com")
        || url.contains("/search?")
        || url.contains("duckduckgo.com/?")
}

fn canonical_url(url: &str) -> String {
    let mut out = decode_html_entities_basic(url).to_lowercase();
    if let Some((left, _)) = out.split_once('#') {
        out = left.to_string();
    }
    while out.ends_with('/') {
        out.pop();
    }
    out
}

fn is_valid_result_url(url: &str) -> bool {
    (url.starts_with("http://") || url.starts_with("https://"))
        && !url.contains("duckduckgo.com/?")
        && !url.contains("bing.com/search?")
}

#[allow(dead_code)]
fn normalize_duckduckgo_href(href: &str) -> String {
    let href = decode_html_entities_basic(href);
    if href.starts_with("http://") || href.starts_with("https://") {
        return href;
    }

    if let Some(pos) = href.find("uddg=") {
        let encoded = &href[pos + 5..];
        let encoded = encoded.split('&').next().unwrap_or(encoded);
        if let Ok(decoded) = urlencoding::decode(encoded) {
            return decoded.into_owned();
        }
    }

    String::new()
}

fn normalize_bing_href(href: &str) -> String {
    let href = decode_html_entities_basic(href);
    if !(href.starts_with("http://") || href.starts_with("https://")) {
        return String::new();
    }

    let Ok(parsed) = reqwest::Url::parse(&href) else {
        return href;
    };
    let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
    let path = parsed.path().to_ascii_lowercase();

    if host.ends_with("bing.com") && path.starts_with("/ck/") {
        if let Some((_, value)) = parsed.query_pairs().find(|(k, _)| k == "u") {
            if let Some(decoded) = decode_bing_u_param(value.as_ref()) {
                return decoded;
            }
        }
    }

    href
}

#[allow(dead_code)]
fn normalize_generic_href(href: &str) -> String {
    let href = decode_html_entities_basic(href);
    if !(href.starts_with("http://") || href.starts_with("https://")) {
        return String::new();
    }
    href
}

fn decode_bing_u_param(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        return Some(raw.to_string());
    }

    let mut candidates = vec![raw];
    if raw.len() > 2 {
        candidates.push(&raw[2..]);
    }

    for candidate in candidates {
        let mut encoded = candidate.trim().to_string();
        if encoded.is_empty() {
            continue;
        }

        let rem = encoded.len() % 4;
        if rem != 0 {
            encoded.push_str(&"=".repeat(4 - rem));
        }

        if let Ok(bytes) = URL_SAFE.decode(encoded.as_bytes()) {
            if let Ok(text) = String::from_utf8(bytes) {
                if text.starts_with("http://") || text.starts_with("https://") {
                    return Some(text);
                }
            }
        }
    }

    None
}

fn decode_html_entities_basic(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&#38;", "&")
        .replace("&quot;", "\"")
}

fn debug_log_hits(stage: &str, query: &str, hits: &[SearchHit], limit: usize) {
    for (idx, hit) in hits.iter().take(limit).enumerate() {
        let title = truncate_text(&normalize_whitespace(&hit.title), 64);
        println!(
            "[DEBUG] {} query={} hit[{}] source={} url={} title={}",
            stage,
            query,
            idx + 1,
            hit.source,
            hit.url,
            title
        );
    }
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
