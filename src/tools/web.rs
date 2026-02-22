use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE, Engine as _};
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashSet;

const DEFAULT_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
const MAX_FETCH_CHARS: usize = 5000;
const MAX_SEARCH_PREVIEW_CHARS: usize = 700;

#[derive(Debug, Clone)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
    source: &'static str,
}

pub async fn search_web(client: &Client, query: &str, debug: bool) -> Result<String> {
    let query = query.trim();
    if query.is_empty() {
        bail!("query is empty");
    }

    if debug {
        println!("[DEBUG] tool.search_web query={query}");
    }

    let variants = build_query_variants(query);
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

    let mut hits = Vec::new();
    for variant in &variants {
        let before = hits.len();
        match search_bing(client, variant, debug).await {
            Ok(mut found) => hits.append(&mut found),
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
                hits.len().saturating_sub(before),
                hits.len()
            );
        }
    }

    if let Some(token) = strict_identifier.as_ref() {
        let mut matched = hits
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
                match search_bing(client, &q, debug).await {
                    Ok(mut found) => hits.append(&mut found),
                    Err(err) => {
                        if debug {
                            println!("[DEBUG] recovery bing failed query={q}: {err}");
                        }
                    }
                }
            }

            matched = hits
                .iter()
                .filter(|hit| hit_matches_identifier(hit, token))
                .count();
            if debug {
                println!("[DEBUG] bing strict identifier recovery matches={matched}");
            }
            if matched == 0 {
                return Ok(format!(
                    "未在 Bing 中国检索到包含标识符 `{token}` 的有效结果（已尝试精确匹配与加号强制检索）。\n建议：\n1) 直接附上目标主页链接\n2) 改搜 `+{token}` 或 `+\"{token}\"`\n3) 检查是否存在大小写/下划线/连字符差异"
                ));
            }
        }
    }

    if debug {
        println!("[DEBUG] search.raw total_hits={}", hits.len());
        debug_log_hits("search.raw", query, &hits, 10);
    }
    if debug {
        if let Some(token) = &strict_identifier {
            let matched = hits
                .iter()
                .filter(|hit| hit_matches_identifier(hit, token))
                .count();
            println!(
                "[DEBUG] search strict identifier matches={}/{} token={}",
                matched,
                hits.len(),
                token
            );
        }
    }

    let raw_hits = hits.clone();
    let mut ranked = rank_search_hits(&variants, hits.clone(), strict_identifier.as_deref());
    if ranked.is_empty() && strict_identifier.is_some() {
        if debug {
            println!(
                "[DEBUG] strict identifier filter produced 0 hits, fallback to relaxed ranking"
            );
        }
        ranked = rank_search_hits(&variants, hits.clone(), None);
    }
    if ranked.is_empty() {
        if debug {
            println!("[DEBUG] search.ranked empty after fallback");
            debug_log_hits("search.ranked.empty.raw", query, &raw_hits, 10);
        }

        let fallback_hits = take_unique_hits_preserve_order(raw_hits, 5);
        if fallback_hits.is_empty() {
            return Ok(format!("未检索到可用结果。query={query}"));
        }

        if debug {
            println!(
                "[DEBUG] search fallback to raw order hits={}",
                fallback_hits.len()
            );
        }

        let mut out = format!("Web 搜索结果（query: {query}，按原始结果回退）:\n");
        for (idx, hit) in fallback_hits.iter().enumerate() {
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
        return Ok(out.trim().to_string());
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

    Ok(out.trim().to_string())
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

async fn search_bing(client: &Client, query: &str, debug: bool) -> Result<Vec<SearchHit>> {
    let encoded = urlencoding::encode(query);
    let mut query_params = format!("q={encoded}&setlang=zh-Hans");
    if prefers_english_search(query) {
        query_params.push_str("&ensearch=1&mkt=en-US&cc=US");
    } else {
        query_params.push_str("&cc=CN&ensearch=0");
    }
    let search_url = format!("https://cn.bing.com/search?{query_params}");
    let accept_language = if prefers_english_search(query) {
        "en-US,en;q=0.9,zh-CN;q=0.7"
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
    query_variants: &[String],
    hits: Vec<SearchHit>,
    strict_identifier: Option<&str>,
) -> Vec<SearchHit> {
    let terms = build_query_terms(query_variants);
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

        let canonical = canonical_url(&hit.url);
        if canonical.is_empty() || !seen.insert(canonical) {
            continue;
        }

        let score = score_hit(&hit, query_variants, &terms);
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
    push_plus_variants(&mut variants, base);
    push_query_variant(&mut variants, base);

    if let Some(token) = extract_identifier_token(base) {
        push_query_variant(&mut variants, &format!("\"{token}\""));
        push_plus_variants(&mut variants, &token);

        if token.contains('_') {
            let alt = token.replace('_', "-");
            push_query_variant(&mut variants, &alt);
            push_plus_variants(&mut variants, &alt);
        }
        if token.contains('-') {
            let alt = token.replace('-', "_");
            push_query_variant(&mut variants, &alt);
            push_plus_variants(&mut variants, &alt);
        }
    }

    if normalized != query.trim() {
        push_plus_variants(&mut variants, &normalized);
        push_query_variant(&mut variants, &normalized);
    }

    if query_needs_recency(query) {
        let news_variant = format!("{base} 新闻 最新");
        push_query_variant(&mut variants, &news_variant);
        push_plus_variants(&mut variants, &news_variant);
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

fn score_hit(hit: &SearchHit, query_variants: &[String], terms: &[String]) -> i32 {
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
