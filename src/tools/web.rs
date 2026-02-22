use anyhow::{anyhow, bail, Context, Result};
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashSet;

const DEFAULT_UA: &str = "Mozilla/5.0 (compatible; XzBot/1.0; +https://example.local)";
const MAX_FETCH_CHARS: usize = 5000;

pub async fn search_web(client: &Client, query: &str, debug: bool) -> Result<String> {
    let query = query.trim();
    if query.is_empty() {
        bail!("query is empty");
    }

    let encoded = urlencoding::encode(query);
    let search_url = format!("https://www.bing.com/search?q={encoded}");

    if debug {
        println!("[DEBUG] tool.search_web query={query}");
    }

    let response = client
        .get(&search_url)
        .header("User-Agent", DEFAULT_UA)
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

    let mut lines = parse_bing_results(&body)?;
    if lines.is_empty() {
        lines = search_web_fallback_duckduckgo(client, query).await?;
    }

    if lines.is_empty() {
        return Ok(format!("未检索到可用结果。query={query}"));
    }

    lines.truncate(5);
    let mut out = format!("Web 搜索结果（query: {query}）:\n");
    for (idx, line) in lines.into_iter().enumerate() {
        out.push_str(&format!("{}. {line}\n", idx + 1));
    }
    Ok(out.trim().to_string())
}

pub async fn fetch_url(client: &Client, url: &str, debug: bool) -> Result<String> {
    let url = url.trim();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("invalid url (must start with http:// or https://): {url}");
    }

    if debug {
        println!("[DEBUG] tool.fetch_url url={url}");
    }

    let response = client
        .get(url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("fetch request failed: {url}"))?;
    let status = response.status();
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
        return Ok(format!("URL: {url}\nTitle: {title}\nContent:\n{text}"));
    }

    let text = truncate_text(&normalize_whitespace(&body), MAX_FETCH_CHARS);
    Ok(format!("URL: {url}\nContent:\n{text}"))
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
                .to_string();
            if !trimmed.is_empty() && seen.insert(trimmed.clone()) {
                out.push(trimmed);
            }
        }
    }

    out
}

fn parse_bing_results(html: &str) -> Result<Vec<String>> {
    let doc = Html::parse_document(html);
    let item_sel = Selector::parse("li.b_algo")
        .map_err(|err| anyhow!("failed to parse selector li.b_algo: {err}"))?;
    let title_sel =
        Selector::parse("h2 a").map_err(|err| anyhow!("failed to parse selector h2 a: {err}"))?;
    let snippet_sel =
        Selector::parse("p").map_err(|err| anyhow!("failed to parse selector p: {err}"))?;

    let mut out = Vec::new();
    for item in doc.select(&item_sel).take(8) {
        let title_node = item.select(&title_sel).next();
        let Some(title_node) = title_node else {
            continue;
        };
        let title = normalize_whitespace(&title_node.text().collect::<Vec<_>>().join(" "));
        let href = title_node
            .value()
            .attr("href")
            .unwrap_or("")
            .trim()
            .to_string();
        if href.is_empty() {
            continue;
        }

        let snippet = item
            .select(&snippet_sel)
            .next()
            .map(|n| normalize_whitespace(&n.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        if snippet.is_empty() {
            out.push(format!("{title} - {href}"));
        } else {
            out.push(format!("{title} - {href} - {snippet}"));
        }
    }

    Ok(out)
}

async fn search_web_fallback_duckduckgo(client: &Client, query: &str) -> Result<Vec<String>> {
    let encoded = urlencoding::encode(query);
    let url = format!(
        "https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&no_redirect=1&skip_disambig=1"
    );
    let response = client
        .get(&url)
        .header("User-Agent", DEFAULT_UA)
        .send()
        .await
        .with_context(|| format!("duckduckgo fallback failed: {url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read duckduckgo response body")?;

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
                .unwrap_or("DuckDuckGo");
            let link = value
                .get("AbstractURL")
                .and_then(Value::as_str)
                .unwrap_or("");
            out.push(format!("{source} - {link} - {abstract_text}"));
        }
    }

    if let Some(related) = value.get("RelatedTopics").and_then(Value::as_array) {
        for item in related.iter().take(5) {
            if let Some(text) = item.get("Text").and_then(Value::as_str) {
                let first_url = item.get("FirstURL").and_then(Value::as_str).unwrap_or("");
                out.push(format!("DuckDuckGo - {first_url} - {text}"));
            }
        }
    }

    Ok(out)
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
