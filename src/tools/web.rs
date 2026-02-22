use anyhow::{anyhow, bail, Context, Result};
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashSet;

const DEFAULT_UA: &str = "Mozilla/5.0 (compatible; XzBot/1.0; +https://example.local)";
const MAX_FETCH_CHARS: usize = 5000;

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

    let mut hits = Vec::new();
    hits.extend(search_bing(client, query, debug).await?);

    if hits.len() < 3 {
        hits.extend(search_duckduckgo_html(client, query, debug).await?);
    }

    if hits.len() < 3 {
        hits.extend(search_duckduckgo_api(client, query).await?);
    }

    let ranked = rank_search_hits(query, hits);
    if ranked.is_empty() {
        return Ok(format!("未检索到可用结果。query={query}"));
    }

    let mut out = format!("Web 搜索结果（query: {query}）:\n");
    for (idx, hit) in ranked.into_iter().take(5).enumerate() {
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
    let search_url =
        format!("https://www.bing.com/search?q={encoded}&setlang=zh-Hans&cc=CN&ensearch=0");

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

    let hits = parse_bing_results(&body)?;
    if debug {
        println!("[DEBUG] search.bing hits={}", hits.len());
    }
    Ok(hits)
}

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
        let href = title_node
            .value()
            .attr("href")
            .map(decode_html_entities_basic)
            .unwrap_or_default();
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

fn rank_search_hits(query: &str, hits: Vec<SearchHit>) -> Vec<SearchHit> {
    let terms = build_query_terms(query);
    if hits.is_empty() {
        return Vec::new();
    }

    let mut seen = HashSet::new();
    let mut scored = Vec::new();

    for hit in hits {
        let canonical = canonical_url(&hit.url);
        if canonical.is_empty() || !seen.insert(canonical) {
            continue;
        }

        let score = score_hit(&hit, query, &terms);
        scored.push((score, hit));
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));

    scored
        .into_iter()
        .filter(|(score, _)| *score > 0)
        .map(|(_, hit)| hit)
        .collect()
}

fn build_query_terms(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    for term in
        query.split(|c: char| c.is_whitespace() || c == ',' || c == '，' || c == ';' || c == '；')
    {
        let t = term.trim().to_lowercase();
        if t.is_empty() {
            continue;
        }
        out.push(t);
    }
    out
}

fn score_hit(hit: &SearchHit, query: &str, terms: &[String]) -> i32 {
    let title = hit.title.to_lowercase();
    let snippet = hit.snippet.to_lowercase();
    let url = hit.url.to_lowercase();
    let mut score = 0i32;

    let query_l = query.to_lowercase();
    if !query_l.is_empty() && (title.contains(&query_l) || snippet.contains(&query_l)) {
        score += 6;
    }

    for term in terms {
        if term.chars().count() < 2 {
            continue;
        }
        if title.contains(term) {
            score += 3;
            continue;
        }
        if snippet.contains(term) {
            score += 2;
            continue;
        }
        if url.contains(term) {
            score += 1;
        }
    }

    if hit.snippet.is_empty() {
        score -= 1;
    }

    if url.contains(".edu") || url.contains(".gov") || url.contains("news") {
        score += 1;
    }

    score
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

fn decode_html_entities_basic(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&#38;", "&")
        .replace("&quot;", "\"")
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
