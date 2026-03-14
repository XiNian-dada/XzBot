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
    path::Path,
    sync::OnceLock,
    time::{Duration, Instant},
};
use tokio::{process::Command, time::timeout};

const DEFAULT_UA: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
const BROWSER_SEC_CH_UA: &str =
    "\"Chromium\";v=\"122\", \"Not(A:Brand\";v=\"24\", \"Google Chrome\";v=\"122\"";
use crate::config::{NetworkConfig, SearchConfig, SearchProvider};

const MAX_FETCH_CHARS: usize = 5000;
const MAX_SEARCH_PREVIEW_CHARS: usize = 700;
const SEARCH_CACHE_TTL_SECS: u64 = 90;
const BROWSER_FETCH_TIMEOUT_SECS: u64 = 15;

/// 支持通过公开 API 读取的知乎页面类型。
///
/// 这里优先覆盖最常见、也是最值得抓取的三类：
/// - 问题页
/// - 单个回答页
/// - 专栏文章页
enum ZhihuApiTarget {
    Question {
        question_id: String,
        answer_limit: usize,
        answer_offset: usize,
    },
    Answer {
        answer_id: String,
    },
    Article {
        article_id: String,
    },
}

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

fn is_zhihu_api_url(url: &str) -> bool {
    let Some(parsed) = reqwest::Url::parse(url).ok() else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    if host != "www.zhihu.com" && host != "zhihu.com" {
        return false;
    };
    let Some(mut segments) = parsed.path_segments() else {
        return false;
    };
    matches!(
        (segments.next(), segments.next(), segments.next()),
        (Some("api"), Some("v4"), Some(_))
    )
}

/// 抓取网页正文，并在必要时自动回退到 reader/proxy 方案。
pub async fn fetch_url(
    client: &Client,
    url: &str,
    debug: bool,
    network: &NetworkConfig,
) -> Result<String> {
    let url = normalize_url_input(url);
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("invalid url (must start with http:// or https://): {url}");
    }

    if debug {
        println!("[DEBUG] tool.fetch_url url={url}");
    }

    // 知乎普通页面对静态抓取非常不友好，而且静态链路最终通常会退到 reader proxy。
    // 这里直接短路到公开 API，避免把知乎内容再交给 r.jina.ai 一类代理。
    if looks_like_zhihu_url(&url) || is_zhihu_api_url(&url) {
        return fetch_zhihu_via_api(client, &url, debug, false)
            .await
            .with_context(|| format!("zhihu api fetch failed for url: {url}"));
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
            if let Ok(v) = fetch_via_headless_browser(&url, debug, network).await {
                return Ok(v);
            }
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
        if let Ok(v) = fetch_via_headless_browser(&url, debug, network).await {
            return Ok(v);
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
            if let Ok(v) = fetch_via_headless_browser(&url, debug, network).await {
                return Ok(v);
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

async fn fetch_via_headless_browser(
    url: &str,
    debug: bool,
    network: &NetworkConfig,
) -> Result<String> {
    let lightpanda_err = match run_lightpanda_dump(url, debug, network).await {
        Ok(v) => return Ok(v),
        Err(err) => {
            if debug {
                println!(
                    "[DEBUG] fetch.browser lightpanda failed url={} err={:#}",
                    url, err
                );
            }
            err
        }
    };

    match run_chromium_dump(url, debug, network).await {
        Ok(v) => Ok(v),
        Err(err) => {
            if debug {
                println!(
                    "[DEBUG] fetch.browser chromium failed url={} err={:#}",
                    url, err
                );
            }
            Err(err.context(format!("lightpanda failed earlier: {lightpanda_err:#}")))
        }
    }
}

async fn run_lightpanda_dump(url: &str, debug: bool, network: &NetworkConfig) -> Result<String> {
    let output = run_browser_command(
        "lightpanda",
        &[
            "fetch",
            "--dump",
            "--log_level",
            "error",
            "--http_timeout",
            "10000",
            url,
        ],
        debug,
        network,
    )
    .await?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let html = browser_dump_to_html(&stdout);
    let (title, text) = extract_html_text(&html)?;
    let title = if title.trim().is_empty() {
        "(no title)".to_string()
    } else {
        title
    };
    let text = truncate_text(&normalize_whitespace(&text), MAX_FETCH_CHARS);
    if text.is_empty() || should_fallback_to_reader(&html, &text) {
        bail!("lightpanda returned blocked/empty page");
    }
    Ok(format!(
        "URL Requested: {url}\nURL Final: {url}\nFetch Mode: lightpanda\nTitle: {title}\nContent:\n{text}"
    ))
}

async fn run_chromium_dump(url: &str, debug: bool, network: &NetworkConfig) -> Result<String> {
    let args_base = [
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--dump-dom",
    ];
    let proxy_arg = network
        .proxy_enabled
        .then(|| format!("--proxy-server={}", network.proxy_url.trim()));

    let mut last_err: Option<anyhow::Error> = None;
    for binary in chromium_candidates() {
        let mut args: Vec<String> = args_base.iter().map(|v| (*v).to_string()).collect();
        if let Some(proxy) = &proxy_arg {
            args.push(proxy.clone());
        }
        args.push(url.to_string());
        let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();

        match run_browser_command(binary, &arg_refs, debug, network).await {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let html = browser_dump_to_html(&stdout);
                let (title, text) = extract_html_text(&html)?;
                let title = if title.trim().is_empty() {
                    "(no title)".to_string()
                } else {
                    title
                };
                let text = truncate_text(&normalize_whitespace(&text), MAX_FETCH_CHARS);
                if text.is_empty() || should_fallback_to_reader(&html, &text) {
                    last_err = Some(anyhow!("browser returned blocked/empty page"));
                    continue;
                }
                return Ok(format!(
                    "URL Requested: {url}\nURL Final: {url}\nFetch Mode: chromium\nTitle: {title}\nContent:\n{text}"
                ));
            }
            Err(err) => last_err = Some(err),
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("no chromium executable found")))
}

async fn run_browser_command(
    binary: &str,
    args: &[&str],
    debug: bool,
    network: &NetworkConfig,
) -> Result<std::process::Output> {
    let exists = Path::new(binary).is_absolute() && Path::new(binary).exists();
    if Path::new(binary).is_absolute() && !exists {
        bail!("browser executable not found: {binary}");
    }

    let mut cmd = Command::new(binary);
    cmd.args(args);
    apply_browser_proxy_env(&mut cmd, network);

    if debug {
        println!(
            "[DEBUG] fetch.browser exec binary={} args={}",
            binary,
            args.join(" ")
        );
    }

    let output = timeout(
        Duration::from_secs(BROWSER_FETCH_TIMEOUT_SECS),
        cmd.output(),
    )
    .await
    .context("browser command timeout")?
    .with_context(|| format!("failed to run browser command: {binary}"))?;

    if !output.status.success() {
        let stderr = truncate_text(&String::from_utf8_lossy(&output.stderr), 300);
        bail!(
            "browser command failed status={} stderr={stderr}",
            output.status
        );
    }

    Ok(output)
}

fn apply_browser_proxy_env(cmd: &mut Command, network: &NetworkConfig) {
    if !network.proxy_enabled {
        return;
    }
    let proxy = network.proxy_url.trim();
    if proxy.is_empty() {
        return;
    }
    cmd.env("http_proxy", proxy)
        .env("https_proxy", proxy)
        .env("HTTP_PROXY", proxy)
        .env("HTTPS_PROXY", proxy)
        .env("ALL_PROXY", proxy)
        .env("all_proxy", proxy);
}

fn chromium_candidates() -> &'static [&'static str] {
    &[
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
}

fn browser_dump_to_html(stdout: &str) -> String {
    if let Some(idx) = stdout.find("<!DOCTYPE") {
        return stdout[idx..].to_string();
    }
    if let Some(idx) = stdout.find("<html") {
        return stdout[idx..].to_string();
    }
    stdout.to_string()
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

fn looks_like_zhihu_url(url: &str) -> bool {
    let Some(parsed) = reqwest::Url::parse(url).ok() else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    matches!(host, "www.zhihu.com" | "zhihu.com" | "zhuanlan.zhihu.com")
}

fn parse_zhihu_api_target(url: &str) -> Option<ZhihuApiTarget> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    let segments: Vec<_> = parsed.path_segments()?.collect();
    let answer_limit = query_usize(&parsed, "limit").unwrap_or(5);
    let answer_offset = query_usize(&parsed, "offset").unwrap_or(0);

    match host.as_str() {
        "www.zhihu.com" | "zhihu.com" => match segments.as_slice() {
            ["question", question_id] if !question_id.is_empty() => {
                Some(ZhihuApiTarget::Question {
                    question_id: (*question_id).to_string(),
                    answer_limit,
                    answer_offset,
                })
            }
            ["api", "v4", "questions", question_id] if !question_id.is_empty() => {
                Some(ZhihuApiTarget::Question {
                    question_id: (*question_id).to_string(),
                    answer_limit,
                    answer_offset,
                })
            }
            ["api", "v4", "questions", question_id, "answers"] if !question_id.is_empty() => {
                Some(ZhihuApiTarget::Question {
                    question_id: (*question_id).to_string(),
                    answer_limit,
                    answer_offset,
                })
            }
            ["question", _, "answer", answer_id] if !answer_id.is_empty() => {
                Some(ZhihuApiTarget::Answer {
                    answer_id: (*answer_id).to_string(),
                })
            }
            ["api", "v4", "answers", answer_id] if !answer_id.is_empty() => {
                Some(ZhihuApiTarget::Answer {
                    answer_id: (*answer_id).to_string(),
                })
            }
            ["api", "v4", "articles", article_id] if !article_id.is_empty() => {
                Some(ZhihuApiTarget::Article {
                    article_id: (*article_id).to_string(),
                })
            }
            _ => None,
        },
        "zhuanlan.zhihu.com" => match segments.as_slice() {
            ["p", article_id] if !article_id.is_empty() => Some(ZhihuApiTarget::Article {
                article_id: (*article_id).to_string(),
            }),
            _ => None,
        },
        _ => None,
    }
}

async fn fetch_zhihu_via_api(
    client: &Client,
    url: &str,
    debug: bool,
    preview_mode: bool,
) -> Result<String> {
    let target = parse_zhihu_api_target(url)
        .ok_or_else(|| anyhow!("unsupported zhihu url for api mode: {url}"))?;

    if debug {
        println!(
            "[DEBUG] fetch.zhihu_api mode={} url={}",
            if preview_mode { "preview" } else { "full" },
            url
        );
    }

    match target {
        ZhihuApiTarget::Question {
            question_id,
            answer_limit,
            answer_offset,
        } => {
            fetch_zhihu_question(
                client,
                url,
                debug,
                &question_id,
                answer_limit,
                answer_offset,
                preview_mode,
            )
            .await
        }
        ZhihuApiTarget::Answer { answer_id } => {
            fetch_zhihu_answer(client, url, debug, &answer_id, preview_mode).await
        }
        ZhihuApiTarget::Article { article_id } => {
            fetch_zhihu_article(client, url, debug, &article_id, preview_mode).await
        }
    }
}

async fn fetch_zhihu_question(
    client: &Client,
    url: &str,
    debug: bool,
    question_id: &str,
    answer_limit: usize,
    answer_offset: usize,
    preview_mode: bool,
) -> Result<String> {
    let answers_limit = if preview_mode {
        answer_limit.min(2).max(1)
    } else {
        answer_limit.max(1)
    };
    let answers_api = format!(
        "https://www.zhihu.com/api/v4/questions/{question_id}/answers?limit={answers_limit}&offset={answer_offset}&include=data[*].content,voteup_count,comment_count,created_time,updated_time,excerpt"
    );
    if debug {
        println!("[DEBUG] fetch.zhihu_api answers_api={answers_api}");
    }
    let answers = fetch_json_value(client, &answers_api).await?;
    let answers_array = answers
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("zhihu question api missing data array"))?;
    let paging = answers.get("paging").cloned().unwrap_or(Value::Null);
    let total_answers = paging
        .get("totals")
        .and_then(Value::as_i64)
        .unwrap_or(answers_array.len() as i64);
    let title = answers_array
        .first()
        .and_then(|answer| answer.get("question"))
        .and_then(|question| question.get("title"))
        .and_then(Value::as_str)
        .unwrap_or("(无标题)");

    let mut out = format!(
        "URL Requested: {url}\nURL Final: {answers_api}\nFetch Mode: zhihu_api\nZhihu Type: question\nQuestion Title: {title}\nAnswer Count: {total_answers}\n"
    );

    if answers_array.is_empty() {
        out.push_str("Answers: (none)");
        return Ok(out);
    }

    out.push_str("Answers:\n");
    for (idx, answer) in answers_array.iter().enumerate() {
        let author = answer
            .get("author")
            .and_then(|v| v.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("匿名用户");
        let votes = answer
            .get("voteup_count")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let comments = answer
            .get("comment_count")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let content = answer
            .get("content")
            .and_then(Value::as_str)
            .map(html_fragment_to_text)
            .or_else(|| {
                answer
                    .get("excerpt")
                    .and_then(Value::as_str)
                    .map(html_fragment_to_text)
            })
            .unwrap_or_default();
        let content = truncate_text(&content, if preview_mode { 220 } else { 900 });
        out.push_str(&format!(
            "{}. Author: {} | Votes: {} | Comments: {}\n{}\n",
            idx + 1,
            author,
            votes,
            comments,
            content
        ));
    }

    Ok(out)
}

async fn fetch_zhihu_answer(
    client: &Client,
    url: &str,
    debug: bool,
    answer_id: &str,
    preview_mode: bool,
) -> Result<String> {
    let answer_api = format!(
        "https://www.zhihu.com/api/v4/answers/{answer_id}?include=content,voteup_count,comment_count,created_time,updated_time,excerpt,author.name,question.title"
    );
    if debug {
        println!("[DEBUG] fetch.zhihu_api answer_api={answer_api}");
    }
    let answer = fetch_json_value(client, &answer_api).await?;

    let question_title = answer
        .get("question")
        .and_then(|v| v.get("title"))
        .and_then(Value::as_str)
        .unwrap_or("(无标题)");
    let author = answer
        .get("author")
        .and_then(|v| v.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("匿名用户");
    let votes = answer
        .get("voteup_count")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let comments = answer
        .get("comment_count")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let content = answer
        .get("content")
        .and_then(Value::as_str)
        .map(html_fragment_to_text)
        .or_else(|| {
            answer
                .get("excerpt")
                .and_then(Value::as_str)
                .map(html_fragment_to_text)
        })
        .unwrap_or_default();
    let content = truncate_text(
        &content,
        if preview_mode {
            MAX_SEARCH_PREVIEW_CHARS
        } else {
            MAX_FETCH_CHARS
        },
    );

    Ok(format!(
        "URL Requested: {url}\nURL Final: {answer_api}\nFetch Mode: zhihu_api\nZhihu Type: answer\nQuestion Title: {question_title}\nAuthor: {author}\nVotes: {votes}\nComments: {comments}\nContent:\n{content}"
    ))
}

async fn fetch_zhihu_article(
    client: &Client,
    url: &str,
    debug: bool,
    article_id: &str,
    preview_mode: bool,
) -> Result<String> {
    let article_api = format!(
        "https://www.zhihu.com/api/v4/articles/{article_id}?include=title,excerpt,content,voteup_count,comment_count"
    );
    if debug {
        println!("[DEBUG] fetch.zhihu_api article_api={article_api}");
    }
    let article = fetch_json_value(client, &article_api).await?;
    let title = json_str(&article, "title").unwrap_or("(无标题)");
    let votes = article
        .get("voteup_count")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let comments = article
        .get("comment_count")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let content = article
        .get("content")
        .and_then(Value::as_str)
        .map(html_fragment_to_text)
        .or_else(|| {
            article
                .get("excerpt")
                .and_then(Value::as_str)
                .map(html_fragment_to_text)
        })
        .unwrap_or_default();
    let content = truncate_text(
        &content,
        if preview_mode {
            MAX_SEARCH_PREVIEW_CHARS
        } else {
            MAX_FETCH_CHARS
        },
    );

    Ok(format!(
        "URL Requested: {url}\nURL Final: {article_api}\nFetch Mode: zhihu_api\nZhihu Type: article\nTitle: {title}\nVotes: {votes}\nComments: {comments}\nContent:\n{content}"
    ))
}

async fn fetch_json_value(client: &Client, url: &str) -> Result<Value> {
    let response = client
        .get(url)
        .header("User-Agent", DEFAULT_UA)
        .header("Accept", "application/json,text/plain,*/*")
        .header("Accept-Language", "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7")
        .header("Referer", "https://www.zhihu.com/")
        .send()
        .await
        .with_context(|| format!("zhihu api request failed: {url}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read zhihu api body")?;
    if !status.is_success() {
        let brief = truncate_text(&normalize_whitespace(&body), 240);
        bail!("zhihu api returned {status}: {brief}");
    }
    serde_json::from_str(&body).context("failed to parse zhihu api json")
}

fn json_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn query_usize(url: &reqwest::Url, key: &str) -> Option<usize> {
    url.query_pairs()
        .find_map(|(k, v)| (k == key).then(|| v.parse::<usize>().ok()).flatten())
}

fn html_fragment_to_text(html: &str) -> String {
    if !html.contains('<') {
        return normalize_whitespace(html);
    }
    let fragment = Html::parse_fragment(html);
    let text = fragment.root_element().text().collect::<Vec<_>>().join(" ");
    normalize_whitespace(&text)
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
