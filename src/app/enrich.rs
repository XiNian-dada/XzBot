//! 事件富化子模块：补全图片 URL、引用文本和上下文附加信息。
//!
//! 这里的职责是把 OneBot 原始事件里“不够直接可用”的上下文补齐出来，
//! 但不篡改原始消息语义：
//! - 原始上报仍保留在 `raw_message` / `message`
//! - 运行时额外解析出来的引用文本、图片 URL 统一放进 `MessageEvent::enriched_parts`
//!
//! 这样后续无论是 AI、插件还是普通路由，只要走 `event.text()`，
//! 就能看到同一份稳定的富化结果。

use super::*;

/// 富化 OneBot 消息事件中的“图片”和“引用消息”上下文。
///
/// 这里的目标不是修改原始语义，而是尽量把 NapCat/OneBot 给出的零散信息
/// 整理成更适合后续 AI 读取的输入：
/// - 能直接拿到 URL 的图片，补成统一形式
/// - 只有 file id 的图片，尝试调用 `get_image` 解析
/// - 引用消息里的文字，也追加到事件的富化上下文里
pub(super) async fn enrich_event_images(
    event: &mut MessageEvent,
    bridge: &WsActionBridge,
    debug: bool,
) -> anyhow::Result<()> {
    let mut urls = Vec::new();
    let mut seen = HashSet::new();
    let mut quote_texts = Vec::new();
    let mut seen_quote_texts = HashSet::new();

    // 0) 直接读取 raw_message 里已有的 CQ image url（有些群直接给 url）。
    for image_ref in extract_cq_image_refs(&event.raw_message) {
        if let Some(url) = image_ref.url {
            let url = normalize_image_ref(&url);
            if looks_like_http_url(&url) && seen.insert(url.clone()) {
                urls.push(url);
            }
        }
    }

    // 0.1) 结构化 segments 里带 url 的情况。
    if let MessagePayload::Segments(segments) = &event.message {
        for segment in segments {
            if segment.kind != "image" {
                continue;
            }
            if let Some(url) = segment.data.get("url").and_then(Value::as_str) {
                let url = normalize_image_ref(url);
                if looks_like_http_url(&url) && seen.insert(url.clone()) {
                    urls.push(url);
                }
            }
        }
    }

    // 1) 当前消息里的图片 file id -> 尝试 get_image 解析 URL。
    for file_id in event.image_file_ids().into_iter().take(4) {
        if let Some(url) = resolve_image_url(bridge, &file_id, debug).await {
            if seen.insert(url.clone()) {
                urls.push(url);
            }
        }
    }

    // 2) 引用回复里的图片：通过 get_msg 拉取被引用消息，再解析其中图片。
    for reply_id in event.reply_message_ids().into_iter().take(3) {
        let response = match bridge
            .call_action("get_msg", json!({ "message_id": reply_id }))
            .await
        {
            Ok(v) => v,
            Err(err) => {
                log_debug(debug, format!("get_msg failed reply_id={reply_id}: {err}"));
                continue;
            }
        };

        let data = response.get("data").cloned().unwrap_or(Value::Null);
        let (quoted_urls, quoted_files) = collect_image_refs_from_message_data(&data);
        if let Some(quote_text) = extract_quote_text_from_message_data(&data) {
            let normalized = quote_text.split_whitespace().collect::<Vec<_>>().join(" ");
            if !normalized.is_empty() && seen_quote_texts.insert(normalized.clone()) {
                quote_texts.push(format!("[引用消息] {}", trim_for_context(&normalized, 220)));
            }
        }

        for url in quoted_urls {
            if seen.insert(url.clone()) {
                urls.push(url);
            }
        }

        for file_id in quoted_files.into_iter().take(4) {
            if let Some(url) = resolve_image_url(bridge, &file_id, debug).await {
                if seen.insert(url.clone()) {
                    urls.push(url);
                }
            }
        }
    }

    let quote_count = quote_texts.len();
    if quote_count > 0 {
        for quote_text in quote_texts {
            event.push_enriched_part(quote_text);
        }
        log_debug(
            debug,
            format!("event quote context enriched count={quote_count}"),
        );
    }

    let url_count = urls.len();
    if url_count > 0 {
        for url in &urls {
            // 使用内部统一的 `[IMAGE:...]` 标记，保证 AI / 插件链路都能一致识别。
            event.push_enriched_part(format!("[IMAGE:url={url}]"));
        }
        log_debug(
            debug,
            format!("event image context enriched urls={url_count}"),
        );
    }

    Ok(())
}

/// 从 `get_msg` 返回数据里提取适合拼进上下文的引用文本。
fn extract_quote_text_from_message_data(data: &Value) -> Option<String> {
    if let Some(message) = data.get("message") {
        if let Some(text) = message_value_to_text(message) {
            if !text.trim().is_empty() {
                return Some(text);
            }
        }
    }

    data.get("raw_message")
        .and_then(Value::as_str)
        .map(strip_cq_to_text)
        .filter(|v| !v.trim().is_empty())
}

/// 把 OneBot 的 `message` 字段统一转换成纯文本。
///
/// 这里同时兼容：
/// - 直接字符串消息
/// - segments 数组消息
///
/// 图片会被替换成 `[图片]` 占位符，避免后续文本链路直接丢失上下文。
fn message_value_to_text(message: &Value) -> Option<String> {
    match message {
        Value::String(raw) => {
            let text = strip_cq_to_text(raw);
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Value::Array(segments) => {
            let mut out = String::new();
            for seg in segments {
                let kind = seg.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "text" => {
                        if let Some(text) = seg
                            .get("data")
                            .and_then(|d| d.get("text"))
                            .and_then(Value::as_str)
                        {
                            out.push_str(text);
                        }
                    }
                    "image" => {
                        if !out.ends_with(' ') && !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str("[图片]");
                    }
                    "at" => {
                        let qq_text = seg.get("data").and_then(|d| d.get("qq")).and_then(|v| {
                            v.as_str()
                                .map(str::to_string)
                                .or_else(|| v.as_i64().map(|n| n.to_string()))
                        });
                        if let Some(qq) = qq_text {
                            if !out.ends_with(' ') && !out.is_empty() {
                                out.push(' ');
                            }
                            out.push('@');
                            out.push_str(&qq);
                        }
                    }
                    _ => {}
                }
            }
            let normalized = out.split_whitespace().collect::<Vec<_>>().join(" ");
            if normalized.is_empty() {
                None
            } else {
                Some(normalized)
            }
        }
        _ => None,
    }
}

/// 去掉 CQ 码，只保留普通文本和少量可读占位符。
fn strip_cq_to_text(raw: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0;

    while let Some(start_rel) = raw[cursor..].find("[CQ:") {
        let start = cursor + start_rel;
        out.push_str(&raw[cursor..start]);

        let Some(end_rel) = raw[start..].find(']') else {
            out.push_str(&raw[start..]);
            cursor = raw.len();
            break;
        };
        let end = start + end_rel;
        let segment = &raw[start + 1..end];
        if segment.starts_with("CQ:image") {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            out.push_str("[图片]");
        }
        cursor = end + 1;
    }

    out.push_str(&raw[cursor..]);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 对过长引用文本做截断，避免上下文无限膨胀。
fn trim_for_context(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    chars.into_iter().take(max_chars).collect::<String>() + "...(truncated)"
}

/// 把图片引用解析为可进一步使用的 URL / base64 / file URL。
///
/// OneBot 侧图片可能是：
/// - 直接 http(s) URL
/// - base64/data URL
/// - 本地文件路径
/// - 需要 `get_image` 二次解析的 file id
pub(super) async fn resolve_image_url(
    bridge: &WsActionBridge,
    image_ref: &str,
    debug: bool,
) -> Option<String> {
    let value = normalize_image_ref(image_ref);
    if value.is_empty() {
        return None;
    }
    if looks_like_http_url(&value) {
        return Some(value.to_string());
    }
    if value.starts_with("base64://") || value.starts_with("data:image/") {
        return Some(value.to_string());
    }
    if value.starts_with("file://") {
        return Some(value.to_string());
    }

    let response = bridge
        .call_action("get_image", json!({ "file": value }))
        .await
        .ok();
    let response = match response {
        Some(v) => v,
        None => {
            log_debug(debug, format!("get_image failed file={image_ref}"));
            return None;
        }
    };
    let data = response.get("data")?;

    for key in ["url", "file"] {
        if let Some(v) = data.get(key).and_then(Value::as_str) {
            let v = normalize_image_ref(v);
            if looks_like_http_url(&v) || v.starts_with("base64://") || v.starts_with("file://") {
                return Some(v.to_string());
            }
            if looks_like_local_path(&v) {
                return Some(format!("file://{v}"));
            }
        }
    }

    if debug {
        log_debug(
            debug,
            format!(
                "get_image unresolved file={} data={}",
                image_ref,
                response.get("data").cloned().unwrap_or(Value::Null)
            ),
        );
    }
    None
}

/// 从 `get_msg` 返回体中收集图片 URL 和 file 引用。
fn collect_image_refs_from_message_data(data: &Value) -> (Vec<String>, Vec<String>) {
    let mut urls = Vec::new();
    let mut files = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut seen_files = HashSet::new();

    if let Some(message) = data.get("message") {
        collect_image_refs_from_message_value(
            message,
            &mut urls,
            &mut files,
            &mut seen_urls,
            &mut seen_files,
        );
    }

    if let Some(raw) = data.get("raw_message").and_then(Value::as_str) {
        for image_ref in extract_cq_image_refs(raw) {
            if let Some(url) = image_ref.url {
                let url = normalize_image_ref(&url);
                if !url.is_empty() && seen_urls.insert(url.clone()) {
                    urls.push(url);
                }
            }
            if let Some(file) = image_ref.file {
                let file = normalize_image_ref(&file);
                if !file.is_empty() && seen_files.insert(file.clone()) {
                    files.push(file);
                }
            }
        }
    }

    (urls, files)
}

/// 从单个 `message` 值里解析图片引用。
///
/// 这里单独拆函数，是为了让“字符串消息”和“segments 消息”两条解析路径
/// 都能复用同样的去重容器。
fn collect_image_refs_from_message_value(
    message: &Value,
    urls: &mut Vec<String>,
    files: &mut Vec<String>,
    seen_urls: &mut HashSet<String>,
    seen_files: &mut HashSet<String>,
) {
    match message {
        Value::String(raw) => {
            for image_ref in extract_cq_image_refs(raw) {
                if let Some(url) = image_ref.url {
                    let url = normalize_image_ref(&url);
                    if !url.is_empty() && seen_urls.insert(url.clone()) {
                        urls.push(url);
                    }
                }
                if let Some(file) = image_ref.file {
                    let file = normalize_image_ref(&file);
                    if !file.is_empty() && seen_files.insert(file.clone()) {
                        files.push(file);
                    }
                }
            }
        }
        Value::Array(segments) => {
            for seg in segments {
                if seg.get("type").and_then(Value::as_str) != Some("image") {
                    continue;
                }
                let Some(seg_data) = seg.get("data") else {
                    continue;
                };
                if let Some(url) = seg_data.get("url").and_then(Value::as_str) {
                    let url = normalize_image_ref(url);
                    if !url.is_empty() && seen_urls.insert(url.clone()) {
                        urls.push(url);
                    }
                }
                if let Some(file) = seg_data.get("file").and_then(Value::as_str) {
                    let file = normalize_image_ref(file);
                    if !file.is_empty() && seen_files.insert(file.clone()) {
                        files.push(file);
                    }
                }
            }
        }
        _ => {}
    }
}

/// 判断一个字符串是否像 HTTP(S) URL。
fn looks_like_http_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

/// 判断一个字符串是否像本地绝对路径。
fn looks_like_local_path(value: &str) -> bool {
    value.starts_with('/') || value.contains(":\\")
}

/// 规范化 CQ 图片字段中的引用值。
///
/// 主要处理两类问题：
/// - 多余引号
/// - `&amp;` 这类 HTML 转义导致的错误 URL
pub(super) fn normalize_image_ref(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace("&amp;", "&")
        .replace("&#38;", "&")
}

/// 判断消息是否是内部 POST 推送产生的“系统代发消息”。
///
/// 命中后，主流程会直接跳过：
/// - 不进入 AI 上下文
/// - 不触发插件
/// - 不参与普通消息路由
pub(super) fn is_post_context_marker_message(event: &MessageEvent) -> bool {
    if event.raw_message.starts_with(POST_CONTEXT_MARKER) {
        return true;
    }

    if let MessagePayload::Text(text) = &event.message {
        if text.starts_with(POST_CONTEXT_MARKER) {
            return true;
        }
    }

    event.text().starts_with(POST_CONTEXT_MARKER)
}
