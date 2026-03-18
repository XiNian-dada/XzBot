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
    network: &NetworkConfig,
) -> anyhow::Result<()> {
    let mut urls = Vec::new();
    let mut seen = HashSet::new();
    let mut quote_texts = Vec::new();
    let mut seen_quote_texts = HashSet::new();
    let mut file_contexts = Vec::new();
    let mut seen_file_contexts = HashSet::new();
    let mut forward_contexts = Vec::new();
    let mut seen_forward_contexts = HashSet::new();
    let file_client = build_client(20_000, network, false)?;

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

    // 1.1) 当前消息里的文件：如果是文本类文件，就把裁剪后的内容补进上下文。
    for file_ref in event.file_refs().into_iter().take(3) {
        if let Some(file_context) = load_file_context(
            &file_client,
            bridge,
            &event.message_type,
            event.group_id,
            &file_ref,
            false,
            debug,
        )
        .await
        {
            let dedup = file_context
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if !dedup.is_empty() && seen_file_contexts.insert(dedup) {
                file_contexts.push(file_context);
            }
        }
    }

    // 1.2) 当前消息里的转发聊天记录。
    for forward_id in event.forward_ids().into_iter().take(2) {
        if let Some(bundle) = load_forward_context_bundle(bridge, &forward_id, false, debug).await {
            if let Some(forward_context) = bundle.text {
                let dedup = forward_context
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                if !dedup.is_empty() && seen_forward_contexts.insert(dedup) {
                    forward_contexts.push(forward_context);
                }
            }
            for url in bundle.urls {
                if seen.insert(url.clone()) {
                    urls.push(url);
                }
            }
            for file_id in bundle.files.into_iter().take(8) {
                if let Some(url) = resolve_image_url(bridge, &file_id, debug).await {
                    if seen.insert(url.clone()) {
                        urls.push(url);
                    }
                }
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
        for forward_id in collect_forward_ids_from_message_data(&data)
            .into_iter()
            .take(2)
        {
            if let Some(bundle) =
                load_forward_context_bundle(bridge, &forward_id, true, debug).await
            {
                if let Some(forward_context) = bundle.text {
                    let dedup = forward_context
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !dedup.is_empty() && seen_forward_contexts.insert(dedup) {
                        forward_contexts.push(forward_context);
                    }
                }
                for url in bundle.urls {
                    if seen.insert(url.clone()) {
                        urls.push(url);
                    }
                }
                for file_id in bundle.files.into_iter().take(8) {
                    if let Some(url) = resolve_image_url(bridge, &file_id, debug).await {
                        if seen.insert(url.clone()) {
                            urls.push(url);
                        }
                    }
                }
            }
        }
        for file_ref in collect_file_refs_from_message_data(&data)
            .into_iter()
            .take(3)
        {
            if let Some(file_context) = load_file_context(
                &file_client,
                bridge,
                &event.message_type,
                event.group_id,
                &file_ref,
                true,
                debug,
            )
            .await
            {
                let dedup = file_context
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                if !dedup.is_empty() && seen_file_contexts.insert(dedup) {
                    file_contexts.push(file_context);
                }
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

    let file_count = file_contexts.len();
    if file_count > 0 {
        for file_context in file_contexts {
            event.push_enriched_part(file_context);
        }
        log_debug(
            debug,
            format!("event file context enriched count={file_count}"),
        );
    }

    let forward_count = forward_contexts.len();
    if forward_count > 0 {
        for forward_context in forward_contexts {
            event.push_enriched_part(forward_context);
        }
        log_debug(
            debug,
            format!("event forward context enriched count={forward_count}"),
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

const FILE_FETCH_MAX_BYTES: u64 = 10 * 1024 * 1024;
const FILE_INLINE_MAX_BYTES: usize = 768 * 1024;
const FILE_PREVIEW_MAX_CHARS: usize = 30_000;

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
                    "file" => {
                        if !out.ends_with(' ') && !out.is_empty() {
                            out.push(' ');
                        }
                        let name = seg
                            .get("data")
                            .and_then(|d| d.get("name").or_else(|| d.get("file_name")))
                            .and_then(Value::as_str)
                            .unwrap_or("文件");
                        out.push_str(&format!("[文件:{name}]"));
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
        } else if segment.starts_with("CQ:file") {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            out.push_str("[文件]");
        } else if segment.starts_with("CQ:forward") {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            out.push_str("[转发消息]");
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

/// 从 `get_msg` 返回体中收集文件引用。
fn collect_file_refs_from_message_data(data: &Value) -> Vec<MessageFileRef> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    if let Some(message) = data.get("message") {
        collect_file_refs_from_message_value(message, &mut out, &mut seen);
    }

    if let Some(raw) = data.get("raw_message").and_then(Value::as_str) {
        for file_ref in extract_cq_file_refs(raw) {
            let dedup_key = file_ref_dedup_key(&file_ref);
            if !dedup_key.is_empty() && seen.insert(dedup_key) {
                out.push(file_ref);
            }
        }
    }

    out
}

/// 从 `get_msg` 返回体中收集转发消息 id。
fn collect_forward_ids_from_message_data(data: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    if let Some(message) = data.get("message") {
        collect_forward_ids_from_message_value(message, &mut out, &mut seen);
    }

    if let Some(raw) = data.get("raw_message").and_then(Value::as_str) {
        for id in extract_cq_forward_ids(raw) {
            if seen.insert(id.clone()) {
                out.push(id);
            }
        }
    }

    out
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

/// 从单个 `message` 值里解析文件引用。
fn collect_file_refs_from_message_value(
    message: &Value,
    out: &mut Vec<MessageFileRef>,
    seen: &mut HashSet<String>,
) {
    match message {
        Value::String(raw) => {
            for file_ref in extract_cq_file_refs(raw) {
                let dedup_key = file_ref_dedup_key(&file_ref);
                if !dedup_key.is_empty() && seen.insert(dedup_key) {
                    out.push(file_ref);
                }
            }
        }
        Value::Array(segments) => {
            for seg in segments {
                if seg.get("type").and_then(Value::as_str) != Some("file") {
                    continue;
                }
                let Some(seg_data) = seg.get("data") else {
                    continue;
                };
                let file_ref = MessageFileRef {
                    name: seg_data
                        .get("name")
                        .or_else(|| seg_data.get("file_name"))
                        .and_then(Value::as_str)
                        .map(normalize_image_ref),
                    file_id: seg_data
                        .get("file_id")
                        .or_else(|| seg_data.get("id"))
                        .and_then(Value::as_str)
                        .map(normalize_image_ref),
                    file: seg_data
                        .get("file")
                        .and_then(Value::as_str)
                        .map(normalize_image_ref),
                    url: seg_data
                        .get("url")
                        .and_then(Value::as_str)
                        .map(normalize_image_ref),
                    busid: seg_data
                        .get("busid")
                        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.parse::<i64>().ok())),
                    size: seg_data.get("size").and_then(|v| {
                        v.as_u64()
                            .or_else(|| v.as_i64().and_then(|n| u64::try_from(n).ok()))
                            .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
                    }),
                };
                let dedup_key = file_ref_dedup_key(&file_ref);
                if !dedup_key.is_empty() && seen.insert(dedup_key) {
                    out.push(file_ref);
                }
            }
        }
        _ => {}
    }
}

/// 从单个 `message` 值里解析转发消息 id。
fn collect_forward_ids_from_message_value(
    message: &Value,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    match message {
        Value::String(raw) => {
            for id in extract_cq_forward_ids(raw) {
                if seen.insert(id.clone()) {
                    out.push(id);
                }
            }
        }
        Value::Array(segments) => {
            for seg in segments {
                if seg.get("type").and_then(Value::as_str) != Some("forward") {
                    continue;
                }
                let Some(seg_data) = seg.get("data") else {
                    continue;
                };
                let Some(id) = seg_data
                    .get("id")
                    .and_then(|v| {
                        v.as_str()
                            .map(normalize_image_ref)
                            .or_else(|| v.as_i64().map(|n| n.to_string()))
                            .or_else(|| v.as_u64().map(|n| n.to_string()))
                    })
                    .filter(|v| !v.is_empty())
                else {
                    continue;
                };
                if seen.insert(id.clone()) {
                    out.push(id);
                }
            }
        }
        _ => {}
    }
}

/// 构造文件引用的去重键。
fn file_ref_dedup_key(file_ref: &MessageFileRef) -> String {
    [
        file_ref.name.clone().unwrap_or_default(),
        file_ref.file_id.clone().unwrap_or_default(),
        file_ref.file.clone().unwrap_or_default(),
        file_ref.url.clone().unwrap_or_default(),
    ]
    .join("|")
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

/// 解析文件引用、拉取内容并生成适合注入上下文的文本块。
async fn load_file_context(
    client: &reqwest::Client,
    bridge: &WsActionBridge,
    message_type: &str,
    group_id: Option<i64>,
    file_ref: &MessageFileRef,
    quoted: bool,
    debug: bool,
) -> Option<String> {
    let file_label = file_ref
        .name
        .clone()
        .or_else(|| file_ref.file_id.clone())
        .or_else(|| file_ref.file.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let Some(source) = resolve_file_source(bridge, message_type, group_id, file_ref, debug).await
    else {
        return None;
    };

    let Some(downloaded) = download_file_preview(client, &source, file_ref.size, debug).await
    else {
        return Some(format!(
            "{} {}（文件过大或无法读取）",
            if quoted { "[引用文件]" } else { "[文件]" },
            file_label
        ));
    };

    if !looks_like_text_file(
        &file_label,
        downloaded.content_type.as_deref(),
        &downloaded.bytes,
    ) {
        return Some(format!(
            "{} {}（非文本文件，未注入内容）",
            if quoted { "[引用文件]" } else { "[文件]" },
            file_label
        ));
    }

    let mut text = String::from_utf8_lossy(&downloaded.bytes).to_string();
    if downloaded.truncated_by_bytes {
        text.push_str("\n...(file truncated by byte limit)");
    }
    let preview = trim_for_context(&text, FILE_PREVIEW_MAX_CHARS);
    Some(format!(
        "{} {}\n{}",
        if quoted {
            "[引用文件内容]"
        } else {
            "[文件内容]"
        },
        file_label,
        preview
    ))
}

const FORWARD_MAX_NODES: usize = 12;
const FORWARD_MAX_IMAGES: usize = 8;
const FORWARD_PREVIEW_MAX_CHARS: usize = 12_000;

struct ForwardContextBundle {
    text: Option<String>,
    urls: Vec<String>,
    files: Vec<String>,
}

/// 读取一条转发聊天记录，并把其中可读文本与图片引用整理出来。
async fn load_forward_context_bundle(
    bridge: &WsActionBridge,
    forward_id: &str,
    quoted: bool,
    debug: bool,
) -> Option<ForwardContextBundle> {
    let response = match bridge
        .call_action("get_forward_msg", json!({ "id": forward_id }))
        .await
    {
        Ok(v) => v,
        Err(err) => {
            log_debug(
                debug,
                format!("get_forward_msg failed id={forward_id}: {err}"),
            );
            return None;
        }
    };

    let data = response.get("data").cloned().unwrap_or(Value::Null);
    let mut lines = Vec::new();
    let mut urls = Vec::new();
    let mut files = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut seen_files = HashSet::new();
    collect_forward_lines_from_value(&data, &mut lines, 0);
    collect_forward_image_refs_from_value(
        &data,
        &mut urls,
        &mut files,
        &mut seen_urls,
        &mut seen_files,
        0,
    );

    let text = if lines.is_empty() {
        Some(format!(
            "{} [转发消息:id={forward_id}]（未解析到文本内容）",
            if quoted {
                "[引用转发消息]"
            } else {
                "[转发消息内容]"
            }
        ))
    } else {
        let text = lines
            .into_iter()
            .take(FORWARD_MAX_NODES)
            .collect::<Vec<_>>()
            .join("\n");
        Some(format!(
            "{}\n{}",
            if quoted {
                "[引用转发消息]"
            } else {
                "[转发消息内容]"
            },
            trim_for_context(&text, FORWARD_PREVIEW_MAX_CHARS)
        ))
    };

    Some(ForwardContextBundle {
        text,
        urls: urls.into_iter().take(FORWARD_MAX_IMAGES).collect(),
        files: files.into_iter().take(FORWARD_MAX_IMAGES).collect(),
    })
}

/// 递归解析 `get_forward_msg` 返回体里的转发节点文本。
fn collect_forward_lines_from_value(value: &Value, out: &mut Vec<String>, depth: usize) {
    if depth > 4 || out.len() >= FORWARD_MAX_NODES {
        return;
    }

    match value {
        Value::Array(items) => {
            for item in items {
                collect_forward_lines_from_value(item, out, depth + 1);
                if out.len() >= FORWARD_MAX_NODES {
                    break;
                }
            }
        }
        Value::Object(map) => {
            if let Some(messages) = map
                .get("messages")
                .or_else(|| map.get("message"))
                .or_else(|| map.get("content"))
            {
                match messages {
                    Value::Array(_) | Value::Object(_) => {
                        collect_forward_lines_from_value(messages, out, depth + 1);
                    }
                    Value::String(raw) => {
                        let text = strip_cq_to_text(raw);
                        let line = attach_forward_sender_prefix(map, text);
                        if !line.is_empty() {
                            out.push(trim_for_context(&line, 500));
                        }
                    }
                    _ => {}
                }
            }

            if out.len() >= FORWARD_MAX_NODES {
                return;
            }

            if let Some(text) = message_value_to_text(value) {
                let line = attach_forward_sender_prefix(map, text);
                if !line.is_empty() {
                    out.push(trim_for_context(&line, 500));
                }
            }
        }
        Value::String(raw) => {
            let text = strip_cq_to_text(raw);
            if !text.trim().is_empty() {
                out.push(trim_for_context(text.trim(), 500));
            }
        }
        _ => {}
    }
}

/// 递归解析 `get_forward_msg` 返回体里的图片引用。
fn collect_forward_image_refs_from_value(
    value: &Value,
    urls: &mut Vec<String>,
    files: &mut Vec<String>,
    seen_urls: &mut HashSet<String>,
    seen_files: &mut HashSet<String>,
    depth: usize,
) {
    if depth > 4 || (urls.len() + files.len()) >= FORWARD_MAX_IMAGES {
        return;
    }

    match value {
        Value::Array(items) => {
            for item in items {
                collect_forward_image_refs_from_value(
                    item,
                    urls,
                    files,
                    seen_urls,
                    seen_files,
                    depth + 1,
                );
                if (urls.len() + files.len()) >= FORWARD_MAX_IMAGES {
                    break;
                }
            }
        }
        Value::Object(map) => {
            if let Some(messages) = map.get("messages").or_else(|| map.get("message")) {
                collect_image_refs_from_message_value(messages, urls, files, seen_urls, seen_files);
                collect_forward_image_refs_from_value(
                    messages,
                    urls,
                    files,
                    seen_urls,
                    seen_files,
                    depth + 1,
                );
            }
            if let Some(content) = map.get("content") {
                collect_image_refs_from_message_value(content, urls, files, seen_urls, seen_files);
                collect_forward_image_refs_from_value(
                    content,
                    urls,
                    files,
                    seen_urls,
                    seen_files,
                    depth + 1,
                );
            }
        }
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
                if (urls.len() + files.len()) >= FORWARD_MAX_IMAGES {
                    break;
                }
            }
        }
        _ => {}
    }
}

/// 给转发节点文本补上发送者前缀，避免摘要失去说话人信息。
fn attach_forward_sender_prefix(node: &serde_json::Map<String, Value>, text: String) -> String {
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }

    let sender = node
        .get("nickname")
        .or_else(|| node.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .or_else(|| {
            node.get("sender")
                .and_then(Value::as_object)
                .and_then(|sender| {
                    sender
                        .get("nickname")
                        .or_else(|| sender.get("name"))
                        .and_then(Value::as_str)
                })
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        });

    match sender {
        Some(sender) => format!("[{sender}] {text}"),
        None => text.to_string(),
    }
}

/// 文件下载来源，可以是 URL，也可以是本地路径。
enum FileSource {
    Url(String),
    LocalPath(String),
}

struct DownloadedFilePreview {
    bytes: Vec<u8>,
    content_type: Option<String>,
    truncated_by_bytes: bool,
}

/// 尽量把 OneBot/NapCat 文件引用解析成可读源。
async fn resolve_file_source(
    bridge: &WsActionBridge,
    message_type: &str,
    group_id: Option<i64>,
    file_ref: &MessageFileRef,
    debug: bool,
) -> Option<FileSource> {
    if let Some(url) = file_ref.url.as_deref() {
        let url = normalize_image_ref(url);
        if looks_like_http_url(&url) {
            return Some(FileSource::Url(url));
        }
        if looks_like_local_path(&url) {
            return Some(FileSource::LocalPath(url));
        }
        if url.starts_with("file://") {
            return Some(FileSource::LocalPath(
                url.trim_start_matches("file://").to_string(),
            ));
        }
    }

    if let Some(file) = file_ref.file.as_deref() {
        let file = normalize_image_ref(file);
        if looks_like_http_url(&file) {
            return Some(FileSource::Url(file));
        }
        if looks_like_local_path(&file) {
            return Some(FileSource::LocalPath(file));
        }
    }

    let api_response = if message_type == "group" {
        let group_id = group_id?;
        let file_id = file_ref
            .file_id
            .as_ref()
            .or(file_ref.file.as_ref())
            .cloned()
            .unwrap_or_default();
        if file_id.is_empty() {
            None
        } else {
            let mut payload = json!({
                "group_id": group_id,
                "file_id": file_id,
            });
            if let Some(busid) = file_ref.busid {
                payload["busid"] = json!(busid);
            }
            bridge.call_action("get_group_file_url", payload).await.ok()
        }
    } else {
        let file_id = file_ref
            .file_id
            .as_ref()
            .or(file_ref.file.as_ref())
            .cloned()
            .unwrap_or_default();
        if file_id.is_empty() {
            None
        } else {
            bridge
                .call_action("get_private_file_url", json!({ "file_id": file_id }))
                .await
                .ok()
        }
    };

    if let Some(response) = api_response {
        if let Some(source) = file_source_from_onebot_response(&response) {
            return Some(source);
        }
        log_debug(
            debug,
            format!(
                "file url resolution returned unsupported data: {}",
                response.get("data").cloned().unwrap_or(Value::Null)
            ),
        );
    }

    None
}

/// 从 OneBot 文件接口返回体中提取 URL 或本地路径。
fn file_source_from_onebot_response(response: &Value) -> Option<FileSource> {
    let data = response.get("data")?;
    if let Some(url) = data.as_str() {
        return file_source_from_string(url);
    }

    for key in ["url", "download_url", "file_url", "file", "path"] {
        if let Some(value) = data.get(key).and_then(Value::as_str) {
            if let Some(source) = file_source_from_string(value) {
                return Some(source);
            }
        }
    }

    None
}

fn file_source_from_string(value: &str) -> Option<FileSource> {
    let value = normalize_image_ref(value);
    if value.is_empty() {
        return None;
    }
    if looks_like_http_url(&value) {
        return Some(FileSource::Url(value));
    }
    if value.starts_with("file://") {
        return Some(FileSource::LocalPath(
            value.trim_start_matches("file://").to_string(),
        ));
    }
    if looks_like_local_path(&value) {
        return Some(FileSource::LocalPath(value));
    }
    None
}

/// 下载文件预览，允许大文件下载但限制注入上下文的字节数。
async fn download_file_preview(
    client: &reqwest::Client,
    source: &FileSource,
    declared_size: Option<u64>,
    debug: bool,
) -> Option<DownloadedFilePreview> {
    if declared_size.unwrap_or(0) > FILE_FETCH_MAX_BYTES {
        log_debug(
            debug,
            format!(
                "skip oversized file declared_size={}",
                declared_size.unwrap_or(0)
            ),
        );
        return None;
    }

    match source {
        FileSource::LocalPath(path) => {
            let bytes = fs::read(path).await.ok()?;
            if bytes.len() as u64 > FILE_FETCH_MAX_BYTES {
                return None;
            }
            let truncated_by_bytes = bytes.len() > FILE_INLINE_MAX_BYTES;
            let bytes = bytes.into_iter().take(FILE_INLINE_MAX_BYTES).collect();
            Some(DownloadedFilePreview {
                bytes,
                content_type: None,
                truncated_by_bytes,
            })
        }
        FileSource::Url(url) => {
            let response = client
                .get(url)
                .header("User-Agent", "Mozilla/5.0")
                .send()
                .await
                .ok()?;
            if !response.status().is_success() {
                log_debug(
                    debug,
                    format!(
                        "file download failed url={} status={}",
                        url,
                        response.status()
                    ),
                );
                return None;
            }
            if response.content_length().unwrap_or(0) > FILE_FETCH_MAX_BYTES {
                return None;
            }
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.to_string());
            let mut response = response;
            let mut bytes = Vec::new();
            let mut truncated_by_bytes = false;
            while let Ok(Some(chunk)) = response.chunk().await {
                let remain = FILE_INLINE_MAX_BYTES.saturating_sub(bytes.len());
                if remain == 0 {
                    truncated_by_bytes = true;
                    break;
                }
                if chunk.len() > remain {
                    bytes.extend_from_slice(&chunk[..remain]);
                    truncated_by_bytes = true;
                    break;
                }
                bytes.extend_from_slice(&chunk);
                if bytes.len() as u64 >= FILE_FETCH_MAX_BYTES {
                    truncated_by_bytes = true;
                    break;
                }
            }
            Some(DownloadedFilePreview {
                bytes,
                content_type,
                truncated_by_bytes,
            })
        }
    }
}

/// 粗判文件是否像可直接注入上下文的文本文件。
fn looks_like_text_file(name: &str, content_type: Option<&str>, bytes: &[u8]) -> bool {
    if let Some(content_type) = content_type {
        let lower = content_type.to_lowercase();
        if lower.starts_with("text/")
            || lower.contains("json")
            || lower.contains("xml")
            || lower.contains("yaml")
            || lower.contains("toml")
            || lower.contains("csv")
            || lower.contains("javascript")
        {
            return true;
        }
    }

    if let Some(ext) = std::path::Path::new(name)
        .extension()
        .and_then(|v| v.to_str())
        .map(|v| v.to_lowercase())
    {
        if matches!(
            ext.as_str(),
            "txt"
                | "md"
                | "json"
                | "toml"
                | "yaml"
                | "yml"
                | "csv"
                | "log"
                | "rs"
                | "py"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "java"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "hpp"
                | "go"
                | "lua"
                | "php"
                | "rb"
                | "swift"
                | "sh"
                | "sql"
                | "html"
                | "css"
                | "xml"
        ) {
            return true;
        }
    }

    // 兜底：前几个 KB 内如果没有大量 NUL 字节，则把它当文本试着读。
    let sample = &bytes[..bytes.len().min(4096)];
    !sample.iter().any(|b| *b == 0)
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
