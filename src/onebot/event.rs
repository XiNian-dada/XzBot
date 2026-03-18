//! OneBot 事件模型：负责事件反序列化和消息内容提取。

use serde::Deserialize;
use serde_json::Value;

/// Minimal message event fields used by the bot runtime.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageEvent {
    /// OneBot post type, only `message` is handled by router.
    pub post_type: String,
    /// `private` or `group`.
    pub message_type: String,
    /// Bot QQ id.
    pub self_id: i64,
    /// Sender QQ id.
    pub user_id: i64,
    /// Sender profile info.
    #[serde(default)]
    pub sender: SenderInfo,
    /// Group id for group messages.
    #[serde(default)]
    pub group_id: Option<i64>,
    /// Raw CQ-coded text.
    #[serde(default)]
    pub raw_message: String,
    /// Structured message payload when provider sends segment array.
    #[serde(default)]
    pub message: MessagePayload,
    /// Runtime-enriched text/image context appended after base message text.
    ///
    /// 这部分不来自 OneBot 原始上报，而是在运行时通过 `get_msg` / `get_image`
    /// 等补充出来，供后续 AI 与路由统一读取。
    #[serde(skip)]
    pub enriched_parts: Vec<String>,
}

/// Message payload variants used by different OneBot implementations.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessagePayload {
    /// Plain text payload.
    Text(String),
    /// Segment payload (`text`, `at`, `image`, ...).
    Segments(Vec<MessageSegment>),
}

impl Default for MessagePayload {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// One segment inside structured message payload.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageSegment {
    /// Segment type.
    #[serde(rename = "type")]
    pub kind: String,
    /// Segment data map.
    #[serde(default)]
    pub data: Value,
}

/// Sender display metadata.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SenderInfo {
    /// QQ nickname.
    #[serde(default)]
    pub nickname: String,
    /// Group card.
    #[serde(default)]
    pub card: String,
}

/// Parsed CQ image reference with optional url/file fields.
#[derive(Debug, Clone)]
pub struct CqImageRef {
    pub url: Option<String>,
    pub file: Option<String>,
}

/// Parsed file reference extracted from OneBot message segments.
#[derive(Debug, Clone)]
pub struct MessageFileRef {
    pub name: Option<String>,
    pub file_id: Option<String>,
    pub file: Option<String>,
    pub url: Option<String>,
    pub busid: Option<i64>,
    pub size: Option<u64>,
}

impl MessageEvent {
    /// Returns normalized text with image markers merged from segments and raw CQ message.
    pub fn text(&self) -> String {
        let mut text = self.base_text();
        for part in &self.enriched_parts {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(trimmed);
        }
        text
    }

    /// Returns normalized text produced only from original OneBot payload.
    ///
    /// 与 `text()` 的区别在于：这里不包含运行时补充的引用文本、解析后图片 URL、
    /// 最近群聊摘录等附加上下文。
    ///
    /// 这个方法主要给“触发判定”使用：
    /// - 是否需要响应群消息
    /// - 是否命中插件命令
    /// - 是否属于加一复读
    ///
    /// 换句话说，运行时补充出来的环境信息只应该帮助 AI 理解上下文，
    /// 不应该反过来改变“这条消息本身是否触发机器人”。
    pub fn original_text(&self) -> String {
        self.base_text()
    }

    /// Returns normalized text produced only from original OneBot payload.
    ///
    /// 与 `text()` 的区别在于：这里不包含运行时补充的引用文本、解析后图片 URL 等附加上下文。
    fn base_text(&self) -> String {
        // 优先使用结构化消息段，图片/at 等内容在这里更完整。
        if let MessagePayload::Segments(segments) = &self.message {
            let mut normalized = segments_to_text(segments);
            // 某些实现只在 raw_message 的 CQ 码里带 url，补齐到标准占位。
            for image_ref in extract_cq_image_refs(&self.raw_message) {
                if !contains_image_ref_marker(&normalized, &image_ref) {
                    let marker = image_ref_to_marker(&image_ref);
                    if !normalized.is_empty() {
                        normalized.push(' ');
                    }
                    normalized.push_str(&marker);
                }
            }
            if !normalized.trim().is_empty() {
                return normalized;
            }
        }

        if let MessagePayload::Text(text) = &self.message {
            let mut normalized = text.clone();
            for image_ref in extract_cq_image_refs(&self.raw_message) {
                if !contains_image_ref_marker(&normalized, &image_ref) {
                    let marker = image_ref_to_marker(&image_ref);
                    if !normalized.is_empty() {
                        normalized.push(' ');
                    }
                    normalized.push_str(&marker);
                }
            }
            if !normalized.trim().is_empty() {
                return normalized;
            }
        }

        if !self.raw_message.trim().is_empty() {
            return self.raw_message.clone();
        }

        String::new()
    }

    /// Appends one runtime-generated context fragment.
    ///
    /// 这里的片段会被 `text()` 自动接到原始消息后面，而不会污染 `raw_message`。
    pub fn push_enriched_part(&mut self, part: impl Into<String>) {
        let part = part.into();
        if part.trim().is_empty() {
            return;
        }
        self.enriched_parts.push(part);
    }

    /// Returns best-effort sender display name.
    pub fn display_name(&self) -> String {
        // 群聊优先用群名片，私聊/兜底用昵称，最终回退到 QQ 号。
        if self.message_type == "group" {
            let card = self.sender.card.trim();
            if !card.is_empty() {
                return card.to_string();
            }
        }

        let nickname = self.sender.nickname.trim();
        if !nickname.is_empty() {
            return nickname.to_string();
        }

        self.user_id.to_string()
    }

    /// Returns deduplicated image file ids in current event.
    pub fn image_file_ids(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();

        if let MessagePayload::Segments(segments) = &self.message {
            for segment in segments {
                if segment.kind != "image" {
                    continue;
                }
                if let Some(file) = segment.data.get("file").and_then(Value::as_str) {
                    let file = file.trim();
                    if !file.is_empty() && seen.insert(file.to_string()) {
                        out.push(file.to_string());
                    }
                }
            }
        }

        for image_ref in extract_cq_image_refs(&self.raw_message) {
            if let Some(file) = image_ref.file {
                if !file.is_empty() && seen.insert(file.clone()) {
                    out.push(file);
                }
            }
        }

        out
    }

    /// Returns deduplicated replied message ids in current event.
    pub fn reply_message_ids(&self) -> Vec<i64> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();

        if let MessagePayload::Segments(segments) = &self.message {
            for segment in segments {
                if segment.kind != "reply" {
                    continue;
                }
                if let Some(id) = extract_i64_from_value(segment.data.get("id")) {
                    if seen.insert(id) {
                        out.push(id);
                    }
                }
            }
        }

        for id in extract_cq_reply_ids(&self.raw_message) {
            if seen.insert(id) {
                out.push(id);
            }
        }

        out
    }

    /// Returns deduplicated file references in current event.
    pub fn file_refs(&self) -> Vec<MessageFileRef> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();

        if let MessagePayload::Segments(segments) = &self.message {
            for segment in segments {
                if segment.kind != "file" {
                    continue;
                }
                let file_ref = file_ref_from_segment_data(&segment.data);
                if let Some(file_ref) = dedup_file_ref(file_ref, &mut seen) {
                    out.push(file_ref);
                }
            }
        }

        if let MessagePayload::Text(text) = &self.message {
            for file_ref in extract_cq_file_refs(text) {
                if let Some(file_ref) = dedup_file_ref(file_ref, &mut seen) {
                    out.push(file_ref);
                }
            }
        }

        for file_ref in extract_cq_file_refs(&self.raw_message) {
            if let Some(file_ref) = dedup_file_ref(file_ref, &mut seen) {
                out.push(file_ref);
            }
        }

        out
    }

    /// Returns deduplicated forward-message ids in current event.
    pub fn forward_ids(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();

        if let MessagePayload::Segments(segments) = &self.message {
            for segment in segments {
                if segment.kind != "forward" {
                    continue;
                }
                if let Some(id) = extract_string_like_value(segment.data.get("id")) {
                    if seen.insert(id.clone()) {
                        out.push(id);
                    }
                }
            }
        }

        if let MessagePayload::Text(text) = &self.message {
            for id in extract_cq_forward_ids(text) {
                if seen.insert(id.clone()) {
                    out.push(id);
                }
            }
        }

        for id in extract_cq_forward_ids(&self.raw_message) {
            if seen.insert(id.clone()) {
                out.push(id);
            }
        }

        out
    }
}

/// Converts segment array into normalized message text.
fn segments_to_text(segments: &[MessageSegment]) -> String {
    let mut out = String::new();

    for segment in segments {
        match segment.kind.as_str() {
            "text" => {
                if let Some(text) = segment.data.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
            "at" => {
                if let Some(qq) = segment.data.get("qq").and_then(Value::as_i64) {
                    out.push_str(&format!("[CQ:at,qq={qq}]"));
                } else if let Some(qq) = segment.data.get("qq").and_then(Value::as_str) {
                    out.push_str(&format!("[CQ:at,qq={qq}]"));
                }
            }
            "image" => {
                let url = segment
                    .data
                    .get("url")
                    .and_then(Value::as_str)
                    .map(normalize_image_ref_value);
                let file = segment
                    .data
                    .get("file")
                    .and_then(Value::as_str)
                    .map(normalize_image_ref_value);
                out.push_str(&image_ref_to_marker(&CqImageRef { url, file }));
            }
            "face" => {
                if let Some(id) = extract_i64_from_value(segment.data.get("id")) {
                    out.push_str(&format!("[CQ:face,id={id}]"));
                }
            }
            "file" => {
                let file_ref = file_ref_from_segment_data(&segment.data);
                out.push_str(&file_ref_to_marker(&file_ref));
            }
            "forward" => out.push_str("[转发消息]"),
            _ => {}
        }
    }

    out
}

/// Extracts `[CQ:image,...]` references from raw message.
pub fn extract_cq_image_refs(raw: &str) -> Vec<CqImageRef> {
    let mut out = Vec::new();
    let mut cursor = 0;
    let mut seen = std::collections::HashSet::new();

    while let Some(start_rel) = raw[cursor..].find("[CQ:image,") {
        let start = cursor + start_rel;
        let Some(end_rel) = raw[start..].find(']') else {
            break;
        };
        let end = start + end_rel;
        let segment = &raw[start + 1..end]; // CQ:image,...
        let mut url = None::<String>;
        let mut file = None::<String>;
        for field in segment.split(',') {
            if let Some(value) = field.trim().strip_prefix("url=") {
                let value = normalize_image_ref_value(value);
                if !value.is_empty() {
                    url = Some(value);
                }
            }
            if let Some(value) = field.trim().strip_prefix("file=") {
                let value = normalize_image_ref_value(value);
                if !value.is_empty() {
                    file = Some(value);
                }
            }
        }

        let image_ref = CqImageRef { url, file };
        let dedup_key = format!(
            "{}|{}",
            image_ref.url.clone().unwrap_or_default(),
            image_ref.file.clone().unwrap_or_default()
        );
        if seen.insert(dedup_key) {
            out.push(image_ref);
        }
        cursor = end + 1;
    }

    out
}

/// Extracts `[CQ:reply,id=...]` ids from raw message.
pub fn extract_cq_reply_ids(raw: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let mut cursor = 0;
    let mut seen = std::collections::HashSet::new();

    while let Some(start_rel) = raw[cursor..].find("[CQ:reply,") {
        let start = cursor + start_rel;
        let Some(end_rel) = raw[start..].find(']') else {
            break;
        };
        let end = start + end_rel;
        let segment = &raw[start + 1..end]; // CQ:reply,...

        for field in segment.split(',') {
            if let Some(id_str) = field.trim().strip_prefix("id=") {
                if let Ok(id) = id_str
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .parse::<i64>()
                {
                    if seen.insert(id) {
                        out.push(id);
                    }
                }
            }
        }

        cursor = end + 1;
    }

    out
}

/// Extracts `[CQ:file,...]` references from raw message.
pub fn extract_cq_file_refs(raw: &str) -> Vec<MessageFileRef> {
    let mut out = Vec::new();
    let mut cursor = 0;
    let mut seen = std::collections::HashSet::new();

    while let Some(start_rel) = raw[cursor..].find("[CQ:file,") {
        let start = cursor + start_rel;
        let Some(end_rel) = raw[start..].find(']') else {
            break;
        };
        let end = start + end_rel;
        let segment = &raw[start + 1..end];

        let mut file_ref = MessageFileRef {
            name: None,
            file_id: None,
            file: None,
            url: None,
            busid: None,
            size: None,
        };

        for field in segment.split(',') {
            let field = field.trim();
            if let Some(value) = field.strip_prefix("name=") {
                let value = normalize_message_ref_value(value);
                if !value.is_empty() {
                    file_ref.name = Some(value);
                }
            }
            if let Some(value) = field.strip_prefix("file_id=") {
                let value = normalize_message_ref_value(value);
                if !value.is_empty() {
                    file_ref.file_id = Some(value);
                }
            }
            if let Some(value) = field.strip_prefix("file=") {
                let value = normalize_message_ref_value(value);
                if !value.is_empty() {
                    file_ref.file = Some(value);
                }
            }
            if let Some(value) = field.strip_prefix("url=") {
                let value = normalize_message_ref_value(value);
                if !value.is_empty() {
                    file_ref.url = Some(value);
                }
            }
            if let Some(value) = field.strip_prefix("busid=") {
                file_ref.busid = value.trim().parse::<i64>().ok();
            }
            if let Some(value) = field.strip_prefix("size=") {
                file_ref.size = value.trim().parse::<u64>().ok();
            }
        }

        if let Some(file_ref) = dedup_file_ref(file_ref, &mut seen) {
            out.push(file_ref);
        }
        cursor = end + 1;
    }

    out
}

/// Extracts `[CQ:forward,id=...]` ids from raw message.
pub fn extract_cq_forward_ids(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0;
    let mut seen = std::collections::HashSet::new();

    while let Some(start_rel) = raw[cursor..].find("[CQ:forward,") {
        let start = cursor + start_rel;
        let Some(end_rel) = raw[start..].find(']') else {
            break;
        };
        let end = start + end_rel;
        let segment = &raw[start + 1..end];

        for field in segment.split(',') {
            if let Some(id) = field.trim().strip_prefix("id=") {
                let id = normalize_message_ref_value(id);
                if !id.is_empty() && seen.insert(id.clone()) {
                    out.push(id);
                }
            }
        }

        cursor = end + 1;
    }

    out
}

/// Encodes parsed image reference into normalized internal marker.
fn image_ref_to_marker(image_ref: &CqImageRef) -> String {
    // 优先保留 url，避免同时携带 file 导致后续重复加载/无效 file 引用。
    match (&image_ref.url, &image_ref.file) {
        (Some(url), Some(_)) => format!("[IMAGE:url={url}]"),
        (Some(url), None) => format!("[IMAGE:url={url}]"),
        (None, Some(file)) => format!("[IMAGE:file={file}]"),
        (None, None) => "[IMAGE]".to_string(),
    }
}

/// Encodes parsed file reference into a normalized internal marker.
fn file_ref_to_marker(file_ref: &MessageFileRef) -> String {
    let label = file_ref
        .name
        .as_deref()
        .or(file_ref.file_id.as_deref())
        .or(file_ref.file.as_deref())
        .unwrap_or("unknown");
    format!("[FILE:name={label}]")
}

/// Reads i64 from either numeric JSON or string JSON.
fn extract_i64_from_value(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    if let Some(v) = value.as_i64() {
        return Some(v);
    }
    if let Some(v) = value.as_str() {
        return v.trim().parse::<i64>().ok();
    }
    None
}

/// Reads string-ish value from JSON numbers or strings.
fn extract_string_like_value(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(v) = value.as_str() {
        let v = normalize_message_ref_value(v);
        if !v.is_empty() {
            return Some(v);
        }
    }
    if let Some(v) = value.as_i64() {
        return Some(v.to_string());
    }
    if let Some(v) = value.as_u64() {
        return Some(v.to_string());
    }
    None
}

/// Normalizes escaped/quoted image reference values from CQ fields.
fn normalize_image_ref_value(value: &str) -> String {
    normalize_message_ref_value(value)
}

/// Normalizes escaped/quoted CQ field values.
fn normalize_message_ref_value(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace("&amp;", "&")
        .replace("&#38;", "&")
}

/// Checks whether a normalized marker already exists in text.
fn contains_image_ref_marker(text: &str, image_ref: &CqImageRef) -> bool {
    let url_hit = image_ref
        .url
        .as_ref()
        .map(|url| text.contains(&format!("url={url}")))
        .unwrap_or(false);
    let file_hit = image_ref
        .file
        .as_ref()
        .map(|file| text.contains(&format!("file={file}")))
        .unwrap_or(false);

    match (&image_ref.url, &image_ref.file) {
        (Some(_), Some(_)) => url_hit && file_hit,
        (Some(_), None) => url_hit,
        (None, Some(_)) => file_hit,
        (None, None) => false,
    }
}

/// Parses a structured `file` segment into a normalized file reference.
fn file_ref_from_segment_data(data: &Value) -> MessageFileRef {
    let name = data
        .get("name")
        .or_else(|| data.get("file_name"))
        .and_then(Value::as_str)
        .map(normalize_message_ref_value)
        .filter(|v| !v.is_empty());
    let file_id = data
        .get("file_id")
        .or_else(|| data.get("id"))
        .and_then(Value::as_str)
        .map(normalize_message_ref_value)
        .filter(|v| !v.is_empty());
    let file = data
        .get("file")
        .and_then(Value::as_str)
        .map(normalize_message_ref_value)
        .filter(|v| !v.is_empty());
    let url = data
        .get("url")
        .and_then(Value::as_str)
        .map(normalize_message_ref_value)
        .filter(|v| !v.is_empty());
    let busid = extract_i64_from_value(data.get("busid"));
    let size = data
        .get("size")
        .or_else(|| data.get("file_size"))
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
                .or_else(|| value.as_str().and_then(|v| v.trim().parse::<u64>().ok()))
        });

    MessageFileRef {
        name,
        file_id,
        file,
        url,
        busid,
        size,
    }
}

/// Deduplicates file refs by their strongest identity fields and drops empty refs.
fn dedup_file_ref(
    file_ref: MessageFileRef,
    seen: &mut std::collections::HashSet<String>,
) -> Option<MessageFileRef> {
    let key = format!(
        "{}|{}|{}|{}",
        file_ref.name.clone().unwrap_or_default(),
        file_ref.file_id.clone().unwrap_or_default(),
        file_ref.file.clone().unwrap_or_default(),
        file_ref.url.clone().unwrap_or_default()
    );
    if key == "|||" || !seen.insert(key) {
        return None;
    }
    Some(file_ref)
}
