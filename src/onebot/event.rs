use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct MessageEvent {
    pub post_type: String,
    pub message_type: String,
    pub self_id: i64,
    pub user_id: i64,
    #[serde(default)]
    pub sender: SenderInfo,
    #[serde(default)]
    pub group_id: Option<i64>,
    #[serde(default)]
    pub raw_message: String,
    #[serde(default)]
    pub message: MessagePayload,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessagePayload {
    Text(String),
    Segments(Vec<MessageSegment>),
}

impl Default for MessagePayload {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessageSegment {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub data: Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SenderInfo {
    #[serde(default)]
    pub nickname: String,
    #[serde(default)]
    pub card: String,
}

impl MessageEvent {
    pub fn text(&self) -> String {
        // 优先使用结构化消息段，图片/at 等内容在这里更完整。
        if let MessagePayload::Segments(segments) = &self.message {
            let mut normalized = segments_to_text(segments);
            // 某些实现只在 raw_message 的 CQ 码里带 url，补齐到标准占位。
            for url in extract_cq_image_urls(&self.raw_message) {
                let marker = format!("[IMAGE:url={url}]");
                if !normalized.contains(&marker) {
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
            if !text.trim().is_empty() {
                return text.clone();
            }
        }

        if !self.raw_message.trim().is_empty() {
            return self.raw_message.clone();
        }

        String::new()
    }

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
}

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
                if let Some(url) = segment.data.get("url").and_then(Value::as_str) {
                    out.push_str(&format!("[IMAGE:url={url}]"));
                } else if let Some(file) = segment.data.get("file").and_then(Value::as_str) {
                    out.push_str(&format!("[IMAGE:file={file}]"));
                } else {
                    out.push_str("[IMAGE]");
                }
            }
            _ => {}
        }
    }

    out
}

fn extract_cq_image_urls(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0;

    while let Some(start_rel) = raw[cursor..].find("[CQ:image,") {
        let start = cursor + start_rel;
        let Some(end_rel) = raw[start..].find(']') else {
            break;
        };
        let end = start + end_rel;
        let segment = &raw[start + 1..end]; // CQ:image,...
        for field in segment.split(',') {
            if let Some(url) = field.trim().strip_prefix("url=") {
                let url = url.trim().trim_matches('"').trim_matches('\'');
                if !url.is_empty() && !out.iter().any(|v| v == url) {
                    out.push(url.to_string());
                }
            }
        }
        cursor = end + 1;
    }

    out
}
