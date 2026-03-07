//! 用户消息拆分工具：把文本、图片标记和附件引用拆成结构化片段。

use std::collections::HashSet;

/// Parsed representation of user message containing text and image references.
#[derive(Debug, Clone)]
pub struct ParsedUserContent {
    /// Text content with image markers replaced by `[图片]`.
    pub text: String,
    /// Deduplicated image URLs extracted from markers.
    pub image_urls: Vec<String>,
    /// Deduplicated image file ids extracted from markers.
    pub image_files: Vec<String>,
}

/// Parses internal `[IMAGE:...]` markers out of user content.
pub fn parse_user_content(raw: &str) -> ParsedUserContent {
    let mut text = String::new();
    let mut image_urls = Vec::new();
    let mut image_files = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut seen_files = HashSet::new();

    let mut cursor = 0;
    while let Some(start_rel) = raw[cursor..].find("[IMAGE") {
        let start = cursor + start_rel;
        text.push_str(&raw[cursor..start]);

        let Some(end_rel) = raw[start..].find(']') else {
            text.push_str(&raw[start..]);
            cursor = raw.len();
            break;
        };
        let end = start + end_rel;

        let marker = &raw[start + 1..end]; // strip leading '[' and trailing ']'
        if let Some(payload) = marker.strip_prefix("IMAGE:") {
            parse_image_marker_payload(
                payload,
                &mut image_urls,
                &mut seen_urls,
                &mut image_files,
                &mut seen_files,
            );
        }

        text.push_str("[图片]");
        cursor = end + 1;
    }

    text.push_str(&raw[cursor..]);

    ParsedUserContent {
        text: compact_spaces(&text),
        image_urls,
        image_files,
    }
}

/// Parses marker payload fields (`url=...`, `file=...`).
fn parse_image_marker_payload(
    payload: &str,
    image_urls: &mut Vec<String>,
    seen_urls: &mut HashSet<String>,
    image_files: &mut Vec<String>,
    seen_files: &mut HashSet<String>,
) {
    for part in payload.split(',') {
        let item = part.trim();
        if let Some(url) = item.strip_prefix("url=") {
            let v = clean_marker_value(url);
            if !v.is_empty() && seen_urls.insert(v.clone()) {
                image_urls.push(v);
            }
            continue;
        }
        if let Some(file) = item.strip_prefix("file=") {
            let v = clean_marker_value(file);
            if !v.is_empty() && seen_files.insert(v.clone()) {
                image_files.push(v);
            }
        }
    }
}

/// Cleans and decodes one marker value.
fn clean_marker_value(value: &str) -> String {
    let value = value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string();
    decode_html_entities_basic(&value)
}

/// Decodes minimal HTML entities that commonly appear in CQ payloads.
fn decode_html_entities_basic(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&#38;", "&")
        .replace("&quot;", "\"")
}

/// Compacts consecutive whitespace characters to single spaces.
fn compact_spaces(input: &str) -> String {
    let mut out = String::new();
    let mut prev_space = false;
    for ch in input.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}
