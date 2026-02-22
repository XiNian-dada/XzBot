use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct ParsedUserContent {
    pub text: String,
    pub image_urls: Vec<String>,
    pub image_files: Vec<String>,
}

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

fn clean_marker_value(value: &str) -> String {
    let value = value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string();
    decode_html_entities_basic(&value)
}

fn decode_html_entities_basic(input: &str) -> String {
    input
        .replace("&amp;", "&")
        .replace("&#38;", "&")
        .replace("&quot;", "\"")
}

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
