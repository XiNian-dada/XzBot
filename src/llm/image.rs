//! 图片加载工具：把 URL、本地路径或 base64 引用转换为模型可消费的内容。

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use serde_json::Value;
use tokio::fs;

const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
const DEFAULT_UA: &str = "Mozilla/5.0 (compatible; XzBot/1.0; +https://example.local)";

/// Normalized image payload passed into model-compatible request formats.
#[derive(Debug, Clone)]
pub struct ImageBinary {
    /// MIME type, e.g. `image/jpeg`.
    pub media_type: String,
    /// Base64-encoded image data.
    pub data_base64: String,
}

impl ImageBinary {
    /// Converts image payload to RFC2397 data URL.
    pub fn as_data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.data_base64)
    }
}

/// Resolves an image reference (`http`, `base64://`, `data:`, `file://`) to binary payload.
pub async fn load_image_for_llm(
    client: &Client,
    image_ref: &str,
    debug: bool,
) -> Result<ImageBinary> {
    let image_ref = normalize_image_ref(image_ref);
    if image_ref.starts_with("http://") || image_ref.starts_with("https://") {
        return fetch_image_by_url(client, &image_ref, debug).await;
    }
    if let Some(payload) = image_ref.strip_prefix("base64://") {
        return image_from_base64_payload(payload);
    }
    if let Some(data_url_payload) = image_ref.strip_prefix("data:") {
        return image_from_data_url_payload(data_url_payload);
    }
    if let Some(path) = image_ref.strip_prefix("file://") {
        return image_from_local_path(path).await;
    }

    bail!("unsupported image ref format: {image_ref}");
}

/// Fetches remote image with anti-hotlink headers and payload validation.
async fn fetch_image_by_url(client: &Client, url: &str, debug: bool) -> Result<ImageBinary> {
    if debug {
        println!("[DEBUG] fetch image for llm url={url}");
    }

    let mut request = client
        .get(url)
        .header("User-Agent", DEFAULT_UA)
        .header(
            "Accept",
            "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        )
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8");
    if url.contains("multimedia.nt.qq.com.cn") {
        request = request
            .header("Referer", "https://im.qq.com/")
            .header("Origin", "https://im.qq.com");
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("failed to fetch image: {url}"))?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .context("failed to read image response body")?;

    if !status.is_success() {
        bail!("image endpoint returned {status}");
    }
    if bytes.is_empty() {
        bail!("image response is empty");
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        bail!(
            "image is too large: {} bytes (limit: {} bytes)",
            bytes.len(),
            MAX_IMAGE_BYTES
        );
    }

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let image_magic = has_supported_image_magic(&bytes);
    if debug {
        println!(
            "[DEBUG] image fetch status={} content_type={} bytes={} magic={}",
            status,
            content_type,
            bytes.len(),
            image_magic
        );
    }

    if is_json_or_text_content_type(&content_type)
        || (!image_magic && !content_type.starts_with("image/"))
    {
        if let Some(api_err) = extract_remote_image_error(&bytes) {
            bail!("image endpoint returned non-image payload: {api_err}");
        }
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(160)
            .collect::<String>();
        bail!("image endpoint returned non-image payload: {preview}");
    }

    if !content_type.starts_with("image/") && !image_magic {
        bail!(
            "image endpoint returned unknown binary payload (content-type={})",
            content_type
        );
    }

    let media_type = normalize_media_type(&content_type, url, &bytes);
    Ok(ImageBinary {
        media_type,
        data_base64: STANDARD.encode(bytes),
    })
}

/// Decodes `base64://...` payload into normalized image.
fn image_from_base64_payload(payload: &str) -> Result<ImageBinary> {
    let bytes = STANDARD
        .decode(payload.trim())
        .context("failed to decode base64 image payload")?;
    ensure_image_size(&bytes)?;
    let media_type = detect_media_type_from_bytes(&bytes);
    Ok(ImageBinary {
        media_type,
        data_base64: STANDARD.encode(bytes),
    })
}

/// Decodes `data:image/...;base64,...` payload.
fn image_from_data_url_payload(payload: &str) -> Result<ImageBinary> {
    let Some((meta, b64)) = payload.split_once(",") else {
        bail!("invalid data URL");
    };
    let media_type = meta
        .split(';')
        .next()
        .unwrap_or("image/jpeg")
        .trim()
        .to_string();
    if !meta.contains(";base64") {
        bail!("data URL image must be base64 encoded");
    }
    let bytes = STANDARD
        .decode(b64.trim())
        .context("failed to decode data URL image base64 payload")?;
    ensure_image_size(&bytes)?;
    let media_type = if media_type.starts_with("image/") {
        media_type
    } else {
        detect_media_type_from_bytes(&bytes)
    };
    Ok(ImageBinary {
        media_type,
        data_base64: STANDARD.encode(bytes),
    })
}

/// Loads local image file referenced by `file://`.
async fn image_from_local_path(path: &str) -> Result<ImageBinary> {
    let path = path.trim();
    if path.is_empty() {
        bail!("empty local image path");
    }

    let decoded = urlencoding::decode(path)
        .map(|v| v.into_owned())
        .unwrap_or_else(|_| path.to_string());
    let bytes = fs::read(&decoded)
        .await
        .with_context(|| format!("failed to read local image path: {decoded}"))?;
    ensure_image_size(&bytes)?;
    let media_type = detect_media_type_from_bytes(&bytes);
    Ok(ImageBinary {
        media_type,
        data_base64: STANDARD.encode(bytes),
    })
}

/// Enforces common image size constraints.
fn ensure_image_size(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        bail!("image response is empty");
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        bail!(
            "image is too large: {} bytes (limit: {} bytes)",
            bytes.len(),
            MAX_IMAGE_BYTES
        );
    }
    Ok(())
}

/// Chooses final media type using header, URL hint, then magic bytes fallback.
fn normalize_media_type(content_type: &str, url: &str, bytes: &[u8]) -> String {
    let ct = content_type.split(';').next().unwrap_or("").trim();
    if is_supported_media_type(ct) {
        return ct.to_string();
    }

    let lower_url = url.to_ascii_lowercase();
    if lower_url.ends_with(".png") {
        return "image/png".to_string();
    }
    if lower_url.ends_with(".gif") {
        return "image/gif".to_string();
    }
    if lower_url.ends_with(".webp") {
        return "image/webp".to_string();
    }
    detect_media_type_from_bytes(bytes)
}

/// Detects media type from image magic bytes.
fn detect_media_type_from_bytes(bytes: &[u8]) -> String {
    if bytes.len() >= 3 && bytes[0..3] == [0xFF, 0xD8, 0xFF] {
        return "image/jpeg".to_string();
    }
    if bytes.len() >= 8
        && bytes[0] == 0x89
        && bytes[1] == b'P'
        && bytes[2] == b'N'
        && bytes[3] == b'G'
    {
        return "image/png".to_string();
    }
    if bytes.len() >= 6 && (&bytes[0..6] == b"GIF87a" || &bytes[0..6] == b"GIF89a") {
        return "image/gif".to_string();
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp".to_string();
    }
    "image/jpeg".to_string()
}

/// Returns true when MIME type is allowed for model upload.
fn is_supported_media_type(value: &str) -> bool {
    matches!(
        value,
        "image/jpeg" | "image/jpg" | "image/png" | "image/gif" | "image/webp"
    )
}

/// Checks whether bytes start with supported image signatures.
fn has_supported_image_magic(bytes: &[u8]) -> bool {
    if bytes.len() >= 3 && bytes[0..3] == [0xFF, 0xD8, 0xFF] {
        return true;
    }
    if bytes.len() >= 8
        && bytes[0] == 0x89
        && bytes[1] == b'P'
        && bytes[2] == b'N'
        && bytes[3] == b'G'
    {
        return true;
    }
    if bytes.len() >= 6 && (&bytes[0..6] == b"GIF87a" || &bytes[0..6] == b"GIF89a") {
        return true;
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return true;
    }
    false
}

/// Returns true for text/json payload content types.
fn is_json_or_text_content_type(content_type: &str) -> bool {
    content_type.contains("application/json")
        || content_type.contains("application/problem+json")
        || content_type.starts_with("text/")
}

/// Extracts structured remote error payload from non-image JSON body.
fn extract_remote_image_error(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?.trim();
    if text.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_str(text).ok()?;
    let retmsg = value
        .get("retmsg")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let retcode = value
        .get("retcode")
        .and_then(Value::as_i64)
        .map(|v| v.to_string())
        .unwrap_or_default();
    if retmsg.is_empty() && retcode.is_empty() {
        return None;
    }
    Some(format!("retcode={} retmsg={}", retcode, retmsg))
}

/// Normalizes CQ/image reference string (quotes + HTML ampersands).
fn normalize_image_ref(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace("&amp;", "&")
        .replace("&#38;", "&")
}
