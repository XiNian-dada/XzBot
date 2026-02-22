use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use tokio::fs;

const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
const DEFAULT_UA: &str = "Mozilla/5.0 (compatible; XzBot/1.0; +https://example.local)";

#[derive(Debug, Clone)]
pub struct ImageBinary {
    pub media_type: String,
    pub data_base64: String,
}

impl ImageBinary {
    pub fn as_data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.data_base64)
    }
}

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

async fn fetch_image_by_url(client: &Client, url: &str, debug: bool) -> Result<ImageBinary> {
    if debug {
        println!("[DEBUG] fetch image for llm url={url}");
    }

    let response = client
        .get(url)
        .header("User-Agent", DEFAULT_UA)
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
    let media_type = normalize_media_type(&content_type, url, &bytes);
    Ok(ImageBinary {
        media_type,
        data_base64: STANDARD.encode(bytes),
    })
}

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

fn is_supported_media_type(value: &str) -> bool {
    matches!(
        value,
        "image/jpeg" | "image/jpg" | "image/png" | "image/gif" | "image/webp"
    )
}

fn normalize_image_ref(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace("&amp;", "&")
        .replace("&#38;", "&")
}
