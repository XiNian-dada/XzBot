use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;

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

pub async fn fetch_image_for_llm(client: &Client, url: &str, debug: bool) -> Result<ImageBinary> {
    let url = url.trim();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("image url must start with http:// or https://");
    }

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
    let media_type = normalize_media_type(&content_type, url);
    let data_base64 = STANDARD.encode(bytes);

    Ok(ImageBinary {
        media_type,
        data_base64,
    })
}

fn normalize_media_type(content_type: &str, url: &str) -> String {
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
    "image/jpeg".to_string()
}

fn is_supported_media_type(value: &str) -> bool {
    matches!(
        value,
        "image/jpeg" | "image/jpg" | "image/png" | "image/gif" | "image/webp"
    )
}
