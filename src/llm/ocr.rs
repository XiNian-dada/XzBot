//! OCR fallback pipeline for image inputs when model vision is unavailable.

use std::{
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use tokio::{fs, process::Command, time::timeout};

use crate::config::OcrProvider;
use crate::llm::image::load_image_for_llm;

/// OCR runtime settings assembled from `Config.ai`.
#[derive(Debug, Clone)]
pub struct OcrSettings {
    /// OCR backend provider.
    pub provider: OcrProvider,
    /// OCR command path (tesseract mode).
    pub cmd: String,
    /// OCR language pack list.
    pub lang: String,
    /// OCR timeout in milliseconds.
    pub timeout_ms: u64,
    /// Paddle OCR endpoint.
    pub paddle_endpoint: String,
    /// Paddle OCR token.
    pub paddle_token: String,
    /// Paddle file type (`0` pdf, `1` image).
    pub paddle_file_type: u8,
    /// Paddle optional parameter.
    pub paddle_use_doc_orientation_classify: bool,
    /// Paddle optional parameter.
    pub paddle_use_doc_unwarping: bool,
    /// Paddle optional parameter.
    pub paddle_use_chart_recognition: bool,
    /// Whether to use global proxy for Paddle OCR request.
    pub paddle_use_proxy: bool,
}

/// Runs OCR for up to 3 image refs and returns merged readable text summary.
pub async fn ocr_images_to_text(
    client: &Client,
    image_refs: &[String],
    settings: &OcrSettings,
    debug: bool,
) -> String {
    if image_refs.is_empty() {
        return String::new();
    }

    let mut results = Vec::new();
    let mut success = 0usize;
    for (idx, image_ref) in image_refs.iter().take(3).enumerate() {
        let result = match settings.provider {
            OcrProvider::Tesseract => ocr_single_image_tesseract(client, image_ref, settings, debug)
                .await,
            OcrProvider::Paddle => {
                ocr_single_image_paddle(client, image_ref, settings, debug).await
            }
        };
        match result {
            Ok(text) => {
                let text = text.trim().to_string();
                if !text.is_empty() {
                    success += 1;
                    results.push(format!("图片{}:\n{}", idx + 1, text));
                }
            }
            Err(err) => {
                if debug {
                    println!(
                        "[DEBUG] ocr failed image_ref={} err={}",
                        image_ref, err
                    );
                }
            }
        }
    }

    if success == 0 {
        return "图片OCR失败或未识别到文字。".to_string();
    }

    format!("以下为图片 OCR 识别结果（可能有误）：\n{}", results.join("\n\n"))
}

/// OCR one image by downloading and invoking local tesseract.
async fn ocr_single_image_tesseract(
    client: &Client,
    image_ref: &str,
    settings: &OcrSettings,
    debug: bool,
) -> Result<String> {
    if settings.cmd.trim().is_empty() {
        bail!("ocr_cmd is empty");
    }

    let image = load_image_for_llm(client, image_ref, debug).await?;
    let bytes = STANDARD
        .decode(image.data_base64.as_bytes())
        .context("failed to decode image base64 payload for ocr")?;
    let ext = extension_from_media_type(&image.media_type);
    let path = temp_image_path(ext);
    fs::write(&path, &bytes)
        .await
        .with_context(|| format!("failed to write temp image {}", path.display()))?;

    let output = run_tesseract(&path, settings).await;
    let _ = fs::remove_file(&path).await;
    output
}

/// OCR one image via Paddle OCR HTTP API.
async fn ocr_single_image_paddle(
    client: &Client,
    image_ref: &str,
    settings: &OcrSettings,
    debug: bool,
) -> Result<String> {
    if settings.paddle_endpoint.trim().is_empty() {
        bail!("paddle_ocr_endpoint is empty");
    }
    if settings.paddle_token.trim().is_empty() {
        bail!("paddle_ocr_token is empty");
    }

    let image = load_image_for_llm(client, image_ref, debug).await?;
    let file_data = image.data_base64;

    let payload = serde_json::json!({
        "file": file_data,
        "fileType": settings.paddle_file_type,
        "useDocOrientationClassify": settings.paddle_use_doc_orientation_classify,
        "useDocUnwarping": settings.paddle_use_doc_unwarping,
        "useChartRecognition": settings.paddle_use_chart_recognition,
    });

    let paddle_client = if settings.paddle_use_proxy {
        None
    } else {
        if debug {
            println!("[DEBUG] paddle ocr bypass proxy");
        }
        Some(
            reqwest::Client::builder()
                .timeout(Duration::from_millis(settings.timeout_ms))
                .build()
                .context("failed to build paddle ocr client without proxy")?,
        )
    };
    let client_ref = paddle_client.as_ref().unwrap_or(client);

    let request = client_ref
        .post(settings.paddle_endpoint.trim())
        .header("Authorization", format!("token {}", settings.paddle_token.trim()))
        .header("Content-Type", "application/json")
        .header(
            "User-Agent",
            "Mozilla/5.0 (compatible; XzBot/1.0; +https://example.local)",
        )
        .header("Accept", "application/json")
        .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
        .json(&payload);

    let response = timeout(Duration::from_millis(settings.timeout_ms), request.send())
        .await
        .map_err(|_| anyhow!("paddle ocr timeout after {} ms", settings.timeout_ms))?
        .with_context(|| {
            format!(
                "failed to call paddle ocr endpoint {}",
                settings.paddle_endpoint
            )
        })?;

    let status = response.status();
    let body = response.text().await.context("failed to read paddle ocr body")?;
    if !status.is_success() {
        let preview = body.chars().take(300).collect::<String>();
        bail!("paddle ocr http {}: {}", status, preview);
    }

    let value: serde_json::Value =
        serde_json::from_str(&body).context("failed to parse paddle ocr json")?;
    let result = value
        .get("result")
        .ok_or_else(|| anyhow!("paddle ocr missing result field"))?;
    let layouts = result
        .get("layoutParsingResults")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("paddle ocr missing layoutParsingResults"))?;

    let mut texts = Vec::new();
    for layout in layouts {
        if let Some(text) = layout
            .get("markdown")
            .and_then(|v| v.get("text"))
            .and_then(|v| v.as_str())
        {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                texts.push(trimmed.to_string());
            }
        }
    }

    if texts.is_empty() {
        bail!("paddle ocr returned empty text");
    }
    Ok(texts.join("\n\n"))
}

/// Executes local tesseract command with timeout guard.
async fn run_tesseract(path: &PathBuf, settings: &OcrSettings) -> Result<String> {
    let mut cmd = Command::new(&settings.cmd);
    cmd.kill_on_drop(true)
        .arg(path)
        .arg("stdout");
    if !settings.lang.trim().is_empty() {
        cmd.arg("-l").arg(settings.lang.trim());
    }

    let output = timeout(Duration::from_millis(settings.timeout_ms), cmd.output())
        .await
        .map_err(|_| anyhow!("ocr timeout after {} ms", settings.timeout_ms))?
        .with_context(|| format!("failed to run {}", settings.cmd))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("ocr command failed: {}", stderr.trim());
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(text)
}

/// Builds unique temp path for OCR image materialization.
fn temp_image_path(ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    path.push(format!("xzbot_ocr_{}_{}.{}", pid, nanos, ext));
    path
}

/// Maps media type to reasonable file extension for OCR temp files.
fn extension_from_media_type(media_type: &str) -> &'static str {
    let lower = media_type.to_ascii_lowercase();
    if lower.contains("png") {
        "png"
    } else if lower.contains("jpeg") || lower.contains("jpg") {
        "jpg"
    } else if lower.contains("webp") {
        "webp"
    } else if lower.contains("gif") {
        "gif"
    } else {
        "img"
    }
}
