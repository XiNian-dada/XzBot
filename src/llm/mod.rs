//! 大模型抽象层：定义统一接口并注册各类后端实现。

use async_trait::async_trait;

/// Anthropic-compatible backend implementation.
pub mod anthropic_compatible;
/// Image loading and normalization helpers.
pub mod image;
/// User message parsing helpers (text + image markers).
pub mod message_parts;
/// Local mock backend for offline testing.
pub mod mock;
/// OCR fallback pipeline for non-multimodal models.
pub mod ocr;
/// OpenAI-compatible backend implementation.
pub mod openai_compatible;

/// Unified chat interface used by the bot runtime.
#[async_trait]
pub trait Llm: Send + Sync {
    /// Generates the assistant reply for one turn.
    async fn chat(
        &self,
        session_id: String,
        messages: Vec<(String, String)>,
    ) -> anyhow::Result<String>;
}
