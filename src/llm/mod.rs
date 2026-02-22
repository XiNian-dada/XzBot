use async_trait::async_trait;

pub mod anthropic_compatible;
pub mod image;
pub mod message_parts;
pub mod mock;
pub mod openai_compatible;

#[async_trait]
pub trait Llm: Send + Sync {
    async fn chat(
        &self,
        session_id: String,
        messages: Vec<(String, String)>,
    ) -> anyhow::Result<String>;
}
