//! 大模型抽象层：定义统一接口并注册各类后端实现。

use anyhow::{anyhow, Error};
use async_trait::async_trait;
use std::fmt::Display;
use std::sync::Arc;

use crate::logger::{warn as log_warn, warn_err as log_warn_err};

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

    /// Generates one short progress acknowledgement for slow tasks.
    ///
    /// 这条回复只用于“先说一句我正在处理”，不应包含最终结论，也不应触发工具。
    /// 默认实现返回 `None`，表示当前后端不提供这项能力。
    async fn progress_ack(
        &self,
        _session_id: String,
        _messages: Vec<(String, String)>,
    ) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
}

/// 多模型回退包装器。
///
/// 它不负责“负载均衡”，只负责“按顺序兜底”：
/// 1. 先尝试主模型
/// 2. 如果失败，再尝试下一个回退模型
/// 3. 全部失败后，对瞬时网络类错误返回统一提示，对非瞬时错误保留原始报错
pub struct FallbackLlm {
    candidates: Vec<FallbackCandidate>,
    debug: bool,
}

struct FallbackCandidate {
    label: String,
    llm: Arc<dyn Llm>,
}

impl FallbackLlm {
    /// 根据候选模型列表创建回退链。
    pub fn new(candidates: Vec<(String, Arc<dyn Llm>)>, debug: bool) -> Self {
        Self {
            candidates: candidates
                .into_iter()
                .map(|(label, llm)| FallbackCandidate { label, llm })
                .collect(),
            debug,
        }
    }
}

#[async_trait]
impl Llm for FallbackLlm {
    async fn chat(
        &self,
        session_id: String,
        messages: Vec<(String, String)>,
    ) -> anyhow::Result<String> {
        let mut last_err: Option<Error> = None;
        let mut all_transient = true;

        for (idx, candidate) in self.candidates.iter().enumerate() {
            if self.debug && self.candidates.len() > 1 {
                println!(
                    "[DEBUG] llm fallback attempt {}/{} model={}",
                    idx + 1,
                    self.candidates.len(),
                    candidate.label
                );
            }

            match candidate
                .llm
                .chat(session_id.clone(), messages.clone())
                .await
            {
                Ok(reply) => {
                    if self.debug && idx > 0 {
                        println!(
                            "[DEBUG] llm fallback recovered with model={} after {} failures",
                            candidate.label, idx
                        );
                    }
                    return Ok(reply);
                }
                Err(err) => {
                    let transient = is_transient_llm_error(&err);
                    all_transient &= transient;
                    if self.candidates.len() > 1 {
                        log_warn_err(
                            format!(
                                "llm candidate failed model={} transient={} session fallback continues",
                                candidate.label, transient
                            ),
                            &err,
                        );
                    } else if self.debug {
                        println!(
                            "[DEBUG] llm candidate failed model={} transient={} err={:#}",
                            candidate.label, transient, err
                        );
                    }
                    last_err = Some(err);
                }
            }
        }

        if all_transient {
            log_warn("all llm candidates failed with transient errors, returning temporary reply");
            return Ok(temporary_network_reply());
        }

        Err(last_err.unwrap_or_else(|| anyhow!("no LLM candidates configured")))
    }

    async fn progress_ack(
        &self,
        session_id: String,
        messages: Vec<(String, String)>,
    ) -> anyhow::Result<Option<String>> {
        let mut last_err: Option<Error> = None;

        for (idx, candidate) in self.candidates.iter().enumerate() {
            if self.debug && self.candidates.len() > 1 {
                println!(
                    "[DEBUG] llm progress-ack attempt {}/{} model={}",
                    idx + 1,
                    self.candidates.len(),
                    candidate.label
                );
            }

            match candidate
                .llm
                .progress_ack(session_id.clone(), messages.clone())
                .await
            {
                Ok(Some(reply)) => return Ok(Some(reply)),
                Ok(None) => continue,
                Err(err) => {
                    if self.candidates.len() > 1 {
                        log_warn_err(
                            format!(
                                "llm progress-ack candidate failed model={} fallback continues",
                                candidate.label
                            ),
                            &err,
                        );
                    }
                    last_err = Some(err);
                }
            }
        }

        if let Some(err) = last_err {
            return Err(err);
        }
        Ok(None)
    }
}

/// 统一的瞬时失败回复文案。
pub fn temporary_network_reply() -> String {
    "网不太好，我这边请求超时了，等会再试试。".to_string()
}

/// Extracts group id from a chat session identifier when the current turn is a group session.
///
/// Session ids are assembled as `provider:model:group:<group_id>:<user_id>` for group chats.
pub fn session_group_id(session_id: &str) -> Option<i64> {
    // Parse from tail so model/provider names containing ':' do not break group detection.
    let mut tail = session_id.rsplit(':');
    let _user_id = tail.next()?;
    let group_id = tail.next()?;
    let chat_type = tail.next()?;
    if chat_type != "group" {
        return None;
    }
    group_id.parse::<i64>().ok()
}

/// Formats one tool failure together with a machine-readable next-step hint for the model.
///
/// 这些提示不是给用户看的，而是给模型下一轮观察用的：
/// - 当前工具为什么失败
/// - 更合理的下一步替代动作是什么
pub(crate) fn tool_error_with_hint(tool: &str, message: impl Display, next_hint: &str) -> String {
    let next_hint = next_hint.trim();
    if next_hint.is_empty() {
        format!("{tool} error: {message}")
    } else {
        format!("{tool} error: {message}\n\n[next_hint]\n{next_hint}")
    }
}

/// 粗判 LLM 错误是否属于“适合切换模型再试一次”的瞬时失败。
fn is_transient_llm_error(err: &Error) -> bool {
    if err.chain().any(|cause| {
        cause
            .downcast_ref::<reqwest::Error>()
            .map(|e| e.is_timeout() || e.is_connect() || e.is_body() || e.is_request())
            .unwrap_or(false)
    }) {
        return true;
    }

    let text = err.to_string().to_lowercase();
    if [
        "invalid value",
        "invalid_request_error",
        "missing required",
        "unsupported value",
        "unsupported type",
        "schema",
        "must be at the beginning",
        "context length",
        "maximum context length",
        "请求参数异常",
        "升级客户端后重试",
    ]
    .iter()
    .any(|needle| text.contains(needle))
    {
        return false;
    }

    [
        "timeout",
        "timed out",
        "too many requests",
        "429",
        "502",
        "503",
        "504",
        "upstream_error",
        "upstream request failed",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "server error",
        "failed to call http",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}
