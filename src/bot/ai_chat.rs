//! AI 对话插件：负责上下文拼装、LLM 调用和回复动作生成。
//!
//! 这个模块是“协议层”和“模型层”之间的胶水：
//! - 从事件里提取当前用户真正想说的话
//! - 读取并维护会话历史
//! - 把回复转换成 OneBot Action
//!
//! 它本身不关心 HTTP、WebSocket 或搜索实现细节，只协调对话过程。

use std::sync::Arc;

use anyhow::Result;
use dashmap::{mapref::entry::Entry, DashMap};
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};

use crate::{
    config::Config,
    llm::Llm,
    logger::{debug as log_debug, info as log_info, warn as log_warn},
    onebot::{action::ActionRequest, event::MessageEvent},
    store::memory::{MemoryStore, SessionKey},
    token_stats,
};

/// Chat plugin that manages session history and LLM calls.
pub struct AiChatPlugin {
    store: Arc<MemoryStore>,
    llm: Arc<dyn Llm>,
    config: Arc<Config>,
    // 按会话串行化 AI 生成：群聊按 group_id，私聊按 user_id。
    reply_locks: Arc<DashMap<String, Arc<Semaphore>>>,
}

impl AiChatPlugin {
    /// Creates a new AI chat plugin instance.
    pub fn new(store: Arc<MemoryStore>, llm: Arc<dyn Llm>, config: Arc<Config>) -> Self {
        Self {
            store,
            llm,
            config,
            reply_locks: Arc::new(DashMap::new()),
        }
    }

    /// Handles one message event and returns reply actions.
    ///
    /// 这里既支持传统的“最终只回一条”，也支持慢任务场景下的“双阶段回复”：
    /// - 先通过 `progress_tx` 立即发一条轻量确认消息
    /// - 再在模型/工具完成后返回最终回复动作
    pub async fn handle_message(
        &self,
        event: MessageEvent,
        progress_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<Vec<ActionRequest>> {
        let raw_text = event.text();
        let trimmed = raw_text.trim();

        if trimmed.is_empty() {
            log_debug(self.config.debug, "empty message ignored");
            return Ok(Vec::new());
        }

        match event.message_type.as_str() {
            "private" => {
                self.handle_private_message(event, trimmed, progress_tx)
                    .await
            }
            "group" => {
                self.handle_group_message(event, raw_text.clone(), progress_tx)
                    .await
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Handles private chat flow, including `/reset` and lock gate.
    async fn handle_private_message(
        &self,
        event: MessageEvent,
        trimmed: &str,
        progress_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<Vec<ActionRequest>> {
        let key = SessionKey::new("private", None, event.user_id);

        if trimmed == "/reset" {
            self.store.reset(&key);
            log_debug(
                self.config.debug,
                format!("session reset private user_id={}", event.user_id),
            );
            return Ok(vec![ActionRequest::send_private_msg(
                event.user_id,
                "会话上下文已重置。".to_string(),
            )]);
        }

        log_debug(
            self.config.debug,
            format!("private prompt user_id={} text={}", event.user_id, trimmed),
        );
        let Some(_permit) = self.try_acquire_reply_lock(&key) else {
            return Ok(vec![ActionRequest::send_private_msg(
                event.user_id,
                "别急别急，我还在回复上一条消息。".to_string(),
            )]);
        };
        self.maybe_emit_progress_ack(
            progress_tx.as_ref(),
            &key,
            trimmed,
            ActionRequest::send_private_msg(event.user_id, String::new()),
            None,
        )
        .await;
        let reply = self.generate_ai_reply(key, trimmed.to_string()).await?;
        Ok(vec![ActionRequest::send_private_msg(event.user_id, reply)])
    }

    /// Handles group chat flow with mention/prefix normalization.
    async fn handle_group_message(
        &self,
        event: MessageEvent,
        raw_text: String,
        progress_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<Vec<ActionRequest>> {
        let Some(group_id) = event.group_id else {
            return Ok(Vec::new());
        };
        let sender_name = event.display_name();

        // 群会话按 group_id 共享，同群不同用户不再隔离。
        let key = SessionKey::new("group", Some(group_id), 0);
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        let mut prompt = raw_text.replace(&at_me, "").trim().to_string();

        if prompt.starts_with("/reset") {
            self.store.reset(&key);
            log_debug(
                self.config.debug,
                format!(
                    "session reset group_id={} user_id={}",
                    group_id, event.user_id
                ),
            );
            let message = if self.config.group.mention_sender {
                format!("[CQ:at,qq={}] 会话上下文已重置。", event.user_id)
            } else {
                "会话上下文已重置。".to_string()
            };
            return Ok(vec![ActionRequest::send_group_msg(group_id, message)]);
        }

        // 去掉配置中的命令前缀，保留关键字触发下的原始内容。
        for prefix in &self.config.group.prefixes {
            let trimmed = prompt.trim_start();
            if !prefix.is_empty() && trimmed.starts_with(prefix) {
                prompt = trimmed.replacen(prefix, "", 1).trim().to_string();
                break;
            }
        }

        let prompt_body = prompt.trim().to_string();
        if prompt_body.is_empty() {
            log_debug(self.config.debug, "group prompt empty after normalize");
            return Ok(Vec::new());
        }

        let prompt = if sender_name == event.user_id.to_string() {
            format!("[群成员 {}] {}", event.user_id, prompt_body)
        } else {
            format!(
                "[群成员 {}({})] {}",
                sender_name, event.user_id, prompt_body
            )
        };
        log_debug(
            self.config.debug,
            format!(
                "group prompt group_id={} user_id={} text={}",
                group_id, event.user_id, prompt
            ),
        );
        let Some(_permit) = self.try_acquire_reply_lock(&key) else {
            let busy = if self.config.group.mention_sender {
                format!(
                    "[CQ:at,qq={}] 别急别急，我还在回复上一条消息。",
                    event.user_id
                )
            } else {
                "别急别急，我还在回复上一条消息。".to_string()
            };
            return Ok(vec![ActionRequest::send_group_msg(group_id, busy)]);
        };
        self.maybe_emit_progress_ack(
            progress_tx.as_ref(),
            &key,
            &prompt_body,
            ActionRequest::send_group_msg(group_id, String::new()),
            if self.config.group.mention_sender {
                Some(format!("[CQ:at,qq={}] ", event.user_id))
            } else {
                None
            },
        )
        .await;
        let reply = self.generate_ai_reply(key, prompt).await?;
        let message = if self.config.group.mention_sender {
            format!("[CQ:at,qq={}] {}", event.user_id, reply)
        } else {
            reply
        };
        Ok(vec![ActionRequest::send_group_msg(group_id, message)])
    }

    /// Builds prompt history, injects persona, calls LLM, and updates context.
    async fn generate_ai_reply(&self, key: SessionKey, user_input: String) -> Result<String> {
        let mut history = self
            .store
            .push_user_message_and_snapshot(key.clone(), user_input);

        let (resolved_prompt, matched_group_override) =
            self.config.persona.resolve_system_for_group(key.group_id);
        let system_prompt = resolved_prompt.trim().to_string();
        if let Some(group_id) = key.group_id {
            if let Some(matched) = matched_group_override {
                log_debug(
                    self.config.debug,
                    format!("persona override applied group_id={group_id} matched={matched}"),
                );
            } else {
                log_debug(
                    self.config.debug,
                    format!("persona override not found group_id={group_id}, fallback default"),
                );
            }
        }
        if !system_prompt.is_empty() {
            history.insert(0, ("system".to_string(), system_prompt));
        }

        let session_id = format!(
            "{}:{}:{}",
            self.config.ai.provider.as_str(),
            self.config.ai.model,
            key.session_id()
        );
        log_debug(
            self.config.debug,
            format!(
                "calling llm provider={} session={}",
                self.config.ai.provider.as_str(),
                session_id
            ),
        );
        let token_before = token_stats::snapshot();
        let reply = self.llm.chat(session_id, history).await?;
        let token_after = token_stats::snapshot();
        let token_delta = token_stats::diff(token_before, token_after);
        self.store.push_assistant_message(key, reply.clone());
        log_debug(
            self.config.debug,
            format!("llm reply length={}", reply.len()),
        );
        log_info(format!(
            "[TOKEN] this_call: prompt={} completion={} total={} | cumulative: prompt={} completion={} total={}",
            token_delta.prompt,
            token_delta.completion,
            token_delta.total,
            token_after.prompt,
            token_after.completion,
            token_after.total
        ));
        Ok(reply)
    }

    /// Emits one immediate progress action via the live websocket sender.
    ///
    /// 这里不再硬编码固定文案，而是让模型先决定：
    /// - 是否需要“先回一句”
    /// - 如果需要，该如何用当前人设自然地说这一句
    ///
    /// 若模型返回 `SKIP` 或空串，则本轮不发送前置 ack。
    async fn maybe_emit_progress_ack(
        &self,
        progress_tx: Option<&mpsc::UnboundedSender<String>>,
        key: &SessionKey,
        user_text: &str,
        action: ActionRequest,
        prefix: Option<String>,
    ) {
        let Some(progress_tx) = progress_tx else {
            return;
        };
        if !self.should_request_progress_ack(user_text) {
            return;
        }

        let Some(ack) = self.generate_progress_ack(key, user_text).await else {
            return;
        };

        let ack = match prefix {
            Some(prefix) => format!("{prefix}{ack}"),
            None => ack,
        };

        let action = match action.action.as_str() {
            "send_private_msg" => {
                let user_id = action
                    .params
                    .get("user_id")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or_default();
                ActionRequest::send_private_msg(user_id, ack)
            }
            "send_group_msg" => {
                let group_id = action
                    .params
                    .get("group_id")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or_default();
                ActionRequest::send_group_msg(group_id, ack)
            }
            _ => return,
        };

        let encoded = match serde_json::to_string(&action) {
            Ok(v) => v,
            Err(err) => {
                log_warn(format!("failed to encode progress action: {err}"));
                return;
            }
        };

        if progress_tx.send(encoded).is_err() {
            log_warn("failed to send progress action to websocket");
            return;
        }

        log_debug(self.config.debug, "progress ack action sent");
    }

    /// Heuristic gate for deciding whether this turn is likely to be slow/tool-heavy.
    ///
    /// 这里仍然由宿主控制“是否值得两阶段回复”，但具体前置文案交给模型来写。
    fn should_request_progress_ack(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        text.contains("[IMAGE:")
            || lower.contains("http://")
            || lower.contains("https://")
            || lower.contains("天气")
            || lower.contains("热搜")
            || lower.contains("热榜")
            || lower.contains("热度")
            || lower.contains("新闻")
            || lower.contains("搜")
            || lower.contains("查")
            || lower.contains("看看")
            || lower.contains("链接")
            || lower.contains("网页")
            || lower.contains("知乎")
            || lower.contains("微博")
            || lower.contains("状态")
            || lower.contains("日志")
            || lower.contains("系统")
            || lower.contains("进程")
            || lower.contains("图片")
            || lower.contains("分析")
            || lower.contains("analyze")
            || lower.contains("fetch")
            || lower.contains("search")
    }

    /// Ask the model whether this turn deserves a short “processing…” acknowledgement.
    async fn generate_progress_ack(&self, key: &SessionKey, user_text: &str) -> Option<String> {
        let trimmed = user_text.trim();
        if trimmed.is_empty() {
            return None;
        }

        let (resolved_prompt, _) = self.config.persona.resolve_system_for_group(key.group_id);
        let progress_system = format!(
            "{}\n\n附加任务：你现在不是正式回答问题，只做一件事：生成一句很短的处理中过渡语，表示你已经开始查了。\n- 这句是发给用户的前置回复，不是最终答案。\n- 禁止给出任何事实结论。\n- 禁止解释原因。\n- 禁止使用 markdown。\n- 语气要符合当前人设。\n- 尽量口语化，最多 18 个字。",
            resolved_prompt.trim()
        );

        let progress_messages = vec![
            ("system".to_string(), progress_system),
            ("user".to_string(), trimmed.to_string()),
        ];
        let session_id = format!(
            "{}:{}:{}:progress_ack",
            self.config.ai.provider.as_str(),
            self.config.ai.model,
            key.session_id()
        );

        match self.llm.progress_ack(session_id, progress_messages).await {
            Ok(Some(reply)) => sanitize_progress_ack(&reply),
            Ok(None) => None,
            Err(err) => {
                log_warn(format!("progress ack generation failed: {err}"));
                None
            }
        }
    }

    /// Acquires per-session reply semaphore without waiting.
    fn try_acquire_reply_lock(&self, key: &SessionKey) -> Option<OwnedSemaphorePermit> {
        let lock_key = key.session_id();
        let semaphore = match self.reply_locks.entry(lock_key.clone()) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let semaphore = Arc::new(Semaphore::new(1));
                entry.insert(semaphore.clone());
                semaphore
            }
        };

        match semaphore.try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => {
                log_debug(
                    self.config.debug,
                    format!("reply lock busy lock_key={lock_key}"),
                );
                None
            }
        }
    }
}

/// Sanitizes model-generated progress text into a single short line.
fn sanitize_progress_ack(text: &str) -> Option<String> {
    let first_line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    if first_line.is_empty() {
        return None;
    }

    let compact = first_line.replace(['`', '*', '#'], "").trim().to_string();
    if compact.is_empty() {
        return None;
    }

    let shortened: String = compact.chars().take(24).collect();
    Some(shortened)
}
