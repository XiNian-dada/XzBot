use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{
    config::Config,
    llm::Llm,
    logger::debug as log_debug,
    onebot::{action::ActionRequest, event::MessageEvent},
    store::memory::{MemoryStore, SessionKey},
};

pub struct AiChatPlugin {
    store: Arc<MemoryStore>,
    llm: Arc<dyn Llm>,
    config: Arc<Config>,
    // 全局串行化 AI 生成，避免并发请求串上下文。
    reply_lock: Arc<Semaphore>,
}

impl AiChatPlugin {
    pub fn new(store: Arc<MemoryStore>, llm: Arc<dyn Llm>, config: Arc<Config>) -> Self {
        Self {
            store,
            llm,
            config,
            reply_lock: Arc::new(Semaphore::new(1)),
        }
    }

    pub async fn handle_message(&self, event: MessageEvent) -> Result<Option<ActionRequest>> {
        let raw_text = event.text();
        let trimmed = raw_text.trim();

        if trimmed.is_empty() {
            log_debug(self.config.debug, "empty message ignored");
            return Ok(None);
        }

        match event.message_type.as_str() {
            "private" => self.handle_private_message(event, trimmed).await,
            "group" => self.handle_group_message(event, raw_text.clone()).await,
            _ => Ok(None),
        }
    }

    async fn handle_private_message(
        &self,
        event: MessageEvent,
        trimmed: &str,
    ) -> Result<Option<ActionRequest>> {
        let key = SessionKey::new("private", None, event.user_id);

        if trimmed == "/reset" {
            self.store.reset(&key);
            log_debug(
                self.config.debug,
                format!("session reset private user_id={}", event.user_id),
            );
            return Ok(Some(ActionRequest::send_private_msg(
                event.user_id,
                "会话上下文已重置。".to_string(),
            )));
        }

        log_debug(
            self.config.debug,
            format!("private prompt user_id={} text={}", event.user_id, trimmed),
        );
        let Some(_permit) = self.try_acquire_reply_lock() else {
            return Ok(Some(ActionRequest::send_private_msg(
                event.user_id,
                "别急别急，我还在回复上一条消息。".to_string(),
            )));
        };
        let reply = self.generate_ai_reply(key, trimmed.to_string()).await?;
        Ok(Some(ActionRequest::send_private_msg(event.user_id, reply)))
    }

    async fn handle_group_message(
        &self,
        event: MessageEvent,
        raw_text: String,
    ) -> Result<Option<ActionRequest>> {
        let Some(group_id) = event.group_id else {
            return Ok(None);
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
            return Ok(Some(ActionRequest::send_group_msg(group_id, message)));
        }

        // 去掉配置中的命令前缀，保留关键字触发下的原始内容。
        for prefix in &self.config.group.prefixes {
            let trimmed = prompt.trim_start();
            if !prefix.is_empty() && trimmed.starts_with(prefix) {
                prompt = trimmed.replacen(prefix, "", 1).trim().to_string();
                break;
            }
        }

        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            log_debug(self.config.debug, "group prompt empty after normalize");
            return Ok(None);
        }

        let prompt = if sender_name == event.user_id.to_string() {
            format!("[群成员 {}] {}", event.user_id, prompt)
        } else {
            format!("[群成员 {}({})] {}", sender_name, event.user_id, prompt)
        };
        log_debug(
            self.config.debug,
            format!(
                "group prompt group_id={} user_id={} text={}",
                group_id, event.user_id, prompt
            ),
        );
        let Some(_permit) = self.try_acquire_reply_lock() else {
            let busy = if self.config.group.mention_sender {
                format!(
                    "[CQ:at,qq={}] 别急别急，我还在回复上一条消息。",
                    event.user_id
                )
            } else {
                "别急别急，我还在回复上一条消息。".to_string()
            };
            return Ok(Some(ActionRequest::send_group_msg(group_id, busy)));
        };
        let reply = self.generate_ai_reply(key, prompt).await?;
        let message = if self.config.group.mention_sender {
            format!("[CQ:at,qq={}] {}", event.user_id, reply)
        } else {
            reply
        };
        Ok(Some(ActionRequest::send_group_msg(group_id, message)))
    }

    async fn generate_ai_reply(&self, key: SessionKey, user_input: String) -> Result<String> {
        self.store.push_user_message(key.clone(), user_input);
        let mut history = self.store.messages(&key);

        if !self.config.persona.system.trim().is_empty() {
            history.insert(
                0,
                ("system".to_string(), self.config.persona.system.clone()),
            );
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
        let reply = self.llm.chat(session_id, history).await?;
        self.store.push_assistant_message(key, reply.clone());
        log_debug(
            self.config.debug,
            format!("llm reply length={}", reply.len()),
        );
        Ok(reply)
    }

    fn try_acquire_reply_lock(&self) -> Option<OwnedSemaphorePermit> {
        match self.reply_lock.clone().try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => {
                log_debug(self.config.debug, "reply lock busy");
                None
            }
        }
    }
}
