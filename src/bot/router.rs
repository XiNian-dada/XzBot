use std::sync::Arc;

use anyhow::Result;
use dashmap::DashSet;

use crate::{
    bot::ai_chat::AiChatPlugin,
    config::{Config, PermissionMode, TriggerMode},
    logger::debug as log_debug,
    onebot::{action::ActionRequest, event::MessageEvent},
};

pub struct BotRouter {
    ai_chat: AiChatPlugin,
    config: Arc<Config>,
    runtime_group_blacklist: DashSet<i64>,
}

impl BotRouter {
    pub fn new(ai_chat: AiChatPlugin, config: Arc<Config>) -> Self {
        let runtime_group_blacklist = DashSet::new();
        for &group_id in &config.group.blacklist {
            runtime_group_blacklist.insert(group_id);
        }

        Self {
            ai_chat,
            config,
            runtime_group_blacklist,
        }
    }

    pub async fn route_message(&self, event: MessageEvent) -> Result<Option<ActionRequest>> {
        if event.post_type != "message" {
            log_debug(
                self.config.debug,
                format!("skip post_type={} (not message)", event.post_type),
            );
            return Ok(None);
        }

        if let Some(action) = self.handle_blacklist_command(&event) {
            return Ok(Some(action));
        }

        if !self.allowed_by_permission(&event) {
            log_debug(
                self.config.debug,
                format!(
                    "blocked by permission message_type={} user_id={} group_id={:?}",
                    event.message_type, event.user_id, event.group_id
                ),
            );
            return Ok(None);
        }

        if event.message_type == "group" && !self.should_trigger_group(&event) {
            log_debug(
                self.config.debug,
                format!("group trigger miss group_id={:?}", event.group_id),
            );
            return Ok(None);
        }

        log_debug(
            self.config.debug,
            format!(
                "route to ai_chat message_type={} user_id={} group_id={:?}",
                event.message_type, event.user_id, event.group_id
            ),
        );
        self.ai_chat.handle_message(event).await
    }

    fn allowed_by_permission(&self, event: &MessageEvent) -> bool {
        if event.message_type == "group" {
            let Some(group_id) = event.group_id else {
                return false;
            };
            if self.runtime_group_blacklist.contains(&group_id) {
                return false;
            }
        }

        match self.config.policy.permission {
            PermissionMode::None => true,
            PermissionMode::OwnerOnly => {
                event.message_type == "private" && event.user_id == self.config.owner.qq
            }
            PermissionMode::Whitelist => {
                if event.message_type == "private" {
                    true
                } else if event.message_type == "group" {
                    event
                        .group_id
                        .map(|group_id| self.config.group.whitelist.contains(&group_id))
                        .unwrap_or(false)
                } else {
                    false
                }
            }
        }
    }

    fn should_trigger_group(&self, event: &MessageEvent) -> bool {
        let raw_text = event.text();
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        let mentioned = raw_text.contains(&at_me);
        let normalized = raw_text.replace(&at_me, "").trim().to_string();

        if normalized.is_empty() {
            return false;
        }

        // reset 命令独立于触发模式，仍然受 permission / blacklist / require_at 控制。
        if normalized.starts_with("/reset") {
            return !self.config.group.require_at || mentioned;
        }

        if self.config.group.require_at && !mentioned {
            return false;
        }

        let prefix_hit = self
            .config
            .group
            .prefixes
            .iter()
            .any(|prefix| !prefix.is_empty() && normalized.starts_with(prefix));
        let keyword_hit = self
            .config
            .group
            .keywords
            .iter()
            .any(|keyword| keyword_match_with_boundary(&normalized, keyword));

        match self.config.group.trigger_mode {
            TriggerMode::At => mentioned,
            TriggerMode::Prefix => prefix_hit,
            TriggerMode::Keyword => keyword_hit,
            TriggerMode::Mixed => mentioned || prefix_hit || keyword_hit,
        }
    }

    fn handle_blacklist_command(&self, event: &MessageEvent) -> Option<ActionRequest> {
        let text = normalize_command_text(event);
        let command = parse_blacklist_command(&text)?;

        if event.user_id != self.config.owner.qq {
            return Some(self.reply_to_event(event, "仅主人可使用 /blacklist 指令。".to_string()));
        }

        match command {
            BlacklistCommand::List => {
                let mut groups: Vec<i64> = self
                    .runtime_group_blacklist
                    .iter()
                    .map(|v| *v.key())
                    .collect();
                groups.sort_unstable();
                groups.dedup();

                let message = if groups.is_empty() {
                    "当前黑名单为空。".to_string()
                } else {
                    format!(
                        "当前黑名单群号：{}",
                        groups
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                Some(self.reply_to_event(event, message))
            }
            BlacklistCommand::Add(group_id_opt) => {
                let target = group_id_opt.or(event.group_id);
                let Some(group_id) = target else {
                    return Some(self.reply_to_event(
                        event,
                        "用法: /blacklist add <group_id> (群聊里可省略 group_id)".to_string(),
                    ));
                };
                self.runtime_group_blacklist.insert(group_id);
                Some(self.reply_to_event(event, format!("已加入黑名单: {group_id}")))
            }
            BlacklistCommand::Remove(group_id_opt) => {
                let target = group_id_opt.or(event.group_id);
                let Some(group_id) = target else {
                    return Some(self.reply_to_event(
                        event,
                        "用法: /blacklist remove <group_id> (群聊里可省略 group_id)".to_string(),
                    ));
                };

                if self.config.group.blacklist.contains(&group_id) {
                    return Some(self.reply_to_event(
                        event,
                        format!("群 {group_id} 在配置黑名单中，运行时无法移除（需修改配置文件）。"),
                    ));
                }

                let removed = self.runtime_group_blacklist.remove(&group_id).is_some();
                if removed {
                    Some(self.reply_to_event(event, format!("已移除黑名单: {group_id}")))
                } else {
                    Some(self.reply_to_event(event, format!("群 {group_id} 不在运行时黑名单中。")))
                }
            }
        }
    }

    fn reply_to_event(&self, event: &MessageEvent, message: String) -> ActionRequest {
        match event.message_type.as_str() {
            "group" => {
                if let Some(group_id) = event.group_id {
                    ActionRequest::send_group_msg(group_id, message)
                } else {
                    ActionRequest::send_private_msg(event.user_id, message)
                }
            }
            _ => ActionRequest::send_private_msg(event.user_id, message),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BlacklistCommand {
    Add(Option<i64>),
    Remove(Option<i64>),
    List,
}

fn normalize_command_text(event: &MessageEvent) -> String {
    let raw = event.text();
    if event.message_type == "group" {
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        raw.replace(&at_me, "").trim().to_string()
    } else {
        raw.trim().to_string()
    }
}

fn parse_blacklist_command(text: &str) -> Option<BlacklistCommand> {
    let mut parts = text.split_whitespace();
    let head = parts.next()?;
    if head != "/blacklist" {
        return None;
    }

    let action = parts.next().unwrap_or("list");
    match action {
        "list" => Some(BlacklistCommand::List),
        "add" => Some(BlacklistCommand::Add(
            parts.next().and_then(|s| s.parse().ok()),
        )),
        "remove" | "del" | "delete" => Some(BlacklistCommand::Remove(
            parts.next().and_then(|s| s.parse().ok()),
        )),
        _ => Some(BlacklistCommand::List),
    }
}

fn keyword_match_with_boundary(text: &str, keyword: &str) -> bool {
    if keyword.trim().is_empty() {
        return false;
    }

    if match_with_boundary(text, keyword) {
        return true;
    }

    let text_lower = text.to_lowercase();
    let keyword_lower = keyword.to_lowercase();
    match_with_boundary(&text_lower, &keyword_lower)
}

fn match_with_boundary(text: &str, keyword: &str) -> bool {
    for (idx, _) in text.match_indices(keyword) {
        let start_ok = idx == 0
            || text[..idx]
                .chars()
                .last()
                .map(is_word_boundary)
                .unwrap_or(true);
        let end = idx + keyword.len();
        let end_ok = end == text.len()
            || text[end..]
                .chars()
                .next()
                .map(is_word_boundary)
                .unwrap_or(true);
        if start_ok && end_ok {
            return true;
        }
    }

    false
}

fn is_word_boundary(ch: char) -> bool {
    !ch.is_alphanumeric() && ch != '_'
}
