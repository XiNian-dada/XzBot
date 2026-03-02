//! High-level message routing: permission check, plugins, AI fallback.

use std::sync::Arc;

use anyhow::Result;
use dashmap::{DashMap, DashSet};

use crate::{
    bot::ai_chat::AiChatPlugin,
    config::{Config, PermissionMode, TriggerMode},
    logger::debug as log_debug,
    onebot::{action::ActionRequest, event::MessageEvent},
    plugins::PluginManager,
};

/// Central router that dispatches events to plugins or AI chat.
pub struct BotRouter {
    ai_chat: AiChatPlugin,
    plugins: PluginManager,
    config: Arc<Config>,
    runtime_group_blacklist: DashSet<i64>,
    group_repeat_state: DashMap<i64, GroupRepeatState>,
}

impl BotRouter {
    /// Builds a router with runtime state initialized from config.
    pub fn new(ai_chat: AiChatPlugin, plugins: PluginManager, config: Arc<Config>) -> Self {
        let runtime_group_blacklist = DashSet::new();
        for &group_id in &config.group.blacklist {
            runtime_group_blacklist.insert(group_id);
        }

        Self {
            ai_chat,
            plugins,
            config,
            runtime_group_blacklist,
            group_repeat_state: DashMap::new(),
        }
    }

    /// Routes one incoming message event to zero or more OneBot actions.
    pub async fn route_message(&self, event: MessageEvent) -> Result<Vec<ActionRequest>> {
        if event.post_type != "message" {
            log_debug(
                self.config.debug,
                format!("skip post_type={} (not message)", event.post_type),
            );
            return Ok(Vec::new());
        }

        if let Some(action) = self.handle_blacklist_command(&event) {
            return Ok(vec![action]);
        }

        if !self.allowed_by_permission(&event) {
            log_debug(
                self.config.debug,
                format!(
                    "blocked by permission message_type={} user_id={} group_id={:?}",
                    event.message_type, event.user_id, event.group_id
                ),
            );
            return Ok(Vec::new());
        }

        let plugin_actions = self.plugins.try_handle(&event).await?;
        if !plugin_actions.is_empty() {
            log_debug(
                self.config.debug,
                format!(
                    "route to plugin message_type={} user_id={} group_id={:?}",
                    event.message_type, event.user_id, event.group_id
                ),
            );
            return Ok(plugin_actions);
        }

        if let Some(action) = self.try_group_repeat(&event) {
            return Ok(vec![action]);
        }

        if event.message_type == "group" && !self.should_trigger_group(&event) {
            log_debug(
                self.config.debug,
                format!("group trigger miss group_id={:?}", event.group_id),
            );
            return Ok(Vec::new());
        }

        log_debug(
            self.config.debug,
            format!(
                "route to ai_chat message_type={} user_id={} group_id={:?}",
                event.message_type, event.user_id, event.group_id
            ),
        );
        Ok(self.ai_chat.handle_message(event).await?.into_iter().collect())
    }

    /// Gracefully shuts down all managed plugins.
    pub async fn shutdown_plugins(&self) {
        self.plugins.shutdown().await;
    }

    /// Applies permission and group blacklist policy.
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

    /// Evaluates group trigger policy (`@`, prefix, keyword, mixed).
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

    /// Handles runtime `/blacklist` management command.
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

    /// Implements group "加一" behavior with per-group mute-after-repeat state.
    fn try_group_repeat(&self, event: &MessageEvent) -> Option<ActionRequest> {
        if event.message_type != "group" {
            return None;
        }
        let group_id = event.group_id?;

        // 仅在配置白名单群启用“加一”。
        if !self.config.group.whitelist.contains(&group_id) {
            return None;
        }

        // 不处理机器人自己发出的消息，避免循环。
        if event.user_id == event.self_id {
            return None;
        }

        let normalized = normalize_repeat_text(&event.text())?;
        let mut state = self
            .group_repeat_state
            .entry(group_id)
            .or_insert_with(GroupRepeatState::default);

        // 已经“加一”过同一句，直到出现不同内容前不再触发。
        if state.muted_text.as_deref() == Some(normalized.as_str()) {
            state.last_text = Some(normalized);
            state.streak = state.streak.saturating_add(1);
            return None;
        }

        // 出现不同内容，解除静音词。
        if state.muted_text.is_some() {
            state.muted_text = None;
        }

        if state.last_text.as_deref() == Some(normalized.as_str()) {
            state.streak = state.streak.saturating_add(1);
        } else {
            state.last_text = Some(normalized.clone());
            state.streak = 1;
        }

        if state.streak >= 2 {
            state.muted_text = Some(normalized.clone());
            log_debug(
                self.config.debug,
                format!("group repeat +1 group_id={group_id} text={normalized}"),
            );
            return Some(ActionRequest::send_group_msg(group_id, normalized));
        }

        None
    }

    /// Builds reply action according to source chat type.
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

#[derive(Debug, Clone, Default)]
struct GroupRepeatState {
    /// Most recent normalized text.
    last_text: Option<String>,
    /// Text already echoed once and temporarily muted.
    muted_text: Option<String>,
    /// Consecutive count for current `last_text`.
    streak: u32,
}

/// Supported `/blacklist` command variants.
#[derive(Debug, Clone, Copy)]
enum BlacklistCommand {
    Add(Option<i64>),
    Remove(Option<i64>),
    List,
}

/// Normalizes command text by stripping bot mention in groups.
fn normalize_command_text(event: &MessageEvent) -> String {
    let raw = event.text();
    if event.message_type == "group" {
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        raw.replace(&at_me, "").trim().to_string()
    } else {
        raw.trim().to_string()
    }
}

/// Parses `/blacklist` command and optional target group id.
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

/// Keyword matching with lightweight word-boundary check and case-insensitive fallback.
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

/// Returns normalized text used by group repeat feature.
fn normalize_repeat_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    // 带 CQ 码的消息（@、图片、回复等）不参与加一，避免误触发和噪音。
    if trimmed.contains("[CQ:") {
        return None;
    }
    Some(trimmed.to_string())
}

/// Checks whether `keyword` occurs with both-side boundary constraints.
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

/// Returns true when char should be treated as token boundary.
fn is_word_boundary(ch: char) -> bool {
    !ch.is_alphanumeric() && ch != '_'
}
