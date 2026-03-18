//! 机器人消息路由：负责权限判断、插件分发和 AI 回退处理。
//!
//! 路由层只做“决策”，不直接实现复杂业务：
//! - 先判断消息类型、权限与群触发条件
//! - 再尝试让插件接管
//! - 最后才回落到 AI 对话插件
//!
//! 这样可以保证后续新增插件时，不需要把判断逻辑散落到多个业务模块。

use std::sync::Arc;

use anyhow::Result;
use dashmap::{DashMap, DashSet};
use tokio::sync::mpsc;

use crate::{
    bot::ai_chat::{AiChatPlugin, ProgressSession},
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

        // Ignore bot self-message events to avoid polluting AI context and feedback loops.
        if event.user_id == event.self_id {
            log_debug(
                self.config.debug,
                format!(
                    "skip self message user_id=self_id={} message_type={} group_id={:?}",
                    event.self_id, event.message_type, event.group_id
                ),
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

        let plugin_event = self.plugins.dispatch_event(&event).await?;
        if plugin_event.stop_propagation {
            log_debug(
                self.config.debug,
                format!(
                    "plugin stopped propagation message_type={} user_id={} group_id={:?}",
                    event.message_type, event.user_id, event.group_id
                ),
            );
            return Ok(plugin_event.actions);
        }

        let mut plugin_actions = plugin_event.actions;
        let command_actions = self.plugins.try_handle_command(&event).await?;
        if !command_actions.is_empty() {
            plugin_actions.extend(command_actions);
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
        let mut actions = plugin_actions;
        actions.extend(self.ai_chat.handle_message(event).await?);
        Ok(actions)
    }

    /// Starts an AI progress session before expensive enrichment when this turn is very likely to reach AI.
    pub async fn maybe_start_progress_session(
        &self,
        event: &MessageEvent,
        progress_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Option<ProgressSession> {
        let progress_tx = progress_tx?;
        if !self.should_prepare_ai_progress(event) {
            return None;
        }
        self.ai_chat
            .start_progress_session(event, progress_tx)
            .await
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
        // 触发判定只能看“当前这条消息原本说了什么”，不能被运行时追加的
        // 最近群聊/引用展开等上下文污染，否则普通聊天也可能被误判为触发。
        let raw_text = event.original_text();
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

        // “加一”只看用户这一条消息的原始内容，不能把最近群聊上下文一并算进去。
        let normalized = normalize_repeat_text(&event.original_text())?;
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

    /// Fast preflight used before enrichment so staged agent updates can start as early as possible.
    fn should_prepare_ai_progress(&self, event: &MessageEvent) -> bool {
        if event.post_type != "message" || event.user_id == event.self_id {
            return false;
        }
        if self.handle_blacklist_command(event).is_some() {
            return false;
        }
        if !self.allowed_by_permission(event) {
            return false;
        }
        if event.message_type == "group" && !self.should_trigger_group(event) {
            return false;
        }

        let normalized = normalize_command_text(event);
        if normalized.is_empty() || normalized == "/reset" {
            return false;
        }
        // 事件插件可能会完全接管这条消息；为了避免先发一条 AI 风格进度消息再被插件拦截，
        // 这里先保守跳过 staged progress。宁可少发，不要把宿主边界打乱。
        if self.plugins.has_matching_event_subscriber(event) {
            return false;
        }
        if self.plugins.has_matching_command(&normalized) {
            return false;
        }

        true
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
    // 命令解析同样只能基于原始消息。
    let raw = event.original_text();
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
    // 仅允许纯文本和 QQ 基础表情参与“加一”。
    // 其它 CQ 码（@、图片、回复、文件等）继续忽略，避免误触发和噪音。
    if contains_non_face_cq(trimmed) {
        return None;
    }
    Some(trimmed.to_string())
}

/// Returns true when the message contains CQ segments other than `[CQ:face,id=...]`.
fn contains_non_face_cq(text: &str) -> bool {
    let mut cursor = 0usize;
    while let Some(start_rel) = text[cursor..].find("[CQ:") {
        let start = cursor + start_rel;
        let Some(end_rel) = text[start..].find(']') else {
            return true;
        };
        let end = start + end_rel;
        let segment = &text[start + 1..end];
        if !is_supported_repeat_cq(segment) {
            return true;
        }
        cursor = end + 1;
    }
    false
}

/// Returns true only for basic QQ face segments that can be safely echoed back.
fn is_supported_repeat_cq(segment: &str) -> bool {
    let Some(rest) = segment.strip_prefix("CQ:face,") else {
        return false;
    };

    rest.split(',')
        .find_map(|field| field.trim().strip_prefix("id="))
        .and_then(|value| value.trim().parse::<i64>().ok())
        .is_some()
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
