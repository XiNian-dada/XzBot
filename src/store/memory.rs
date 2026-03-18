//! 内存会话存储：维护对话历史、锁状态和上下文裁剪。

use dashmap::DashMap;

/// Stable key used to isolate conversation context by chat scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    /// `private` or `group`.
    pub chat_type: String,
    /// Group id when in group chat.
    pub group_id: Option<i64>,
    /// Sender id (private) or primary actor id (group session strategy dependent).
    pub user_id: i64,
}

impl SessionKey {
    /// Creates a new session key.
    pub fn new(chat_type: impl Into<String>, group_id: Option<i64>, user_id: i64) -> Self {
        Self {
            chat_type: chat_type.into(),
            group_id,
            user_id,
        }
    }

    /// Returns the canonical session identifier string.
    pub fn session_id(&self) -> String {
        match self.group_id {
            Some(group_id) => format!("{}:{}:{}", self.chat_type, group_id, self.user_id),
            None => format!("{}:{}", self.chat_type, self.user_id),
        }
    }
}

/// Thread-safe in-memory session store with bounded context length.
pub struct MemoryStore {
    // 保存 (role, content)，并在写入时裁剪到最近 10 轮（20 条消息）。
    sessions: DashMap<SessionKey, Vec<(String, String)>>,
    // 保存每个群最近的原始聊天摘录，供被 @ 时补作“群内最近发生了什么”的环境上下文。
    recent_group_messages: DashMap<i64, Vec<String>>,
}

impl MemoryStore {
    /// Builds an empty memory store.
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            recent_group_messages: DashMap::new(),
        }
    }

    /// Clears the full conversation context for one session.
    pub fn reset(&self, key: &SessionKey) {
        self.sessions.remove(key);
    }

    /// Appends one user message and returns the updated history snapshot.
    ///
    /// AI 主链路里最常见的模式是：
    /// 1. 先把用户消息写入上下文
    /// 2. 立刻读取完整历史去调用模型
    ///
    /// 单独 `push` 再 `get` 会多做一次 `DashMap` 查找，因此这里提供一个组合接口。
    pub fn push_user_message_and_snapshot(
        &self,
        key: SessionKey,
        content: String,
    ) -> Vec<(String, String)> {
        self.push_message_and_snapshot(key, "user", content)
    }

    /// Appends one assistant message and trims old history if needed.
    pub fn push_assistant_message(&self, key: SessionKey, content: String) {
        self.push_message(key, "assistant", content);
    }

    /// Internal append helper with fixed window truncation.
    fn push_message(&self, key: SessionKey, role: &str, content: String) {
        let mut entry = self.sessions.entry(key).or_insert_with(Vec::new);
        entry.push((role.to_string(), content));

        if entry.len() > 20 {
            let overflow = entry.len() - 20;
            entry.drain(0..overflow);
        }
    }

    /// Internal append helper that also clones the updated snapshot for immediate reuse.
    fn push_message_and_snapshot(
        &self,
        key: SessionKey,
        role: &str,
        content: String,
    ) -> Vec<(String, String)> {
        let mut entry = self.sessions.entry(key).or_insert_with(Vec::new);
        entry.push((role.to_string(), content));

        if entry.len() > 20 {
            let overflow = entry.len() - 20;
            entry.drain(0..overflow);
        }

        entry.clone()
    }

    /// Appends one summarized group message into the rolling recent-message buffer.
    ///
    /// 这条缓冲区和 AI 对话历史不同：
    /// - 它保存的是“群里最近说过什么”
    /// - 目的是让机器人在被 @ 时，能看到刚刚几条并非直接对它说的话
    pub fn push_recent_group_message(&self, group_id: i64, content: String) {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return;
        }

        let mut entry = self
            .recent_group_messages
            .entry(group_id)
            .or_insert_with(Vec::new);
        entry.push(trimmed.to_string());

        if entry.len() > 30 {
            let overflow = entry.len() - 30;
            entry.drain(0..overflow);
        }
    }

    /// Returns up to `limit` recent summarized group messages in chronological order.
    pub fn recent_group_messages(&self, group_id: i64, limit: usize) -> Vec<String> {
        self.recent_group_messages
            .get(&group_id)
            .map(|entry| {
                let len = entry.len();
                let start = len.saturating_sub(limit);
                entry[start..].to_vec()
            })
            .unwrap_or_default()
    }

    /// Renders a compact recent-group-context block for model/tool consumption.
    pub fn recent_group_context(
        &self,
        group_id: i64,
        limit: usize,
        max_chars: usize,
    ) -> Option<String> {
        let recent = self.recent_group_messages(group_id, limit);
        if recent.is_empty() {
            return None;
        }

        let mut out = String::from("[最近群聊]\n");
        for (idx, line) in recent.iter().enumerate() {
            let item = format!("{}. {}\n", idx + 1, line);
            if out.len() + item.len() > max_chars {
                break;
            }
            out.push_str(&item);
        }

        let trimmed = out.trim().to_string();
        if trimmed == "[最近群聊]" {
            None
        } else {
            Some(trimmed)
        }
    }
}
