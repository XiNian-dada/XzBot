//! In-memory conversation context storage.

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
}

impl MemoryStore {
    /// Builds an empty memory store.
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    /// Clears the full conversation context for one session.
    pub fn reset(&self, key: &SessionKey) {
        self.sessions.remove(key);
    }

    /// Returns a snapshot of current conversation messages for one session.
    pub fn messages(&self, key: &SessionKey) -> Vec<(String, String)> {
        self.sessions
            .get(key)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Appends one user message and trims old history if needed.
    pub fn push_user_message(&self, key: SessionKey, content: String) {
        self.push_message(key, "user", content);
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
}
