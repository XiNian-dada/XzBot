use dashmap::DashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    pub chat_type: String,
    pub group_id: Option<i64>,
    pub user_id: i64,
}

impl SessionKey {
    pub fn new(chat_type: impl Into<String>, group_id: Option<i64>, user_id: i64) -> Self {
        Self {
            chat_type: chat_type.into(),
            group_id,
            user_id,
        }
    }

    pub fn session_id(&self) -> String {
        match self.group_id {
            Some(group_id) => format!("{}:{}:{}", self.chat_type, group_id, self.user_id),
            None => format!("{}:{}", self.chat_type, self.user_id),
        }
    }
}

pub struct MemoryStore {
    // 保存 (role, content)，并在写入时裁剪到最近 10 轮（20 条消息）。
    sessions: DashMap<SessionKey, Vec<(String, String)>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    pub fn reset(&self, key: &SessionKey) {
        self.sessions.remove(key);
    }

    pub fn messages(&self, key: &SessionKey) -> Vec<(String, String)> {
        self.sessions
            .get(key)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    pub fn push_user_message(&self, key: SessionKey, content: String) {
        self.push_message(key, "user", content);
    }

    pub fn push_assistant_message(&self, key: SessionKey, content: String) {
        self.push_message(key, "assistant", content);
    }

    fn push_message(&self, key: SessionKey, role: &str, content: String) {
        let mut entry = self.sessions.entry(key).or_insert_with(Vec::new);
        entry.push((role.to_string(), content));

        if entry.len() > 20 {
            let overflow = entry.len() - 20;
            entry.drain(0..overflow);
        }
    }
}
