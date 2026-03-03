//! Chat-bound POST token storage for external push delivery.

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use tokio::{fs, sync::RwLock};

/// Token binding record for one chat target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTokenEntry {
    /// Chat type: `private` or `group`.
    pub chat_type: String,
    /// Private target user id.
    pub user_id: i64,
    /// Group target id when `chat_type = group`.
    pub group_id: Option<i64>,
    /// API token used by external POST calls.
    pub token: String,
    /// Unix timestamp when token was created.
    pub created_at: u64,
    /// Unix timestamp when token was last updated/regenerated.
    pub updated_at: u64,
}

impl ChatTokenEntry {
    /// Returns stable key for one chat binding.
    pub fn chat_key(&self) -> String {
        chat_key(&self.chat_type, self.user_id, self.group_id)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedChatTokenData {
    entries: Vec<ChatTokenEntry>,
}

/// In-memory token registry with JSON persistence.
#[derive(Debug, Clone)]
pub struct ChatTokenStore {
    path: Arc<PathBuf>,
    inner: Arc<RwLock<HashMap<String, ChatTokenEntry>>>,
}

impl ChatTokenStore {
    /// Loads token store from disk or creates empty store when file is absent.
    pub async fn load(path: PathBuf) -> Result<Self> {
        let mut map = HashMap::new();
        match fs::read_to_string(&path).await {
            Ok(raw) => {
                let data: PersistedChatTokenData = serde_json::from_str(&raw)
                    .with_context(|| format!("failed to parse token store {}", path.display()))?;
                for item in data.entries {
                    map.insert(item.chat_key(), item);
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("failed to read token store {}", path.display()))
            }
        }

        Ok(Self {
            path: Arc::new(path),
            inner: Arc::new(RwLock::new(map)),
        })
    }

    /// Gets current token for chat, or creates one if absent.
    pub async fn get_or_create(
        &self,
        chat_type: &str,
        user_id: i64,
        group_id: Option<i64>,
    ) -> Result<(ChatTokenEntry, bool)> {
        let key = chat_key(chat_type, user_id, group_id);
        let mut guard = self.inner.write().await;
        if let Some(existing) = guard.get(&key).cloned() {
            return Ok((existing, false));
        }

        let now = now_unix();
        let token = generate_token(&guard)?;
        let entry = ChatTokenEntry {
            chat_type: chat_type.to_string(),
            user_id,
            group_id,
            token,
            created_at: now,
            updated_at: now,
        };
        guard.insert(key, entry.clone());
        self.persist_locked(&guard).await?;
        Ok((entry, true))
    }

    /// Re-generates token for a chat (create if absent).
    pub async fn regenerate(
        &self,
        chat_type: &str,
        user_id: i64,
        group_id: Option<i64>,
    ) -> Result<ChatTokenEntry> {
        let key = chat_key(chat_type, user_id, group_id);
        let mut guard = self.inner.write().await;
        let now = now_unix();
        let token = generate_token(&guard)?;

        let entry = if let Some(existing) = guard.get_mut(&key) {
            existing.token = token;
            existing.updated_at = now;
            existing.clone()
        } else {
            let entry = ChatTokenEntry {
                chat_type: chat_type.to_string(),
                user_id,
                group_id,
                token,
                created_at: now,
                updated_at: now,
            };
            guard.insert(key, entry.clone());
            entry
        };

        self.persist_locked(&guard).await?;
        Ok(entry)
    }

    /// Deletes token for a chat target.
    pub async fn remove(
        &self,
        chat_type: &str,
        user_id: i64,
        group_id: Option<i64>,
    ) -> Result<bool> {
        let key = chat_key(chat_type, user_id, group_id);
        let mut guard = self.inner.write().await;
        let removed = guard.remove(&key).is_some();
        if removed {
            self.persist_locked(&guard).await?;
        }
        Ok(removed)
    }

    /// Looks up chat binding by token.
    pub async fn lookup_token(&self, token: &str) -> Option<ChatTokenEntry> {
        let token = token.trim();
        if token.is_empty() {
            return None;
        }

        let guard = self.inner.read().await;
        guard.values().find(|v| v.token == token).cloned()
    }

    async fn persist_locked(&self, map: &HashMap<String, ChatTokenEntry>) -> Result<()> {
        let mut entries = map.values().cloned().collect::<Vec<_>>();
        entries.sort_by(|a, b| a.chat_key().cmp(&b.chat_key()));
        let data = PersistedChatTokenData { entries };
        let json = serde_json::to_string_pretty(&data).context("failed to encode token store")?;

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).await.with_context(|| {
                format!("failed to create token store dir {}", parent.display())
            })?;
        }
        fs::write(&*self.path, json)
            .await
            .with_context(|| format!("failed to write token store {}", self.path.display()))?;
        Ok(())
    }
}

/// Returns display text for bound chat target.
pub fn chat_target_desc(entry: &ChatTokenEntry) -> String {
    if entry.chat_type == "group" {
        format!("group:{}", entry.group_id.unwrap_or_default())
    } else {
        format!("private:{}", entry.user_id)
    }
}

fn chat_key(chat_type: &str, user_id: i64, group_id: Option<i64>) -> String {
    if chat_type == "group" {
        format!("group:{}", group_id.unwrap_or_default())
    } else {
        format!("private:{user_id}")
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generate_token(existing: &HashMap<String, ChatTokenEntry>) -> Result<String> {
    for _ in 0..5 {
        let mut bytes = [0u8; 24];
        fill_random(&mut bytes)?;
        let token = URL_SAFE_NO_PAD.encode(bytes);
        if existing.values().all(|v| v.token != token) {
            return Ok(token);
        }
    }
    Err(anyhow::anyhow!("failed to generate unique token"))
}

fn fill_random(buf: &mut [u8]) -> Result<()> {
    let mut file =
        File::open("/dev/urandom").context("failed to open /dev/urandom for token generation")?;
    file.read_exact(buf)
        .context("failed to read random bytes for token generation")
}
