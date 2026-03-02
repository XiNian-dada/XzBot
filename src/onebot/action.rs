//! OneBot v11 action request payload builders.

use serde::Serialize;
use serde_json::json;

/// Generic OneBot action request envelope sent over websocket.
#[derive(Debug, Serialize)]
pub struct ActionRequest {
    /// OneBot action name, for example `send_group_msg`.
    pub action: String,
    /// Action parameters payload.
    pub params: serde_json::Value,
    /// Optional request correlation id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<String>,
}

impl ActionRequest {
    /// Builds a private message reply action.
    pub fn send_private_msg(user_id: i64, message: String) -> Self {
        Self {
            action: "send_private_msg".to_string(),
            params: json!({
                "user_id": user_id,
                "message": message
            }),
            echo: None,
        }
    }

    /// Builds a group message reply action.
    pub fn send_group_msg(group_id: i64, message: String) -> Self {
        Self {
            action: "send_group_msg".to_string(),
            params: json!({
                "group_id": group_id,
                "message": message
            }),
            echo: None,
        }
    }

    /// Builds a group file upload action.
    pub fn upload_group_file(group_id: i64, file: String, name: Option<String>) -> Self {
        let mut params = json!({
            "group_id": group_id,
            "file": file
        });
        if let Some(name) = name {
            params["name"] = json!(name);
        }
        Self {
            action: "upload_group_file".to_string(),
            params,
            echo: None,
        }
    }

    /// Builds a private file upload action.
    pub fn upload_private_file(user_id: i64, file: String, name: Option<String>) -> Self {
        let mut params = json!({
            "user_id": user_id,
            "file": file
        });
        if let Some(name) = name {
            params["name"] = json!(name);
        }
        Self {
            action: "upload_private_file".to_string(),
            params,
            echo: None,
        }
    }
}
