use serde::Serialize;
use serde_json::json;

#[derive(Debug, Serialize)]
pub struct ActionRequest {
    pub action: String,
    pub params: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<String>,
}

impl ActionRequest {
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
}
