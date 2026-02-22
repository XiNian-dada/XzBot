use std::{collections::HashMap, sync::Arc};

use anyhow::Context;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::{
    bot::{ai_chat::AiChatPlugin, router::BotRouter},
    config::{AiProvider, Config},
    llm::{
        anthropic_compatible::AnthropicCompatibleLlm, mock::MockLlm,
        openai_compatible::OpenAiCompatibleLlm, Llm,
    },
    logger::{debug as log_debug, error as log_error, info as log_info, warn as log_warn},
    onebot::event::MessageEvent,
    store::memory::MemoryStore,
};

mod bot;
mod config;
mod llm;
mod logger;
mod onebot;
mod store;
mod tools;

#[derive(Clone)]
struct AppState {
    router: Arc<BotRouter>,
    config: Arc<Config>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let exe_path = std::env::current_exe().context("failed to get current executable path")?;
    let exe_dir = exe_path
        .parent()
        .context("failed to resolve executable directory")?;
    let config_path = exe_dir.join("config").join("config.toml");

    let config = Arc::new(Config::load(&config_path)?);
    let store = Arc::new(MemoryStore::new());
    let llm: Arc<dyn Llm> = match config.ai.provider {
        AiProvider::Mock => Arc::new(MockLlm::new(config.debug, config.ai.timeout_ms)?),
        AiProvider::OpenaiCompatible => {
            Arc::new(OpenAiCompatibleLlm::from_config(&config.ai, config.debug)?)
        }
        AiProvider::AnthropicCompatible => Arc::new(AnthropicCompatibleLlm::from_config(
            &config.ai,
            config.debug,
        )?),
    };
    let ai_chat = AiChatPlugin::new(store, llm, config.clone());
    let router = Arc::new(BotRouter::new(ai_chat, config.clone()));
    let ws_path = config.server.ws_path.clone();
    let bind_addr = format!("{}:{}", config.server.host, config.server.port);

    let app = Router::new()
        .route(ws_path.as_str(), get(onebot_ws_handler))
        .with_state(AppState {
            router,
            config: config.clone(),
        });

    let listener = TcpListener::bind(&bind_addr).await?;
    log_info(format!("OneBot reverse WS server listening on {bind_addr}"));
    log_info(format!("Using config: {}", config_path.display()));
    log_info(format!("LLM provider: {}", config.ai.provider.as_str()));
    log_debug(config.debug, "debug log is enabled");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn onebot_ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    if state.config.server.verify_token {
        let expected = state.config.server.access_token.trim();
        let provided = extract_access_token(&headers, &query);
        if provided.as_deref() != Some(expected) {
            log_warn("websocket rejected: invalid access token");
            return (StatusCode::UNAUTHORIZED, "invalid access token").into_response();
        }
        log_debug(state.config.debug, "websocket token verified");
    }

    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    log_info("websocket connected");
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let debug = state.config.debug;

    let writer = tokio::spawn(async move {
        while let Some(action_text) = rx.recv().await {
            if sender
                .send(Message::Text(action_text.into()))
                .await
                .is_err()
            {
                log_warn("failed to send action to websocket peer");
                break;
            }
            log_debug(debug, "onebot action sent");
        }
    });

    while let Some(result) = receiver.next().await {
        let message = match result {
            Ok(msg) => msg,
            Err(err) => {
                log_error(format!("websocket receive error: {err}"));
                break;
            }
        };

        match message {
            Message::Text(text) => {
                log_debug(
                    state.config.debug,
                    format!("incoming ws text length={}", text.len()),
                );
                let tx = tx.clone();
                let state = state.clone();
                tokio::spawn(async move {
                    if let Some(action_text) = process_incoming(&state, &text).await {
                        let _ = tx.send(action_text);
                    }
                });
            }
            Message::Binary(bytes) => {
                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                    log_debug(
                        state.config.debug,
                        format!("incoming ws binary utf8 length={}", text.len()),
                    );
                    let tx = tx.clone();
                    let state = state.clone();
                    tokio::spawn(async move {
                        if let Some(action_text) = process_incoming(&state, &text).await {
                            let _ = tx.send(action_text);
                        }
                    });
                }
            }
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    drop(tx);
    let _ = writer.await;
    log_info("websocket disconnected");
}

async fn process_incoming(state: &AppState, payload: &str) -> Option<String> {
    let value: Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(err) => {
            log_error(format!("invalid json payload: {err}"));
            return None;
        }
    };

    if value.get("post_type").and_then(Value::as_str) != Some("message") {
        return None;
    }

    let event: MessageEvent = match serde_json::from_value(value) {
        Ok(v) => v,
        Err(err) => {
            log_error(format!("invalid message event: {err}"));
            return None;
        }
    };

    log_debug(
        state.config.debug,
        format!(
            "event message_type={} user_id={} group_id={:?}",
            event.message_type, event.user_id, event.group_id
        ),
    );

    let action = match state.router.route_message(event).await {
        Ok(action) => action,
        Err(err) => {
            log_error(format!("route message failed: {err}"));
            return None;
        }
    };

    action.and_then(|v| serde_json::to_string(&v).ok())
}

fn extract_access_token(headers: &HeaderMap, query: &HashMap<String, String>) -> Option<String> {
    if let Some(token) = header_token(headers, "authorization") {
        if let Some(bearer) = token.strip_prefix("Bearer ") {
            return Some(bearer.trim().to_string());
        }
        if let Some(bearer) = token.strip_prefix("bearer ") {
            return Some(bearer.trim().to_string());
        }
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }

    if let Some(token) = header_token(headers, "x-access-token") {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }

    if let Some(token) = header_token(headers, "access_token") {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }

    if let Some(token) = query.get("access_token").or_else(|| query.get("token")) {
        if !token.trim().is_empty() {
            return Some(token.trim().to_string());
        }
    }

    None
}

fn header_token(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string())
}
