//! Application entrypoint, websocket server lifecycle and message pipeline.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Json, Query, State,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::{fs, net::TcpListener};
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::time::timeout;

use crate::{
    bot::{ai_chat::AiChatPlugin, router::BotRouter},
    config::{AiProvider, Config, NetworkConfig},
    llm::{
        anthropic_compatible::AnthropicCompatibleLlm, mock::MockLlm,
        openai_compatible::OpenAiCompatibleLlm, Llm,
    },
    logger::{debug as log_debug, error as log_error, info as log_info, warn as log_warn},
    onebot::action::ActionRequest,
    onebot::event::{extract_cq_image_refs, MessageEvent, MessagePayload},
    post_api::{chat_target_desc, ChatTokenStore},
    plugins::PluginManager,
    store::memory::MemoryStore,
    tools::http::build_client,
};

mod bot;
mod config;
mod llm;
mod logger;
mod onebot;
mod post_api;
mod plugins;
mod store;
mod token_stats;
mod tools;

/// Invisible marker inserted into POST-pushed messages so inbound event handling can ignore them.
const POST_CONTEXT_MARKER: &str = "\u{2063}\u{2064}\u{2063}\u{2064}";

/// Shared application state stored in axum router.
#[derive(Clone)]
struct AppState {
    runtime: Arc<RwLock<RuntimeState>>,
    config_path: Arc<PathBuf>,
    store: Arc<MemoryStore>,
    post_tokens: Arc<ChatTokenStore>,
    ws_action_tx: Arc<RwLock<Option<mpsc::UnboundedSender<String>>>>,
}

/// Live runtime object that can be atomically swapped on `/reload`.
struct RuntimeState {
    router: Arc<BotRouter>,
    config: Arc<Config>,
}

/// Bridge for sending OneBot actions and awaiting async action responses.
#[derive(Clone)]
struct WsActionBridge {
    action_tx: mpsc::UnboundedSender<String>,
    pending: Arc<DashMap<String, oneshot::Sender<Value>>>,
    seq: Arc<AtomicU64>,
    debug: bool,
}

impl WsActionBridge {
    /// Creates a new bridge bound to outbound websocket sender.
    fn new(action_tx: mpsc::UnboundedSender<String>, debug: bool) -> Self {
        Self {
            action_tx,
            pending: Arc::new(DashMap::new()),
            seq: Arc::new(AtomicU64::new(1)),
            debug,
        }
    }

    /// Dispatches incoming non-event payload to pending action waiter by `echo`.
    fn try_dispatch_response(&self, payload: &Value) -> bool {
        if payload.get("post_type").is_some() {
            return false;
        }

        let Some(echo) = payload.get("echo").and_then(Value::as_str) else {
            return false;
        };

        if let Some((_, tx)) = self.pending.remove(echo) {
            let _ = tx.send(payload.clone());
            log_debug(
                self.debug,
                format!("received onebot action response echo={echo}"),
            );
            return true;
        }

        false
    }

    /// Sends one OneBot action and waits for echoed response.
    async fn call_action(&self, action: &str, params: Value) -> anyhow::Result<Value> {
        let echo = format!("xzbot-echo-{}", self.seq.fetch_add(1, Ordering::Relaxed));
        let request = json!({
            "action": action,
            "params": params,
            "echo": echo,
        });
        let request_text =
            serde_json::to_string(&request).context("failed to encode onebot action request")?;

        let (tx, rx) = oneshot::channel();
        self.pending.insert(echo.clone(), tx);

        if self.action_tx.send(request_text).is_err() {
            self.pending.remove(&echo);
            return Err(anyhow!("failed to send onebot action request"));
        }

        let response = timeout(Duration::from_millis(8_000), rx)
            .await
            .context("onebot action timeout")?
            .context("onebot action response channel closed")?;
        Ok(response)
    }
}

#[tokio::main]
/// Process entrypoint: load config/runtime, start websocket server.
async fn main() -> anyhow::Result<()> {
    let exe_path = std::env::current_exe().context("failed to get current executable path")?;
    let exe_dir = exe_path
        .parent()
        .context("failed to resolve executable directory")?;
    let config_path = exe_dir.join("config").join("config.toml");

    let config = Arc::new(Config::load(&config_path)?);
    check_proxy_availability(&config.network).await?;
    let store = Arc::new(MemoryStore::new());
    let token_store_path = exe_dir.join("config").join("post_tokens.json");
    let post_tokens = Arc::new(ChatTokenStore::load(token_store_path).await?);
    let ws_action_tx = Arc::new(RwLock::new(None));
    let plugin_root = std::env::current_dir()?.join("Plugins");
    let plugins = PluginManager::load_from_dir(&plugin_root, config.clone())?;
    log_info(format!("Plugins dir: {}", plugin_root.display()));
    log_info(format!("Loaded plugins: {}", plugins.plugin_count()));

    let runtime = Arc::new(RwLock::new(build_runtime(
        config.clone(),
        store.clone(),
        plugins,
    )?));

    let ws_path = config.server.ws_path.clone();
    let bind_addr = format!("{}:{}", config.server.host, config.server.port);

    let app = Router::new()
        .route(ws_path.as_str(), get(onebot_ws_handler))
        .route("/api/post/send", post(post_send_handler))
        .with_state(AppState {
            runtime,
            config_path: Arc::new(config_path.clone()),
            store,
            post_tokens,
            ws_action_tx,
        });

    let listener = TcpListener::bind(&bind_addr).await?;
    log_info(format!("OneBot reverse WS server listening on {bind_addr}"));
    log_info(format!("Using config: {}", config_path.display()));
    log_info(format!("LLM provider: {}", config.ai.provider.as_str()));
    log_debug(config.debug, "debug log is enabled");
    axum::serve(listener, app).await?;

    Ok(())
}

/// Builds runtime components from config/store/plugins.
fn build_runtime(
    config: Arc<Config>,
    store: Arc<MemoryStore>,
    plugins: PluginManager,
) -> anyhow::Result<RuntimeState> {
    let llm: Arc<dyn Llm> = match config.ai.provider {
        AiProvider::Mock => Arc::new(MockLlm::new(
            config.debug,
            config.ai.timeout_ms,
            config.search.clone(),
            config.network.clone(),
        )?),
        AiProvider::OpenaiCompatible => Arc::new(OpenAiCompatibleLlm::from_config(
            &config.ai,
            &config.search,
            &config.network,
            config.debug,
        )?),
        AiProvider::AnthropicCompatible => Arc::new(AnthropicCompatibleLlm::from_config(
            &config.ai,
            &config.search,
            &config.network,
            config.debug,
        )?),
    };
    let ai_chat = AiChatPlugin::new(store, llm, config.clone());
    let router = Arc::new(BotRouter::new(ai_chat, plugins, config.clone()));

    Ok(RuntimeState { router, config })
}

/// Performs startup proxy connectivity check when proxy is enabled.
async fn check_proxy_availability(config: &NetworkConfig) -> anyhow::Result<()> {
    if !config.proxy_enabled {
        return Ok(());
    }
    let client = build_client(config.proxy_timeout_ms, config, false)
        .context("failed to build proxy check client")?;
    let url = config.proxy_test_url.trim();
    if url.is_empty() {
        return Err(anyhow!("proxy_test_url is empty"));
    }
    let response = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await
        .with_context(|| format!("proxy check request failed: {url}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(anyhow!("proxy check failed: status {} for {}", status, url));
    }
    log_info(format!("proxy check ok via {}", url));
    Ok(())
}

/// Axum websocket upgrade handler with optional token verification.
async fn onebot_ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };

    if config.server.verify_token {
        let expected = config.server.access_token.trim();
        let provided = extract_access_token(&headers, &query);
        if provided.as_deref() != Some(expected) {
            log_warn("websocket rejected: invalid access token");
            return (StatusCode::UNAUTHORIZED, "invalid access token").into_response();
        }
        log_debug(config.debug, "websocket token verified");
    }

    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum OneOrManyString {
    One(String),
    Many(Vec<String>),
}

impl OneOrManyString {
    /// Normalizes single-or-list payload into flat string list.
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(v) => vec![v],
            Self::Many(v) => v,
        }
    }
}

/// HTTP payload for external POST push API.
#[derive(Debug, serde::Deserialize)]
struct PostSendRequest {
    token: String,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    image: Option<OneOrManyString>,
    #[serde(default)]
    images: Option<OneOrManyString>,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    file_name: Option<String>,
}

/// JSON response for external POST push API.
#[derive(Debug, serde::Serialize)]
struct PostSendResponse {
    ok: bool,
    sent_actions: usize,
    chat_type: String,
    user_id: i64,
    group_id: Option<i64>,
}

/// POST endpoint: send message/image/file to chat bound by token.
async fn post_send_handler(
    State(state): State<AppState>,
    Json(req): Json<PostSendRequest>,
) -> impl IntoResponse {
    let token = req.token.trim().to_string();
    if token.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": "token is empty" })),
        )
            .into_response();
    }

    let Some(target) = state.post_tokens.lookup_token(&token).await else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "ok": false, "error": "invalid token" })),
        )
            .into_response();
    };

    let sender = {
        let guard = state.ws_action_tx.read().await;
        guard.clone()
    };
    let Some(sender) = sender else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "ok": false, "error": "onebot websocket not connected" })),
        )
            .into_response();
    };

    let mut actions = Vec::new();
    if let Some(file_path) = req.file_path.as_ref().map(|v| v.trim()).filter(|v| !v.is_empty()) {
        let action = if target.chat_type == "group" {
            ActionRequest::upload_group_file(
                target.group_id.unwrap_or_default(),
                file_path.to_string(),
                req.file_name.clone(),
            )
        } else {
            ActionRequest::upload_private_file(target.user_id, file_path.to_string(), req.file_name.clone())
        };
        actions.push(action);
    }

    let mut images = Vec::new();
    if let Some(v) = req.image {
        images.extend(v.into_vec());
    }
    if let Some(v) = req.images {
        images.extend(v.into_vec());
    }
    images = images
        .into_iter()
        .map(|v| normalize_image_ref(&v))
        .filter(|v| !v.is_empty())
        .collect();

    let mut text = req.message.unwrap_or_default();
    for image_ref in images {
        if !text.is_empty() && !text.ends_with(' ') {
            text.push(' ');
        }
        text.push_str(&format!("[CQ:image,file={}]", image_ref));
    }
    if !text.trim().is_empty() {
        // Mark POST-pushed message so inbound pipeline can skip it from AI context.
        text = format!("{POST_CONTEXT_MARKER}{text}");
        let action = if target.chat_type == "group" {
            ActionRequest::send_group_msg(target.group_id.unwrap_or_default(), text)
        } else {
            ActionRequest::send_private_msg(target.user_id, text)
        };
        actions.push(action);
    }

    if actions.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": "no message/image/file provided" })),
        )
            .into_response();
    }

    for action in &actions {
        let encoded = match serde_json::to_string(action) {
            Ok(v) => v,
            Err(err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "ok": false, "error": format!("encode action failed: {err}") })),
                )
                    .into_response();
            }
        };
        if sender.send(encoded).is_err() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "ok": false, "error": "failed to send action to websocket" })),
            )
                .into_response();
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::to_value(PostSendResponse {
            ok: true,
            sent_actions: actions.len(),
            chat_type: target.chat_type,
            user_id: target.user_id,
            group_id: target.group_id,
        })
        .unwrap_or_else(|_| json!({ "ok": true, "sent_actions": actions.len() }))),
    )
        .into_response()
}

/// One websocket connection lifecycle (reader loop + writer task).
async fn handle_socket(socket: WebSocket, state: AppState) {
    log_info("websocket connected");
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    {
        let mut guard = state.ws_action_tx.write().await;
        *guard = Some(tx.clone());
    }
    let debug = {
        let runtime = state.runtime.read().await;
        runtime.config.debug
    };
    let bridge = WsActionBridge::new(tx.clone(), debug);

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
                let debug = {
                    let runtime = state.runtime.read().await;
                    runtime.config.debug
                };
                log_debug(debug, format!("incoming ws text length={}", text.len()));
                handle_incoming_payload(text.as_str(), &state, &bridge, &tx);
            }
            Message::Binary(bytes) => {
                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                    let debug = {
                        let runtime = state.runtime.read().await;
                        runtime.config.debug
                    };
                    log_debug(
                        debug,
                        format!("incoming ws binary utf8 length={}", text.len()),
                    );
                    handle_incoming_payload(&text, &state, &bridge, &tx);
                }
            }
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    drop(tx);
    let _ = writer.await;
    {
        let mut guard = state.ws_action_tx.write().await;
        if let Some(current) = guard.as_ref() {
            if current.same_channel(&bridge.action_tx) {
                *guard = None;
            }
        }
    }
    log_info("websocket disconnected");
}

/// Parses one websocket text payload and dispatches async event processing.
fn handle_incoming_payload(
    payload: &str,
    state: &AppState,
    bridge: &WsActionBridge,
    tx: &mpsc::UnboundedSender<String>,
) {
    let value: Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(err) => {
            log_error(format!("invalid json payload: {err}"));
            return;
        }
    };

    if bridge.try_dispatch_response(&value) {
        return;
    }

    let state = state.clone();
    let bridge = bridge.clone();
    let tx = tx.clone();
    tokio::spawn(async move {
        let action_texts = process_incoming(&state, &bridge, value).await;
        for action_text in action_texts {
            let _ = tx.send(action_text);
        }
    });
}

/// Parses one OneBot payload and routes to bot runtime.
async fn process_incoming(
    state: &AppState,
    bridge: &WsActionBridge,
    payload: Value,
) -> Vec<String> {
    if payload.get("post_type").and_then(Value::as_str) != Some("message") {
        return Vec::new();
    }

    let mut event: MessageEvent = match serde_json::from_value(payload) {
        Ok(v) => v,
        Err(err) => {
            log_error(format!("invalid message event: {err}"));
            return Vec::new();
        }
    };

    let (config, router) = {
        let runtime = state.runtime.read().await;
        (runtime.config.clone(), runtime.router.clone())
    };

    if is_post_context_marker_message(&event) {
        log_debug(
            config.debug,
            format!(
                "skip post-marked message user_id={} group_id={:?}",
                event.user_id, event.group_id
            ),
        );
        return Vec::new();
    }

    if let Err(err) = enrich_event_images(&mut event, bridge, config.debug).await {
        log_warn(format!("failed to enrich image context: {err}"));
    }

    log_debug(
        config.debug,
        format!(
            "event message_type={} user_id={} group_id={:?}",
            event.message_type, event.user_id, event.group_id
        ),
    );

    if let Some(action) = try_reload_config_command(state, &event, &config).await {
        return serde_json::to_string(&action)
            .ok()
            .into_iter()
            .collect();
    }

    if let Some(actions) = try_post_token_command(state, &event, &config).await {
        return actions
            .into_iter()
            .filter_map(|v| serde_json::to_string(&v).ok())
            .collect();
    }

    if let Some(actions) = try_dump_log_command(state, &event, &config).await {
        return actions
            .into_iter()
            .filter_map(|v| serde_json::to_string(&v).ok())
            .collect();
    }

    let actions = match router.route_message(event).await {
        Ok(action) => action,
        Err(err) => {
            log_error(format!("route message failed: {err}"));
            return Vec::new();
        }
    };

    actions
        .into_iter()
        .filter_map(|v| serde_json::to_string(&v).ok())
        .collect()
}

/// Outcome object returned by runtime reload.
struct ReloadOutcome {
    config: Arc<Config>,
    server_rebind_required: bool,
    plugin_names: Vec<String>,
}

/// Handles owner `/reload` command and returns immediate status action.
async fn try_reload_config_command(
    state: &AppState,
    event: &MessageEvent,
    config: &Config,
) -> Option<ActionRequest> {
    if !is_reload_command(event, config.owner.qq) {
        return None;
    }

    match reload_runtime(state).await {
        Ok(outcome) => {
            let plugin_summary = format!(
                "plugins({}): {}",
                outcome.plugin_names.len(),
                if outcome.plugin_names.is_empty() {
                    "-".to_string()
                } else {
                    outcome.plugin_names.join(", ")
                }
            );
            log_info(format!(
                "config reloaded: provider={} model={} {}",
                outcome.config.ai.provider.as_str(),
                outcome.config.ai.model,
                plugin_summary
            ));

            let mut msg = format!(
                "配置已重载。provider={} model={} {}",
                outcome.config.ai.provider.as_str(),
                outcome.config.ai.model,
                plugin_summary
            );
            if outcome.server_rebind_required {
                msg.push_str(
                    "；检测到 server.host/server.port/server.ws_path 变更，需重启进程后生效。",
                );
            }
            Some(reload_reply_action(event, msg))
        }
        Err(err) => {
            log_error(format!("reload config failed: {err:#}"));
            Some(reload_reply_action(event, format!("配置重载失败：{err}")))
        }
    }
}

/// Checks whether current event is owner-issued reload command.
fn is_reload_command(event: &MessageEvent, owner_qq: i64) -> bool {
    if event.user_id != owner_qq {
        return false;
    }

    let text = event.text();
    let trimmed = text.trim();
    if event.message_type == "private" {
        return trimmed == "/reload";
    }

    if event.message_type != "group" {
        return false;
    }

    let at_me = format!("[CQ:at,qq={}]", event.self_id);
    if !text.contains(&at_me) {
        return false;
    }

    text.replace(&at_me, "").trim() == "/reload"
}

/// Builds reply action for reload status message.
fn reload_reply_action(event: &MessageEvent, message: String) -> ActionRequest {
    if event.message_type == "group" {
        if let Some(group_id) = event.group_id {
            return ActionRequest::send_group_msg(
                group_id,
                format!("[CQ:at,qq={}] {}", event.user_id, message),
            );
        }
    }
    ActionRequest::send_private_msg(event.user_id, message)
}

/// Handles owner `/log [N]` command and returns an upload action for recent logs.
async fn try_dump_log_command(
    state: &AppState,
    event: &MessageEvent,
    config: &Config,
) -> Option<Vec<ActionRequest>> {
    let line_limit = parse_log_command(event, config.owner.qq)?;
    let lines = crate::logger::recent_lines(line_limit);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let base_dir = state
        .config_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::temp_dir());
    let log_dir = base_dir.join("logs");
    if let Err(err) = fs::create_dir_all(&log_dir).await {
        return Some(vec![reload_reply_action(
            event,
            format!("导出日志失败：无法创建目录 {} ({err})", log_dir.display()),
        )]);
    }

    let file_name = format!("xzbot-log-{}-{}.txt", ts, lines.len());
    let file_path = log_dir.join(&file_name);
    let mut body = format!(
        "# XzBot Recent Logs\n# generated_at_unix={ts}\n# requested_lines={line_limit}\n# exported_lines={}\n\n",
        lines.len()
    );
    if lines.is_empty() {
        body.push_str("(no buffered logs)\n");
    } else {
        for line in lines {
            body.push_str(&line);
            body.push('\n');
        }
    }

    if let Err(err) = fs::write(&file_path, body).await {
        return Some(vec![reload_reply_action(
            event,
            format!("导出日志失败：无法写入文件 {} ({err})", file_path.display()),
        )]);
    }

    let action = if event.message_type == "group" {
        if let Some(group_id) = event.group_id {
            ActionRequest::upload_group_file(
                group_id,
                file_path.to_string_lossy().to_string(),
                Some(file_name),
            )
        } else {
            ActionRequest::upload_private_file(
                event.user_id,
                file_path.to_string_lossy().to_string(),
                Some(file_name),
            )
        }
    } else {
        ActionRequest::upload_private_file(
            event.user_id,
            file_path.to_string_lossy().to_string(),
            Some(file_name),
        )
    };

    Some(vec![action])
}

/// Parses owner `/log [N]` command and returns requested line count.
fn parse_log_command(event: &MessageEvent, owner_qq: i64) -> Option<usize> {
    if event.user_id != owner_qq {
        return None;
    }

    let text = event.text();
    let normalized = if event.message_type == "private" {
        text.trim().to_string()
    } else if event.message_type == "group" {
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        if !text.contains(&at_me) {
            return None;
        }
        text.replace(&at_me, "").trim().to_string()
    } else {
        return None;
    };

    let mut parts = normalized.split_whitespace();
    if parts.next()? != "/log" {
        return None;
    }
    let requested = parts
        .next()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(100);
    Some(requested.clamp(10, 2000))
}

#[derive(Debug, Clone, Copy)]
enum PostTokenCommand {
    Create,
    Show,
    Regenerate,
    Delete,
}

/// Handles owner `/posttoken` commands for chat-bound external push token management.
async fn try_post_token_command(
    state: &AppState,
    event: &MessageEvent,
    config: &Config,
) -> Option<Vec<ActionRequest>> {
    let command = parse_post_token_command(event, config.owner.qq)?;
    let (chat_type, user_id, group_id) = if event.message_type == "group" {
        ("group", event.user_id, event.group_id)
    } else if event.message_type == "private" {
        ("private", event.user_id, None)
    } else {
        return Some(vec![reload_reply_action(
            event,
            "仅支持私聊或群聊中使用 /posttoken。".to_string(),
        )]);
    };

    let Some(group_id_checked) = (if chat_type == "group" {
        group_id
    } else {
        Some(0)
    }) else {
        return Some(vec![reload_reply_action(
            event,
            "当前群聊缺少 group_id，无法创建 token。".to_string(),
        )]);
    };

    let result = match command {
        PostTokenCommand::Create => state
            .post_tokens
            .get_or_create(chat_type, user_id, group_id)
            .await
            .map(|(entry, created)| {
                let status = if created { "已创建" } else { "已存在" };
                format!(
                    "{} POST token。\n目标会话：{}\nToken: {}",
                    status,
                    chat_target_desc(&entry),
                    entry.token
                )
            }),
        PostTokenCommand::Show => state
            .post_tokens
            .get_or_create(chat_type, user_id, group_id)
            .await
            .map(|(entry, _)| {
                format!(
                    "当前 POST token。\n目标会话：{}\nToken: {}",
                    chat_target_desc(&entry),
                    entry.token
                )
            }),
        PostTokenCommand::Regenerate => state
            .post_tokens
            .regenerate(chat_type, user_id, group_id)
            .await
            .map(|entry| {
                format!(
                    "已重新生成 POST token。\n目标会话：{}\nToken: {}",
                    chat_target_desc(&entry),
                    entry.token
                )
            }),
        PostTokenCommand::Delete => state
            .post_tokens
            .remove(chat_type, user_id, group_id)
            .await
            .map(|removed| {
                if chat_type == "group" {
                    format!(
                        "目标会话：group:{}\n{}",
                        group_id_checked,
                        if removed { "Token 已删除。" } else { "未找到 token。" }
                    )
                } else {
                    format!(
                        "目标会话：private:{}\n{}",
                        user_id,
                        if removed { "Token 已删除。" } else { "未找到 token。" }
                    )
                }
            }),
    };

    let text = match result {
        Ok(v) => v,
        Err(err) => return Some(vec![reload_reply_action(event, format!("操作失败：{err}"))]),
    };

    if matches!(command, PostTokenCommand::Delete) {
        return Some(vec![reload_reply_action(event, text)]);
    }

    if event.message_type == "group" {
        let mut out = Vec::new();
        out.push(ActionRequest::send_private_msg(event.user_id, text));
        if let Some(group_id) = event.group_id {
            out.push(ActionRequest::send_group_msg(
                group_id,
                format!(
                    "[CQ:at,qq={}] token 已发送到你的私聊，请注意保密。",
                    event.user_id
                ),
            ));
        }
        Some(out)
    } else {
        Some(vec![ActionRequest::send_private_msg(event.user_id, text)])
    }
}

/// Parses `/posttoken` command and enforces owner/auth mention rules.
fn parse_post_token_command(event: &MessageEvent, owner_qq: i64) -> Option<PostTokenCommand> {
    if event.user_id != owner_qq {
        return None;
    }

    let text = event.text();
    let normalized = if event.message_type == "private" {
        text.trim().to_string()
    } else if event.message_type == "group" {
        let at_me = format!("[CQ:at,qq={}]", event.self_id);
        if !text.contains(&at_me) {
            return None;
        }
        text.replace(&at_me, "").trim().to_string()
    } else {
        return None;
    };

    let mut parts = normalized.split_whitespace();
    if parts.next()? != "/posttoken" {
        return None;
    }

    match parts.next().unwrap_or("show").to_ascii_lowercase().as_str() {
        "create" => Some(PostTokenCommand::Create),
        "show" => Some(PostTokenCommand::Show),
        "regen" | "regenerate" | "reset" => Some(PostTokenCommand::Regenerate),
        "delete" | "remove" | "del" => Some(PostTokenCommand::Delete),
        _ => Some(PostTokenCommand::Show),
    }
}

/// Reloads config and plugin runtime without process restart.
async fn reload_runtime(state: &AppState) -> anyhow::Result<ReloadOutcome> {
    let config_path = state.config_path.as_ref();
    let new_config = Arc::new(Config::load(config_path)?);
    let plugin_root = std::env::current_dir()?.join("Plugins");
    let plugins = PluginManager::load_from_dir(&plugin_root, new_config.clone())?;
    let plugin_names = plugins.plugin_names();
    let new_runtime = build_runtime(new_config.clone(), state.store.clone(), plugins)?;

    let mut runtime = state.runtime.write().await;
    let old_router = runtime.router.clone();
    let old_config = runtime.config.clone();
    *runtime = new_runtime;

    drop(runtime);
    old_router.shutdown_plugins().await;

    let server_rebind_required = old_config.server.host != new_config.server.host
        || old_config.server.port != new_config.server.port
        || old_config.server.ws_path != new_config.server.ws_path;

    Ok(ReloadOutcome {
        config: new_config,
        server_rebind_required,
        plugin_names,
    })
}

/// Enriches incoming event with resolvable image URLs and quoted message text context.
async fn enrich_event_images(
    event: &mut MessageEvent,
    bridge: &WsActionBridge,
    debug: bool,
) -> anyhow::Result<()> {
    let mut urls = Vec::new();
    let mut seen = HashSet::new();
    let mut quote_texts = Vec::new();
    let mut seen_quote_texts = HashSet::new();

    // 0) 直接读取 raw_message 里已有的 CQ image url（有些群直接给 url）。
    for image_ref in extract_cq_image_refs(&event.raw_message) {
        if let Some(url) = image_ref.url {
            let url = normalize_image_ref(&url);
            if looks_like_http_url(&url) && seen.insert(url.clone()) {
                urls.push(url);
            }
        }
    }

    // 0.1) 结构化 segments 里带 url 的情况。
    if let MessagePayload::Segments(segments) = &event.message {
        for segment in segments {
            if segment.kind != "image" {
                continue;
            }
            if let Some(url) = segment.data.get("url").and_then(Value::as_str) {
                let url = normalize_image_ref(url);
                if looks_like_http_url(&url) && seen.insert(url.clone()) {
                    urls.push(url);
                }
            }
        }
    }

    // 1) 当前消息里的图片 file id -> 尝试 get_image 解析 URL。
    for file_id in event.image_file_ids().into_iter().take(4) {
        if let Some(url) = resolve_image_url(bridge, &file_id, debug).await {
            if seen.insert(url.clone()) {
                urls.push(url);
            }
        }
    }

    // 2) 引用回复里的图片：通过 get_msg 拉取被引用消息，再解析其中图片。
    for reply_id in event.reply_message_ids().into_iter().take(3) {
        let response = match bridge
            .call_action("get_msg", json!({ "message_id": reply_id }))
            .await
        {
            Ok(v) => v,
            Err(err) => {
                log_debug(debug, format!("get_msg failed reply_id={reply_id}: {err}"));
                continue;
            }
        };

        let data = response.get("data").cloned().unwrap_or(Value::Null);
        let (quoted_urls, quoted_files) = collect_image_refs_from_message_data(&data);
        if let Some(quote_text) = extract_quote_text_from_message_data(&data) {
            let normalized = quote_text.split_whitespace().collect::<Vec<_>>().join(" ");
            if !normalized.is_empty() && seen_quote_texts.insert(normalized.clone()) {
                quote_texts.push(format!("[引用消息] {}", trim_for_context(&normalized, 220)));
            }
        }

        for url in quoted_urls {
            if seen.insert(url.clone()) {
                urls.push(url);
            }
        }

        for file_id in quoted_files.into_iter().take(4) {
            if let Some(url) = resolve_image_url(bridge, &file_id, debug).await {
                if seen.insert(url.clone()) {
                    urls.push(url);
                }
            }
        }
    }

    if !quote_texts.is_empty() {
        if !event.raw_message.is_empty() {
            event.raw_message.push(' ');
        }
        event.raw_message.push_str(&quote_texts.join(" "));
        log_debug(
            debug,
            format!("event quote context enriched count={}", quote_texts.len()),
        );
    }

    if !urls.is_empty() {
        if !event.raw_message.is_empty() {
            event.raw_message.push(' ');
        }
        for (idx, url) in urls.iter().enumerate() {
            if idx > 0 {
                event.raw_message.push(' ');
            }
            event.raw_message.push_str(&format!("[CQ:image,url={url}]"));
        }
        log_debug(
            debug,
            format!("event image context enriched urls={}", urls.len()),
        );
    }

    Ok(())
}

/// Extracts readable quote text from `get_msg` response payload.
fn extract_quote_text_from_message_data(data: &Value) -> Option<String> {
    if let Some(message) = data.get("message") {
        if let Some(text) = message_value_to_text(message) {
            if !text.trim().is_empty() {
                return Some(text);
            }
        }
    }

    data.get("raw_message")
        .and_then(Value::as_str)
        .map(strip_cq_to_text)
        .filter(|v| !v.trim().is_empty())
}

/// Converts OneBot `message` field (string or segments) into plain text.
fn message_value_to_text(message: &Value) -> Option<String> {
    match message {
        Value::String(raw) => {
            let text = strip_cq_to_text(raw);
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Value::Array(segments) => {
            let mut out = String::new();
            for seg in segments {
                let kind = seg.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "text" => {
                        if let Some(text) = seg
                            .get("data")
                            .and_then(|d| d.get("text"))
                            .and_then(Value::as_str)
                        {
                            out.push_str(text);
                        }
                    }
                    "image" => {
                        if !out.ends_with(' ') && !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str("[图片]");
                    }
                    "at" => {
                        let qq_text = seg.get("data").and_then(|d| d.get("qq")).and_then(|v| {
                            v.as_str()
                                .map(str::to_string)
                                .or_else(|| v.as_i64().map(|n| n.to_string()))
                        });
                        if let Some(qq) = qq_text {
                            if !out.ends_with(' ') && !out.is_empty() {
                                out.push(' ');
                            }
                            out.push('@');
                            out.push_str(&qq);
                        }
                    }
                    _ => {}
                }
            }
            let normalized = out.split_whitespace().collect::<Vec<_>>().join(" ");
            if normalized.is_empty() {
                None
            } else {
                Some(normalized)
            }
        }
        _ => None,
    }
}

/// Removes CQ segments and keeps plain text/image placeholders.
fn strip_cq_to_text(raw: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0;

    while let Some(start_rel) = raw[cursor..].find("[CQ:") {
        let start = cursor + start_rel;
        out.push_str(&raw[cursor..start]);

        let Some(end_rel) = raw[start..].find(']') else {
            out.push_str(&raw[start..]);
            cursor = raw.len();
            break;
        };
        let end = start + end_rel;
        let segment = &raw[start + 1..end];
        if segment.starts_with("CQ:image") {
            if !out.ends_with(' ') && !out.is_empty() {
                out.push(' ');
            }
            out.push_str("[图片]");
        }
        cursor = end + 1;
    }

    out.push_str(&raw[cursor..]);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Trims long quote context for prompt safety.
fn trim_for_context(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    chars.into_iter().take(max_chars).collect::<String>() + "...(truncated)"
}

/// Resolves image reference to URL/file by optionally calling OneBot `get_image`.
async fn resolve_image_url(
    bridge: &WsActionBridge,
    image_ref: &str,
    debug: bool,
) -> Option<String> {
    let value = normalize_image_ref(image_ref);
    if value.is_empty() {
        return None;
    }
    if looks_like_http_url(&value) {
        return Some(value.to_string());
    }
    if value.starts_with("base64://") || value.starts_with("data:image/") {
        return Some(value.to_string());
    }
    if value.starts_with("file://") {
        return Some(value.to_string());
    }

    let response = bridge
        .call_action("get_image", json!({ "file": value }))
        .await
        .ok();
    let response = match response {
        Some(v) => v,
        None => {
            log_debug(debug, format!("get_image failed file={image_ref}"));
            return None;
        }
    };
    let data = response.get("data")?;

    for key in ["url", "file"] {
        if let Some(v) = data.get(key).and_then(Value::as_str) {
            let v = normalize_image_ref(v);
            if looks_like_http_url(&v) || v.starts_with("base64://") || v.starts_with("file://") {
                return Some(v.to_string());
            }
            if looks_like_local_path(&v) {
                return Some(format!("file://{v}"));
            }
        }
    }

    if debug {
        log_debug(
            debug,
            format!(
                "get_image unresolved file={} data={}",
                image_ref,
                response.get("data").cloned().unwrap_or(Value::Null)
            ),
        );
    }
    None
}

/// Collects image URL/file refs from message data payload.
fn collect_image_refs_from_message_data(data: &Value) -> (Vec<String>, Vec<String>) {
    let mut urls = Vec::new();
    let mut files = Vec::new();
    let mut seen_urls = HashSet::new();
    let mut seen_files = HashSet::new();

    if let Some(message) = data.get("message") {
        collect_image_refs_from_message_value(
            message,
            &mut urls,
            &mut files,
            &mut seen_urls,
            &mut seen_files,
        );
    }

    if let Some(raw) = data.get("raw_message").and_then(Value::as_str) {
        for image_ref in extract_cq_image_refs(raw) {
            if let Some(url) = image_ref.url {
                let url = normalize_image_ref(&url);
                if !url.is_empty() && seen_urls.insert(url.clone()) {
                    urls.push(url);
                }
            }
            if let Some(file) = image_ref.file {
                let file = normalize_image_ref(&file);
                if !file.is_empty() && seen_files.insert(file.clone()) {
                    files.push(file);
                }
            }
        }
    }

    (urls, files)
}

/// Helper for parsing image refs from one `message` value variant.
fn collect_image_refs_from_message_value(
    message: &Value,
    urls: &mut Vec<String>,
    files: &mut Vec<String>,
    seen_urls: &mut HashSet<String>,
    seen_files: &mut HashSet<String>,
) {
    match message {
        Value::String(raw) => {
            for image_ref in extract_cq_image_refs(raw) {
                if let Some(url) = image_ref.url {
                    let url = normalize_image_ref(&url);
                    if !url.is_empty() && seen_urls.insert(url.clone()) {
                        urls.push(url);
                    }
                }
                if let Some(file) = image_ref.file {
                    let file = normalize_image_ref(&file);
                    if !file.is_empty() && seen_files.insert(file.clone()) {
                        files.push(file);
                    }
                }
            }
        }
        Value::Array(segments) => {
            for seg in segments {
                if seg.get("type").and_then(Value::as_str) != Some("image") {
                    continue;
                }
                let Some(seg_data) = seg.get("data") else {
                    continue;
                };
                if let Some(url) = seg_data.get("url").and_then(Value::as_str) {
                    let url = normalize_image_ref(url);
                    if !url.is_empty() && seen_urls.insert(url.clone()) {
                        urls.push(url);
                    }
                }
                if let Some(file) = seg_data.get("file").and_then(Value::as_str) {
                    let file = normalize_image_ref(file);
                    if !file.is_empty() && seen_files.insert(file.clone()) {
                        files.push(file);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Returns true when value looks like HTTP(S) URL.
fn looks_like_http_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

/// Returns true when value looks like absolute local path.
fn looks_like_local_path(value: &str) -> bool {
    value.starts_with('/') || value.contains(":\\")
}

/// Normalizes image reference string from CQ payload fields.
fn normalize_image_ref(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace("&amp;", "&")
        .replace("&#38;", "&")
}

/// Returns true when message contains internal POST marker and should be excluded from routing/context.
fn is_post_context_marker_message(event: &MessageEvent) -> bool {
    if event.raw_message.starts_with(POST_CONTEXT_MARKER) {
        return true;
    }

    if let MessagePayload::Text(text) = &event.message {
        if text.starts_with(POST_CONTEXT_MARKER) {
            return true;
        }
    }

    event.text().starts_with(POST_CONTEXT_MARKER)
}

/// Extracts access token from headers or query params.
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

/// Reads one header value as string.
fn header_token(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string())
}
