use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{anyhow, Context};
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
use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpListener;
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
    plugins::PluginManager,
    store::memory::MemoryStore,
    tools::http::build_client,
};

mod bot;
mod config;
mod llm;
mod logger;
mod onebot;
mod plugins;
mod store;
mod token_stats;
mod tools;

#[derive(Clone)]
struct AppState {
    runtime: Arc<RwLock<RuntimeState>>,
    config_path: Arc<PathBuf>,
    store: Arc<MemoryStore>,
}

struct RuntimeState {
    router: Arc<BotRouter>,
    config: Arc<Config>,
}

#[derive(Clone)]
struct WsActionBridge {
    action_tx: mpsc::UnboundedSender<String>,
    pending: Arc<DashMap<String, oneshot::Sender<Value>>>,
    seq: Arc<AtomicU64>,
    debug: bool,
}

impl WsActionBridge {
    fn new(action_tx: mpsc::UnboundedSender<String>, debug: bool) -> Self {
        Self {
            action_tx,
            pending: Arc::new(DashMap::new()),
            seq: Arc::new(AtomicU64::new(1)),
            debug,
        }
    }

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
async fn main() -> anyhow::Result<()> {
    let exe_path = std::env::current_exe().context("failed to get current executable path")?;
    let exe_dir = exe_path
        .parent()
        .context("failed to resolve executable directory")?;
    let config_path = exe_dir.join("config").join("config.toml");

    let config = Arc::new(Config::load(&config_path)?);
    check_proxy_availability(&config.network).await?;
    let store = Arc::new(MemoryStore::new());
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
        .with_state(AppState {
            runtime,
            config_path: Arc::new(config_path.clone()),
            store,
        });

    let listener = TcpListener::bind(&bind_addr).await?;
    log_info(format!("OneBot reverse WS server listening on {bind_addr}"));
    log_info(format!("Using config: {}", config_path.display()));
    log_info(format!("LLM provider: {}", config.ai.provider.as_str()));
    log_debug(config.debug, "debug log is enabled");
    axum::serve(listener, app).await?;

    Ok(())
}

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

async fn handle_socket(socket: WebSocket, state: AppState) {
    log_info("websocket connected");
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
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
    log_info("websocket disconnected");
}

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
        if let Some(action_text) = process_incoming(&state, &bridge, value).await {
            let _ = tx.send(action_text);
        }
    });
}

async fn process_incoming(
    state: &AppState,
    bridge: &WsActionBridge,
    payload: Value,
) -> Option<String> {
    if payload.get("post_type").and_then(Value::as_str) != Some("message") {
        return None;
    }

    let mut event: MessageEvent = match serde_json::from_value(payload) {
        Ok(v) => v,
        Err(err) => {
            log_error(format!("invalid message event: {err}"));
            return None;
        }
    };

    let (config, router) = {
        let runtime = state.runtime.read().await;
        (runtime.config.clone(), runtime.router.clone())
    };

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
        return serde_json::to_string(&action).ok();
    }

    let action = match router.route_message(event).await {
        Ok(action) => action,
        Err(err) => {
            log_error(format!("route message failed: {err}"));
            return None;
        }
    };

    action.and_then(|v| serde_json::to_string(&v).ok())
}

struct ReloadOutcome {
    config: Arc<Config>,
    server_rebind_required: bool,
}

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
            log_info(format!(
                "config reloaded: provider={} model={}",
                outcome.config.ai.provider.as_str(),
                outcome.config.ai.model
            ));

            let mut msg = format!(
                "配置已重载。provider={} model={}",
                outcome.config.ai.provider.as_str(),
                outcome.config.ai.model
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

async fn reload_runtime(state: &AppState) -> anyhow::Result<ReloadOutcome> {
    let config_path = state.config_path.as_ref();
    let new_config = Arc::new(Config::load(config_path)?);
    let plugin_root = std::env::current_dir()?.join("Plugins");
    let plugins = PluginManager::load_from_dir(&plugin_root, new_config.clone())?;
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
    })
}

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

fn trim_for_context(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    chars.into_iter().take(max_chars).collect::<String>() + "...(truncated)"
}

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

fn looks_like_http_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

fn looks_like_local_path(value: &str) -> bool {
    value.starts_with('/') || value.contains(":\\")
}

fn normalize_image_ref(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace("&amp;", "&")
        .replace("&#38;", "&")
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
