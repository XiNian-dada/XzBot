//! 应用运行时入口：负责 WebSocket 服务、HTTP 推送接口和消息处理主流程。
//!
//! 这个模块承接了二进制启动后的大部分运行时职责，核心包括：
//! 1. 加载配置、初始化 LLM、会话存储、插件管理器等运行组件。
//! 2. 暴露 OneBot 反向 WebSocket 接口与外部 POST 推送接口。
//! 3. 在收到事件后完成权限检查、上下文富化、AI/插件分发和回复下发。
//!
//! 由于这里同时连接了网络层、协议层和业务层，代码天然会偏重编排。
//! 因此注释重点不是解释语法，而是说明各个步骤为什么在这里做、依赖谁做。

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
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::time::timeout;
use tokio::{fs, net::TcpListener};

use crate::{
    bot::{ai_chat::AiChatPlugin, router::BotRouter},
    config::{AiProvider, Config, NetworkConfig},
    llm::{
        anthropic_compatible::AnthropicCompatibleLlm, mock::MockLlm,
        openai_compatible::OpenAiCompatibleLlm, FallbackLlm, Llm,
    },
    logger::{
        debug as log_debug, error as log_error, error_err as log_error_err, info as log_info,
        warn as log_warn, warn_err as log_warn_err,
    },
    onebot::action::ActionRequest,
    onebot::event::{extract_cq_image_refs, MessageEvent, MessagePayload},
    plugins::PluginManager,
    post_api::{chat_target_desc, ChatTokenStore},
    store::memory::MemoryStore,
    tools::http::build_client,
};

mod admin;
mod enrich;

use admin::*;
use enrich::*;

// ===== 运行时共享状态与协议桥接 =====

/// 插入到外部 POST 推送消息里的隐藏标记。
///
/// 这样 OneBot 回流这条消息时，路由层可以识别出它是“系统代发”，而不是用户真实输入。
const POST_CONTEXT_MARKER: &str = "\u{2063}\u{2064}\u{2063}\u{2064}";

/// axum 路由共享状态。
#[derive(Clone)]
struct AppState {
    runtime: Arc<RwLock<RuntimeState>>,
    config_path: Arc<PathBuf>,
    store: Arc<MemoryStore>,
    post_tokens: Arc<ChatTokenStore>,
    ws_action_tx: Arc<RwLock<Option<mpsc::UnboundedSender<String>>>>,
}

/// 可被 `/reload` 原子替换的运行时对象。
struct RuntimeState {
    router: Arc<BotRouter>,
    config: Arc<Config>,
}

/// OneBot 动作桥接器。
///
/// 负责两件事：
/// 1. 给每次动作调用分配唯一 `echo`
/// 2. 在响应回流时把结果派发给对应等待者
#[derive(Clone)]
struct WsActionBridge {
    action_tx: mpsc::UnboundedSender<String>,
    pending: Arc<DashMap<String, oneshot::Sender<Value>>>,
    seq: Arc<AtomicU64>,
    debug: bool,
}

impl WsActionBridge {
    /// 基于当前 WebSocket 出站发送器创建动作桥接器。
    fn new(action_tx: mpsc::UnboundedSender<String>, debug: bool) -> Self {
        Self {
            action_tx,
            pending: Arc::new(DashMap::new()),
            seq: Arc::new(AtomicU64::new(1)),
            debug,
        }
    }

    /// 把回流的动作响应按 `echo` 分发给对应等待中的调用方。
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

    /// 发送一个 OneBot Action，并异步等待带相同 `echo` 的响应。
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

// ===== 启动与运行时构建 =====

/// 启动完整应用运行时：加载配置、构建运行组件并启动 axum 服务。
pub async fn run() -> anyhow::Result<()> {
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

/// 根据配置构建完整运行时组件。
///
/// 这里把 LLM 后端、AI 插件、业务路由和插件管理器组装成一个可替换的运行时快照。
fn build_runtime(
    config: Arc<Config>,
    store: Arc<MemoryStore>,
    plugins: PluginManager,
) -> anyhow::Result<RuntimeState> {
    let llm = build_llm_chain(&config)?;
    let ai_chat = AiChatPlugin::new(store, llm, config.clone());
    let router = Arc::new(BotRouter::new(ai_chat, plugins, config.clone()));

    Ok(RuntimeState { router, config })
}

/// 根据配置构建一条 LLM 回退链。
///
/// 逻辑是：
/// - `ai.model` 作为主模型
/// - `ai.fallback_models` 依次作为候补
/// - 所有候选共享同一套 provider/base_url/api_key 等参数
fn build_llm_chain(config: &Arc<Config>) -> anyhow::Result<Arc<dyn Llm>> {
    let mut candidates = Vec::new();

    for model in config.ai.model_chain() {
        let mut ai_config = config.ai.clone();
        ai_config.model = model.clone();

        let llm: Arc<dyn Llm> = match ai_config.provider {
            AiProvider::Mock => Arc::new(MockLlm::new(
                config.debug,
                ai_config.timeout_ms,
                config.search.clone(),
                config.network.clone(),
            )?),
            AiProvider::OpenaiCompatible => Arc::new(OpenAiCompatibleLlm::from_config(
                &ai_config,
                &config.search,
                &config.network,
                config.debug,
                false,
            )?),
            AiProvider::AnthropicCompatible => Arc::new(AnthropicCompatibleLlm::from_config(
                &ai_config,
                &config.search,
                &config.network,
                config.debug,
            )?),
        };

        candidates.push((model, llm));
    }

    Ok(Arc::new(FallbackLlm::new(candidates, config.debug)))
}

/// 启动时检查代理是否可用。
///
/// 代理一旦不可用，很多下游能力（LLM、搜索、抓取）都会一起失败，所以直接在启动阶段拦截。
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

// ===== HTTP / WebSocket 入口 =====

/// 处理 OneBot 反向 WebSocket 升级请求，并按配置决定是否校验 token。
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

/// 外部 POST 推送接口的请求体。
///
/// 设计成“一个接口同时支持文本、图片、文件”，是为了让外部系统接起来简单：
/// - 普通 webhook 只传 `message`
/// - 图床 / 监控告警可以传 `image` / `images`
/// - 报告导出场景可以传 `file_path`
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

/// 外部 POST 推送接口的响应体。
#[derive(Debug, serde::Serialize)]
struct PostSendResponse {
    ok: bool,
    sent_actions: usize,
    chat_type: String,
    user_id: i64,
    group_id: Option<i64>,
}

/// 通过绑定 token 向对应聊天发送文本、图片或文件。
///
/// 这里故意不触碰 AI/插件逻辑，只做机械转发：
/// 1. 校验 token
/// 2. 找到目标会话
/// 3. 把请求体转换成 OneBot action
/// 4. 直接推给当前 WebSocket 会话
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

    // 先用 token 反查目标聊天。这样调用方只知道 token，不需要知道群号/QQ 号。
    let Some(target) = state.post_tokens.lookup_token(&token).await else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "ok": false, "error": "invalid token" })),
        )
            .into_response();
    };

    // 当前设计里 OneBot 动作只能通过活跃的反向 WS 连接发出，因此这里必须确认连接存在。
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
    // 文件发送和文本/图片发送分开建 action，原因是 OneBot 的文件上传是独立动作。
    if let Some(file_path) = req
        .file_path
        .as_ref()
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
    {
        let action = if target.chat_type == "group" {
            ActionRequest::upload_group_file(
                target.group_id.unwrap_or_default(),
                file_path.to_string(),
                req.file_name.clone(),
            )
        } else {
            ActionRequest::upload_private_file(
                target.user_id,
                file_path.to_string(),
                req.file_name.clone(),
            )
        };
        actions.push(action);
    }

    // `image` 和 `images` 两个字段并存，是为了兼容更简单的 webhook 模板写法。
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
        // 给文本类代发消息打隐藏标记，避免这条消息回流后被当作用户输入写进 AI 上下文。
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

    // 顺序发送 action，确保“先传文件再发说明文字”这类场景能按调用方预期执行。
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
        Json(
            serde_json::to_value(PostSendResponse {
                ok: true,
                sent_actions: actions.len(),
                chat_type: target.chat_type,
                user_id: target.user_id,
                group_id: target.group_id,
            })
            .unwrap_or_else(|_| json!({ "ok": true, "sent_actions": actions.len() })),
        ),
    )
        .into_response()
}

// ===== WebSocket 会话与事件处理 =====

/// One websocket connection lifecycle (reader loop + writer task).
///
/// 一个连接拆成两部分：
/// - writer task：专门负责把内部 action 发回 NapCat
/// - reader loop：持续接收事件/动作回执
///
/// 这样可以避免“边读边写同一个 socket”导致控制流混乱。
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
///
/// 这里先尝试把消息识别成“动作响应”，如果不是，再当事件处理。
/// 事件处理会 `spawn` 到后台，避免某一条慢请求卡住整个 WS 读循环。
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

    // 优先把动作回执分发给等待中的调用方，避免这些回执继续走消息事件逻辑。
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
///
/// 这是消息事件进入业务层前的最后一道编排逻辑，顺序不能乱：
/// 1. 只放行 `post_type=message`
/// 2. 跳过内部 POST 回流消息
/// 3. 富化图片/引用上下文
/// 4. 处理高优先级管理指令
/// 5. 最后才交给 BotRouter
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

    // 每次都从 runtime 快照读取 router/config，确保 `/reload` 后新配置能立刻生效。
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

    // 富化失败不拦截消息主流程，只记录警告。否则图片链路偶发失败会让整个机器人失语。
    if let Err(err) = enrich_event_images(&mut event, bridge, config.debug).await {
        log_warn_err("failed to enrich image context", &err);
    }

    log_debug(
        config.debug,
        format!(
            "event message_type={} user_id={} group_id={:?}",
            event.message_type, event.user_id, event.group_id
        ),
    );

    // 管理指令优先级高于普通 AI/插件路由，避免 owner 指令被错误吞进会话上下文。
    if let Some(action) = try_reload_config_command(state, &event, &config).await {
        return serde_json::to_string(&action).ok().into_iter().collect();
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
            log_error_err("route message failed", &err);
            return Vec::new();
        }
    };

    actions
        .into_iter()
        .filter_map(|v| serde_json::to_string(&v).ok())
        .collect()
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
