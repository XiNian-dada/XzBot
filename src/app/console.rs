//! Web 控制面板：提供带鉴权的配置编辑、日志查看、插件概览与热重载。
//!
//! 这个面板刻意做成“运行时薄壳”：
//! - 配置仍然以 TOML 文件为准，不额外引入第二套配置存储
//! - 页面只负责展示、编辑和触发 reload
//! - 鉴权使用单个共享 token，适合自托管场景快速运维

use super::*;

use axum::{
    http::header::{AUTHORIZATION, COOKIE, SET_COOKIE},
    response::Html,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

const ADMIN_COOKIE_NAME: &str = "xzbot_admin";
const ADMIN_COOKIE_MAX_AGE: u64 = 7 * 24 * 60 * 60;
const DEFAULT_LOG_LINES: usize = 300;
const MAX_LOG_LINES: usize = 2000;
const PLUGIN_FILE_MAX_BYTES: u64 = 512 * 1024;

#[derive(Debug, Deserialize)]
pub(super) struct AdminLoginRequest {
    token: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct AdminSaveFileRequest {
    file_id: String,
    content: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct AdminLogQuery {
    lines: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AdminPluginFileQuery {
    file_id: String,
}

#[derive(Debug, Serialize)]
struct AdminBootstrapResponse {
    title: String,
    provider: String,
    model: String,
    wire_api: String,
    ws_connected: bool,
    config_path: String,
    config_files: Vec<AdminConfigFile>,
    plugins: Vec<crate::plugins::PluginSummary>,
    plugin_files: Vec<AdminPluginFileEntry>,
}

#[derive(Debug, Serialize)]
struct AdminConfigFile {
    file_id: String,
    label: String,
    path: String,
    content: String,
    kind: String,
}

#[derive(Debug, Serialize)]
struct AdminPluginFileEntry {
    file_id: String,
    plugin_name: String,
    label: String,
    path: String,
    size_bytes: u64,
}

#[derive(Debug, Serialize)]
struct AdminPluginFileContentResponse {
    file_id: String,
    path: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct AdminLogResponse {
    lines: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AdminReloadResponse {
    ok: bool,
    provider: String,
    model: String,
    server_rebind_required: bool,
    plugin_names: Vec<String>,
    plugin_tool_names: Vec<String>,
}

/// 返回控制面板 HTML 壳。实际数据由前端再调用 `/api/admin/*` 拉取。
pub(super) async fn admin_console_page(State(state): State<AppState>) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if !config.web_admin.enabled {
        return (StatusCode::NOT_FOUND, "admin console disabled").into_response();
    }

    Html(render_console_html(&config.web_admin.title)).into_response()
}

/// 处理控制面板登录，验证通过后写入 HttpOnly cookie。
pub(super) async fn admin_login_handler(
    State(state): State<AppState>,
    Json(req): Json<AdminLoginRequest>,
) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if !config.web_admin.enabled {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": "admin disabled" })),
        )
            .into_response();
    }

    if req.token.trim() != config.web_admin.token.trim() {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "ok": false, "error": "invalid token" })),
        )
            .into_response();
    }

    let cookie = format!(
        "{name}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={age}",
        name = ADMIN_COOKIE_NAME,
        value = config.web_admin.token.trim(),
        age = ADMIN_COOKIE_MAX_AGE
    );
    (
        StatusCode::OK,
        [(SET_COOKIE, cookie)],
        Json(json!({ "ok": true })),
    )
        .into_response()
}

/// 清除控制面板登录 cookie。
pub(super) async fn admin_logout_handler(State(state): State<AppState>) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if !config.web_admin.enabled {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": "admin disabled" })),
        )
            .into_response();
    }

    let cookie = format!(
        "{name}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0",
        name = ADMIN_COOKIE_NAME
    );
    (
        StatusCode::OK,
        [(SET_COOKIE, cookie)],
        Json(json!({ "ok": true })),
    )
        .into_response()
}

/// 返回控制面板启动所需的全部基础数据。
pub(super) async fn admin_bootstrap_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if let Some(resp) = require_admin_auth(&config, &headers) {
        return resp;
    }

    let (provider, model, wire_api, plugins) = {
        let runtime = state.runtime.read().await;
        (
            runtime.config.ai.provider.as_str().to_string(),
            runtime.config.ai.model.clone(),
            format!("{:?}", runtime.config.ai.wire_api).to_lowercase(),
            runtime.router.plugin_summaries(),
        )
    };
    let ws_connected = state.ws_action_tx.read().await.is_some();

    match (
        read_core_config_files(&state).await,
        read_plugin_file_entries(&state).await,
    ) {
        (Ok(config_files), Ok(plugin_files)) => (
            StatusCode::OK,
            Json(AdminBootstrapResponse {
                title: config.web_admin.title.clone(),
                provider,
                model,
                wire_api,
                ws_connected,
                config_path: state.config_path.display().to_string(),
                config_files,
                plugins,
                plugin_files,
            }),
        )
            .into_response(),
        (Err(err), _) | (_, Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": err.to_string() })),
        )
            .into_response(),
    }
}

/// 返回单个插件文件的文本内容，供文件管理器按需加载。
pub(super) async fn admin_plugin_file_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AdminPluginFileQuery>,
) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if let Some(resp) = require_admin_auth(&config, &headers) {
        return resp;
    }

    let Some(path) = resolve_plugin_editable_path(&state, query.file_id.trim()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": "unknown plugin file" })),
        )
            .into_response();
    };

    match fs::read(&path).await {
        Ok(bytes) => {
            if !bytes_look_like_text(&bytes) {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "ok": false, "error": "plugin file is not text-like" })),
                )
                    .into_response();
            }
            (
                StatusCode::OK,
                Json(AdminPluginFileContentResponse {
                    file_id: query.file_id,
                    path: path.display().to_string(),
                    content: String::from_utf8_lossy(&bytes).to_string(),
                }),
            )
                .into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": format!("failed to read plugin file: {err}") })),
        )
            .into_response(),
    }
}

/// 保存单个配置文件。这里先做 TOML 语法检查，再真正写盘。
pub(super) async fn admin_save_config_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AdminSaveFileRequest>,
) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if let Some(resp) = require_admin_auth(&config, &headers) {
        return resp;
    }

    let file_id = req.file_id.trim();
    let path = resolve_editable_config_path(&state, file_id)
        .or_else(|| resolve_plugin_editable_path(&state, file_id));
    let Some(path) = path else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": "unknown editable file" })),
        )
            .into_response();
    };

    if file_id == "config.toml" && req.content.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": "config.toml cannot be empty" })),
        )
            .into_response();
    }

    if is_core_config_file_id(file_id) && !req.content.trim().is_empty() {
        if let Err(err) = req.content.parse::<toml::Value>() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "ok": false, "error": format!("TOML parse error: {err}") })),
            )
                .into_response();
        }
    }

    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    json!({ "ok": false, "error": format!("failed to create config dir: {err}") }),
                ),
            )
                .into_response();
        }
    }
    if let Err(err) = fs::write(&path, req.content).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": format!("failed to save file: {err}") })),
        )
            .into_response();
    }

    (
        StatusCode::OK,
        Json(json!({ "ok": true, "path": path.display().to_string(), "file_id": file_id })),
    )
        .into_response()
}

/// 触发一次运行时热重载。
pub(super) async fn admin_reload_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if let Some(resp) = require_admin_auth(&config, &headers) {
        return resp;
    }

    match reload_runtime(&state).await {
        Ok(outcome) => (
            StatusCode::OK,
            Json(AdminReloadResponse {
                ok: true,
                provider: outcome.config.ai.provider.as_str().to_string(),
                model: outcome.config.ai.model.clone(),
                server_rebind_required: outcome.server_rebind_required,
                plugin_names: outcome.plugin_names,
                plugin_tool_names: outcome.plugin_tool_names,
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "ok": false, "error": err.to_string() })),
        )
            .into_response(),
    }
}

/// 返回内存日志尾部。
pub(super) async fn admin_logs_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AdminLogQuery>,
) -> impl IntoResponse {
    let config = {
        let runtime = state.runtime.read().await;
        runtime.config.clone()
    };
    if let Some(resp) = require_admin_auth(&config, &headers) {
        return resp;
    }

    let lines = crate::logger::recent_lines(
        query
            .lines
            .unwrap_or(DEFAULT_LOG_LINES)
            .clamp(50, MAX_LOG_LINES),
    );
    (StatusCode::OK, Json(AdminLogResponse { lines })).into_response()
}

fn require_admin_auth(config: &Config, headers: &HeaderMap) -> Option<axum::response::Response> {
    if !config.web_admin.enabled {
        return Some(
            (
                StatusCode::NOT_FOUND,
                Json(json!({ "ok": false, "error": "admin disabled" })),
            )
                .into_response(),
        );
    }

    if admin_token_from_headers(headers)
        .map(|token| token == config.web_admin.token.trim())
        .unwrap_or(false)
    {
        return None;
    }

    Some(
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "ok": false, "error": "unauthorized" })),
        )
            .into_response(),
    )
}

fn admin_token_from_headers(headers: &HeaderMap) -> Option<String> {
    if let Some(auth) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }

    let cookie_header = headers.get(COOKIE).and_then(|value| value.to_str().ok())?;
    for part in cookie_header.split(';') {
        let part = part.trim();
        let Some(value) = part.strip_prefix(&format!("{ADMIN_COOKIE_NAME}=")) else {
            continue;
        };
        if !value.trim().is_empty() {
            return Some(value.trim().to_string());
        }
    }
    None
}

async fn read_core_config_files(state: &AppState) -> anyhow::Result<Vec<AdminConfigFile>> {
    let mut files = Vec::new();
    for file_name in crate::config::known_config_file_names() {
        let Some(path) = resolve_editable_config_path(state, file_name) else {
            continue;
        };
        let content = match fs::read_to_string(&path).await {
            Ok(body) => body,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(err) => {
                return Err(anyhow!(
                    "failed to read config file {}: {}",
                    path.display(),
                    err
                ));
            }
        };
        files.push(AdminConfigFile {
            file_id: file_name.to_string(),
            label: file_name.to_string(),
            path: path.display().to_string(),
            content,
            kind: "core".to_string(),
        });
    }
    Ok(files)
}

fn resolve_editable_config_path(state: &AppState, file_name: &str) -> Option<PathBuf> {
    if file_name == "config.toml" {
        return Some(state.config_path.as_ref().clone());
    }

    if crate::config::known_config_file_names()
        .iter()
        .any(|candidate| *candidate == file_name)
    {
        return state.config_path.parent().map(|dir| dir.join(file_name));
    }

    None
}

async fn read_plugin_file_entries(_state: &AppState) -> anyhow::Result<Vec<AdminPluginFileEntry>> {
    let plugin_root = std::env::current_dir()?.join("Plugins");
    if !plugin_root.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    let mut dirs = vec![plugin_root.clone()];
    while let Some(current_dir) = dirs.pop() {
        let mut entries = fs::read_dir(&current_dir).await.with_context(|| {
            format!("failed to read plugin config dir {}", current_dir.display())
        })?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let meta = entry.metadata().await?;

            if meta.is_dir() {
                dirs.push(path);
                continue;
            }
            if !meta.is_file() || meta.len() > PLUGIN_FILE_MAX_BYTES {
                continue;
            }
            if !looks_like_editable_plugin_file(&path) {
                continue;
            }

            let bytes = match fs::read(&path).await {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            if !bytes_look_like_text(&bytes) {
                continue;
            }

            let relative = match path.strip_prefix(&plugin_root) {
                Ok(relative) => relative,
                Err(_) => continue,
            };
            let file_id = format!("plugin:{}", relative.to_string_lossy().replace('\\', "/"));
            let label = format!("Plugins/{}", relative.to_string_lossy().replace('\\', "/"));
            let plugin_name = relative
                .components()
                .next()
                .map(|part| part.as_os_str().to_string_lossy().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            files.push(AdminPluginFileEntry {
                file_id,
                plugin_name,
                label,
                path: path.display().to_string(),
                size_bytes: meta.len(),
            });
        }
    }
    files.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(files)
}

fn resolve_plugin_editable_path(_state: &AppState, file_id: &str) -> Option<PathBuf> {
    let relative = file_id.strip_prefix("plugin:")?;
    if relative.is_empty() {
        return None;
    }

    let plugin_root = std::env::current_dir().ok()?.join("Plugins");
    let candidate = plugin_root.join(relative);
    if !candidate.starts_with(&plugin_root) {
        return None;
    }
    Some(candidate)
}

fn is_core_config_file_id(file_id: &str) -> bool {
    file_id == "config.toml"
        || crate::config::known_config_file_names()
            .iter()
            .any(|candidate| *candidate == file_id)
}

fn looks_like_editable_plugin_file(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|value| value.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "toml"
            | "json"
            | "yaml"
            | "yml"
            | "txt"
            | "md"
            | "ini"
            | "conf"
            | "cfg"
            | "env"
            | "log"
            | "xml"
    )
}

fn bytes_look_like_text(bytes: &[u8]) -> bool {
    let sample = &bytes[..bytes.len().min(4096)];
    !sample.iter().any(|b| *b == 0)
}

fn render_console_html(title: &str) -> String {
    ADMIN_CONSOLE_HTML.replace("__TITLE__", &escape_html(title))
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

const ADMIN_CONSOLE_HTML: &str = r#"<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root{
      --bg:#0d1117;
      --panel:#111827;
      --panel-soft:#172033;
      --line:#243249;
      --text:#e7edf7;
      --muted:#9fb0c8;
      --accent:#64d2ff;
      --accent-2:#7af0b3;
      --danger:#ff7f8a;
      --warning:#ffc36d;
      --shadow:0 20px 80px rgba(0,0,0,.35);
      --radius:18px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      color:var(--text);
      background:
        radial-gradient(circle at top left, rgba(100,210,255,.15), transparent 28%),
        radial-gradient(circle at top right, rgba(122,240,179,.12), transparent 22%),
        linear-gradient(180deg, #0b1017, #0d1117 40%, #0b0f15);
      font:14px/1.6 "SF Pro Display","PingFang SC","Noto Sans SC",sans-serif;
      min-height:100vh;
    }
    .shell{
      width:min(1400px, calc(100vw - 32px));
      margin:24px auto;
      display:grid;
      grid-template-columns:280px 1fr;
      gap:18px;
    }
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
      border:1px solid var(--line);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
      backdrop-filter:blur(12px);
    }
    .sidebar{padding:22px; position:sticky; top:24px; height:fit-content}
    .brand{font-size:26px; font-weight:800; letter-spacing:.02em}
    .sub{margin-top:8px; color:var(--muted)}
    .nav{margin-top:24px; display:grid; gap:10px}
    .nav button{
      all:unset; cursor:pointer; padding:12px 14px; border-radius:14px;
      border:1px solid transparent; color:var(--muted); transition:.18s ease;
      background:rgba(255,255,255,.02)
    }
    .nav button.active,.nav button:hover{
      color:var(--text); border-color:rgba(100,210,255,.32);
      background:linear-gradient(180deg, rgba(100,210,255,.12), rgba(100,210,255,.04));
    }
    .main{display:grid; gap:18px}
    .topbar{padding:20px 24px; display:flex; align-items:center; justify-content:space-between; gap:16px}
    .topbar h1{margin:0; font-size:24px}
    .actions{display:flex; gap:10px; flex-wrap:wrap}
    button.primary,button.secondary,button.warn{
      border:0; cursor:pointer; border-radius:12px; padding:10px 14px; font-weight:700;
    }
    button.primary{background:linear-gradient(135deg, var(--accent), #4c90ff); color:#07111f}
    button.secondary{background:#1a2435; color:var(--text); border:1px solid var(--line)}
    button.warn{background:linear-gradient(135deg, #ffb86c, #ff7f8a); color:#2a0c12}
    .grid{display:grid; gap:18px; grid-template-columns:repeat(12,1fr)}
    .panel{padding:20px 22px}
    .span-4{grid-column:span 4}
    .span-6{grid-column:span 6}
    .span-8{grid-column:span 8}
    .span-12{grid-column:span 12}
    .label{font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
    .value{margin-top:8px; font-size:22px; font-weight:800}
    .tiny{font-size:12px; color:var(--muted)}
    .section{display:none}
    .section.active{display:block}
    .editor-shell{display:grid; grid-template-columns:minmax(260px, 340px) minmax(0, 1fr); gap:16px; align-items:start}
    .file-list{display:grid; gap:8px; min-width:0}
    .file-item{
      border:1px solid var(--line); padding:12px 14px; border-radius:12px;
      cursor:pointer; color:var(--muted); background:#121a28; text-align:left; min-width:0;
      display:block;
    }
    .file-item.active{color:var(--text); border-color:rgba(122,240,179,.36)}
    .file-item strong{display:block; overflow-wrap:anywhere; word-break:break-word; white-space:normal}
    .file-item .tiny{margin-top:4px; overflow-wrap:anywhere; word-break:break-word; white-space:normal}
    .stack{display:grid; gap:18px}
    textarea, pre{
      width:100%; background:#0b1017; color:#dce8f8; border:1px solid var(--line);
      border-radius:14px; padding:14px; font:13px/1.55 "SFMono-Regular","JetBrains Mono",monospace;
    }
    textarea{min-height:560px; resize:vertical}
    pre{max-height:560px; overflow:auto; white-space:pre-wrap; word-break:break-word}
    table{width:100%; border-collapse:collapse}
    th,td{padding:10px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top}
    th{color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em}
    .badge{display:inline-flex; align-items:center; gap:6px; padding:5px 10px; border-radius:999px; font-size:12px; font-weight:700}
    .ok{background:rgba(122,240,179,.14); color:var(--accent-2)}
    .off{background:rgba(255,255,255,.08); color:var(--muted)}
    .status{display:flex; align-items:center; gap:8px}
    .dot{width:10px; height:10px; border-radius:50%}
    .dot.ok{background:var(--accent-2)}
    .dot.off{background:var(--danger)}
    .toast{
      position:fixed; right:18px; bottom:18px; min-width:240px; max-width:420px;
      background:#131d2b; border:1px solid var(--line); color:var(--text);
      border-radius:14px; padding:14px 16px; box-shadow:var(--shadow); display:none;
    }
    .toast.show{display:block}
    .log-tools{display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px}
    .login{
      max-width:420px; margin:12vh auto 0; padding:28px;
    }
    .login input{
      width:100%; margin-top:14px; border-radius:12px; border:1px solid var(--line);
      background:#0b1017; color:var(--text); padding:12px 14px; font-size:14px;
    }
    .hidden{display:none!important}
    @media (max-width: 1024px){
      .shell{grid-template-columns:1fr}
      .sidebar{position:static}
      .editor-shell{grid-template-columns:1fr}
      .grid{grid-template-columns:1fr}
      .span-4,.span-6,.span-8,.span-12{grid-column:auto}
    }
  </style>
</head>
<body>
  <div id="login" class="card login">
    <div class="brand">__TITLE__</div>
    <div class="sub">登录后就能直接改配置、看日志、看插件并触发热重载。</div>
    <input id="tokenInput" type="password" placeholder="输入控制面板 token" />
    <div class="actions" style="margin-top:16px">
      <button class="primary" id="loginBtn">登录</button>
    </div>
  </div>

  <div id="app" class="shell hidden">
    <aside class="card sidebar">
      <div class="brand">__TITLE__</div>
      <div class="sub" id="subTitle">正在读取运行时状态…</div>
      <div class="nav">
        <button class="active" data-section="overview">总览</button>
        <button data-section="config">配置</button>
        <button data-section="logs">日志</button>
        <button data-section="plugins">插件</button>
      </div>
    </aside>

    <main class="main">
      <div class="card topbar">
        <h1>控制面板</h1>
        <div class="actions">
          <button class="secondary" id="refreshBtn">刷新</button>
          <button class="primary" id="reloadBtn">保存后重载</button>
          <button class="warn" id="logoutBtn">退出登录</button>
        </div>
      </div>

      <section id="section-overview" class="section active">
        <div class="grid">
          <div class="card panel span-4">
            <div class="label">Provider</div>
            <div class="value" id="providerValue">-</div>
          </div>
          <div class="card panel span-4">
            <div class="label">Model</div>
            <div class="value" id="modelValue">-</div>
          </div>
          <div class="card panel span-4">
            <div class="label">OneBot WS</div>
            <div class="value status"><span id="wsDot" class="dot off"></span><span id="wsText">离线</span></div>
          </div>
          <div class="card panel span-8">
            <div class="label">Config Path</div>
            <div class="value" style="font-size:16px" id="configPathValue">-</div>
          </div>
          <div class="card panel span-4">
            <div class="label">插件数</div>
            <div class="value" id="pluginCountValue">0</div>
          </div>
        </div>
      </section>

      <section id="section-config" class="section">
        <div class="card panel">
          <div class="editor-shell">
            <div class="file-list" id="fileList"></div>
            <div>
              <div class="actions" style="margin-bottom:12px">
                <button class="primary" id="saveFileBtn">保存当前文件</button>
                <span class="tiny" id="editingPath">-</span>
              </div>
              <textarea id="configEditor" spellcheck="false"></textarea>
            </div>
          </div>
        </div>
      </section>

      <section id="section-logs" class="section">
        <div class="card panel">
          <div class="log-tools">
            <button class="secondary" id="refreshLogsBtn">刷新日志</button>
            <span class="tiny">日志页自动每 3 秒刷新一次</span>
          </div>
          <pre id="logsView">正在加载日志…</pre>
        </div>
      </section>

      <section id="section-plugins" class="section">
        <div class="stack">
          <div class="card panel">
            <table>
              <thead>
                <tr>
                  <th>插件</th>
                  <th>命令</th>
                  <th>事件</th>
                  <th>工具</th>
                  <th>优先级</th>
                  <th>超时</th>
                </tr>
              </thead>
              <tbody id="pluginsTable"></tbody>
            </table>
          </div>

          <div class="card panel">
            <div class="label" style="margin-bottom:12px">插件文件管理器</div>
            <div class="editor-shell">
              <div class="file-list" id="pluginFileList"></div>
              <div>
                <div class="actions" style="margin-bottom:12px">
                  <button class="secondary" id="loadPluginFileBtn">重新加载当前文件</button>
                  <button class="primary" id="savePluginFileBtn">保存当前插件文件</button>
                  <span class="tiny" id="pluginEditingPath">-</span>
                </div>
                <textarea id="pluginEditor" spellcheck="false" placeholder="从左侧选择一个插件文本文件后，就会在这里显示内容。"></textarea>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <div id="toast" class="toast"></div>

  <script>
    const state = {
      files: [],
      activeFile: null,
      pluginFiles: [],
      activePluginFile: null,
      activePluginFilePath: "",
      activePluginContent: "",
      logsTimer: null
    };

    const el = (id) => document.getElementById(id);
    const toast = (message) => {
      const node = el("toast");
      node.textContent = message;
      node.classList.add("show");
      clearTimeout(window.__toastTimer);
      window.__toastTimer = setTimeout(() => node.classList.remove("show"), 2800);
    };

    async function api(path, options = {}) {
      const resp = await fetch(path, {
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        ...options,
      });
      const text = await resp.text();
      let body = {};
      try { body = text ? JSON.parse(text) : {}; } catch { body = { raw: text }; }
      if (!resp.ok) {
        const error = body.error || body.raw || `${resp.status} ${resp.statusText}`;
        throw new Error(error);
      }
      return body;
    }

    async function login() {
      try {
        await api("/api/admin/login", {
          method: "POST",
          body: JSON.stringify({ token: el("tokenInput").value }),
        });
        el("login").classList.add("hidden");
        el("app").classList.remove("hidden");
        await bootstrap();
        toast("登录成功");
      } catch (err) {
        toast(`登录失败：${err.message}`);
      }
    }

    function renderFiles() {
      const list = el("fileList");
      list.innerHTML = "";
      state.files.forEach((file) => {
        const item = document.createElement("button");
        item.className = `file-item ${state.activeFile === file.file_id ? "active" : ""}`;
        const title = document.createElement("strong");
        title.textContent = file.label;
        const meta = document.createElement("div");
        meta.className = "tiny";
        meta.textContent = "核心配置";
        item.appendChild(title);
        item.appendChild(meta);
        item.onclick = () => {
          state.activeFile = file.file_id;
          el("configEditor").value = file.content;
          el("editingPath").textContent = file.path;
          renderFiles();
        };
        list.appendChild(item);
      });
      if (!state.activeFile && state.files.length) {
        state.activeFile = state.files[0].file_id;
        el("configEditor").value = state.files[0].content;
        el("editingPath").textContent = state.files[0].path;
        renderFiles();
      }
    }

    function formatBytes(value) {
      if (!Number.isFinite(value) || value < 1024) return `${value || 0} B`;
      if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
      return `${(value / (1024 * 1024)).toFixed(1)} MB`;
    }

    function renderPluginFiles() {
      const list = el("pluginFileList");
      list.innerHTML = "";
      if (!state.pluginFiles.length) {
        list.innerHTML = `<div class="tiny">当前没有可编辑的插件文本文件。</div>`;
        return;
      }

      state.pluginFiles.forEach((file) => {
        const item = document.createElement("button");
        item.className = `file-item ${state.activePluginFile === file.file_id ? "active" : ""}`;
        const title = document.createElement("strong");
        title.textContent = file.label;
        const meta = document.createElement("div");
        meta.className = "tiny";
        meta.textContent = `${file.plugin_name} · ${formatBytes(file.size_bytes)}`;
        item.appendChild(title);
        item.appendChild(meta);
        item.onclick = () => loadPluginFile(file.file_id);
        list.appendChild(item);
      });
    }

    function renderPlugins(plugins) {
      const tbody = el("pluginsTable");
      tbody.innerHTML = "";
      if (!plugins.length) {
        tbody.innerHTML = `<tr><td colspan="6" class="tiny">当前没有已加载插件。</td></tr>`;
        return;
      }
      plugins.forEach((plugin) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${plugin.name}</strong><div class="tiny">${plugin.path}</div></td>
          <td>${(plugin.commands || []).join(", ") || "-"}</td>
          <td>${(plugin.subscriptions || []).join(", ") || "-"}</td>
          <td>${(plugin.tool_names || []).join(", ") || "-"}</td>
          <td>${plugin.priority}</td>
          <td>${plugin.timeout_ms}ms</td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function loadPluginFile(fileId, { silent = false } = {}) {
      if (!fileId) return;
      try {
        const data = await api(`/api/admin/plugin-file?file_id=${encodeURIComponent(fileId)}`);
        state.activePluginFile = data.file_id;
        state.activePluginFilePath = data.path;
        state.activePluginContent = data.content || "";
        el("pluginEditor").value = state.activePluginContent;
        el("pluginEditingPath").textContent = data.path || "-";
        renderPluginFiles();
      } catch (err) {
        if (!silent) toast(`读取插件文件失败：${err.message}`);
      }
    }

    async function bootstrap() {
      const data = await api("/api/admin/bootstrap");
      el("providerValue").textContent = data.provider;
      el("modelValue").textContent = data.model;
      el("configPathValue").textContent = data.config_path;
      el("pluginCountValue").textContent = data.plugins.length;
      el("subTitle").textContent = `${data.provider} / ${data.model} / ${data.wire_api}`;
      el("wsDot").className = `dot ${data.ws_connected ? "ok" : "off"}`;
      el("wsText").textContent = data.ws_connected ? "在线" : "离线";
      state.files = data.config_files || [];
      state.activeFile = null;
      state.pluginFiles = data.plugin_files || [];
      state.activePluginFile = null;
      state.activePluginFilePath = "";
      state.activePluginContent = "";
      el("pluginEditor").value = "";
      el("pluginEditingPath").textContent = "-";
      renderFiles();
      renderPlugins(data.plugins || []);
      renderPluginFiles();
      if (state.pluginFiles.length) {
        await loadPluginFile(state.pluginFiles[0].file_id, { silent: true });
      }
      await refreshLogs();
    }

    async function saveCurrentFile() {
      const file = state.files.find((item) => item.file_id === state.activeFile);
      if (!file) return;
      const content = el("configEditor").value;
      await api("/api/admin/config/save", {
        method: "POST",
        body: JSON.stringify({ file_id: file.file_id, content }),
      });
      file.content = content;
      toast(`已保存 ${file.label}`);
    }

    async function saveCurrentPluginFile() {
      const file = state.pluginFiles.find((item) => item.file_id === state.activePluginFile);
      if (!file) return;
      const content = el("pluginEditor").value;
      await api("/api/admin/config/save", {
        method: "POST",
        body: JSON.stringify({ file_id: file.file_id, content }),
      });
      state.activePluginContent = content;
      toast(`已保存 ${file.label}`);
    }

    async function reloadRuntime() {
      const activeSection = document.querySelector(".nav button.active")?.dataset.section;
      if (activeSection === "config") {
        await saveCurrentFile();
      } else if (activeSection === "plugins" && state.activePluginFile) {
        await saveCurrentPluginFile();
      }
      const result = await api("/api/admin/reload", { method: "POST", body: "{}" });
      toast(`已重载：${result.provider} / ${result.model}`);
      if (result.server_rebind_required) {
        toast("检测到监听配置变更，需重启进程后完全生效");
      }
      await bootstrap();
    }

    async function refreshLogs() {
      const data = await api("/api/admin/logs?lines=1200");
      el("logsView").textContent = (data.lines || []).join("\n") || "(no logs)";
    }

    function syncLogAutoRefresh() {
      if (state.logsTimer) {
        clearInterval(state.logsTimer);
        state.logsTimer = null;
      }
      const activeSection = document.querySelector(".nav button.active")?.dataset.section;
      if (activeSection !== "logs") return;
      state.logsTimer = setInterval(() => {
        refreshLogs().catch(() => {});
      }, 3000);
    }

    async function logout() {
      await api("/api/admin/logout", { method: "POST", body: "{}" });
      location.reload();
    }

    document.querySelectorAll(".nav button").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".nav button").forEach((node) => node.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.dataset.section;
        document.querySelectorAll(".section").forEach((node) => node.classList.remove("active"));
        el(`section-${target}`).classList.add("active");
        syncLogAutoRefresh();
      });
    });

    el("loginBtn").onclick = login;
    el("refreshBtn").onclick = bootstrap;
    el("reloadBtn").onclick = reloadRuntime;
    el("saveFileBtn").onclick = saveCurrentFile;
    el("loadPluginFileBtn").onclick = () => loadPluginFile(state.activePluginFile);
    el("savePluginFileBtn").onclick = saveCurrentPluginFile;
    el("refreshLogsBtn").onclick = refreshLogs;
    el("logoutBtn").onclick = logout;
    el("tokenInput").addEventListener("keydown", (event) => {
      if (event.key === "Enter") login();
    });

    (async () => {
      try {
        await bootstrap();
        el("login").classList.add("hidden");
        el("app").classList.remove("hidden");
        syncLogAutoRefresh();
      } catch {
        el("login").classList.remove("hidden");
        el("app").classList.add("hidden");
      }
    })();
  </script>
</body>
</html>
"#;
