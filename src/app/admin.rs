//! 管理指令子模块：处理 owner 级别的 `/reload`、`/log` 和 `/posttoken`。
//!
//! 这些功能都属于“运行时维护能力”，和普通消息对话不同，单独拆出来可以避免主流程继续膨胀。

use super::*;

/// 一次热重载完成后返回给主流程的结果。
///
/// 主模块需要根据这里的信息决定：
/// 1. 回复 owner 什么内容。
/// 2. 是否提示“监听地址变更需要重启”。
/// 3. 当前插件装载情况如何。
pub(super) struct ReloadOutcome {
    pub(super) config: Arc<Config>,
    pub(super) server_rebind_required: bool,
    pub(super) plugin_names: Vec<String>,
}

// ===== 管理指令：reload / log / posttoken =====

/// 处理 owner 发出的 `/reload` 指令。
///
/// 返回 `Some(ActionRequest)` 表示当前消息已经被这个管理指令消费，
/// 主流程不应再继续把它交给 AI 或插件。
pub(super) async fn try_reload_config_command(
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

/// 判断当前消息是否是 owner 发出的 `/reload`。
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

/// 根据消息上下文构造一条“热重载状态回复”。
///
/// 群聊里优先 `@` 原发送者，私聊则直接回复。
pub(super) fn reload_reply_action(event: &MessageEvent, message: String) -> ActionRequest {
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

/// 处理 owner `/log [N]` 指令。
///
/// 逻辑是从内存日志缓冲区取最近 N 行，落到临时文件，再通过 OneBot 上传给当前会话。
pub(super) async fn try_dump_log_command(
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

/// 解析 `/log [N]` 指令，返回需要导出的日志行数。
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

/// 处理 owner `/posttoken` 指令族。
///
/// 每个 token 都绑定到“当前会话”，因此：
/// - 私聊里执行，生成的是私聊 token
/// - 群聊里执行，生成的是群 token
pub(super) async fn try_post_token_command(
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
                        if removed {
                            "Token 已删除。"
                        } else {
                            "未找到 token。"
                        }
                    )
                } else {
                    format!(
                        "目标会话：private:{}\n{}",
                        user_id,
                        if removed {
                            "Token 已删除。"
                        } else {
                            "未找到 token。"
                        }
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

/// 解析 `/posttoken` 子命令，并顺带做 owner / `@机器人` 规则校验。
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

/// 在不重启进程的前提下重建运行时。
///
/// 这里会重新加载：
/// - `config.toml`
/// - 插件目录
/// - 基于新配置构建出的 router / LLM / AI 插件
///
/// 旧插件会在新运行时切换成功后主动关闭，避免出现“新旧插件并存”。
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

// ===== 事件上下文富化：图片与引用消息 =====
