//! OpenAI 兼容模型接入：负责主流程编排、图片处理、工具调用与重试。
//!
//! 这里保留“主干流程”，即：
//! 1. 把上层 `(role, content)` 对话转成请求消息。
//! 2. 根据模型能力决定走多模态、OCR 还是纯文本。
//! 3. 在 Chat Completions / Responses 两种线协议间选择合适的执行路径。
//! 4. 统一处理工具循环、错误重试、空回复兜底和截断续写。
//!
//! 解析细节与 Responses 子流程被拆到子模块，避免主文件继续膨胀。

use std::{collections::HashSet, time::Duration};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{
    config::{AiConfig, NetworkConfig, OpenAiWireApi, SearchConfig, VisionMode},
    llm::Llm,
    llm::{
        image::load_image_for_llm,
        message_parts::{parse_user_content, ParsedUserContent},
        ocr::{ocr_images_to_text, OcrSettings},
    },
    logger::warn_err as log_warn_err,
    token_stats,
    tools::system::{get_process_info, get_system_info},
    tools::weather::get_weather,
    tools::{
        http::build_client,
        web::{extract_urls, fetch_url, search_web},
    },
};

mod helpers;
mod responses;

use helpers::*;

/// OpenAI 兼容模型实现。
///
/// 这个结构体保存了调用外部模型服务所需的全部运行时依赖，包括：
/// - 复用的 HTTP 客户端
/// - 已解析好的接口地址与鉴权信息
/// - 搜索、OCR、代理等辅助子系统配置
/// - 当前模型的温度、最大 token 等生成参数
pub struct OpenAiCompatibleLlm {
    client: reqwest::Client,
    endpoint: String,
    base_url: String,
    wire_api: OpenAiWireApi,
    api_key: String,
    model: String,
    reasoning_effort: String,
    disable_response_storage: bool,
    temperature: f32,
    max_tokens: u32,
    timeout_ms: u64,
    debug: bool,
    suppress_transient_errors: bool,
    search: SearchConfig,
    network: NetworkConfig,
    vision_mode: VisionMode,
    ocr_settings: OcrSettings,
}

impl OpenAiCompatibleLlm {
    /// 当上游因为 token 上限截断回复时，尝试补一次“续写请求”。
    async fn maybe_continue_if_truncated(
        &self,
        session_id: String,
        prior_messages: &[Value],
        reply: String,
        finish_reason: Option<String>,
    ) -> Result<(String, bool)> {
        if finish_reason.as_deref() != Some("length") {
            return Ok((reply, false));
        }
        if reply.trim().is_empty() {
            return Ok((reply, false));
        }

        if self.debug {
            println!(
                "[DEBUG] finish_reason=length, attempting single continuation session={}",
                session_id
            );
        }

        let mut messages = prior_messages.to_vec();
        messages.push(json!({ "role": "assistant", "content": reply }));
        messages.push(json!({
            "role": "system",
            "content": "上一条回复被截断。请直接续写剩余内容，不要重复已输出部分，也不要重新开头。"
        }));
        normalize_system_messages(&mut messages);

        let payload = json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.answer_temperature(),
            "max_tokens": self.max_tokens,
            "user": session_id,
        });

        let value = match self.call_openai(payload).await {
            Ok(v) => v,
            Err(err) => {
                if self.debug {
                    println!("[DEBUG] continuation call failed: {err:#}");
                }
                return Ok((reply, false));
            }
        };

        let (continued, _, _, _) = parse_openai_choice(&value)?;
        let continued = continued.trim();
        if continued.is_empty() {
            return Ok((reply, false));
        }

        Ok((format!("{reply}\n{continued}"), true))
    }

    /// 根据全局配置构造 OpenAI 兼容模型客户端。
    pub fn from_config(
        config: &AiConfig,
        search: &SearchConfig,
        network: &NetworkConfig,
        debug: bool,
        suppress_transient_errors: bool,
    ) -> Result<Self> {
        let base = config.base_url.trim_end_matches('/').to_string();
        let endpoint = resolve_openai_endpoint(&base, config.wire_api);
        let client = build_client(config.timeout_ms, network, false)
            .context("failed to build HTTP client for OpenAI-compatible provider")?;

        Ok(Self {
            client,
            endpoint,
            base_url: config.base_url.clone(),
            wire_api: config.wire_api,
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            reasoning_effort: config.reasoning_effort.clone(),
            disable_response_storage: config.disable_response_storage,
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            timeout_ms: config.timeout_ms,
            debug,
            suppress_transient_errors,
            search: search.clone(),
            network: network.clone(),
            vision_mode: config.vision_mode,
            ocr_settings: OcrSettings {
                provider: config.ocr_provider,
                cmd: config.ocr_cmd.clone(),
                lang: config.ocr_lang.clone(),
                timeout_ms: config.ocr_timeout_ms,
                paddle_endpoint: config.paddle_ocr_endpoint.clone(),
                paddle_token: config.paddle_ocr_token.clone(),
                paddle_file_type: config.paddle_file_type,
                paddle_use_doc_orientation_classify: config.paddle_use_doc_orientation_classify,
                paddle_use_doc_unwarping: config.paddle_use_doc_unwarping,
                paddle_use_chart_recognition: config.paddle_use_chart_recognition,
                paddle_use_proxy: config.paddle_use_proxy,
            },
        })
    }
}

#[async_trait]
impl Llm for OpenAiCompatibleLlm {
    /// 执行一次完整对话回合。
    ///
    /// 流程是：
    /// 1. 构造模型请求消息
    /// 2. 根据接口形态选择 `chat_completions` 或 `responses`
    /// 3. 优先走工具调用分支，失败时回落到纯文本回答
    async fn chat(
        &self,
        session_id: String,
        messages: Vec<(String, String)>,
    ) -> anyhow::Result<String> {
        let mut request_messages = self.build_request_messages(&messages).await?;
        prepend_runtime_system_hint(&mut request_messages);
        prepend_current_turn_focus_hint(&messages, &mut request_messages);
        self.preload_current_turn_url_context(&messages, &mut request_messages)
            .await?;

        let reply = match self.wire_api {
            OpenAiWireApi::ChatCompletions => {
                let with_tools = self
                    .chat_with_function_calls(session_id.clone(), request_messages.clone())
                    .await;

                match with_tools {
                    Ok(v) => v,
                    Err(err) => {
                        log_warn_err(
                            format!(
                                "openai tool mode failed model={} endpoint={}, fallback to plain chat",
                                self.model, self.endpoint
                            ),
                            &err,
                        );
                        if self.debug {
                            println!("[DEBUG] openai tool mode failed, fallback plain chat: {err}");
                        }
                        match self.chat_plain(session_id, request_messages).await {
                            Ok(v) => v,
                            Err(plain_err) => {
                                log_warn_err(
                                    format!(
                                        "openai plain fallback failed model={} endpoint={}, returning transient reply",
                                        self.model, self.endpoint
                                    ),
                                    &plain_err,
                                );
                                if self.debug {
                                    println!(
                                        "[DEBUG] openai plain fallback failed, return transient message: {plain_err:#}"
                                    );
                                }
                                return self.transient_reply_or_err(plain_err);
                            }
                        }
                    }
                }
            }
            OpenAiWireApi::Responses => {
                let with_tools = self
                    .chat_with_function_calls_responses(
                        session_id.clone(),
                        request_messages.clone(),
                    )
                    .await;

                match with_tools {
                    Ok(v) => v,
                    Err(err) => {
                        log_warn_err(
                            format!(
                                "openai responses tool mode failed model={} endpoint={}, fallback to plain chat",
                                self.model, self.endpoint
                            ),
                            &err,
                        );
                        if self.debug {
                            println!(
                                "[DEBUG] openai responses tool mode failed, fallback plain chat: {err}"
                            );
                        }
                        match self
                            .chat_plain_responses(session_id, request_messages)
                            .await
                        {
                            Ok(v) => v,
                            Err(plain_err) => {
                                log_warn_err(
                                    format!(
                                        "openai responses plain fallback failed model={} endpoint={}, returning transient reply",
                                        self.model, self.endpoint
                                    ),
                                    &plain_err,
                                );
                                if self.debug {
                                    println!(
                                        "[DEBUG] openai responses plain fallback failed, return transient message: {plain_err:#}"
                                    );
                                }
                                return self.transient_reply_or_err(plain_err);
                            }
                        }
                    }
                }
            }
        };

        Ok(sanitize_identity_preface(&reply))
    }
}

impl OpenAiCompatibleLlm {
    /// 执行 Chat Completions 风格的工具循环，并返回最终回复。
    async fn chat_with_function_calls(
        &self,
        session_id: String,
        mut messages: Vec<Value>,
    ) -> Result<String> {
        const MAX_TOOL_ROUNDS: usize = 3;
        let tools = openai_tools_schema();
        let mut executed_tool_signatures = HashSet::new();

        for round in 0..=MAX_TOOL_ROUNDS {
            // Some gateways require all system messages to be merged at the beginning.
            normalize_system_messages(&mut messages);
            let stage = if has_openai_tool_results(&messages) {
                "final_answer"
            } else {
                "tool_planning"
            };
            let temperature = if stage == "tool_planning" {
                self.tool_planning_temperature()
            } else {
                self.answer_temperature()
            };

            if self.debug {
                println!(
                    "[DEBUG] calling OpenAI-compatible endpoint={} model={} session={} messages={} round={} stage={} temperature={:.2}",
                    self.endpoint,
                    self.model,
                    session_id,
                    messages.len(),
                    round,
                    stage,
                    temperature
                );
            }

            let payload = json!({
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "user": session_id,
                "tools": tools,
                "tool_choice": "auto",
            });

            let value = self.call_openai(payload).await?;
            let (content, tool_calls, assistant_msg, finish_reason) = parse_openai_choice(&value)?;

            if tool_calls.is_empty() {
                if let Some(call) =
                    recover_tool_call_from_text(&content, &messages, round, self.debug)
                {
                    if self.debug {
                        println!(
                            "[DEBUG] recovered textual tool call name={} args={}",
                            call.name, call.arguments
                        );
                    }
                    messages.push(build_synthetic_assistant_tool_message(&call, &content));
                    let result = self.execute_tool_call(&call).await;
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result
                    }));
                    continue;
                }

                let reply = content.trim().to_string();
                if reply.is_empty() {
                    if let Some(call) = infer_tool_call_from_recent_user(&messages, round) {
                        if self.debug {
                            println!(
                                "[DEBUG] inferred synthetic tool call from recent user name={} args={}",
                                call.name, call.arguments
                            );
                        }
                        messages.push(build_synthetic_assistant_tool_message(&call, ""));
                        let result = self.execute_tool_call(&call).await;
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result
                        }));
                        continue;
                    }

                    if self.debug {
                        let raw = truncate_debug_json(&assistant_msg);
                        println!("[DEBUG] empty assistant message with no tool call: {raw}");
                    }
                    return self
                        .force_final_answer_without_tools(
                            session_id,
                            messages,
                            "empty content after tool round",
                        )
                        .await;
                }
                let (reply, continued) = self
                    .maybe_continue_if_truncated(
                        session_id.clone(),
                        &messages,
                        reply,
                        finish_reason.clone(),
                    )
                    .await?;
                let finish = if continued { None } else { finish_reason };
                return Ok(append_finish_reason_hint(reply, finish));
            }

            if round == MAX_TOOL_ROUNDS {
                if self.debug {
                    println!("[DEBUG] tool call rounds exceeded, forcing final answer");
                }
                return self
                    .force_final_answer_without_tools(session_id, messages, "tool rounds exceeded")
                    .await;
            }

            if self.debug {
                println!("[DEBUG] openai tool calls requested: {}", tool_calls.len());
            }

            messages.push(assistant_msg);
            let mut executed_in_this_round = 0usize;
            let mut skipped_duplicate = 0usize;
            for call in tool_calls {
                let signature = tool_call_signature(&call);
                if !signature.is_empty() && executed_tool_signatures.contains(&signature) {
                    skipped_duplicate += 1;
                    if self.debug {
                        println!(
                            "[DEBUG] skip duplicate tool call name={} id={} signature={}",
                            call.name, call.id, signature
                        );
                    }
                    let duplicate_msg = format!(
                        "duplicate tool call skipped: {}。请基于已有工具结果继续回答，除非用户提供了新的关键词/URL。",
                        call.name
                    );
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": wrap_untrusted_tool_output("tool_guard", duplicate_msg)
                    }));
                    continue;
                }

                if !signature.is_empty() {
                    executed_tool_signatures.insert(signature);
                }
                let result = self.execute_tool_call(&call).await;
                executed_in_this_round += 1;
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                }));
            }

            if executed_in_this_round == 0 && skipped_duplicate > 0 {
                if self.debug {
                    println!(
                        "[DEBUG] all requested tools are duplicates, force final answer to save tokens"
                    );
                }
                return self
                    .force_final_answer_without_tools(
                        session_id,
                        messages,
                        "duplicate tool calls only",
                    )
                    .await;
            }
        }

        self.force_final_answer_without_tools(session_id, messages, "tool loop end")
            .await
    }

    /// 不使用工具，直接请求 Chat Completions 回复。
    async fn chat_plain(&self, session_id: String, messages: Vec<Value>) -> Result<String> {
        let mut messages = messages;
        normalize_system_messages(&mut messages);
        let temperature = self.answer_temperature();
        if self.debug {
            println!(
                "[DEBUG] calling OpenAI-compatible plain endpoint={} model={} session={} messages={} temperature={:.2}",
                self.endpoint,
                self.model,
                session_id,
                messages.len(),
                temperature
            );
        }

        let payload = json!({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "user": session_id,
        });

        let value = self.call_openai(payload).await?;
        let (reply, _, assistant_msg, finish_reason) = parse_openai_choice(&value)?;
        let reply = reply.trim().to_string();

        if reply.is_empty() {
            if self.debug {
                let raw = truncate_debug_json(&assistant_msg);
                println!("[DEBUG] plain chat got empty assistant content: {raw}");
            }

            return self
                .force_final_answer_without_tools(session_id, messages, "plain empty content")
                .await;
        }

        if self.debug {
            println!(
                "[DEBUG] AI response ok model={} timeout_ms={} base_url={}",
                self.model, self.timeout_ms, self.base_url
            );
        }

        let (reply, continued) = self
            .maybe_continue_if_truncated(session_id, &messages, reply, finish_reason.clone())
            .await?;
        let finish = if continued { None } else { finish_reason };
        Ok(append_finish_reason_hint(reply, finish))
    }

    async fn chat_plain_responses(
        &self,
        session_id: String,
        request_messages: Vec<Value>,
    ) -> Result<String> {
        let ctx = self.build_responses_context(request_messages);
        let temperature = self.answer_temperature();
        if self.debug {
            println!(
                "[DEBUG] calling OpenAI-compatible responses plain endpoint={} model={} session={} input={} temperature={:.2}",
                self.endpoint,
                self.model,
                session_id,
                ctx.input.len(),
                temperature
            );
        }

        let payload = self.build_responses_payload(
            &session_id,
            &ctx.instructions,
            &ctx.input,
            temperature,
            None,
        );
        let value = self.call_openai(payload).await?;
        let parsed = parse_responses_output(&value)?;
        let reply = parsed.content.trim().to_string();

        if reply.is_empty() {
            if self.debug {
                let raw = truncate_debug_json(&parsed.assistant_output);
                println!("[DEBUG] responses plain chat got empty assistant content: {raw}");
            }
            return self
                .force_final_answer_without_tools_responses(session_id, ctx, "plain empty content")
                .await;
        }

        if self.debug {
            println!(
                "[DEBUG] AI responses response ok model={} timeout_ms={} base_url={}",
                self.model, self.timeout_ms, self.base_url
            );
        }

        Ok(append_finish_reason_hint(reply, parsed.finish_reason))
    }

    /// 当工具循环没有得到可用正文时，强制模型基于已有信息直接产出最终答案。
    async fn force_final_answer_without_tools(
        &self,
        session_id: String,
        mut messages: Vec<Value>,
        reason: &str,
    ) -> Result<String> {
        let temperature = self.answer_temperature();
        messages.push(json!({
            "role": "system",
            "content": format!(
                "工具流程已结束（reason={reason}）。请基于已有对话与工具结果直接输出最终答复文本。禁止继续调用工具，禁止输出空内容。若信息不足请明确说明缺少哪些信息。"
            )
        }));
        normalize_system_messages(&mut messages);

        if self.debug {
            println!(
                "[DEBUG] forcing final no-tool answer session={} messages={} temperature={:.2} reason={}",
                session_id,
                messages.len(),
                temperature,
                reason
            );
        }

        let retry_messages = messages.clone();
        let payload = json!({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "user": session_id.clone(),
        });

        let value = self.call_openai(payload).await?;
        let (reply, _, assistant_msg, finish_reason) = parse_openai_choice(&value)?;
        let reply = reply.trim().to_string();
        if reply.is_empty() {
            if self.debug {
                let raw = truncate_debug_json(&assistant_msg);
                println!("[DEBUG] forced final still empty: {raw}");
            }
            if has_reasoning_content(&assistant_msg) {
                if self.debug {
                    println!(
                        "[DEBUG] forced final has reasoning_content but empty content, retry with no-reasoning hints"
                    );
                }
                if let Some(retry_reply) = self
                    .retry_plain_answer_no_reasoning(
                        session_id,
                        retry_messages,
                        "forced_final_empty_with_reasoning",
                    )
                    .await?
                {
                    return Ok(retry_reply);
                }
            }
            return self.transient_reply_or_err(anyhow!("AI endpoint returned empty content"));
        }
        let (reply, continued) = self
            .maybe_continue_if_truncated(
                session_id.clone(),
                &messages,
                reply,
                finish_reason.clone(),
            )
            .await?;
        let finish = if continued { None } else { finish_reason };
        Ok(append_finish_reason_hint(reply, finish))
    }

    /// 某些兼容网关会把“思考内容”塞进非标准字段，这里做一次禁用思考的重试。
    async fn retry_plain_answer_no_reasoning(
        &self,
        session_id: String,
        mut messages: Vec<Value>,
        reason: &str,
    ) -> Result<Option<String>> {
        messages.push(json!({
            "role": "system",
            "content": format!(
                "请直接输出最终答复正文（reason={reason}）。不要输出思考过程、分析步骤或 reasoning_content 字段对应内容。"
            )
        }));
        normalize_system_messages(&mut messages);

        let temperature = self.answer_temperature();
        let payload_with_hints = json!({
            "model": self.model,
            "messages": messages.clone(),
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "user": session_id,
            "enable_thinking": false,
            "reasoning_effort": "low",
            "thinking": { "type": "disabled" },
            "extra_body": {
                "enable_thinking": false,
                "reasoning_effort": "low",
                "thinking": { "type": "disabled" },
                "reasoning": { "enabled": false }
            }
        });

        let value = match self.call_openai(payload_with_hints).await {
            Ok(v) => v,
            Err(err) => {
                if self.debug {
                    println!(
                        "[DEBUG] no-reasoning payload rejected, fallback without extra hints: {err:#}"
                    );
                }
                let fallback_payload = json!({
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": self.max_tokens,
                    "user": session_id,
                });
                self.call_openai(fallback_payload).await?
            }
        };

        let (reply, _, assistant_msg, finish_reason) = parse_openai_choice(&value)?;
        let reply = reply.trim().to_string();
        if reply.is_empty() {
            if self.debug {
                let raw = truncate_debug_json(&assistant_msg);
                println!("[DEBUG] no-reasoning retry still empty: {raw}");
            }
            return Ok(None);
        }
        let (reply, continued) = self
            .maybe_continue_if_truncated(session_id, &messages, reply, finish_reason.clone())
            .await?;
        let finish = if continued { None } else { finish_reason };
        Ok(Some(append_finish_reason_hint(reply, finish)))
    }

    async fn force_final_answer_without_tools_responses(
        &self,
        session_id: String,
        mut ctx: ResponsesContext,
        reason: &str,
    ) -> Result<String> {
        let temperature = self.answer_temperature();
        ctx.input.push(build_responses_text_message(
            "system",
            format!(
                "工具流程已结束（reason={reason}）。请基于已有对话与工具结果直接输出最终答复文本。禁止继续调用工具，禁止输出空内容。若信息不足请明确说明缺少哪些信息。"
            ),
        ));

        if self.debug {
            println!(
                "[DEBUG] forcing final responses no-tool answer session={} input={} temperature={:.2} reason={}",
                session_id,
                ctx.input.len(),
                temperature,
                reason
            );
        }

        let payload = self.build_responses_payload(
            &session_id,
            &ctx.instructions,
            &ctx.input,
            temperature,
            None,
        );
        let value = self.call_openai(payload).await?;
        let parsed = parse_responses_output(&value)?;
        let reply = parsed.content.trim().to_string();

        if reply.is_empty() {
            if self.debug {
                let raw = truncate_debug_json(&parsed.assistant_output);
                println!("[DEBUG] forced final responses still empty: {raw}");
            }
            return self.transient_reply_or_err(anyhow!("AI endpoint returned empty content"));
        }

        Ok(append_finish_reason_hint(reply, parsed.finish_reason))
    }

    fn answer_temperature(&self) -> f32 {
        if !self.temperature.is_finite() {
            return 0.55;
        }
        // 回答阶段保持稳定可读，避免过高温度导致跑题。
        self.temperature.clamp(0.3, 0.75)
    }

    fn tool_planning_temperature(&self) -> f32 {
        // 工具选择阶段用更低温度，减少误选工具和幻觉。
        (self.answer_temperature() * 0.4).clamp(0.05, 0.25)
    }

    /// 根据当前实例模式决定：是直接给用户返回临时网络提示，还是把错误继续抛给上层回退链。
    fn transient_reply_or_err(&self, err: anyhow::Error) -> Result<String> {
        if self.suppress_transient_errors {
            log_warn_err(
                format!(
                    "all retries exhausted for model={} endpoint={}, returning transient reply",
                    self.model, self.endpoint
                ),
                &err,
            );
            return Ok(temporary_network_reply());
        }
        Err(err)
    }

    /// 发送一次 HTTP 请求到 OpenAI 兼容网关，并统一做重试与 usage 记录。
    async fn call_openai(&self, payload: Value) -> Result<Value> {
        let payload = self.prepare_openai_payload(payload);
        const MAX_ATTEMPTS: usize = 3;

        for attempt in 1..=MAX_ATTEMPTS {
            let mut request = self
                .client
                .post(&self.endpoint)
                .header("Content-Type", "application/json");

            if !self.api_key.trim().is_empty() {
                request = request.bearer_auth(self.api_key.trim());
            }

            let response = match request.json(&payload).send().await {
                Ok(v) => v,
                Err(err) => {
                    if self.debug {
                        println!(
                            "[DEBUG] openai request transport error attempt={}/{} timeout_ms={} err={}",
                            attempt, MAX_ATTEMPTS, self.timeout_ms, err
                        );
                    }
                    if attempt < MAX_ATTEMPTS && is_retryable_reqwest_error(&err) {
                        let backoff_ms = 300 * attempt as u64;
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                        continue;
                    }
                    return Err(err).with_context(|| format!("failed to call {}", self.endpoint));
                }
            };

            let status = response.status();
            let body = response
                .text()
                .await
                .context("failed to read AI response body")?;

            if !status.is_success() {
                if self.debug {
                    let brief = debug_brief(&body, 240);
                    println!(
                        "[DEBUG] openai non-success status attempt={}/{} status={} body={}",
                        attempt, MAX_ATTEMPTS, status, brief
                    );
                }

                if attempt < MAX_ATTEMPTS && (status.is_server_error() || status.as_u16() == 429) {
                    let backoff_ms = 300 * attempt as u64;
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                bail!("AI endpoint returned {}: {}", status, body);
            }

            let value: Value =
                serde_json::from_str(&body).context("failed to parse AI response JSON")?;
            record_usage_from_openai_response(&value, self.debug);
            return Ok(value);
        }

        bail!("failed to call {} after retries", self.endpoint)
    }

    fn prepare_openai_payload(&self, payload: Value) -> Value {
        match self.wire_api {
            OpenAiWireApi::ChatCompletions => apply_default_no_thinking_hints(payload),
            OpenAiWireApi::Responses => apply_responses_defaults(
                payload,
                &self.reasoning_effort,
                self.disable_response_storage,
            ),
        }
    }

    fn build_responses_context(&self, request_messages: Vec<Value>) -> ResponsesContext {
        let mut messages = request_messages;
        normalize_system_messages(&mut messages);

        let mut instructions = Vec::new();
        let mut input = Vec::new();
        for message in messages {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user");
            match role {
                "system" => {
                    let text = message_content_as_text(&message);
                    if !text.trim().is_empty() {
                        instructions.push(text);
                    }
                }
                "tool" => {
                    let call_id = message
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();
                    let output = message_content_as_text(&message);
                    if !call_id.is_empty() && !output.trim().is_empty() {
                        input.push(build_responses_function_call_output_item(call_id, &output));
                    }
                }
                "assistant" => {
                    if let Some(item) =
                        convert_chat_message_to_responses_item(&message, "assistant")
                    {
                        input.push(item);
                    }
                }
                _ => {
                    if let Some(item) = convert_chat_message_to_responses_item(&message, "user") {
                        input.push(item);
                    }
                }
            }
        }

        ResponsesContext {
            instructions: instructions.join("\n\n"),
            input,
        }
    }

    fn build_responses_payload(
        &self,
        _session_id: &str,
        instructions: &str,
        input: &[Value],
        temperature: f32,
        tools: Option<&Value>,
    ) -> Value {
        let mut payload = json!({
            "model": self.model,
            "input": input,
            "temperature": temperature,
            "max_output_tokens": self.max_tokens,
        });

        if !instructions.trim().is_empty() {
            payload["instructions"] = Value::String(instructions.to_string());
        }
        if let Some(tools) = tools {
            payload["tools"] = tools.clone();
        }

        payload
    }

    /// 把上层统一消息格式转换成模型实际请求消息。
    async fn build_request_messages(&self, messages: &[(String, String)]) -> Result<Vec<Value>> {
        let mut out = Vec::with_capacity(messages.len());
        let last_user_idx = messages.iter().rposition(|(role, _)| role == "user");

        for (idx, (role, content)) in messages.iter().enumerate() {
            if role == "user" {
                let parsed = parse_user_content(content);
                let with_images = Some(idx) == last_user_idx
                    && (!parsed.image_urls.is_empty() || !parsed.image_files.is_empty());
                if with_images {
                    if self.should_send_images() {
                        let payload = self.build_openai_user_content_with_images(parsed).await;
                        out.push(json!({ "role": "user", "content": payload }));
                    } else if self.should_use_ocr() {
                        let text = self.build_user_text_with_ocr(parsed).await;
                        out.push(json!({ "role": "user", "content": text }));
                    } else {
                        out.push(json!({ "role": "user", "content": parsed.text }));
                    }
                } else {
                    out.push(json!({ "role": "user", "content": parsed.text }));
                }
            } else {
                out.push(json!({ "role": role, "content": content }));
            }
        }

        Ok(out)
    }

    /// 预抓取当前轮文本里出现的 URL，并把摘要前置到系统提示中。
    async fn preload_current_turn_url_context(
        &self,
        messages: &[(String, String)],
        request_messages: &mut Vec<Value>,
    ) -> Result<()> {
        let latest_user = messages
            .iter()
            .rev()
            .find(|(role, _)| role == "user")
            .map(|(_, content)| content.as_str())
            .unwrap_or("");

        let mut urls = extract_urls(latest_user);
        if urls.is_empty() {
            return Ok(());
        }

        urls.truncate(2);
        let mut blocks = Vec::new();
        for url in urls {
            match fetch_url(&self.client, &url, self.debug).await {
                Ok(content) => blocks.push(content),
                Err(err) => blocks.push(format!("URL: {url}\n抓取失败: {err}")),
            }
        }

        let merged = blocks.join("\n\n---\n\n");
        if self.debug {
            println!(
                "[DEBUG] preload current-turn url context blocks={}",
                blocks.len()
            );
        }
        request_messages.push(json!({
            "role": "system",
            "content": format!("当前轮用户消息包含 URL。请优先基于以下抓取内容回答；若抓取内容与用户描述冲突，请明确指出并给出依据。\n{merged}")
        }));
        Ok(())
    }

    async fn build_openai_user_content_with_images(
        &self,
        parsed: crate::llm::message_parts::ParsedUserContent,
    ) -> Value {
        let mut blocks = Vec::new();
        blocks.push(json!({
            "type": "text",
            "text": parsed.text
        }));

        let mut image_count = 0usize;
        let mut image_refs = parsed.image_urls.clone();
        for file in &parsed.image_files {
            if !is_loadable_image_ref(file) {
                if self.debug {
                    println!("[DEBUG] skip unresolved image file ref={file}");
                }
                continue;
            }
            if !image_refs.iter().any(|v| v == file) {
                image_refs.push(file.clone());
            }
        }

        for image_ref in image_refs.iter().take(3) {
            match load_image_for_llm(&self.client, image_ref, self.debug).await {
                Ok(image) => {
                    blocks.push(json!({
                        "type": "image_url",
                        "image_url": { "url": image.as_data_url() }
                    }));
                    image_count += 1;
                }
                Err(err) => {
                    if self.debug {
                        println!("[DEBUG] skip invalid image ref={} err={}", image_ref, err);
                    }
                }
            }
        }

        if self.debug {
            println!(
                "[DEBUG] image payload for llm refs={} attached={}",
                image_refs.len(),
                image_count
            );
        }

        if image_count == 0 {
            blocks.push(json!({
                "type": "text",
                "text": "未能读取图片数据，请用户重发图片或附上可访问链接。"
            }));
        }

        Value::Array(blocks)
    }

    async fn build_user_text_with_ocr(&self, parsed: ParsedUserContent) -> String {
        let image_refs = collect_image_refs(&parsed);
        let mut text = parsed.text;
        if image_refs.is_empty() {
            return text;
        }

        if self.debug {
            println!(
                "[DEBUG] using ocr fallback refs={} mode={:?}",
                image_refs.len(),
                self.vision_mode
            );
        }

        let ocr_text =
            ocr_images_to_text(&self.client, &image_refs, &self.ocr_settings, self.debug).await;
        if !ocr_text.is_empty() {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&ocr_text);
        }

        text
    }

    fn should_send_images(&self) -> bool {
        match self.vision_mode {
            VisionMode::Multimodal => true,
            VisionMode::Ocr | VisionMode::Off => false,
            VisionMode::Auto => model_seems_multimodal(&self.model),
        }
    }

    fn should_use_ocr(&self) -> bool {
        match self.vision_mode {
            VisionMode::Ocr => true,
            VisionMode::Auto => !model_seems_multimodal(&self.model),
            VisionMode::Multimodal | VisionMode::Off => false,
        }
    }

    async fn execute_tool_call(&self, call: &OpenAiToolCall) -> String {
        match call.name.as_str() {
            "search_web" => {
                let query = extract_argument_str(&call.arguments, &["query", "q", "keyword"])
                    .or_else(|| {
                        call.arguments
                            .get("raw")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_default();
                let query = strip_sender_prefix(query.trim()).trim().to_string();
                if query.is_empty() {
                    return "search_web error: query is empty".to_string();
                }
                // Weather policy: search web first for multi-day forecast; use get_weather only as fallback.
                let weather_query = weather_intent_from_text(&query.to_lowercase());
                match search_web(
                    &self.client,
                    &query,
                    self.debug,
                    &self.search,
                    &self.network,
                )
                .await
                {
                    Ok(v) => {
                        if weather_query && search_result_looks_empty(&v) {
                            let location = extract_weather_location_hint(&query);
                            match get_weather(&self.client, &location, self.debug).await {
                                Ok(weather) => wrap_untrusted_tool_output(
                                    "search_web",
                                    format!("{}\n\n[weather_fallback]\n{}", v, weather),
                                ),
                                Err(_) => wrap_untrusted_tool_output("search_web", v),
                            }
                        } else {
                            wrap_untrusted_tool_output("search_web", v)
                        }
                    }
                    Err(err) => {
                        if weather_query {
                            let location = extract_weather_location_hint(&query);
                            if let Ok(weather) =
                                get_weather(&self.client, &location, self.debug).await
                            {
                                return wrap_untrusted_tool_output(
                                    "get_weather",
                                    format!(
                                        "search_web error: {err}\n\n[weather_fallback]\n{}",
                                        weather
                                    ),
                                );
                            }
                        }
                        format!("search_web error: {err}")
                    }
                }
            }
            "fetch_url" => {
                let url = extract_argument_str(&call.arguments, &["url", "link", "href"])
                    .or_else(|| {
                        call.arguments
                            .get("raw")
                            .and_then(Value::as_str)
                            .and_then(|raw| extract_urls(raw).into_iter().next())
                    })
                    .unwrap_or_default();
                let url = url.trim().to_string();
                if url.is_empty() {
                    return "fetch_url error: url is empty".to_string();
                }
                match fetch_url(&self.client, &url, self.debug).await {
                    Ok(v) => wrap_untrusted_tool_output("fetch_url", v),
                    Err(err) => format!("fetch_url error: {err}"),
                }
            }
            "get_system_info" => {
                let scope = extract_argument_str(&call.arguments, &["scope", "type"])
                    .unwrap_or_else(|| "all".to_string());
                let scope = scope.trim().to_string();
                match get_system_info(&scope) {
                    Ok(v) => wrap_untrusted_tool_output("get_system_info", v),
                    Err(err) => format!("get_system_info error: {err}"),
                }
            }
            "get_process_info" => match get_process_info() {
                Ok(v) => wrap_untrusted_tool_output("get_process_info", v),
                Err(err) => format!("get_process_info error: {err}"),
            },
            "get_weather" => {
                let location =
                    extract_argument_str(&call.arguments, &["location", "city", "place", "query"])
                        .unwrap_or_default();
                let location = strip_sender_prefix(location.trim()).trim().to_string();
                if location.is_empty() {
                    return "get_weather error: location is empty".to_string();
                }
                match get_weather(&self.client, &location, self.debug).await {
                    Ok(v) => wrap_untrusted_tool_output("get_weather", v),
                    Err(err) => format!("get_weather error: {err}"),
                }
            }
            _ => format!("unknown tool: {}", call.name),
        }
    }
}
