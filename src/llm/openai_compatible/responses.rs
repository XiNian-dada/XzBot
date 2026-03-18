//! OpenAI Responses API 的对话循环与请求构造。
//!
//! Responses API 和传统 Chat Completions 的线协议不同：
//! - 输入使用 `input` 数组而不是 `messages`
//! - 工具调用与工具结果的结构也不同
//! - 某些兼容网关还会对字段支持程度不一致
//!
//! 因此这里单独维护 Responses 的执行分支，避免把所有分支条件塞回主文件。

use super::*;

impl OpenAiCompatibleLlm {
    /// 使用 Responses API 执行“带工具调用”的完整对话回合。
    ///
    /// 这条链路的职责是：
    /// 1. 发送当前 `input`
    /// 2. 解析模型要求的工具调用
    /// 3. 执行工具并把结果以 `function_call_output` 回填
    /// 4. 在达到稳定答案或达到安全上限时收敛
    pub(super) async fn chat_with_function_calls_responses(
        &self,
        session_id: String,
        request_messages: Vec<Value>,
    ) -> Result<String> {
        const MAX_TOOL_ROUNDS: usize = 3;
        let mut ctx = self.build_responses_context(request_messages);
        let tools = openai_responses_tools_schema(self.plugins.openai_responses_tool_schemas());
        let mut executed_tool_signatures = HashSet::new();

        for round in 0..=MAX_TOOL_ROUNDS {
            // 一旦上下文里已经有工具结果，下一轮通常就是“综合工具结果生成最终答案”。
            let stage = if has_responses_tool_results(&ctx.input) {
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
                    "[DEBUG] calling OpenAI-compatible responses endpoint={} model={} session={} input={} round={} stage={} temperature={:.2}",
                    self.endpoint,
                    self.model,
                    session_id,
                    ctx.input.len(),
                    round,
                    stage,
                    temperature
                );
            }

            let payload = self.build_responses_payload(
                &session_id,
                &ctx.instructions,
                &ctx.input,
                temperature,
                Some(&tools),
            );
            let value = self.call_openai(payload).await?;
            let parsed = parse_responses_output(&value)?;

            if parsed.tool_calls.is_empty() {
                // 某些兼容网关不会返回结构化 tool_call，而是把“伪工具调用文本”塞进正文里。
                // 这里做一次恢复，尽量把这种非标准输出拉回正常工具链。
                if let Some(call) =
                    recover_tool_call_from_text(&parsed.content, &ctx.input, round, self.debug)
                {
                    if self.debug {
                        println!(
                            "[DEBUG] recovered textual responses tool call name={} args={}",
                            call.name, call.arguments
                        );
                    }
                    ctx.input.push(build_responses_function_call_item(&call));
                    let result = self.execute_tool_call(&session_id, &call).await;
                    ctx.input
                        .push(build_responses_function_call_output_item(&call.id, &result));
                    continue;
                }

                let reply = parsed.content.trim().to_string();
                if reply.is_empty() {
                    if self.debug {
                        let raw = truncate_debug_json(&parsed.assistant_output);
                        println!("[DEBUG] responses tool mode got empty assistant content: {raw}");
                    }
                    return self
                        .force_final_answer_without_tools_responses(
                            session_id,
                            ctx,
                            "tool mode empty content",
                        )
                        .await;
                }

                return Ok(append_finish_reason_hint(reply, parsed.finish_reason));
            }

            if self.debug {
                println!(
                    "[DEBUG] openai responses tool calls requested: {}",
                    parsed.tool_calls.len()
                );
            }

            // 工具轮次上限是防止“模型无限继续要求搜索/抓取”导致 token 与等待时间失控。
            if round >= MAX_TOOL_ROUNDS {
                if self.debug {
                    println!("[DEBUG] responses tool call rounds exceeded, forcing final answer");
                }
                return self
                    .force_final_answer_without_tools_responses(
                        session_id,
                        ctx,
                        "tool call rounds exceeded",
                    )
                    .await;
            }

            let mut executed_in_this_round = 0usize;
            let mut skipped_duplicate = 0usize;
            for call in parsed.tool_calls {
                // 同签名工具调用只执行一次，避免模型反复要求完全相同的搜索/抓取。
                let signature = tool_call_signature(&call);
                if !signature.is_empty() && executed_tool_signatures.contains(&signature) {
                    skipped_duplicate += 1;
                    if self.debug {
                        println!(
                            "[DEBUG] skip duplicate responses tool call name={} id={} signature={}",
                            call.name, call.id, signature
                        );
                    }
                    continue;
                }

                ctx.input.push(build_responses_function_call_item(&call));
                if !signature.is_empty() {
                    executed_tool_signatures.insert(signature);
                }
                let result = self.execute_tool_call(&session_id, &call).await;
                executed_in_this_round += 1;
                ctx.input
                    .push(build_responses_function_call_output_item(&call.id, &result));
            }

            if executed_in_this_round == 0 && skipped_duplicate > 0 {
                if self.debug {
                    println!(
                        "[DEBUG] all requested responses tools are duplicates, force final answer to save tokens"
                    );
                }
                return self
                    .force_final_answer_without_tools_responses(
                        session_id,
                        ctx,
                        "duplicate tool calls only",
                    )
                    .await;
            }
        }

        self.force_final_answer_without_tools_responses(session_id, ctx, "tool loop end")
            .await
    }
}
