# XzBot

基于 **Rust + Tokio + Axum** 的 QQ 聊天机器人，通过 NapCat OneBot v11 反向 WebSocket 接入，支持多 LLM 后端与 Function Call 工具调用。

## 特性

- **反向 WS 接入** — NapCat 主动连接，无需公网端口映射
- **灵活触发规则** — 支持 @、前缀、关键词、混合模式
- **多 LLM 后端** — OpenAI 兼容 / Anthropic 兼容 / Mock（测试用）
- **上下文会话** — 内存存储，每用户/群保留最近 10 轮
- **图片理解** — 自动提取消息及引用回复中的图片，传递给模型
- **Function Call 工具**
  - `search_web` — 网页搜索
  - `fetch_url` — 抓取 URL 内容
  - `get_system_info` — 只读系统信息
- **权限控制** — 支持 None / OwnerOnly / Whitelist 三种模式
- **运行时指令** — `/reset`、`/blacklist`（仅 owner）

## 快速开始

**依赖：** Rust 1.75+、NapCat（OneBot v11 反向 WS）

```bash
cargo build --release
./target/release/xzbot
```

默认监听 `0.0.0.0:3000`，WS 路径 `/onebot/v11/ws`。

## 配置

程序启动时从**二进制所在目录**读取配置：

```
./config/config.toml
```

首次运行会自动生成模板配置（源自项目根目录的 `config.toml`）。

**NapCat 反向 WS 连接地址：**
```
ws://<主机IP>:3000/onebot/v11/ws
```

> **安全提示：** 真实 `api_key` 只写入运行时配置（已被 `.gitignore` 忽略），切勿提交到 Git。

## 运行时指令

| 指令 | 说明 |
|------|------|
| `/reset` | 清空当前会话上下文 |
| `/blacklist list` | 查看群黑名单 |
| `/blacklist add <group_id>` | 添加群到黑名单 |
| `/blacklist remove <group_id>` | 从黑名单移除群 |

## 项目结构

```
src/
├── main.rs          # WS 服务器、消息路由、图片增强
├── config.rs        # 配置加载与校验
├── bot/             # 消息路由、AI 对话插件
├── llm/             # LLM 后端适配（OpenAI / Anthropic / Mock）
├── onebot/          # OneBot v11 事件与动作
├── store/           # 会话内存存储
└── tools/           # Function Call 工具实现
```
