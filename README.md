# XzBot

一个基于 Rust + Tokio + Axum 的 NapCat OneBot v11 反向 WebSocket 聊天机器人服务。

## 功能
- 反向 WS 接入（NapCat 主动连接）
- 私聊/群聊消息路由与触发规则
- 上下文会话管理（内存，最近 10 轮）
- `/reset` 清空会话
- `/blacklist` 运行时群黑名单管理（仅 owner）
- 多工具 Function Call：
  - `search_web`
  - `fetch_url`
  - `get_system_info`（只读系统信息）
- 图片消息处理：从 OneBot 消息中提取图片 URL，并将图片内容传给模型（而不是只用文件名）

## 运行环境
- Rust 1.75+
- NapCat（OneBot v11 反向 WS）

## 快速启动
```bash
cargo check
cargo build
./target/debug/xzbot
```

启动后默认监听（由配置控制）：
- `0.0.0.0:3000`
- WS 路径：`/onebot/v11/ws`

## 配置说明
项目根目录的 `config.toml` 是模板配置（用于编译内置默认模板）。

程序实际运行时，会在**二进制所在目录**读取/创建配置：
- `./config/config.toml`

例如使用 debug 构建运行时，通常是：
- `target/debug/config/config.toml`

### NapCat 连接要点
- 只需启用 NapCat 的**反向 WebSocket 客户端**
- 连接地址：
  - `ws://<你的主机IP>:<port><ws_path>`
  - 例：`ws://192.168.1.10:3000/onebot/v11/ws`

## 安全建议（重要）
- 不要把真实 `api_key` 提交到 Git。
- 建议只在本地运行时配置里填写真实密钥：
  - `target/debug/config/config.toml`
- 已通过 `.gitignore` 忽略构建产物和运行时配置，避免误提交。

## 常用指令
- `/reset`：清空当前会话上下文
- `/blacklist list`
- `/blacklist add <group_id>`
- `/blacklist remove <group_id>`

## 项目结构
```text
src/
 ├─ main.rs
 ├─ config.rs
 ├─ logger.rs
 ├─ onebot/
 ├─ bot/
 ├─ store/
 ├─ llm/
 └─ tools/
```
