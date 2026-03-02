# XzBot

基于 **Rust + Tokio + Axum** 的 QQ 聊天机器人，通过 NapCat OneBot v11 反向 WebSocket 接入，支持多 LLM 后端、Function Call 工具调用与 OCR 图片识别。

## 特性

- **反向 WS 接入** — NapCat 主动连接，无需公网端口映射
- **灵活触发规则** — 支持 @、前缀、关键词、混合模式
- **多 LLM 后端** — OpenAI 兼容 / Anthropic 兼容 / Mock（测试用）
- **上下文会话** — 内存存储，每用户/群保留最近 10 轮
- **图片理解** — 自动提取消息及引用回复中的图片
- **OCR 兜底** — 非多模态模型自动 OCR（Tesseract / PaddleOCR）
- **Function Call 工具**
  - `search_web` — 网页搜索
  - `fetch_url` — 抓取 URL 内容
  - `get_system_info` — 只读系统信息
  - `get_weather` — 天气查询
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

> **安全提示：** 真实 `api_key` / OCR token 只写入运行时配置（已被 `.gitignore` 忽略），切勿提交到 Git。

### 搜索配置

- 内置 Bing（中国版）
- 自建 SearXNG（启用后不再使用内置搜索）

对应配置项：

```toml
[search]
provider = "builtin"   # 或 "searxng"
searxng_url = ""
```

### OCR 配置

支持两类 OCR：

1. **Tesseract（本地）**
2. **Paddle OCR（HTTP API）**

常用配置示例：

```toml
[ai]
vision_mode = "auto"      # auto / multimodal / ocr / off
ocr_provider = "tesseract" # 或 "paddle"
ocr_cmd = "tesseract"
ocr_lang = "chi_sim+eng"
ocr_timeout_ms = 8000

# Paddle OCR（layout-parsing）
paddle_ocr_endpoint = ""
paddle_ocr_token = ""
paddle_file_type = 1
paddle_use_proxy = true
```

### 代理配置

所有 HTTP 请求默认走同一代理（LLM / 搜索 / OCR / fetch_url 等）。
如需 **仅让 Paddle OCR 直连**，可设置 `paddle_use_proxy=false`。

```toml
[network]
proxy_enabled = false
proxy_url = ""
proxy_test_url = "https://www.baidu.com"
proxy_timeout_ms = 5000
```

## 运行时指令

| 指令 | 说明 |
|------|------|
| `/reset` | 清空当前会话上下文 |
| `/blacklist list` | 查看群黑名单 |
| `/blacklist add <group_id>` | 添加群到黑名单 |
| `/blacklist remove <group_id>` | 从黑名单移除群 |

## 插件（托管进程）

XzBot 启动时会在 **当前工作目录** 创建 `Plugins/` 文件夹，并扫描其中的可执行文件。  
插件进程由 XzBot **统一拉起并常驻管理**（类似 MC/Spigot 插件生命周期）。

```
Plugins/
├── my-plugin            # 插件二进制
└── my-plugin/           # 插件自己使用的目录（配置/缓存）
```

插件需支持：

```
<plugin_binary> --manifest
```

并输出 JSON 清单：

```json
{ "name": "my-plugin", "commands": ["hello"], "timeout_ms": 20000 }
```

通信方式：stdin/stdout 按行 JSON（必须回传 `request_id`）。
插件可返回 `file_path` 以发送文件（适合长报告），也可返回 `image_path` / `image_url` 发送图片（CQ 码）。
`/reload` 可重新扫描 `Plugins/` 并重启插件进程。
插件 stderr 会转发到 XzBot 控制台日志。

详细规范见 `docs/PLUGINS.md`。

## 项目结构

```
src/
├── main.rs          # WS 服务器、消息路由、图片增强
├── config.rs        # 配置加载与校验
├── bot/             # 消息路由、AI 对话插件
├── llm/             # LLM 后端适配（OpenAI / Anthropic / Mock）
├── llm/ocr.rs        # OCR 兜底（Tesseract / Paddle）
├── onebot/          # OneBot v11 事件与动作
├── store/           # 会话内存存储
├── plugins/         # 插件系统（托管进程）
└── tools/           # Function Call 工具实现
```
