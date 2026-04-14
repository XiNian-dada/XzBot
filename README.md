# XzBot

基于 **Rust + Tokio + Axum** 的 QQ 聊天机器人，通过 NapCat OneBot v11 反向 WebSocket 接入，支持多 LLM 后端、Function Call 工具调用与 OCR 图片识别。

- 在线文档（GitHub Pages，多页面 Wiki）：`https://xinian-dada.github.io/XzBot/`
- 仓库文档入口：`docs/index.html`
- 当前 crate 版本：`0.2.2`

## 特性

- **反向 WS 接入** — NapCat 主动连接，无需公网端口映射
- **灵活触发规则** — 支持 @、前缀、关键词、混合模式
- **多 LLM 后端** — OpenAI 兼容 / Anthropic 兼容 / Mock（测试用）
- **上下文会话** — 内存存储，每用户/群保留最近 10 轮
- **图片理解** — 自动提取消息及引用回复中的图片
- **OCR 兜底** — 非多模态模型自动 OCR（Tesseract / PaddleOCR）
- **Function Call 工具**
  - `search_web` — 网页搜索
  - `fetch_url` — 抓取 URL 内容（知乎优先走公开 API，其他站点走浏览器风格请求 + reader 兜底）
  - `get_system_info` — 只读系统信息
  - `get_process_info` — 只读 XzBot 进程资源占用
  - `get_weather` — 天气查询（当前天气，作为兜底）
- **权限控制** — 支持 None / OwnerOnly / Whitelist 三种模式
- **运行时指令** — `/reset`、`/blacklist`、`/reload`、`/log`、`/posttoken`
- **外部推送 API** — 通过聊天绑定 token 向指定会话发送文本/图片/文件

## 快速开始

**依赖：** Rust 1.75+、NapCat（OneBot v11 反向 WS）

```bash
cargo build --release
./target/release/xzbot
```

默认监听 `0.0.0.0:3000`，WS 路径 `/onebot/v11/ws`。

如果你更想看一份按“从零部署 -> 对接 NapCat -> 配置 -> 控制台 -> 插件 -> 升级发版”顺序整理的多页面傻瓜文档，建议直接打开在线文档站：

- GitHub Pages：`https://xinian-dada.github.io/XzBot/`
- 仓库内文件：`/Users/bernard/Code/XzBot/docs/index.html`

## 配置

程序启动时从**二进制所在目录**读取配置：

```
./config/config.toml
```

首次运行会自动生成模板配置（源自项目根目录的 `config.default.toml`）。

### 主配置 + 可选覆盖文件

XzBot 现在采用“一个主配置 + 多个可选覆盖文件”的方式：

1. 普通场景只改：
   - `./config/config.toml`
2. 高级场景可以额外新建：
   - `./config/ai.toml`
   - `./config/persona.toml`
   - `./config/search.toml`
   - `./config/network.toml`
   - 以及 `server.toml` / `group.toml` / `policy.toml` / `owner.toml`

这些覆盖文件现在会自动生成模板，并且每个文件默认都带：

```toml
enabled = false
```

含义是：

- `false`：整份覆盖文件不生效
- `true`：这份文件开始覆盖主配置里对应的部分

加载顺序是：

1. `config.toml`
2. 其他同目录覆盖文件（若存在）

后读取的文件会覆盖前面的同名字段。

这意味着你可以渐进式地拆配置：

1. 先只改 `config.toml`
2. 某一块变长了，比如 persona
3. 再把它搬到 `persona.toml`
4. 把 `persona.toml` 的 `enabled` 改成 `true`

例如，你可以把日常常改的配置留在 `config.toml`，再把很长的人设拆到 `persona.toml`：

```toml
[persona]
system = "主配置里的简短默认人设"
```

然后在 `./config/persona.toml` 里覆盖：

```toml
[persona]
system = """
这里放完整的人设长文本
"""

[[persona.group_overrides]]
groups = [970199915]
system = """
这个群使用单独的人设
"""
```

分群人设机制没有变化，仍然使用 `[[persona.group_overrides]]`。

### 旧版单文件配置迁移

如果你是从旧版本升级而来，程序会自动检查旧布局：

```text
./config.toml
```

当新的主配置：

```text
./config/config.toml
```

不存在时，XzBot 会自动做一次“结构迁移”：

1. 读取旧版单文件配置
2. 把这些部分拆到独立覆盖文件并自动启用：
   - `persona -> config/persona.toml`
   - `ai -> config/ai.toml`
   - `search -> config/search.toml`
   - `network -> config/network.toml`
3. 把剩余部分写入新的主配置：
   - `config/config.toml`

这样迁移后的主配置会更轻，复杂配置会自动分流到对应文件里。

旧版 `config.toml` 不会被删除，你仍然可以自行备份或手动清理。

**NapCat 反向 WS 连接地址：**
```
ws://<主机IP>:3000/onebot/v11/ws
```

> **安全提示：** 真实 `api_key` / OCR token 只写入运行时配置（已被 `.gitignore` 忽略），切勿提交到 Git。

### 搜索配置

- 内置 Bing（中国版）
- 自建 SearXNG（启用后不再使用内置搜索）
- 天气查询场景下，SearXNG 结果会优先 `tianqi.2345.com`，并自动附带天气页预览

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

### OpenAI Responses API 配置

如果你的网关 **不兼容** `/chat/completions`，而是走新版 `/responses`，这样配：

```toml
[ai]
provider = "openai_compatible"
base_url = "https://cacode-sub2api-dev.hf.space/v1"
wire_api = "responses"
api_key = "YOUR_API_KEY"
model = "gpt-5.4"
fallback_models = ["gpt-4.1", "gpt-4o-mini"]
reasoning_effort = "xhigh"
disable_response_storage = true
temperature = 0.7
max_tokens = 4096
timeout_ms = 60000
```

说明：

1. `base_url` 填 API 根路径即可，XzBot 会自动补成 `/responses`
2. 若你已经拿到完整地址，也可以直接填完整的 `.../responses`
3. 老接口仍用 `wire_api = "chat_completions"`
4. `disable_response_storage` 和 `reasoning_effort` 只对支持该字段的网关生效

### OpenAI Chat Streaming 兼容

有些第三方 OpenAI 兼容网关在 `/chat/completions` 下只有 `stream=true` 时才会稳定返回正文。
这类网关可以这样配：

```toml
[ai]
provider = "openai_compatible"
base_url = "https://your-gateway.example/v1"
wire_api = "chat_completions"
stream_chat_completions = true
api_key = "YOUR_API_KEY"
model = "gpt-5.4"
```

XzBot 会在内部把 SSE 流式块重新拼成普通的 Chat Completions 返回结构，所以对上层会话、工具调用和日志行为保持不变。

### 多模型回退

如果同一个 provider / base_url / api_key 下有多个模型可用，可以配置一条回退链：

```toml
[ai]
provider = "openai_compatible"
base_url = "https://your-gateway.example/v1"
api_key = "sk-xxx"
model = "gpt-5.4"
fallback_models = ["gpt-4.1", "gpt-4o-mini"]
```

当前行为：

1. 先尝试 `model`
2. 当前模型请求失败时，再按顺序尝试 `fallback_models`
3. 全部失败后：
   - 如果都是超时 / 429 / 5xx / upstream 这类瞬时错误，返回统一“网不好”提示
   - 如果是确定性的配置或请求错误，保留原始报错

### 代理配置

所有 HTTP 请求默认走同一代理（LLM / 搜索 / OCR / fetch_url 等）。
如需 **仅让 Paddle OCR 直连**，可设置 `paddle_use_proxy=false`。

### Web 控制面板

如果你不想继续手改一堆 TOML，可以开启内置控制面板：

```toml
[web_admin]
enabled = true
token = "请换成你自己的强随机 token"
title = "XzBot Console"
```

启动后访问：

- `http://<host>:<port>/admin`

当前控制面板支持：

1. 登录鉴权
2. 查看当前 provider / model / WS 连接状态
3. 编辑主配置和所有覆盖文件
4. 在“插件”页查看已加载插件、命令、事件订阅和工具
5. 在“插件”页用文件管理器方式浏览并编辑 `Plugins/<插件名>/` 目录里的小体积文本文件
6. 查看最近日志
7. 保存配置后直接触发 `/reload` 对应的运行时热重载

说明：

1. 控制面板使用同一个 HTTP 服务，不额外开端口
2. 修改 `server.host / server.port / server.ws_path` 这类监听项后，面板会提示你“需要重启进程后完全生效”
3. 插件文件不会再混进“配置”页；“配置”页只放核心配置和覆盖文件
4. 插件文件管理器只会扫描 `Plugins/<插件名>/` 目录里的小体积文本文件，不会把插件二进制本体或明显的二进制文件放进编辑器
5. 控制面板本身不引入第二套配置来源，保存后仍然是直接写回磁盘文件

### URL 抓取策略

`fetch_url` 采用分层抓取：
1. 若识别为知乎问题/回答/专栏链接，优先构造知乎公开 API 地址并直接读取 JSON
2. 其他站点走浏览器风格请求（完整请求头）
3. 若失败或命中 JS 门页 / 反爬页，优先尝试无头浏览器抓取：
   - 首选 `lightpanda`
   - 回退 `chromium` / `google-chrome`
4. 仍失败时，最后回退 `reader proxy` 抓取正文

知乎这样处理的原因很直接：普通静态抓取对知乎问题页非常容易命中 403 或 JS 壳页面，
而公开 API 对问题、回答、专栏内容更稳定，也更适合模型读取。

无头浏览器链路的目标不是替代所有普通请求，而是专门处理：
- 强依赖 JS 渲染的页面
- 只返回占位壳页面的站点
- 普通 HTTP 抓取明显不可靠的页面

其中：
- `lightpanda` 作为轻量高性能浏览器优先尝试
- `chromium` 作为兼容性兜底

支持的知乎页面类型：
- `https://www.zhihu.com/question/<id>`
- `https://www.zhihu.com/question/<id>/answer/<id>`
- `https://zhuanlan.zhihu.com/p/<id>`
- `https://www.zhihu.com/api/v4/questions/<id>/answers?...`
- `https://www.zhihu.com/api/v4/answers/<id>...`
- `https://www.zhihu.com/api/v4/articles/<id>...`

如果用户直接给的是知乎 API 链接，XzBot 会优先保留其中的 `limit` / `offset`，
并自动补齐适合正文抓取的 `include` 字段，避免因为字段不全只拿到“空壳 JSON”。

原来的通用链路仍保留，所以知乎 API 临时不可用时，会继续退回普通抓取。

这样在动态站点、反爬站点和知乎这类高门槛页面上稳定性更高。

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
| `/reload`（owner） | 重载配置与插件 |
| `/log [N]`（owner） | 导出最近 N 行日志并发送文件（默认 100） |
| `/posttoken create/show/regen/delete`（owner） | 管理当前会话的外部推送 token |

## 外部推送 API

先由 owner 在目标会话创建 token：

- 私聊：`/posttoken create`
- 群聊：`@机器人 /posttoken create`（token 会私聊发给 owner）

然后调用 HTTP 接口（默认监听同服务端口）：

- `POST /api/post/send`
- `Content-Type: application/json`

请求体示例：

```json
{
  "token": "YOUR_CHAT_TOKEN",
  "message": "你好，这是外部推送",
  "image": "https://example.com/a.jpg",
  "file_path": "/abs/path/report.md",
  "file_name": "report.md"
}
```

字段说明：

- `token`：必填，聊天绑定 token
- `message`：可选，文本消息
- `image` / `images`：可选，图片引用（字符串或字符串数组）
- `file_path`：可选，上传文件路径
- `file_name`：可选，上传显示名

至少需要提供 `message`、`image/images`、`file_path` 中的一项。

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
{
  "name": "my-plugin",
  "commands": ["hello"],
  "subscriptions": ["mention", "image"],
  "tools": [
    {
      "name": "repo_analyze",
      "description": "分析 GitHub 仓库并返回摘要",
      "input_schema": {
        "type": "object",
        "properties": {
          "repo": { "type": "string" }
        },
        "required": ["repo"]
      }
    }
  ],
  "timeout_ms": 20000,
  "priority": 100
}
```

现在的插件既可以做命令型扩展，也可以订阅事件，或向 AI 注册新工具：

- `commands`：处理 `/hello`
- `subscriptions`：监听 `group_message` / `mention` / `image` / `quote` 等事件
- `tools`：向 AI 暴露新的 function call

通信方式：stdin/stdout 按行 JSON（必须回传 `request_id`）。
插件新版推荐返回 `actions` 数组，支持按顺序发送：

- `message`
- `image`
- `file`

也支持 `stop_propagation=true` 阻断后续 AI/插件处理。
旧版 `reply` / `file_path` / `image_path` 协议仍兼容。
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
