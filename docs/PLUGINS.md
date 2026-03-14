# XzBot 插件开发文档（事件驱动宿主）

XzBot 的插件由主进程统一拉起并长期管理。  
插件不是“自己连 OneBot 的机器人”，而是宿主平台上的托管扩展。

当前插件系统支持三类能力：

- 命令型插件：处理 `/analyze`、`/hello` 这类 slash command
- 事件型插件：订阅 `group_message`、`mention`、`image`、`quote` 等事件
- 工具型插件：向 AI 暴露新的 function call，由模型按需调用

宿主边界保持克制：

- 插件不能直接调用任意 OneBot action
- 插件只能返回声明式动作，由 XzBot 代发消息 / 图片 / 文件
- 插件拿到的是只读事件快照，不是主程序内部对象

---

## 1. 目录结构

```text
Plugins/
├── my-plugin            # 插件二进制文件
└── my-plugin/           # 插件自己的配置 / 缓存 / 临时文件目录
```

说明：

- 插件二进制直接放在 `Plugins/` 下
- 插件数据目录固定为 `Plugins/<plugin_name>/`
- XzBot 启动和 `/reload` 时都会扫描 `Plugins/`

---

## 2. Manifest（插件清单）

插件需要支持：

```bash
./my-plugin --manifest
```

并输出 JSON：

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

字段说明：

- `name`：插件名称
- `commands`：该插件接管的 `/命令`
- `subscriptions`：该插件订阅的事件类型
- `tools`：注册给 AI 的工具列表
- `timeout_ms`：单次调用超时，默认 `15000`
- `priority`：优先级，值越大越先处理，默认 `0`

兼容行为：

- 如果不提供 manifest，XzBot 会回退到：
  - `name = <文件名>`
  - `commands = [<文件名>]`
- 老插件只声明 `commands` 也能继续运行

---

## 3. 事件类型

`subscriptions` 可声明这些值：

- `message`
- `group_message`
- `private_message`
- `mention`
- `image`
- `quote`
- `owner`
- `command`
- `*`（订阅全部事件）

说明：

- `mention`：消息中 @ 了机器人
- `image`：消息正文里包含图片引用
- `quote`：消息引用了上一条消息
- `owner`：消息发送者是 owner
- `command`：消息正文是 slash command

插件只会收到自己声明过的事件。

---

## 4. IPC 协议（stdin / stdout）

插件启动后保持常驻，通过 stdin / stdout 按行传 JSON。

### 4.1 宿主发给插件的请求

```json
{
  "request_id": "my-plugin-1",
  "kind": "event",
  "command": "",
  "args": "",
  "raw_text": "[CQ:at,qq=123] 帮我看看这个",
  "text": "帮我看看这个",
  "message_type": "group",
  "user_id": 10001,
  "group_id": 20001,
  "self_id": 30001,
  "display_name": "leeinx",
  "mentioned": true,
  "is_owner": false,
  "event_types": ["message", "group_message", "mention"],
  "image_urls": [],
  "image_files": [],
  "reply_message_ids": [123456],
  "tool_name": null,
  "tool_arguments": null,
  "config_dir": "Plugins/my-plugin"
}
```

字段说明：

- `request_id`：请求唯一 ID，响应时原样带回
- `kind`：`command` / `event` / `tool`
- `command`：命令模式下的命令名；工具模式下也会回填工具名，便于兼容旧插件
- `args`：命令参数；工具模式下为 JSON 参数的字符串形式
- `raw_text`：原始消息文本
- `text`：宿主清洗后的文本（已去掉 @ 机器人等宿主层噪声）
- `display_name`：发送者显示名
- `mentioned`：是否 @ 了机器人
- `is_owner`：是否为 owner
- `event_types`：本次命中的事件标签
- `image_urls` / `image_files`：解析出的图片引用
- `reply_message_ids`：引用消息 ID 列表
- `tool_name` / `tool_arguments`：仅工具模式下有值
- `config_dir`：插件自己的工作目录

### 4.2 插件返回给宿主的响应

新版推荐直接返回 `actions`：

```json
{
  "request_id": "my-plugin-1",
  "actions": [
    {
      "type": "file",
      "file_path": "report.md",
      "file_name": "report.md"
    },
    {
      "type": "message",
      "text": "报告已生成并上传。",
      "mention_sender": true
    }
  ],
  "stop_propagation": true
}
```

支持的动作：

#### `message`

```json
{
  "type": "message",
  "text": "Hello",
  "mention_sender": true
}
```

#### `image`

```json
{
  "type": "image",
  "image_path": "out/chart.png",
  "caption": "趋势图",
  "mention_sender": false
}
```

或：

```json
{
  "type": "image",
  "image_url": "https://example.com/a.png"
}
```

#### `file`

```json
{
  "type": "file",
  "file_path": "report.md",
  "file_name": "result.md"
}
```

路径规则：

- 绝对路径：直接使用
- 相对路径：相对于 `config_dir` 解析

控制字段：

- `stop_propagation = true`
  - 表示该插件已经“吃掉”本次事件
  - 宿主不再继续把这条消息交给后续插件 / 命令插件 / AI

- `tool_result`
  - 工具型插件建议直接返回这个字段
  - 宿主会把它作为 function call 的结果喂回模型

### 4.3 兼容旧协议

老插件仍可返回这些旧字段：

```json
{
  "request_id": "my-plugin-1",
  "reply": "Hello world",
  "mention_sender": true,
  "file_path": "report.md",
  "file_name": "report.md",
  "image_path": "image.png",
  "image_url": null
}
```

宿主会自动把这些旧字段折叠成新版 `actions`，无需立刻迁移旧插件。

---

## 5. 三种插件模式

### 5.1 命令型插件

当用户发送：

```text
/hello world
```

宿主会把：

- `kind = "command"`
- `command = "hello"`
- `args = "world"`

发给声明了 `commands=["hello"]` 的插件。

### 5.2 事件型插件

事件型插件通过 `subscriptions` 订阅消息事件。  
它适合做：

- 监听 @ 机器人
- 监听图片消息
- 监听引用消息
- 自动欢迎 / 自动审核 / 自动复读之外的逻辑

### 5.3 工具型插件

工具型插件会把自己的 `tools` 暴露给 AI。

流程：

1. 插件在 manifest 中声明工具和 JSON Schema
2. 宿主把这些工具注册到 LLM 请求里
3. 模型选择调用某个工具
4. 宿主把调用参数转成 `kind = "tool"` 请求发给插件
5. 插件返回 `tool_result`
6. 宿主把结果回填给模型，生成最终回答

这类插件适合做：

- GitHub / 仓库分析
- 外部平台 API 对接
- 业务系统查询
- 特定站点抓取或转换

---

## 6. 日志与生命周期

- 插件进程由 XzBot 启动并常驻
- 插件异常退出后，宿主会在下次调用时自动重启
- 插件 `stderr` 会被转发到 XzBot 控制台日志，格式类似：

```text
[PLUGIN:my-plugin] ...
```

- `/reload` 会：
  - 重新扫描 `Plugins/`
  - 重建插件索引
  - 重启旧插件进程

---

## 7. 开发建议

建议插件遵守这些原则：

- stdout 只输出协议 JSON，不要混日志
- 日志写 stderr
- 结果尽量通过 `actions` 明确表达，不要靠字符串约定
- 事件插件如果只是旁路观察，不要随意 `stop_propagation`
- 工具插件要返回稳定、短小、可复用的 `tool_result`

---

## 8. 本地调试

### 调试命令型插件

```bash
echo '{"request_id":"demo-1","kind":"command","command":"hello","args":"world","raw_text":"/hello world","text":"hello world","message_type":"private","user_id":1,"group_id":null,"self_id":2,"display_name":"tester","mentioned":false,"is_owner":false,"event_types":["message","private_message","command"],"image_urls":[],"image_files":[],"reply_message_ids":[],"tool_name":null,"tool_arguments":null,"config_dir":"./Plugins/my-plugin"}' | ./my-plugin
```

### 调试工具型插件

```bash
echo '{"request_id":"demo-2","kind":"tool","command":"repo_analyze","args":"{\"repo\":\"owner/repo\"}","raw_text":"","text":"","message_type":"","user_id":0,"group_id":null,"self_id":0,"display_name":"","mentioned":false,"is_owner":false,"event_types":[],"image_urls":[],"image_files":[],"reply_message_ids":[],"tool_name":"repo_analyze","tool_arguments":{"repo":"owner/repo"},"config_dir":"./Plugins/my-plugin"}' | ./my-plugin
```

---

如果你要写 Rust 插件模板，建议直接做成：

- `--manifest` 输出 manifest
- 常驻读 stdin
- 按 `request_id` 回 JSON

如果需要，我可以再给一份最小 Rust 插件模板。
