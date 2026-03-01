# XzBot 插件开发文档（托管进程）

XzBot 的插件**由主进程启动并长期管理**（类似 MC/Spigot 插件的生命周期）。  
插件为独立可执行文件，放在 `Plugins/` 目录下，运行期间保持常驻。

---

## 1. 目录结构（MC 风格）

```
Plugins/
├── my-plugin            # 插件二进制文件（直接放在 Plugins 下）
└── my-plugin/           # 插件自己使用的目录（配置/缓存/日志）
    └── ...
```

说明：
- **二进制文件**直接放 `Plugins/`
- **插件数据目录**为 `Plugins/<plugin_name>/`（由插件自行管理）

---

## 2. 插件发现与清单

XzBot 启动时扫描 `Plugins/` 下所有**可执行文件**。

插件需支持：

```
<plugin_binary> --manifest
```

并输出 JSON 清单：

```json
{
  "name": "my-plugin",
  "commands": ["hello", "ping"],
  "timeout_ms": 20000
}
```

字段说明：
- `name`：插件名称（用于目录与日志）
- `commands`：触发命令（用户输入 `/hello` / `/ping`）
- `timeout_ms`：单次请求超时（可选，默认 15000ms）

如未提供清单，XzBot 会退回到默认策略：
- `name = <文件名>`
- `commands = [<文件名>]`

---

## 3. 通信协议（stdin / stdout）

插件启动后保持常驻，通过 stdin/stdout 进行 JSON 行通信。

### 输入（stdin，每行一个 JSON）

```json
{
  "request_id": "my-plugin-1",
  "command": "hello",
  "args": "world",
  "raw_text": "/hello world",
  "message_type": "group",
  "user_id": 123,
  "group_id": 456,
  "self_id": 789,
  "config_dir": "Plugins/my-plugin"
}
```

字段说明：
- `request_id`：请求唯一 ID（必须回传）
- `command`：命令名（不含 `/`）
- `args`：命令后参数（原样）
- `raw_text`：原始消息
- `message_type`：`group` / `private`
- `config_dir`：插件数据目录

### 输出（stdout，每行一个 JSON）

```json
{
  "request_id": "my-plugin-1",
  "reply": "Hello world",
  "mention_sender": true
}
```

- `request_id`：必须与输入一致
- `reply`：回复文本
- `mention_sender`：是否 @ 发送者（可选）

**发送文件（可选）**
```json
{
  "request_id": "my-plugin-1",
  "file_path": "report.md",
  "file_name": "report.md"
}
```

- `file_path`：要发送的文件路径（相对路径会基于 `config_dir` 解析）
- `file_name`：发送时展示的文件名（可选，默认取文件名）

> 仅当插件严格输出 JSON 行，XzBot 才能正确匹配请求。  
> 请勿在 stdout 输出日志，日志请写入 stderr 或文件。  
> XzBot 会把插件 **stderr** 输出转发到控制台（带前缀）。

---

## 4. 触发规则

- 只响应以 `/` 开头的命令
- 命令名必须在 `manifest.commands` 中
- 若群聊配置 `require_at = true`，插件同样要求 @ 机器人

---

## 5. 生命周期

1. XzBot 启动 → 扫描插件 → 逐个启动进程  
2. 插件常驻，等待 stdin 输入  
3. 插件异常退出 → 下次请求时自动拉起  

> 使用 `/reload` 会重新扫描 `Plugins/` 并重启插件进程，实现热重载。

---

## 6. 示例：/analyze 插件

目标：支持 `/analyze <repo>`

流程：
1. 校验 GitHub URL
2. 临时目录 `git clone --depth 1`
3. 检查大小（超限拒绝）
4. `fuck-u-code analyze <repo> --format markdown --locale zh`
5. 输出 Markdown 结果
6. 删除临时目录

---

## 7. 本地调试

可直接模拟输入：

```bash
echo '{"request_id":"demo-1","command":"hello","args":"world","raw_text":"/hello world","message_type":"private","user_id":1,"group_id":null,"self_id":2,"config_dir":"./my-plugin"}' | ./my-plugin
```

---

如果你需要 **Rust 插件模板**，我可以直接生成一个最小实现。  
只要告诉我插件名和命令即可。
