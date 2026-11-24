# codex-2-agy

无损、严格只读地将 **Codex（Codex App + Codex CLI，`~/.codex/`）** 的项目、历史会话（含 `.jsonl.zst` zstd 压缩日志与内嵌 Base64 截图）、子代理（Subagent）与全文搜索索引迁移到 **Antigravity（`agy`，`~/.gemini/`）**。

## 核心特性

- **严格只读 Codex**：通过 SQLite `?mode=ro&immutable=1` 打开 `state_5.sqlite` / `thread_history_1.sqlite`，在内存中流式解压 `.jsonl.zst`，绝不向 `~/.codex/` 写入或修改任何字节。
- **活体进程安全锁（防自杀）**：向真实 `~/.gemini` 写入、清理或回滚前，自动检测运行中的 Antigravity 桌面端与 `language_server` 进程（支持 macOS / Linux / Windows）；如果 `agy` 正在运行，立即拦截并退出（返回码 `2`）。
- **历史 `mtime` 精确同步 + Antigravity 500 会话硬上限保护**：
  - Antigravity `language_server` 启动时对 `~/.gemini/antigravity/conversations/*.db` 硬编码了 **500 个文件上限**（按 OS 文件系统 `mtime` 淘汰最旧会话，且主会话与独立 Subagent 合并计数）。
  - 本工具写入每个 `.db`（以及 `.pbtxt` 和 `brain/<cid>`）后强制调用 `os.utime()` 同步为该会话在 Codex 里的**真实历史修改时间**，确保历史会话绝不会覆盖/挤掉较新的原生 agy 会话。
  - 每次迁移自动扣除**预留槽位（可通过 `--reserve-slots N` 自定义，默认 `50`，即总目标上限 `450/500`）**与磁盘已有 `.db` 占用，按 Codex 会话**更新时间倒序**自动选择可迁入的最新会话；同时在写入层和校验层均有硬上限拦截。
- **零槽位内联 Subagent（默认 `--subagent-mode inline`）**：
  - 默认不再为 Codex `thread_spawn` 子代理单独创建 `.db`（避免 400+ 子代理吃光 500 槽位），也不产生会报 `Conversation unavailable` 的死链 `AgentRef`。
  - 在父会话对应步骤中内联渲染折叠卡片（任务指令、执行摘要、最终结论），并将子代理完整对话过程无损存档为 `brain/<父会话id>/subagents/<role>_<id>.md`（不计入 500 槽位）。
  - 如需为少数重点会话生成原生独立子代理面板，可显式指定 `--subagent-mode separate`；也可使用 `--subagent-mode skip` 跳过子代理。
- **项目目录内独立备份与状态管理（不污染 `~/.gemini/antigravity/`）**：
  - 所有迁移备份快照与清单默认保存在**当前项目目录** `./migration_state/` 下（`migration_state/backups/<run_id>` 与 `migration_state/manifest.json`，可通过 `--state-dir` 自定义），绝不在 `~/.gemini/antigravity/` 中写入工具元数据。
  - 内置 `clean-orphans` 子命令，可一键清理无对应 `.db` 的孤儿索引、锁文件，并将旧版遗留在 `~/.gemini/antigravity/` 下的备份迁回 `./migration_state/legacy_backups/`。
- **智能合并已有 agy 项目**：
  - 按规范化目录路径匹配现有 Antigravity 项目（自动忽略 `antigravity-vitals` / `swarm_board` 等 sidecar 项目）。
  - 目录一致的直接复用原有 agy 项目 UUID、名称与权限设置；如 Codex 侧包含额外根目录，自动追加到原项目 `projectResources.resources`。
  - Codex 普通对话（`projectless` 或未归属任何项目）原样映射为 agy 的普通对话（`project_id = "outside-of-project"`）。
- **尽可能无损的数据映射**：
  - **`.jsonl.zst` 双源融合**：结合 `thread_history_1.sqlite` 结构化 item 与 `.jsonl.zst` 原始 rollout，还原过期截图到 `brain/<cid>/.user_uploaded/`，补充 `token_usage_record` 与模型名；缺失 history DB 记录时自动走多轮 rollout 回退解析。
  - **全量工具与 UI 卡片**：支持全部 16 种 Codex item 类型，严格保持 `PLANNER_RESPONSE.tool_calls` ↔ `GENERIC` 顺序配对，支持迁移后继续追问。
  - **Codex 专有指令与思考过程清洗**：自动将 Codex App 专有的 `::code-comment{...}`、`::codex-file-citation{...}` 转换为标准 Markdown 卡片与可点击 `file://` 链接，剥离 `::git-commit` / `::git-stage` 等桌面端按钮占位符；自动将延迟落盘在最终回答后的同轮 `reasoning` 归并回回复上方，并清洗重复思考标题与 `<!-- -->` 分隔符。
  - **大输出不截断 + FTS5 搜索 + Transcript**：超过 32KB 的输出自动落盘至 `brain/<cid>/.system_generated/steps/<idx>/output.txt` 并写入 `full_output_uri`；同步生成 `conversations_fts`（FTS5 全文检索）、`annotations/<cid>.pbtxt` 与 `transcript_full.jsonl` / `transcript.jsonl`。
- **零额外依赖**：仅使用 Python 3.10+ 标准库；解压 `.jsonl.zst` 时按 Python 3.14+ 内置 `compression.zstd` → `zstandard` 包 → 系统 `zstd` CLI 依次自动回退。

## 目录结构

```text
codex-2-agy/
├── codex-2-agy                  # 可直接运行的 CLI 入口
├── pyproject.toml               # 包配置
├── migration_state/             # 运行后自动生成：备份快照 backups/ 与迁移清单 manifest.json
├── codex_2_agy/
│   ├── __main__.py              # 支持 `python3 -m codex_2_agy`
│   ├── protobuf_wire.py         # 零依赖 Protobuf 二进制编解码器
│   ├── agy_proto.py             # Antigravity Protobuf Schema 构造器
│   ├── agy_store.py             # agy 项目 / 会话 DB / mtime 同步 / 500 锁 / 孤儿清理
│   ├── codex_reader.py          # 严格只读 Codex 读取器（含 .jsonl.zst 流式解压）
│   ├── mapper.py                # 16 种 Codex item + inline/separate Subagent 无损映射器
│   ├── project_merger.py        # 项目合并、9 级归属解析与 500 槽位倒序预算规划器
│   ├── safety.py                # 进程安全锁、原子写、快照备份与回滚
│   ├── verify.py                # SQLite / 500 上限 / mtime / Protobuf / 配对 / 媒体 / FTS5 校验器
│   └── cli.py                   # plan / migrate / verify / clean-orphans / rollback / inspect
└── tests/                       # 单元测试与端到端合成集成测试
```

## 快速开始

> **重要提醒**：
> 1. 正式写入、清理或回滚真实 `~/.gemini` 前，**务必先完全退出 Antigravity（macOS 按 `Cmd+Q`）**，再在系统终端（Terminal / iTerm2 / Shell）中运行命令。
> 2. 如果 Codex 正在运行，建议也先退出或静置几秒，确保其 SQLite WAL 已刷盘，以读到最新一轮对话。

进入项目目录：

```bash
cd codex-2-agy
```

### 1. 只读预览槽位预算与迁移计划（随时可跑，无需退出 agy）

```bash
python3 -m codex_2_agy plan
```

### 2. 清理历史孤儿索引与旧版残留（请先 `Cmd+Q` 退出 Antigravity）

```bash
python3 -m codex_2_agy clean-orphans --dry-run   # 先预览
python3 -m codex_2_agy clean-orphans             # 执行清理并将旧备份移回 ./migration_state/
```

### 3. 执行安全迁移（默认 `inline` Subagent + 预留 50 个槽位 + 倒序选最新会话）

请先 `Cmd+Q` 退出 Antigravity：

```bash
python3 -m codex_2_agy migrate
# 如需自定义预留槽位数（例如仅预留 10 个槽位，最多迁入至 490/500）：
python3 -m codex_2_agy migrate --reserve-slots 10
```

迁移完成后工具会自动运行完整性校验（含 500 上限校验与历史 `mtime` 一致性校验），并打印保存在 `./migration_state/backups/<run_id>` 的快照目录与一键回滚命令。

### 4. 一键回滚

先 `Cmd+Q` 退出 Antigravity，再运行迁移输出中给出的回滚命令：

```bash
python3 -m codex_2_agy rollback --backup-dir ./migration_state/backups/<run_id>
```

## 常用参数

| 命令 / 参数 | 说明 |
| --- | --- |
| `plan [-v]` | 只读预览 500 槽位预算、项目合并与按时间倒序入选的会话 |
| `migrate --dry-run` | 全流程演练，不写入任何文件 |
| `plan / migrate --reserve-slots N` | 在 500 硬上限内为原生会话预留 N 个槽位（可选，默认 `50`，即上限 `450/500`） |
| `migrate --max-total N` | 直接指定迁移后总 `.db` 上限（最大 `500`） |
| `migrate --subagent-mode inline\|separate\|skip` | Subagent 策略（默认 `inline` 占 0 槽位并存档 Markdown） |
| `migrate --state-dir <dir>` | 自定义备份与清单存放目录（默认 `./migration_state`） |
| `migrate --agy-home /tmp/test-agy` | 迁移到独立测试目录（无需退出 agy） |
| `migrate --only <uuid> [...]` | 仅迁移指定 Codex 线程 |
| `migrate --project "<项目名>"` | 仅迁移单个项目（支持 `standalone` 表示普通对话） |
| `migrate --limit N` | 仅迁移最近 `N` 条顶层会话（按更新时间倒序） |
| `migrate --update-existing` | 增量同步在 Codex 端有新改动的已迁会话 |
| `clean-orphans [--dry-run]` | 清理无对应 `.db` 的孤儿索引/锁文件并迁移旧版残留备份 |
| `verify [--only <uuid>]` | 独立校验 500 上限、历史 `mtime`、SQLite、Protobuf、工具配对、FTS5 与媒体 |
| `inspect <uuid>` | 查看单条 Codex 线程的解析详情 |

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 兼容性与免责声明

- 本工具基于对当前 Codex（`~/.codex/`）与 Antigravity（`~/.gemini/`）本地 SQLite / Protobuf 存储结构的逆向适配，与 OpenAI 或 Google 无官方关联。
- 客户端升级后内部 Schema 或会话上限策略可能发生变化。首次迁移或客户端大版本升级后，建议先使用 `plan -v` 和 `--agy-home /tmp/test-agy` 演练，并妥善保留 `./migration_state/backups/` 快照以便随时回滚。
