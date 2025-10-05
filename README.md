# Text2Dialog

> 将长文本（如小说、剧本、纪实文本）自动抽取为结构化的**角色对话和引用关系**，并一键完成**质量校验 → 角色对配对 → ChatML 数据集导出**。提供命令行、FastAPI 服务与可视化前端（含一键启动器）。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110%2B-009688" />
  <img alt="OpenAI SDK" src="https://img.shields.io/badge/SDK-openai%20compatible-5b9bd5" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen" />
</p>

---

## ✨ 功能亮点

- **长文本分块**：按 Token 上限智能分块，支持**跨块重叠上下文**，减轻切断语义带来的误判。
- **多平台 LLM 兼容**：通过 OpenAI 兼容 SDK 适配多个平台（DeepSeek、硅基流动 SiliconFlow、阿里云百炼/通义、Kimi/Moonshot、OpenAI、Gemini、AWS Bedrock、自定义 BaseURL）。
- **高质量抽取**：统一提示词与 TypeScript 风格 schema，输出 `[{role, dialogue, reply}]`；自动剥离推理型模型的“思考”前缀。
- **引用关系(reply)**：`reply.target_index` 仅在**同一 chunk 内向前引用**，可配置回溯窗口与置信度阈值。
- **并发与续跑**：多线程并发处理、**断点续跑**、进度与 ETA 估算，支持**暂停/继续/取消**。
- **全链路工具**：严格**校验器** → **角色对配对**（A→B / B→A）→ **ChatML 导出**（支持 pair 模式与多轮 stitch 模式）。
- **一键启动器**：`launcher.py` 图形化：创建虚拟环境、安装依赖、启动/停止服务、写入 `.env`、打开前端和帮助文档。

---

## 🗂 目录结构

```
Text2Dialog/
├─ launcher.py                 # Tk GUI：配置环境、管理 FastAPI、打开前端/文档
├─ run_server.(sh|bat)         # 一键启动 uvicorn 服务
└─ text2dialog/
   ├─ server.py                # FastAPI：作业管理、抽取/校验/配对/导出、静态前端
   ├─ dialogue_chain.py        # 核心对话抽取：分块/并发/重试/校验/进度/续跑/控制
   ├─ config.py                # 配置与平台抽象（默认模型、提示词模板、schema 等）
   ├─ validate_output.py       # 输出校验：结构、索引、reply 合法性、置信度范围
   ├─ pair_dataset_builder.py  # 从 JSONL 生成“有向角色对”数据集（过滤/严格模式）
   ├─ pair_to_chatml.py        # 将配对样本转为 ChatML（pair 或 stitch 多轮）
   └─ static/                  # 可视化前端（原生 HTML/CSS/JS）
```

---

## 🚀 安装与运行

### 1) 环境要求
- Python **3.9+**（推荐 3.10–3.12）  
- `pip` 可访问 pypi.org

### 2) 一键启动（GUI）
```bash
cd Text2Dialog
python launcher.py
```
- 点击“**① 一键配置/修复环境**”：自动创建 `.venv` 并安装 `text2dialog/requirements.txt`。
- 点击“**启动服务**”，再点击“**打开前端**”，进入可视化控制台。
- 在“**保存 API 配置(.env)**”中写入平台密钥与默认模型。

### 3) 直接启动服务（CLI）
```bash
# macOS / Linux
cd Text2Dialog
bash run_server.sh

# Windows
cd Text2Dialog
.\run_server.bat
```
手动方式：
```bash
cd Text2Dialog/text2dialog
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 4) 纯命令行抽取（不启用服务/前端）
```bash
cd Text2Dialog/text2dialog

# 最简用法：输入文本 → 输出 JSONL
python dialogue_chain.py input.txt -o output.jsonl --concurrent -t 8

# 常用选项（示例）：
python dialogue_chain.py input.txt -o output.jsonl \
  --platform siliconflow --concurrent -t 8 --save-chunk-text \
  --sort-output --stats --reply-window 6 --reply-confidence-th 0.65
```

---

## 🧪 效果示例

以下为基于 **Text2Dialog** 构建数据并微调后的 LLM 一个**模型间对话**片段：

> **A**：你在等我醒来？  
> **B**：不。我在等你永远睡去。  
> **A**：好大的口气。就凭你？  
> **B**：就凭我的剑。  
> **A**：剑？这世上用剑的人太多，能杀死我的太少。  
> **B**：也许我正好是那一个。  
> **A**：为了赏金吗？  
> **B**：为了一个名字。

**由该片段抽取得到的结构化结果（节选，JSONL）：**
```json
{"chunk_id": 0, "dialogue_index": 0, "role": "A", "dialogue": "你在等我醒来？", "reply": null}
{"chunk_id": 0, "dialogue_index": 1, "role": "B", "dialogue": "不。我在等你永远睡去。", "reply": {"target_index": 0, "target_role": "A", "confidence": 0.96}}
{"chunk_id": 0, "dialogue_index": 2, "role": "A", "dialogue": "好大的口气。就凭你？", "reply": {"target_index": 1, "target_role": "B", "confidence": 0.93}}
{"chunk_id": 0, "dialogue_index": 3, "role": "B", "dialogue": "就凭我的剑。", "reply": {"target_index": 2, "target_role": "A", "confidence": 0.95}}
{"chunk_id": 0, "dialogue_index": 4, "role": "A", "dialogue": "剑？这世上用剑的人太多，能杀死我的太少。", "reply": {"target_index": 3, "target_role": "B", "confidence": 0.92}}
{"chunk_id": 0, "dialogue_index": 5, "role": "B", "dialogue": "也许我正好是那一个。", "reply": {"target_index": 4, "target_role": "A", "confidence": 0.94}}
{"chunk_id": 0, "dialogue_index": 6, "role": "A", "dialogue": "为了赏金吗？", "reply": {"target_index": 5, "target_role": "B", "confidence": 0.91}}
{"chunk_id": 0, "dialogue_index": 7, "role": "B", "dialogue": "为了一个名字。", "reply": {"target_index": 6, "target_role": "A", "confidence": 0.95}}
```

**导出的 ChatML（pair 模式，示例一对）：**
```json
{"messages": [
  {"role": "system", "content": "用B的语气回复输入内容。"},
  {"role": "user", "content": "你在等我醒来？"},
  {"role": "assistant", "content": "不。我在等你永远睡去。"}
]}
```

> 更复杂的场景可用 `--mode stitch --max-turns 4` 将同一 chunk 的连续样本拼接为多轮对话，以训练长程承接能力。

---

## 🔧 配置 LLM 平台

复制 `text2dialog/env.example` 为 `.env` 并按注释填入：

- **平台选择**：`LLM_PLATFORM=<deepseek|siliconflow|bailian|moonshot|openai|gemini|aws_bedrock|custom>`  
- **常见变量**（按平台有所不同）：`*_API_KEY`、`*_BASE_URL`（OpenAI 兼容地址）、`*_MODEL_NAME`。

示例（节选）：
```ini
# = DeepSeek =
DEEPSEEK_API="your-deepseek-api-key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL_NAME="deepseek-chat"
LLM_PLATFORM=deepseek

# = SiliconFlow =
SILICONFLOW_API_KEY="your-siliconflow-api-key"
SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
LLM_PLATFORM=siliconflow

# = OpenAI =
OPENAI_API_KEY="sk-..."
LLM_PLATFORM=openai
```
> 前端“高级设置”亦支持**临时覆盖**（平台、API Key、BaseURL、模型名），仅对本次作业生效。

---

## 🧭 推荐使用流程（前端）

1. **上传文本**（`.txt`，建议 UTF‑8）。  
2. **设置平台与模型**（可在“高级设置”里覆盖 `.env`）。  
3. 点击“**开始抽取**”，观察进度条与 ETA；支持**暂停/继续/取消**。  
4. 抽取完成后依次：**校验输出** → **角色对配对** → **导出 ChatML**。  
5. 在“下载”区获取 `extraction.jsonl`、`pair_datasets/`、`chatml.jsonl`。

---

## 📦 数据格式

### 1) 抽取结果（JSONL，每行一条）
```json
{
  "chunk_id": 12,
  "dialogue_index": 3,
  "role": "张三",
  "dialogue": "我到了。",
  "reply": {
    "target_index": 2,
    "target_role": "李四",
    "confidence": 0.91
  }
}
```
- `reply` 可为 `null` 或对象；若为对象，`target_index` **必须早于**当前 `dialogue_index` 且在**同一 chunk** 内。

### 2) 角色对样本（pair_dataset_builder.py 产出）
```json
{
  "source": {"chunk_id": 12, "dialogue_index": 2, "role": "李四", "text": "到哪儿了？"},
  "reply":  {"chunk_id": 12, "dialogue_index": 3, "role": "张三", "text": "我到了。"},
  "pair":   {"from": "李四", "to": "张三"},
  "confidence": 0.91
}
```

### 3) ChatML（SFT messages）
```json
{"messages": [
  {"role": "system", "content": "你现在扮演 张三，仅围绕文本对话。"},
  {"role": "user", "content": "到哪儿了？"},
  {"role": "assistant", "content": "我到了。"}
]}
```

---

## 🖥️ 命令行工具

### 1) 抽取（dialogue_chain.py）
常用参数：
- `-o/--output`：输出 JSONL。
- `-t/--threads` 与 `--concurrent/--no-concurrent`：并发控制（默认开启）。
- `--save-chunk-text/--no-save-chunk-text`：是否在输出中一并保存原始 chunk 文本。
- `--sort-output`：完成后按 `chunk_id` 排序写回。
- `--stats/--no-stats`：打印总体/分块统计。
- `-p/--platform`、`--list-platforms`：选择或列出平台。
- `--reply-window`：reply 可回溯窗口（仅看**本 chunk**内的更早发言）。
- `--reply-confidence-th`：低于阈值将清除 `reply`。

### 2) 角色对配对（pair_dataset_builder.py）
```bash
# 列出角色与频次
python pair_dataset_builder.py -i extraction.jsonl --list-roles

# 指定有序对（可多次指定） → 输出到目录
python pair_dataset_builder.py -i extraction.jsonl \
  --pairs "张三,李四" --pairs "李四,张三" \
  -o ./pair_datasets --min-confidence 0.80 --strict

# 也可以合并为一个文件
python pair_dataset_builder.py -i extraction.jsonl --merge-out ./all_pairs.jsonl \
  --all-ordered-pairs --roles "张三" --roles "李四" --roles "王五"
```
常用参数要点：
- `--pairs` / `--roles` + `--all-ordered-pairs`：显式或按角色集**全排列**生成有序对（自反对除外）。
- 过滤：`--min-confidence`（默认 **0.8**）、`--require-confidence`、文本长度上下限与 `--deny-pattern` 黑名单正则。
- **严格模式**（默认启用）：仅保留能**确定**为 `A→B` 的样本；`--no-strict` 可放宽到启发式回退。

### 3) ChatML 导出（pair_to_chatml.py）
```bash
# 多文件/目录/通配符输入 → 单个 ChatML JSONL
python pair_to_chatml.py -i ./pair_datasets -o ./chatml.jsonl \
  --mode pair --dedupe --min-confidence 0.85 \
  --system-template "你现在扮演 {to_role}，仅围绕文本对话。"

# Stitch 模式（同 chunk 连续样本拼接为多轮对话）
python pair_to_chatml.py -i ./pair_datasets -o ./chatml_stitch.jsonl \
  --mode stitch --max-turns 4 --include-meta
```
参数要点：
- `--mode {pair|stitch}`、`--max-turns`、`--min-confidence`、`--dedupe`、`--reverse`、`--include-meta`。
- 系统提示：`--system` 传文本或 `@path/to/file`；`--system-template` 可用 `{from_role}/{to_role}/{src_role}/{tgt_role}`。

---

## 🌐 REST API（FastAPI）

### 作业生命周期
- `POST /api/jobs/create`：表单上传 `file=@input.txt` → `{ "job_id": "..." }`
- `POST /api/jobs/{job_id}/extract`：请求体示例
  ```json
  {
    "platform": "openai|deepseek|...|custom",
    "api_key": "sk-...",
    "base_url": "https://.../v1",
    "model_name": "gpt-4o-mini",
    "concurrent": true,
    "threads": 8,
    "save_chunk_text": false,
    "sort_output": false,
    "MAX_TOKEN_LEN": 1000,
    "COVER_CONTENT": 100,
    "TEMPERATURE": 0.6,
    "REPLY_WINDOW": 6,
    "REPLY_CONFIDENCE_TH": 0.65
  }
  ```
- `GET /api/jobs/{job_id}/progress`：返回进度、速度、ETA 与状态（running/paused/cancelling/succeeded/failed/done）。
- `POST /api/jobs/{job_id}/control`：`{ "action": "pause|resume|cancel|force-cancel", "reason": "..." }`
- `GET /api/jobs/{job_id}/download?which=extract|pairs|chatml`：下载阶段性产物。

### 校验 / 配对 / 导出
- `POST /api/validate`：`{ "job_id": "...", "input_path": "可选" }` → `{ "ok": true|false, "log": "..." }`
- `POST /api/pairs`：`{ "job_id": "...", "pairs": ["A,B"], "min_confidence": 0.8, "strict": true, ... }`
- `POST /api/chatml`：`{ "job_id": "...", "inputs": ["dir|glob|file"], "mode": "pair|stitch", ... }`

> 另有 `GET /api/defaults` 用于拉取默认配置；静态前端路由位于 `/` 与 `/static/*`。

**最小可用 cURL**：
```bash
# 1) 上传并创建作业
curl -F "file=@input.txt" http://localhost:8000/api/jobs/create

# 2) 触发抽取（填入 job_id 与你的平台参数）
curl -X POST http://localhost:8000/api/jobs/<job_id>/extract \
  -H "Content-Type: application/json" \
  -d '{"platform":"siliconflow","api_key":"...","base_url":"https://api.siliconflow.cn/v1","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507","threads":8,"concurrent":true}'

# 3) 轮询进度
curl http://localhost:8000/api/jobs/<job_id>/progress
```

---

## ⚙️ 实现要点

- **分块**：`tiktoken` 估算 Token（默认 `cl100k_base`）；`MAX_TOKEN_LEN` 与 `COVER_CONTENT` 支持配置。
- **调用栈**：优先使用 OpenAI **Responses API**（可用时），否则回退到 **Chat Completions**。
- **鲁棒性**：自动重试、统一剥离“思考/推理”前缀、解析双通道输出（reasoning vs content）。
- **并发写回**：内部缓冲 + `next_expected_chunk_id` 确保输出顺序稳定。
- **进度持久化**：`.cache/progress.json`；控制文件 `.cache/control.json` 支持暂停/继续/取消。
- **校验规则**（节选）：
  - `dialogue_index`：**从 0 连续递增**；重复或跳号判错。
  - `role` 与 `dialogue`：**非空**。
  - `reply`：`null` 或对象；若为对象：`target_index` **早于当前**且位于同 chunk；`confidence ∈ [0,1]`。

---

## ❓FAQ / 疑难排查

- **输出为空或换行异常**：确认输入 `.txt` 的编码（推荐 UTF‑8），并检查前处理是否删除了所有换行。
- **模型频繁报错/限流**：调低并发 `-t/--threads`，增大重试间隔；确认 `base_url` 与 `model_name` 是否匹配平台。
- **reply 错配**：适当调小 `REPLY_WINDOW`，或提高 `REPLY_CONFIDENCE_TH`；必要时在配对阶段启用 `--strict` 并提高 `--min-confidence`。
- **跨 chunk 引用**：当前 `reply` **只在本 chunk 内向前引用**；跨块关系不在范围内。


## 📄 许可（License）

本项目基于 **MIT License** 开源发布。

