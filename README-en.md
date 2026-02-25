# Text2Dialog: Converting Long Texts into Trainable Dialogue Data

文档语言 / Languages / 言語：
[简体中文](./README.md) · [English](./README-en.md) · [日本語](./README-ja.md)


> Automatically extract long-form text (e.g., novels, screenplays, nonfiction) into structured character dialogues with reply links, and complete: quality validation → role pairing → ChatML dataset export in one click. Provides a command line tool, FastAPI service, and a visual frontend (with a one‑click launcher).

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110%2B-009688" />
  <img alt="OpenAI SDK" src="https://img.shields.io/badge/SDK-openai%20compatible-5b9bd5" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen" />
</p>

---

## ✨ Highlights

- **Long‑text chunking**: intelligently splits by token limits and supports overlapping context across chunks to reduce errors caused by hard cuts in semantics.
- **Multi‑platform LLM compatibility**: via an OpenAI‑compatible SDK, supports multiple platforms (DeepSeek, SiliconFlow, Alibaba Bailian/Tongyi, Kimi/Moonshot, OpenAI, Gemini, AWS Bedrock, custom BaseURL).
- **High‑quality extraction**: unified prompts and a TypeScript‑style schema, outputting `[{role, dialogue, reply}]`; automatically strips “thinking” prefixes from reasoning‑style models.
- **Reply links (`reply`)**: `reply.target_index` only points backward within the same chunk; configurable look‑back window and confidence threshold.
- **Concurrency & resume**: multithreaded processing, checkpoint resume, progress & ETA estimation; supports pause / resume / cancel.
- **End‑to‑end tooling**: strict validator → role‑pair builder (A→B / B→A) → ChatML export (supports pair mode and multi‑turn stitch mode).
- **One‑click launcher**: GUI `launcher.py` to create a virtualenv, install dependencies, start/stop the service, write `.env`, and open the frontend & docs.

---

## 🗂 Project Structure

```
Text2Dialog/
├─ launcher.py                 # Tk GUI: configure env, manage FastAPI, open frontend/docs
├─ run_server.(sh|bat)         # One-click start for uvicorn
└─ text2dialog/
   ├─ server.py                # FastAPI: job mgmt, extract/validate/pair/export, static frontend
   ├─ dialogue_chain.py        # Core dialogue extraction: chunking/concurrency/retry/validation/progress/resume/control
   ├─ config.py                # Config & platform abstraction (default models, prompt templates, schemas, etc.)
   ├─ validate_output.py       # Output validation: structure, indices, reply legality, confidence ranges
   ├─ pair_dataset_builder.py  # Build “directed role-pair” datasets from JSONL (filtering/strict mode)
   ├─ pair_to_chatml.py        # Convert paired samples to ChatML (pair or multi-turn stitch)
   └─ static/                  # Visual frontend (vanilla HTML/CSS/JS)
```

---

## 🚀 Installation & Run

### 1) Requirements
- Python 3.9+ (recommended 3.10–3.12)  
- `pip` can access pypi.org

### 2) One‑click start (GUI)
```bash
cd Text2Dialog
python launcher.py
```
- Click “① One‑Click Setup / Repair Environment”: automatically creates `.venv` and installs `text2dialog/requirements.txt`.
- Click “Start Service”, then “Open Frontend” to enter the visual console.
- In “Save API Config (.env)”, enter platform keys and the default model.

### 3) Start the service directly (CLI)
```bash
# macOS / Linux
cd Text2Dialog
bash run_server.sh

# Windows
cd Text2Dialog
.\run_server.bat
```
Manual approach:
```bash
cd Text2Dialog/text2dialog
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

### 4) CLI‑only extraction (no service/frontend)
```bash
cd Text2Dialog/text2dialog

# Minimal usage: input text → output JSONL
python dialogue_chain.py input.txt -o output.jsonl --concurrent -t 8

# Common options (example):
python dialogue_chain.py input.txt -o output.jsonl \
  --platform siliconflow --concurrent -t 8 --save-chunk-text \
  --sort-output --stats --reply-window 6 --reply-confidence-th 0.65
```

---

## 🧪 Example

Below is a short inter‑model dialogue from an LLM fine‑tuned on data built with Text2Dialog:

> **A**: Are you waiting for me to wake up?  
> **B**: No. I’m waiting for you to sleep forever.  
> **A**: Bold words. Just you?  
> **B**: With my sword.  
> **A**: A sword? Many wield swords; few can kill me.  
> **B**: Perhaps I’m exactly that one.  
> **A**: For the bounty?  
> **B**: For a name.

### Structured extraction from the snippet (excerpt, JSONL)
```json
{"chunk_id": 0, "dialogue_index": 0, "role": "A", "dialogue": "Are you waiting for me to wake up?", "reply": null}
{"chunk_id": 0, "dialogue_index": 1, "role": "B", "dialogue": "No. I'm waiting for you to sleep forever.", "reply": {"target_index": 0, "target_role": "A", "confidence": 0.96}}
{"chunk_id": 0, "dialogue_index": 2, "role": "A", "dialogue": "Bold words. Just you?", "reply": {"target_index": 1, "target_role": "B", "confidence": 0.93}}
{"chunk_id": 0, "dialogue_index": 3, "role": "B", "dialogue": "With my sword.", "reply": {"target_index": 2, "target_role": "A", "confidence": 0.95}}
{"chunk_id": 0, "dialogue_index": 4, "role": "A", "dialogue": "A sword? Many wield swords; few can kill me.", "reply": {"target_index": 3, "target_role": "B", "confidence": 0.92}}
{"chunk_id": 0, "dialogue_index": 5, "role": "B", "dialogue": "Perhaps I'm exactly that one.", "reply": {"target_index": 4, "target_role": "A", "confidence": 0.94}}
{"chunk_id": 0, "dialogue_index": 6, "role": "A", "dialogue": "For the bounty?", "reply": {"target_index": 5, "target_role": "B", "confidence": 0.91}}
{"chunk_id": 0, "dialogue_index": 7, "role": "B", "dialogue": "For a name.", "reply": {"target_index": 6, "target_role": "A", "confidence": 0.95}}
```

### Exported ChatML (pair mode, one example pair)
```json
{"messages": [
  {"role": "system", "content": "Reply to the input in B’s voice."},
  {"role": "user", "content": "Are you waiting for me to wake up?"},
  {"role": "assistant", "content": "No. I'm waiting for you to sleep forever."}
]}
```

> For more complex scenarios, use `--mode stitch --max-turns 4` to stitch consecutive samples within the same chunk into multi‑turn dialogues and train long‑range coherence.

---

## 🔧 Configure LLM Platforms

Copy `text2dialog/env.example` to `.env` and fill it according to the comments:

- **Platform selection**: `LLM_PLATFORM=<deepseek|siliconflow|bailian|moonshot|openai|gemini|aws_bedrock|custom>`  
- **Common variables** (vary by platform): `*_API_KEY`, `*_BASE_URL` (OpenAI‑compatible endpoint), `*_MODEL_NAME`.

Example (excerpt):
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
> The frontend’s “Advanced Settings” also support temporary overrides (platform, API key, BaseURL, model name) that apply to the current job only.

---

## 🧭 Recommended Workflow (Frontend)

1. Upload text (`.txt`, UTF‑8 recommended).  
2. Set platform & model (you can override `.env` in “Advanced Settings”).  
3. Click “Start Extraction”, monitor the progress bar & ETA; supports pause / resume / cancel.  
4. After extraction, run: Validate Output → Role Pairing → Export ChatML.  
5. In “Downloads”, fetch `extraction.jsonl`, `pair_datasets/`, `chatml.jsonl`.

---

## 📦 Data Formats

### 1) Extraction result (JSONL, one row per line)
```json
{
  "chunk_id": 12,
  "dialogue_index": 3,
  "role": "Zhang San",
  "dialogue": "I've arrived.",
  "reply": {
    "target_index": 2,
    "target_role": "Li Si",
    "confidence": 0.91
  }
}
```
- `reply` may be `null` or an object. If an object, `target_index` must be earlier than the current `dialogue_index` and be in the same chunk.

### 2) Role‑pair samples (produced by `pair_dataset_builder.py`)
```json
{
  "source": {"chunk_id": 12, "dialogue_index": 2, "role": "Li Si", "text": "Where are you now?"},
  "reply":  {"chunk_id": 12, "dialogue_index": 3, "role": "Zhang San", "text": "I've arrived."},
  "pair":   {"from": "Li Si", "to": "Zhang San"},
  "confidence": 0.91
}
```

### 3) ChatML (SFT messages)
```json
{"messages": [
  {"role": "system", "content": "You are now Zhang San. Reply strictly about the text dialogue."},
  {"role": "user", "content": "Where are you now?"},
  {"role": "assistant", "content": "I've arrived."}
]}
```

---

## 🖥️ Command‑Line Tools

### 1) Extraction (`dialogue_chain.py`)
Common arguments:
- `-o/--output`: output JSONL path.
- `-t/--threads` and `--concurrent/--no-concurrent`: concurrency control (enabled by default).
- `--save-chunk-text/--no-save-chunk-text`: whether to also store the original chunk text in the output.
- `--sort-output`: sort by `chunk_id` before writing back.
- `--stats/--no-stats`: print overall/per‑chunk statistics.
- `-p/--platform`, `--list-platforms`: select or list platforms.
- `--reply-window`: reply look‑back window (only earlier turns within this chunk).
- `--reply-confidence-th`: remove `reply` if below the threshold.

### 2) Role pairing (`pair_dataset_builder.py`)
```bash
# List roles and their frequencies
python pair_dataset_builder.py -i extraction.jsonl --list-roles

# Specify ordered pairs (repeatable) → output to directory
python pair_dataset_builder.py -i extraction.jsonl \
  --pairs "Zhang San,Li Si" --pairs "Li Si,Zhang San" \
  -o ./pair_datasets --min-confidence 0.80 --strict

# Alternatively, merge into one file
python pair_dataset_builder.py -i extraction.jsonl --merge-out ./all_pairs.jsonl \
  --all-ordered-pairs --roles "Zhang San" --roles "Li Si" --roles "Wang Wu"
```
Key arguments:
- `--pairs` / `--roles` + `--all-ordered-pairs`: explicitly specify or permute the role set to generate ordered pairs (excluding reflexive pairs).
- Filtering: `--min-confidence` (default 0.8), `--require-confidence`, text length bounds, and `--deny-pattern` blacklist regex.
- **Strict mode** (enabled by default): keep only samples unambiguously determined as `A→B`; use `--no-strict` to relax to heuristics.

### 3) ChatML export (`pair_to_chatml.py`)
```bash
# Multiple files/dirs/globs → single ChatML JSONL
python pair_to_chatml.py -i ./pair_datasets -o ./chatml.jsonl \
  --mode pair --dedupe --min-confidence 0.85 \
  --system-template "You are now {to_role}. Reply strictly about the text dialogue."

# Stitch mode (stitch consecutive samples within the same chunk into multi-turn)
python pair_to_chatml.py -i ./pair_datasets -o ./chatml_stitch.jsonl \
  --mode stitch --max-turns 4 --include-meta
```
Notes:
- `--mode {pair|stitch}`, `--max-turns`, `--min-confidence`, `--dedupe`, `--reverse`, `--include-meta`.
- System prompt: pass text via `--system` or `@path/to/file`; `--system-template` supports `{from_role}/{to_role}/{src_role}/{tgt_role}` placeholders.

---

## 🌐 REST API (FastAPI)

### Job lifecycle
- `POST /api/jobs/create`: form upload `file=@input.txt` → `{ "job_id": "..." }`
- `POST /api/jobs/{job_id}/extract`: request body example
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
- `GET /api/jobs/{job_id}/progress`: returns progress, throughput, ETA, and status (running/paused/cancelling/succeeded/failed/done).
- `POST /api/jobs/{job_id}/control`: `{ "action": "pause|resume|cancel|force-cancel", "reason": "..." }`
- `GET /api/jobs/{job_id}/download?which=extract|pairs|chatml`: download intermediate artifacts.

### Validate / Pair / Export
- `POST /api/validate`: `{ "job_id": "...", "input_path": "optional" }` → `{ "ok": true|false, "log": "..." }`
- `POST /api/pairs`: `{ "job_id": "...", "pairs": ["A,B"], "min_confidence": 0.8, "strict": true, ... }`
- `POST /api/chatml`: `{ "job_id": "...", "inputs": ["dir|glob|file"], "mode": "pair|stitch", ... }`

> There is also `GET /api/defaults` to fetch default configuration; static frontend routes are served at `/` and `/static/*`.

### Minimal end‑to‑end cURL
```bash
# 1) Upload and create a job
curl -F "file=@input.txt" http://localhost:8000/api/jobs/create

# 2) Trigger extraction (fill in job_id and your platform params)
curl -X POST http://localhost:8000/api/jobs/<job_id>/extract \
  -H "Content-Type: application/json" \
  -d '{"platform":"siliconflow","api_key":"...","base_url":"https://api.siliconflow.cn/v1","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507","threads":8,"concurrent":true}'

# 3) Poll progress
curl http://localhost:8000/api/jobs/<job_id>/progress
```

---

## ⚙️ Implementation Notes

- **Chunking**: token estimation via `tiktoken` (default `cl100k_base`); `MAX_TOKEN_LEN` and `COVER_CONTENT` are configurable.
- **Call stack**: prefer OpenAI Responses API (when available), otherwise fall back to Chat Completions.
- **Robustness**: automatic retries, unified stripping of “thinking/reasoning” prefixes, and parsing of dual‑channel outputs (reasoning vs. content).
- **Concurrent writeback**: internal buffer + `next_expected_chunk_id` ensure stable output ordering.
- **Progress persistence**: `.cache/progress.json`; control file `.cache/control.json` supports pause/resume/cancel.
- **Validation rules** (partial):
  - `dialogue_index`: starts at 0 and increments contiguously; duplicates or gaps are errors.
  - `role` and `dialogue`: non‑empty.
  - `reply`: `null` or object. If object: `target_index` earlier than current and in the same chunk; `confidence ∈ [0,1]`.

---

## ❓FAQ / Troubleshooting

- **Empty output or broken line breaks**: ensure the input `.txt` encoding (UTF‑8 recommended) and check that preprocessing didn’t remove all newlines.
- **Frequent model errors / rate limiting**: lower concurrency (`-t/--threads`), increase retry intervals; ensure `base_url` and `model_name` match the platform.
- **Reply mismatches**: decrease `REPLY_WINDOW`, or raise `REPLY_CONFIDENCE_TH`; during pairing, enable `--strict` and increase `--min-confidence` if needed.
- **Cross‑chunk references**: currently `reply` only links backward within the same chunk; cross‑chunk relations are out of scope.

## 📄 License

Released under the MIT License.
