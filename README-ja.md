# Text2Dialog

**文档语言 / Languages / 言語**：
[简体中文](./README.md) · [English](./README-en.md) · [日本語](./README-ja.md)


> 長文（小説・脚本・ノンフィクション等）から構造化された**登場人物の対話 + 参照関係**を自動抽出し、**品質検査 → 役割ペア化 → ChatML データセット出力**をワンクリックで完了します。コマンドライン、FastAPI サービス、可視化フロントエンド（ワンクリック・ランチャー付き）を提供。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110%2B-009688" />
  <img alt="OpenAI SDK" src="https://img.shields.io/badge/SDK-openai%20compatible-5b9bd5" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen" />
</p>

---

## ✨ 機能ハイライト

- **長文の分割**：トークン上限に基づくスマート分割。**チャンク間の重なり文脈**をサポートし、切断による誤判定を抑制。
- **複数プラットフォーム LLM 対応**：OpenAI 互換 SDK で複数プラットフォーム（DeepSeek、SiliconFlow、阿里雲百煉/通義、Kimi/Moonshot、OpenAI、Gemini、AWS Bedrock、カスタム BaseURL）に適合。
- **高品質抽出**：統一プロンプトと TypeScript 風スキーマで `[{role, dialogue, reply}]` を出力。推論型モデルの「思考」前置きは自動で剥離。
- **参照関係（reply）**：`reply.target_index` は**同一チャンク内で過去のみ**を参照。回溯ウィンドウと信頼度しきい値は設定可能。
- **併行実行と続行**：マルチスレッド併行処理、**中断からの続行**、進捗・ETA 推定。**一時停止/再開/取消**に対応。
- **フルパイプライン**：厳格な**検証器** → **役割ペア化**（A→B / B→A）→ **ChatML 出力**（pair モードと複数ターン stitch モード）。
- **ワンクリック・ランチャー**：`launcher.py` の GUI で仮想環境作成、依存関係インストール、サービス起動/停止、`.env` 設定、フロント/ヘルプを開く。

---

## 🗂 ディレクトリ構成

```
Text2Dialog/
├─ launcher.py                 # Tk GUI：環境設定、FastAPI 管理、フロント/文書を開く
├─ run_server.(sh|bat)         # ワンクリックで uvicorn サービス起動
└─ text2dialog/
   ├─ server.py                # FastAPI：ジョブ管理、抽出/検証/ペア化/出力、静的フロント
   ├─ dialogue_chain.py        # コア対話抽出：分割/併行/再試行/検証/進捗/続行/制御
   ├─ config.py                # 設定とプラットフォーム抽象（既定モデル、プロンプト、スキーマ等）
   ├─ validate_output.py       # 出力検証：構造・索引・reply の合法性・信頼度範囲
   ├─ pair_dataset_builder.py  # JSONL から「有向役割ペア」データセット生成（フィルタ/厳格モード）
   ├─ pair_to_chatml.py        # ペア化サンプルを ChatML に変換（pair または stitch 多輪）
   └─ static/                  # 可視化フロント（素の HTML/CSS/JS）
```

---

## 🚀 インストールと実行

### 1) 動作環境
- Python **3.9+**（推奨 3.10–3.12）  
- `pip` が pypi.org にアクセス可能

### 2) ワンクリック起動（GUI）
```bash
cd Text2Dialog
python launcher.py
```
- **「① ワンクリック設定/環境修復」**をクリック：`.venv` を自動作成し、`text2dialog/requirements.txt` をインストール。
- **「サービス起動」**→**「フロントを開く」**の順にクリックして可視化コンソールへ。
- **「API 設定を保存（.env）」**でプラットフォームのキーと既定モデルを記入。

### 3) 直接起動（CLI）
```bash
# macOS / Linux
cd Text2Dialog
bash run_server.sh

# Windows
cd Text2Dialog
.
un_server.bat
```
手動手順：
```bash
cd Text2Dialog/text2dialog
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 4) 純コマンドライン抽出（サービス/フロントなし）
```bash
cd Text2Dialog/text2dialog

# 最小例：テキスト入力 → JSONL 出力
python dialogue_chain.py input.txt -o output.jsonl --concurrent -t 8

# よく使うオプション（例）：
python dialogue_chain.py input.txt -o output.jsonl   --platform siliconflow --concurrent -t 8 --save-chunk-text   --sort-output --stats --reply-window 6 --reply-confidence-th 0.65
```

---

## 🧪 効果例

以下は **Text2Dialog** で構築したデータを用いて微調整した LLM による**モデル間対話**の一片です：

> **A**：私が目覚めるのを待っていたのか？  
> **B**：いや。君が永遠に眠るのを待っている。  
> **A**：大きな口を叩くな。お前ごときに？  
> **B**：この剣さ。  
> **A**：剣？この世に剣を使う者は多いが、俺を殺せる者は少ない。  
> **B**：たまたま俺がその一人かもな。  
> **A**：賞金のためか？  
> **B**：ひとつの名のためだ。

**この断片から抽出された構造化結果（抜粋、JSONL）：**
```json
{"chunk_id": 0, "dialogue_index": 0, "role": "A", "dialogue": "私が目覚めるのを待っていたのか？", "reply": null}
{"chunk_id": 0, "dialogue_index": 1, "role": "B", "dialogue": "いや。君が永遠に眠るのを待っている。", "reply": {"target_index": 0, "target_role": "A", "confidence": 0.96}}
{"chunk_id": 0, "dialogue_index": 2, "role": "A", "dialogue": "大きな口を叩くな。お前ごときに？", "reply": {"target_index": 1, "target_role": "B", "confidence": 0.93}}
{"chunk_id": 0, "dialogue_index": 3, "role": "B", "dialogue": "この剣さ。", "reply": {"target_index": 2, "target_role": "A", "confidence": 0.95}}
{"chunk_id": 0, "dialogue_index": 4, "role": "A", "dialogue": "剣？この世に剣を使う者は多いが、俺を殺せる者は少ない。", "reply": {"target_index": 3, "target_role": "B", "confidence": 0.92}}
{"chunk_id": 0, "dialogue_index": 5, "role": "B", "dialogue": "たまたま俺がその一人かもな。", "reply": {"target_index": 4, "target_role": "A", "confidence": 0.94}}
{"chunk_id": 0, "dialogue_index": 6, "role": "A", "dialogue": "賞金のためか？", "reply": {"target_index": 5, "target_role": "B", "confidence": 0.91}}
{"chunk_id": 0, "dialogue_index": 7, "role": "B", "dialogue": "ひとつの名のためだ。", "reply": {"target_index": 6, "target_role": "A", "confidence": 0.95}}
```

**出力された ChatML（pair モード、1 組の例）：**
```json
{"messages": [
  {"role": "system", "content": "Bの口調で入力に返答せよ。"},
  {"role": "user", "content": "私が目覚めるのを待っていたのか？"},
  {"role": "assistant", "content": "いや。君が永遠に眠るのを待っている。"}
]}
```

> より複雑な場面では `--mode stitch --max-turns 4` を用い、同一チャンクの連続サンプルを多輪対話に連結して長距離の文脈継承能力を鍛えられます。

---

## 🔧 LLM プラットフォームの設定

`text2dialog/env.example` を `.env` にコピーし、コメントに従って値を記入：

- **プラットフォーム選択**：`LLM_PLATFORM=<deepseek|siliconflow|bailian|moonshot|openai|gemini|aws_bedrock|custom>`  
- **共通変数**（プラットフォームにより異なる）：`*_API_KEY`、`*_BASE_URL`（OpenAI 互換アドレス）、`*_MODEL_NAME`

例（抜粋）：
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
> フロントエンドの「詳細設定」から**一時的に上書き**（プラットフォーム、API Key、BaseURL、モデル名）も可能で、当該ジョブにのみ適用されます。

---

## 🧭 推奨ワークフロー（フロントエンド）

1. **テキストをアップロード**（`.txt`、UTF‑8 推奨）。  
2. **プラットフォームとモデルを設定**（「詳細設定」で `.env` を上書き可能）。  
3. **「抽出開始」**をクリックし、進捗バーと ETA を確認。**一時停止/再開/取消**に対応。  
4. 抽出完了後に順に：**出力検証** → **役割ペア化** → **ChatML 出力**。  
5. 「ダウンロード」で `extraction.jsonl`、`pair_datasets/`、`chatml.jsonl` を取得。

---

## 📦 データ形式

### 1) 抽出結果（JSONL、1 行 1 レコード）
```json
{
  "chunk_id": 12,
  "dialogue_index": 3,
  "role": "张三",
  "dialogue": "着いた。",
  "reply": {
    "target_index": 2,
    "target_role": "李四",
    "confidence": 0.91
  }
}
```
- `reply` は `null` またはオブジェクト。オブジェクトの場合、`target_index` は現在より**前**かつ**同一チャンク**内でなければならない。

### 2) 役割ペアサンプル（pair_dataset_builder.py の出力）
```json
{
  "source": {"chunk_id": 12, "dialogue_index": 2, "role": "李四", "text": "どこまで来た？"},
  "reply":  {"chunk_id": 12, "dialogue_index": 3, "role": "张三", "text": "着いた。"},
  "pair":   {"from": "李四", "to": "张三"},
  "confidence": 0.91
}
```

### 3) ChatML（SFT messages）
```json
{"messages": [
  {"role": "system", "content": "あなたは今 張三 を演じ、テキストの対話のみに集中してください。"},
  {"role": "user", "content": "どこまで来た？"},
  {"role": "assistant", "content": "着いた。"}
]}
```

---

## 🖥️ コマンドラインツール

### 1) 抽出（dialogue_chain.py）
主な引数：
- `-o/--output`：出力 JSONL。
- `-t/--threads` と `--concurrent/--no-concurrent`：併行制御（既定で有効）。
- `--save-chunk-text/--no-save-chunk-text`：出力に元チャンク本文を含めるか。
- `--sort-output`：完了後に `chunk_id` 順で整列して書き戻す。
- `--stats/--no-stats`：全体/チャンク別統計を表示。
- `-p/--platform`、`--list-platforms`：プラットフォームの選択/一覧。
- `--reply-window`：reply の遡及ウィンドウ（**同一チャンク**内のより前の発言のみ）。
- `--reply-confidence-th`：しきい値未満の `reply` を削除。

### 2) 役割ペア化（pair_dataset_builder.py）
```bash
# 役割と出現頻度を列挙
python pair_dataset_builder.py -i extraction.jsonl --list-roles

# 有向ペア（複数指定可） → ディレクトリへ出力
python pair_dataset_builder.py -i extraction.jsonl   --pairs "张三,李四" --pairs "李四,张三"   -o ./pair_datasets --min-confidence 0.80 --strict

# 1 ファイルに結合することも可能
python pair_dataset_builder.py -i extraction.jsonl --merge-out ./all_pairs.jsonl   --all-ordered-pairs --roles "张三" --roles "李四" --roles "王五"
```
主なパラメータ要点：
- `--pairs` / `--roles` + `--all-ordered-pairs`：明示または役割集合の**全順列**で有向ペア生成（自己ペアは除外）。
- フィルタ：`--min-confidence`（既定 **0.8**）、`--require-confidence`、テキスト長上下限、`--deny-pattern` ブラックリスト正規表現。
- **厳格モード**（既定有効）：`A→B` が**確実**なサンプルのみ保持。`--no-strict` でヒューリスティックに緩和可。

### 3) ChatML 出力（pair_to_chatml.py）
```bash
# 複数ファイル/ディレクトリ/ワイルドカード入力 → 単一 ChatML JSONL
python pair_to_chatml.py -i ./pair_datasets -o ./chatml.jsonl   --mode pair --dedupe --min-confidence 0.85   --system-template "あなたは {to_role} を演じ、テキストの対話のみに集中してください。"

# Stitch モード（同一チャンクの連続サンプルを多輪に連結）
python pair_to_chatml.py -i ./pair_datasets -o ./chatml_stitch.jsonl   --mode stitch --max-turns 4 --include-meta
```
パラメータ要点：
- `--mode {pair|stitch}`、`--max-turns`、`--min-confidence`、`--dedupe`、`--reverse`、`--include-meta`
- システムプロンプト：`--system` は本文または `@path/to/file` を許容。`--system-template` では `{from_role}/{to_role}/{src_role}/{tgt_role}` を利用可能。

---

## 🌐 REST API（FastAPI）

### ジョブのライフサイクル
- `POST /api/jobs/create`：フォームで `file=@input.txt` を送信 → `{ "job_id": "..." }`
- `POST /api/jobs/{job_id}/extract`：リクエスト例
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
- `GET /api/jobs/{job_id}/progress`：進捗、速度、ETA、状態（running/paused/cancelling/succeeded/failed/done）を返す。
- `POST /api/jobs/{job_id}/control`：`{ "action": "pause|resume|cancel|force-cancel", "reason": "..." }`
- `GET /api/jobs/{job_id}/download?which=extract|pairs|chatml`：段階成果物をダウンロード。

### 検証 / ペア化 / 出力
- `POST /api/validate`：`{ "job_id": "...", "input_path": "任意" }` → `{ "ok": true|false, "log": "..." }`
- `POST /api/pairs`：`{ "job_id": "...", "pairs": ["A,B"], "min_confidence": 0.8, "strict": true, ... }`
- `POST /api/chatml`：`{ "job_id": "...", "inputs": ["dir|glob|file"], "mode": "pair|stitch", ... }`

> さらに `GET /api/defaults` で既定設定を取得可能。静的フロントのルートは `/` および `/static/*`。

**最小限の cURL**
```bash
# 1) アップロードしてジョブ作成
curl -F "file=@input.txt" http://localhost:8000/api/jobs/create

# 2) 抽出をトリガー（job_id とプラットフォーム設定を投入）
curl -X POST http://localhost:8000/api/jobs/<job_id>/extract   -H "Content-Type: application/json"   -d '{"platform":"siliconflow","api_key":"...","base_url":"https://api.siliconflow.cn/v1","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507","threads":8,"concurrent":true}'

# 3) 進捗をポーリング
curl http://localhost:8000/api/jobs/<job_id>/progress
```

---

## ⚙️ 実装要点

- **分割**：`tiktoken` でトークン数を見積（既定 `cl100k_base`）。`MAX_TOKEN_LEN` と `COVER_CONTENT` は設定可能。
- **呼び出し**：可能なら OpenAI **Responses API** を優先。不可の場合は **Chat Completions** にフォールバック。
- **頑健性**：自動リトライ、「思考/推論」前置きの統一剥離、二経路出力（reasoning vs content）の解析。
- **併行書き戻し**：内部バッファ＋`next_expected_chunk_id` により出力順序を安定化。
- **進捗の永続化**：`.cache/progress.json`。制御ファイル `.cache/control.json` で一時停止/再開/取消をサポート。
- **検証規則**（抜粋）：
  - `dialogue_index`：**0 から連続昇順**。重複や飛び番号はエラー。
  - `role` と `dialogue`：**空不可**。
  - `reply`：`null` またはオブジェクト。オブジェクトの場合：`target_index` は**現在より前**で**同一チャンク**内、`confidence ∈ [0,1]`。

---

## ❓FAQ / トラブルシューティング

- **出力が空/改行異常**：入力 `.txt` のエンコーディング（UTF‑8 推奨）を確認。前処理で改行をすべて削除していないか確認。
- **モデルのエラー/レート制限**：併行度 `-t/--threads` を下げ、リトライ間隔を延ばす。`base_url` と `model_name` の整合も確認。
- **reply の誤対応**：`REPLY_WINDOW` を縮小、または `REPLY_CONFIDENCE_TH` を引き上げる。必要に応じてペア化段階で `--strict` を有効にし、`--min-confidence` を上げる。
- **チャンク間参照**：現状の `reply` は**同一チャンク内で過去のみ**を対象。チャンクをまたぐ関係は対象外。

## 📄 ライセンス

本プロジェクトは **MIT License** で公開されています。
