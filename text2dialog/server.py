#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — FastAPI 服务，为 text2dialog 提供可视化前端 API。
运行：
  pip install -r requirements.txt -i https://pypi.org/simple
  uvicorn server:app --host 0.0.0.0 --port 8000

"""
import os
import io
import re
import sys
import json
import time
import uuid
import signal
from typing import Optional, Dict, Any, List
try:
    # Python 3.8+
    from typing import Literal
except Exception:  # pragma: no cover
    # 兼容旧版本
    from typing_extensions import Literal  # type: ignore

from pathlib import Path
from multiprocessing import Process

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, "static")
JOBS_DIR = os.path.join(APP_ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# ---- 将“可能的项目目录”加入 sys.path，确保能导入 ----
POSSIBLE_DIRS = [
    APP_ROOT,
    os.path.dirname(APP_ROOT),
    os.path.join(APP_ROOT, ".."),
    os.path.join(APP_ROOT, "../dialogue-chain"),
]
for d in POSSIBLE_DIRS:
    d = os.path.abspath(d)
    if os.path.exists(os.path.join(d, "dialogue_chain.py")) and d not in sys.path:
        sys.path.insert(0, d)

# ---- 延迟导入（修复脚本中可能出现的中文引号等问题） ----
# 如果你是专业人士或者特别在意复杂度的算竟选手可以将这部分删去:)
DialogueChain = None
Config = None
ModelPlatform = None
validator = None
pair_builder = None
p2c = None
CancelledErrorCls = None  # 新增：用于识别对话链中的取消异常

def _import_project_modules():
    global DialogueChain, Config, ModelPlatform, validator, pair_builder, p2c, CancelledErrorCls
    if all([DialogueChain, Config, ModelPlatform, validator, pair_builder, p2c, CancelledErrorCls]):
        return
    try:
        from dialogue_chain import DialogueChain as _DC, CancelledError as _CE  # 新增 CancelledError
        from config import Config as _CFG, ModelPlatform as _MP
        import validate_output as _V
        import pair_dataset_builder as _PB
        import pair_to_chatml as _P2C
        DialogueChain, Config, ModelPlatform = _DC, _CFG, _MP
        validator, pair_builder, p2c = _V, _PB, _P2C
        CancelledErrorCls = _CE
    except SyntaxError:
        # 若因全角符号/智能引号导致语法错误，尝试修复 dialogue_chain.py 后再导入一次
        cand = None
        for d in sys.path:
            fp = os.path.join(d, "dialogue_chain.py")
            if os.path.exists(fp):
                cand = fp; break
        if not cand:
            raise
        # 粗略修正智能引号与破折号
        txt = open(cand, "r", encoding="utf-8").read()
        orig = txt
        txt = txt.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        txt = txt.replace("—", "-").replace("…", "...")
        if txt != orig:
            with open(cand, "w", encoding="utf-8") as f: f.write(txt)
        from dialogue_chain import DialogueChain as _DC2, CancelledError as _CE2
        from config import Config as _CFG2, ModelPlatform as _MP2
        import validate_output as _V2
        import pair_dataset_builder as _PB2
        import pair_to_chatml as _P2C2
        DialogueChain, Config, ModelPlatform = _DC2, _CFG2, _MP2
        validator, pair_builder, p2c = _V2, _PB2, _P2C2
        CancelledErrorCls = _CE2

# ---- 工具函数 ----
_JOBS: Dict[str, Dict[str, Any]] = {}

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _job_dir(job_id: str) -> str:
    d = os.path.join(JOBS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d

def _save_job(job_id: str) -> None:
    d = _job_dir(job_id); fp = os.path.join(d, "job.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(_JOBS[job_id], f, ensure_ascii=False, indent=2)

def _load_job(job_id: str) -> Dict[str, Any]:
    # 始终以磁盘为准，避免多进程更新不一致
    d = _job_dir(job_id); fp = os.path.join(d, "job.json")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            job = json.load(f)
        _JOBS[job_id] = job
        return job
    raise KeyError(job_id)

def _update_job(job_id: str, **patch: Any) -> None:
    job = _load_job(job_id); job.update(patch); _save_job(job_id)

def _read_progress(cache_dir: str) -> Dict[str, Any]:
    if not cache_dir: return {}
    try:
        fp = os.path.join(cache_dir, "progress.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: pass
    return {}

def _set_platform_env(platform: str, api_key: Optional[str], base_url: Optional[str], model_name: Optional[str]) -> None:
    if not platform: return
    os.environ["LLM_PLATFORM"] = platform
    # 获取该平台的变量名
    cfg = ModelPlatform.get_platform_config(platform)
    if api_key:  os.environ[cfg["api_key_env"]] = api_key
    if base_url: os.environ[cfg["base_url_env"]] = base_url
    if model_name: os.environ[f"{platform.upper()}_MODEL_NAME"] = model_name

def _apply_overrides(overrides: Dict[str, Any]) -> None:
    mapping = {
        "MAX_TOKEN_LEN": ("MAX_TOKEN_LEN", int),
        "COVER_CONTENT": ("COVER_CONTENT", int),
        "TEMPERATURE": ("TEMPERATURE", float),
        "MAX_WORKERS": ("MAX_WORKERS", int),
        "REPLY_WINDOW": ("REPLY_WINDOW", int),
        "REPLY_CONFIDENCE_TH": ("REPLY_CONFIDENCE_TH", float),
        "SAVE_CHUNK_TEXT": ("SAVE_CHUNK_TEXT", bool),
        "DEFAULT_CONCURRENT": ("DEFAULT_CONCURRENT", bool),
        "DEFAULT_SORT_OUTPUT": ("DEFAULT_SORT_OUTPUT", bool),
    }
    for k, v in overrides.items():
        if v is None or k not in mapping: continue
        attr, caster = mapping[k]
        try: setattr(Config, attr, caster(v))
        except Exception: setattr(Config, attr, v)

# ---- 控制文件（暂停/继续/取消）辅助函数 ----
def _control_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), ".cache", "control.json")

def _write_control(job_id: str, state: str, reason: Optional[str] = None) -> Dict[str, Any]:
    ctrl = {"state": state, "reason": reason or "", "ts": _now_str()}
    path = _control_path(job_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ctrl, f, ensure_ascii=False)
    os.replace(tmp, path)
    return ctrl

def _read_control(job_id: str) -> Dict[str, Any]:
    path = _control_path(job_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"state": "running", "reason": "", "ts": _now_str()}

# ---------- FastAPI ----------
app = FastAPI(title="dialogue-chain UI", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ExtractReq(BaseModel):
    platform: Optional[str] = Field(None)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    concurrent: Optional[bool] = True
    threads: Optional[int] = None
    save_chunk_text: Optional[bool] = None
    sort_output: Optional[bool] = None
    MAX_TOKEN_LEN: Optional[int] = None
    COVER_CONTENT: Optional[int] = None
    TEMPERATURE: Optional[float] = None
    REPLY_WINDOW: Optional[int] = None
    REPLY_CONFIDENCE_TH: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/ping")
def ping(): return {"ok": True, "time": _now_str()}

@app.get("/api/defaults")
def api_defaults():
    _import_project_modules()
    platforms = ModelPlatform.list_platforms()
    cfg = {
        "MAX_TOKEN_LEN": Config.MAX_TOKEN_LEN,
        "COVER_CONTENT": Config.COVER_CONTENT,
        "TEMPERATURE": Config.TEMPERATURE,
        "MAX_RETRIES": Config.MAX_RETRIES,
        "RETRY_DELAY": Config.RETRY_DELAY,
        "MAX_WORKERS": Config.MAX_WORKERS,
        "REPLY_WINDOW": Config.REPLY_WINDOW,
        "REPLY_CONFIDENCE_TH": Config.REPLY_CONFIDENCE_TH,
        "DEFAULT_CONCURRENT": getattr(Config, "DEFAULT_CONCURRENT", True),
        "DEFAULT_SORT_OUTPUT": getattr(Config, "DEFAULT_SORT_OUTPUT", False),
        "SAVE_CHUNK_TEXT": getattr(Config, "SAVE_CHUNK_TEXT", False),
        "CACHE_DIR": Config.CACHE_DIR,
    }
    return {"platforms": platforms, "config": cfg, "schema_default": Config.DEFAULT_SCHEMA}

@app.post("/api/jobs/create")
async def create_job(file: UploadFile = File(...)):
    _import_project_modules()
    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    infile = os.path.join(d, "input.txt")
    with open(infile, "wb") as f: f.write(await file.read())
    _JOBS[job_id] = {"id": job_id, "status": "created", "created_at": _now_str(), "input_file": infile,
                     "message": "已创建，待运行", "progress": {"processed": 0, "total": 0}, "artifacts": {}}
    _save_job(job_id)
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    _import_project_modules()
    try: return _load_job(job_id)
    except KeyError: raise HTTPException(status_code=404, detail="job not found")

@app.get("/api/jobs/{job_id}/progress")
def poll_progress(job_id: str):
    _import_project_modules()
    job = _load_job(job_id)
    prog = _read_progress(job.get("cache_dir"))
    return {"progress": prog, "status": job.get("status"), "message": job.get("message")}

@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, which: str):
    _import_project_modules()
    job = _load_job(job_id)
    art = job.get("artifacts", {})
    path = art.get(which)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{which} not found")
    return FileResponse(path, filename=os.path.basename(path))

# ---- 子进程：执行抽取 ----
def _worker_extract(job_id: str, req_body: Dict[str, Any]) -> None:
    _import_project_modules()
    try:
        # 环境与配置仅影响子进程
        cache_dir = os.path.join(_job_dir(job_id), ".cache")
        os.makedirs(cache_dir, exist_ok=True)

        # 初始化控制状态为 running，便于前端/子进程读取
        try:
            _write_control(job_id, "running", "作业启动")
        except Exception:
            pass

        # 平台注入 & Config 覆盖（仅在子进程生效）
        _set_platform_env(req_body.get("platform") or os.getenv("LLM_PLATFORM"), req_body.get("api_key"), req_body.get("base_url"), req_body.get("model_name"))
        overrides = dict(
            MAX_TOKEN_LEN=req_body.get("MAX_TOKEN_LEN"),
            COVER_CONTENT=req_body.get("COVER_CONTENT"),
            TEMPERATURE=req_body.get("TEMPERATURE"),
            REPLY_WINDOW=req_body.get("REPLY_WINDOW"),
            REPLY_CONFIDENCE_TH=req_body.get("REPLY_CONFIDENCE_TH"),
            SAVE_CHUNK_TEXT=req_body.get("save_chunk_text"),
            DEFAULT_SORT_OUTPUT=req_body.get("sort_output"),
            DEFAULT_CONCURRENT=req_body.get("concurrent"),
            MAX_WORKERS=req_body.get("threads"),
        )
        _apply_overrides(overrides)
        Config.CACHE_DIR = cache_dir

        out_file = os.path.join(_job_dir(job_id), "extraction.jsonl")
        existing_artifacts = dict(_load_job(job_id).get("artifacts", {}))
        existing_artifacts["extraction"] = out_file
        _update_job(job_id, status="running", message="???????", cache_dir=cache_dir, artifacts=existing_artifacts)

        extractor = DialogueChain(
            schema=None,
            platform=req_body.get("platform") or os.getenv("LLM_PLATFORM"),
            max_workers=req_body.get("threads"),
            save_chunk_text=req_body.get("save_chunk_text"),
            cache_dir=cache_dir,
        )

        try:
            if req_body.get("concurrent", True):
                extractor.extract_dialogues_concurrent(_load_job(job_id)["input_file"], out_file)
            else:
                extractor.extract_dialogues(_load_job(job_id)["input_file"], out_file)
        except Exception as e:
            # 识别“取消”并退出
            if CancelledErrorCls and isinstance(e, CancelledErrorCls):
                _update_job(job_id, status="cancelled", message="已取消")
                return
            # 其他异常继续抛出
            raise

        if req_body.get("sort_output") or getattr(Config, "DEFAULT_SORT_OUTPUT", False):
            sorted_path = extractor.sort_dialogues(out_file)
            art = _load_job(job_id).get("artifacts", {})
            art["extraction"] = sorted_path
            _update_job(job_id, artifacts=art)

        stats = extractor.get_statistics(_load_job(job_id)["artifacts"]["extraction"])
        _update_job(job_id, status="succeeded", message="对话提取完成", stats=stats)
    except Exception as e:
        _update_job(job_id, status="failed", message=f"提取失败：{e}")
    finally:
        # 清理进度文件
        try:
            pf = os.path.join(_job_dir(job_id), ".cache", "progress.json")
            if os.path.exists(pf): os.remove(pf)
        except Exception:
            pass

@app.post("/api/jobs/{job_id}/extract")
async def run_extract(job_id: str, request: Request):
    _import_project_modules()
    # 立即读取并验证
    _ = _load_job(job_id)
    body = await request.json()
    req = ExtractReq(**body)
    cache_dir = os.path.join(_job_dir(job_id), ".cache"); os.makedirs(cache_dir, exist_ok=True)

    # 启动前将控制状态置为 running
    try:
        _write_control(job_id, "running", "准备启动")
    except Exception:
        pass

    # 启动子进程（互不影响）
    p = Process(target=_worker_extract, args=(job_id, req.dict()))
    p.daemon = True
    p.start()
    _update_job(job_id, message="已启动作业", pid=p.pid, cache_dir=cache_dir, status="running")
    return {"ok": True, "pid": p.pid}

class ControlReq(BaseModel):
    action: Literal["pause", "resume", "cancel", "force-cancel"]
    reason: Optional[str] = None

@app.post("/api/jobs/{job_id}/control")
def control_job(job_id: str, req: ControlReq):
    """作业运行时控制：暂停/继续/取消/强制终止"""
    _import_project_modules()
    try:
        job = _load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

    action = req.action
    if action == "pause":
        _write_control(job_id, "paused", req.reason)
        _update_job(job_id, status="paused", message=req.reason or "已暂停")
        return {"ok": True, "status": "paused"}
    if action == "resume":
        _write_control(job_id, "running", req.reason)
        _update_job(job_id, status="running", message=req.reason or "继续运行")
        return {"ok": True, "status": "running"}
    if action == "cancel":
        _write_control(job_id, "cancelling", req.reason)
        _update_job(job_id, status="cancelling", message=req.reason or "正在取消…")
        return {"ok": True, "status": "cancelling"}

    # force-cancel
    pid = job.get("pid")
    if pid:
        try:
            try:
                _write_control(job_id, "cancelling", req.reason or "强制终止")
            except Exception:
                pass
            os.kill(int(pid), getattr(signal, "SIGTERM", signal.SIGINT))
        except Exception as e:
            # 进程可能已退出
            return {"ok": False, "error": f"终止失败：{e}"}
    # 清理进度文件，声明作业已取消
    try:
        pf = os.path.join(_job_dir(job_id), ".cache", "progress.json")
        if os.path.exists(pf): os.remove(pf)
    except Exception:
        pass
    _update_job(job_id, status="cancelled", message=req.reason or "已强制终止")
    return {"ok": True, "status": "cancelled"}

class ValidateReq(BaseModel):
    job_id: str
    input_path: Optional[str] = None

@app.post("/api/validate")
def run_validate(req: ValidateReq):
    _import_project_modules()
    job = _load_job(req.job_id)
    src = req.input_path or job.get("artifacts", {}).get("extraction")
    if not src or not os.path.exists(src): raise HTTPException(status_code=400, detail="没有可校验的文件")
    import contextlib
    buf = io.StringIO()
    ok = False
    with contextlib.redirect_stderr(buf):
        try:
            code = validator.validate(src); ok = (code == 0)
        except Exception as e:
            msg = f"validate 运行错误：{e}"
            return {"ok": False, "log": msg + "\n" + buf.getvalue()}
    log = buf.getvalue()
    if ok:
        # 写一份校验后的文件
        dst = os.path.join(_job_dir(req.job_id), "extraction.validated.jsonl")
        with open(src, "r", encoding="utf-8") as fr, open(dst, "w", encoding="utf-8") as fw:
            for line in fr:
                fw.write(line)
        arts = {**_load_job(req.job_id)["artifacts"], "validated": dst}
        _update_job(req.job_id, artifacts=arts)
        return {"ok": True, "log": log}
    return {"ok": False, "log": log}

class PairBuildReq(BaseModel):
    job_id: str
    input_path: Optional[str] = None
    out_dir: Optional[str] = None
    merge_out: Optional[str] = None
    pairs: Optional[List[str]] = None
    roles: Optional[List[str]] = None
    all_ordered_pairs: Optional[bool] = False
    min_confidence: Optional[float] = 0.8
    require_confidence: Optional[bool] = False
    strict: Optional[bool] = True
    min_src_chars: Optional[int] = 1
    min_reply_chars: Optional[int] = 1
    max_src_chars: Optional[int] = None
    max_reply_chars: Optional[int] = None
    deny_pattern: Optional[List[str]] = None
    list_roles: Optional[bool] = False

@app.post("/api/pairs")
def build_pairs(req: PairBuildReq):
    _import_project_modules()
    job = _load_job(req.job_id)
    src = req.input_path or job.get("artifacts", {}).get("validated") or job.get("artifacts", {}).get("extraction")
    if not src or not os.path.exists(src):
        raise HTTPException(status_code=400, detail="没有可用的 JSONL 输入")
    out_dir = req.out_dir or os.path.join(_job_dir(req.job_id), "pair_datasets")
    os.makedirs(out_dir, exist_ok=True)

    # 支持列出角色
    if req.list_roles:
        try:
            counts = pair_builder.list_roles(Path(src))
            roles = [r for (r, _) in counts.most_common()]
        except Exception:
            roles = []
        return {"roles": roles}

    # 展开 req.pairs 中的通配 ALL/全部 为显式 'A,B'
    def _all_roles() -> List[str]:
        try:
            counts = pair_builder.list_roles(Path(src))
            return [r for (r, _) in counts.most_common()]
        except Exception:
            try:
                return list(counts.keys())
            except Exception:
                return []

    if req.pairs:
        roles_for_expand = req.roles or _all_roles()
        expanded_pairs: List[str] = []
        for item in req.pairs:
            s = item.strip()
            if s.upper() in ("ALL", "全部"):
                for a in roles_for_expand:
                    for b in roles_for_expand:
                        if a != b: expanded_pairs.append(f"{a},{b}")
            elif "," in s:
                expanded_pairs.append(s)
        # 去重
        seen = set()
        req.pairs = [p for p in expanded_pairs if (p not in seen and not seen.add(p))]

    argv: List[str] = ["--input", src, "--out", out_dir]
    if req.merge_out: argv = ["--input", src, "--merge-out", req.merge_out]
    if req.pairs:
        argv.extend(["--pairs"] + req.pairs)

    # 统一处理 roles，避免重复 --roles 导致 argparse 覆盖
    roles_args: List[str] = []
    if req.all_ordered_pairs:
        roles_args = req.roles or _all_roles()
    elif req.roles:
        roles_args = req.roles
    if roles_args:
        argv.extend(["--roles"] + roles_args)
    if req.all_ordered_pairs: argv.append("--all-ordered-pairs")

    if req.min_confidence is not None: argv += ["--min-confidence", str(req.min_confidence)]
    if req.require_confidence: argv.append("--require-confidence")
    if req.strict: argv.append("--strict")
    else: argv.append("--no-strict")
    if req.min_src_chars is not None: argv += ["--min-src-chars", str(req.min_src_chars)]
    if req.min_reply_chars is not None: argv += ["--min-reply-chars", str(req.min_reply_chars)]
    if req.max_src_chars is not None: argv += ["--max-src-chars", str(req.max_src_chars)]
    if req.max_reply_chars is not None: argv += ["--max-reply-chars", str(req.max_reply_chars)]
    if req.deny_pattern: [argv.extend(["--deny-pattern", pat]) for pat in req.deny_pattern]
    if req.list_roles: argv.append("--list-roles")

    code = pair_builder.main(argv)
    if code != 0: return {"ok": False, "log": "pair_dataset_builder: 非零退出"}

    zip_path = os.path.join(_job_dir(req.job_id), "pairs.zip")
    if os.path.exists(out_dir):
        # 简单打包
        import zipfile
        with zipfile.ZipFile(zip_path, "w") as zf:
            for root, _, files in os.walk(out_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    zf.write(fp, arcname=os.path.relpath(fp, out_dir))
    if os.path.exists(zip_path):
        _update_job(req.job_id, artifacts={**_load_job(req.job_id)["artifacts"], "pairs_zip": zip_path})
    return {"ok": True, "out_dir": out_dir, "zip": zip_path}

class ChatMLReq(BaseModel):
    job_id: str
    input: Optional[str] = None
    mode: Optional[str] = "pair"
    out: Optional[str] = None
    min_confidence: Optional[float] = None
    reverse: Optional[bool] = False
    include_meta: Optional[bool] = False
    max_turns: Optional[int] = None
    dedupe: Optional[bool] = False
    system_text: Optional[str] = None
    system_template: Optional[str] = None

@app.post("/api/chatml")
def build_chatml(req: ChatMLReq):
    _import_project_modules()
    job = _load_job(req.job_id)
    inputs = req.input or os.path.join(_job_dir(req.job_id), "pair_datasets")
    out = req.out or os.path.join(_job_dir(req.job_id), "chatml.jsonl")

    # 通过 pair_to_chatml.py 的 CLI 入口
    argv = [
        "-i", inputs,
        "-o", out,
        "--mode", req.mode or "pair",
    ]
    if req.min_confidence is not None:
        argv.extend(["--min-confidence", str(req.min_confidence)])
    if req.max_turns is not None:
        argv.extend(["--max-turns", str(max(1, int(req.max_turns)))])
    if req.dedupe:
        argv.append("--dedupe")
    if req.reverse: argv.append("--reverse")
    if req.include_meta: argv.append("--include-meta")
    if req.system_template:
        argv.extend(["--system-template", req.system_template])
    elif req.system_text:
        argv.extend(["--system", req.system_text])

    code = p2c.main(argv)
    if code != 0:
        return {"ok": False, "log": "pair_to_chatml: 非零退出"}
    _update_job(req.job_id, artifacts={**job.get("artifacts", {}), "chatml": out})
    return {"ok": True, "out": out}

def _all_roles_from_pairs(job_id: str) -> Dict[str, int]:
    # 尝试从 pair_datasets 汇总角色；兼容旧/新字段
    def _collect_roles(obj: Dict[str, Any]) -> List[str]:
        roles: List[str] = []
        # 优先使用标准的 pair: {"from": "...", "to": "..."}
        pair = obj.get("pair")
        if isinstance(pair, dict):
            roles.extend([pair.get("from"), pair.get("to")])
        # 兼容字段：source/reply 里的 role
        src = obj.get("source")
        if isinstance(src, dict):
            roles.append(src.get("role"))
        tgt = obj.get("reply")
        if isinstance(tgt, dict):
            roles.append(tgt.get("role"))
        # 最后兜底旧字段
        roles.extend([obj.get("from_role"), obj.get("to_role")])
        return [r for r in roles if r]

    try:
        ds_dir = os.path.join(_job_dir(job_id), "pair_datasets")
        if not os.path.isdir(ds_dir):
            return {}
        from collections import Counter

        cnt = Counter()
        for root, _, files in os.walk(ds_dir):
            for fn in files:
                if not fn.endswith(".jsonl"):
                    continue
                fp = os.path.join(root, fn)
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            for role in _collect_roles(obj):
                                cnt[role] += 1
                        except Exception:
                            # 忽略坏行，确保枚举不中断
                            continue
        return dict(cnt.most_common())
    except Exception:
        return {}

@app.get("/api/roles")
def list_roles(job_id: str):
    try:
        roles = list(_all_roles_from_pairs(job_id).keys())
    except Exception:
        roles = []
    return {"roles": roles}

# 静态资源
@app.get("/static/{path:path}")
def static_assets(path: str):
    # 防止路径穿越：限制到 STATIC_DIR 内部
    static_root = os.path.abspath(STATIC_DIR)
    fp = os.path.abspath(os.path.normpath(os.path.join(static_root, path)))
    try:
        if os.path.commonpath([static_root, fp]) != static_root:
            raise HTTPException(status_code=404, detail="not found")
    except Exception:
        raise HTTPException(status_code=404, detail="not found")
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="not found")
    with open(fp, "rb") as f:
        data = f.read()
    media = "text/plain"
    if path.endswith(".css"): media = "text/css"
    elif path.endswith(".js"): media = "application/javascript"
    elif path.endswith(".html"): media = "text/html"
    return Response(content=data, media_type=media)
