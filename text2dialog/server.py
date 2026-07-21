#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py — FastAPI 服务，为 text2dialog 提供可视化前端 API。
运行：
  pip install -r requirements.txt -i https://pypi.org/simple
  uvicorn server:app --host 127.0.0.1 --port 8000

"""
import os
import io
import re
import sys
import json
import time
import uuid
import signal
import hmac
import ipaddress
import codecs
import shutil
import socket
import threading
from collections import Counter
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Set
from urllib.parse import quote, urlsplit
try:
    # Python 3.8+
    from typing import Literal
except Exception:  # pragma: no cover
    # 兼容旧版本
    from typing_extensions import Literal  # type: ignore

from pathlib import Path
from multiprocessing import Process

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from . import __version__ as PACKAGE_VERSION
except ImportError:  # pragma: no cover - direct ``uvicorn server:app`` usage
    from text2dialog import __version__ as PACKAGE_VERSION

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.getenv("TEXT2DIALOG_STATIC_DIR") or os.path.join(APP_ROOT, "static"))
JOBS_DIR = os.path.abspath(os.getenv("TEXT2DIALOG_JOBS_DIR") or os.path.join(APP_ROOT, "jobs"))
os.makedirs(JOBS_DIR, exist_ok=True)

_TRUE_SET = {"1", "true", "yes", "on"}
_JOB_ID_RE = re.compile(r"^[0-9a-fA-F]{12}$")
_ARTIFACT_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

MAX_UPLOAD_BYTES = int(os.getenv("TEXT2DIALOG_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 64 * 1024
MAX_JSON_BODY_BYTES = 256 * 1024
MAX_PREVIEW_BYTES = 64 * 1024
MAX_PREVIEW_LINES = 100
MAX_CONCURRENT_JOBS = int(os.getenv("TEXT2DIALOG_MAX_CONCURRENT_JOBS", "4"))
MAX_STORED_JOBS = int(os.getenv("TEXT2DIALOG_MAX_STORED_JOBS", "200"))
JOB_RETENTION_SECONDS = int(os.getenv("TEXT2DIALOG_JOB_RETENTION_SECONDS", str(7 * 24 * 60 * 60)))
MAX_JOB_DISK_BYTES = int(os.getenv("TEXT2DIALOG_MAX_JOB_DISK_BYTES", str(512 * 1024 * 1024)))
MAX_PAIR_ROLES = 64
MAX_PAIR_COMBINATIONS = 256
MAX_PAIR_FILES = 512
MAX_PAIR_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_REGEX_PATTERNS = 16
MAX_REGEX_LENGTH = 128
PROCESS_START_GRACE_SECONDS = 10.0
PROCESS_UNKNOWN_GRACE_SECONDS = 60.0
_PREVIEW_ARTIFACTS = {
    "extraction",
    "validated",
    "quality_report",
    "validation_report",
    "pair_diagnostics",
    "chatml",
}
_KNOWN_PLATFORMS = {
    "deepseek", "siliconflow", "bailian", "moonshot", "openai", "gemini",
    "aws_bedrock", "custom",
}
_TERMINAL_STATUSES = {"cancelled", "succeeded", "failed"}
_ACTIVE_STATUSES = {"running", "paused", "cancelling"}

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_SET

ALLOW_REMOTE = _env_flag("TEXT2DIALOG_ALLOW_REMOTE", default=False)
ALLOW_EXTERNAL_PATHS = _env_flag("TEXT2DIALOG_ALLOW_EXTERNAL_PATHS", default=False)
TRUST_PROXY_HEADERS = _env_flag("TEXT2DIALOG_TRUST_PROXY_HEADERS", default=False)
ALLOW_LOCAL_MODEL_ENDPOINTS = _env_flag("TEXT2DIALOG_ALLOW_LOCAL_MODEL_ENDPOINTS", default=False)
REMOTE_API_TOKEN = (os.getenv("TEXT2DIALOG_API_TOKEN") or "").strip()
_START_CAPACITY_LOCK = threading.Lock()

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

# ---- 延迟导入 ----
DialogueChain = None
Config = None
ModelPlatform = None
validator = None
pair_builder = None
p2c = None
CancelledErrorCls = None

def _import_project_modules():
    global DialogueChain, Config, ModelPlatform, validator, pair_builder, p2c, CancelledErrorCls
    if all([DialogueChain, Config, ModelPlatform, validator, pair_builder, p2c, CancelledErrorCls]):
        return
    try:
        try:
            from .dialogue_chain import DialogueChain as _DC, CancelledError as _CE
            from .config import Config as _CFG, ModelPlatform as _MP
            from . import validate_output as _V
            from . import pair_dataset_builder as _PB
            from . import pair_to_chatml as _P2C
        except ImportError:
            from dialogue_chain import DialogueChain as _DC, CancelledError as _CE
            from config import Config as _CFG, ModelPlatform as _MP
            import validate_output as _V
            import pair_dataset_builder as _PB
            import pair_to_chatml as _P2C
        DialogueChain, Config, ModelPlatform = _DC, _CFG, _MP
        validator, pair_builder, p2c = _V, _PB, _P2C
        CancelledErrorCls = _CE
    except SyntaxError as exc:
        raise RuntimeError("导入项目模块失败，请检查 dialogue_chain.py 语法") from exc

# ---- 工具函数 ----
_JOBS: Dict[str, Dict[str, Any]] = {}

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _require_job_id(job_id: str) -> str:
    jid = (job_id or "").strip()
    if not _JOB_ID_RE.fullmatch(jid):
        raise HTTPException(status_code=400, detail="invalid job_id")
    return jid.lower()

def _is_within(base_dir: str, target_path: str) -> bool:
    try:
        base = os.path.normcase(os.path.realpath(os.path.abspath(base_dir)))
        target = os.path.normcase(os.path.realpath(os.path.abspath(target_path)))
        return os.path.commonpath([base, target]) == base
    except Exception:
        return False

def _resolve_job_scoped_path(job_id: str, raw_path: str, field_name: str) -> str:
    text = (raw_path or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    job_root = os.path.realpath(os.path.abspath(_job_dir(job_id)))
    if os.path.isabs(text):
        resolved = os.path.realpath(os.path.abspath(os.path.expanduser(text)))
    else:
        resolved = os.path.realpath(os.path.abspath(os.path.join(job_root, text)))
    if not ALLOW_EXTERNAL_PATHS and not _is_within(job_root, resolved):
        raise HTTPException(status_code=400, detail=f"{field_name} must be inside this job directory")
    return resolved

def _normalize_optional_at_file(job_id: str, value: Optional[str], field_name: str) -> Optional[str]:
    if not value:
        return value
    s = value.strip()
    if not s.startswith("@"):
        return value
    safe_path = _resolve_job_scoped_path(job_id, s[1:], field_name)
    return "@" + safe_path

def _client_host(request: Request) -> str:
    host = (request.client.host if request.client else "") or ""
    # Forwarded headers are meaningful only when the immediate peer is local.
    # Otherwise a remote caller could claim to be 127.0.0.1 and bypass auth.
    if TRUST_PROXY_HEADERS and _is_loopback(host):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            host = xff.split(",", 1)[0].strip()
    return host

def _is_loopback(host: str) -> bool:
    h = (host or "").strip()
    if not h:
        return False
    if h.lower() in {"localhost", "testclient", "testserver"}:
        return True
    if "%" in h:
        h = h.split("%", 1)[0]
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _process_state(pid: Any) -> str:
    """Return alive/dead/unknown without treating access denial as success."""
    if isinstance(pid, bool):
        return "dead"
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return "dead"
    if process_id <= 0:
        return "dead"
    try:
        os.kill(process_id, 0)
        return "alive"
    except PermissionError:
        return "unknown"
    except (ProcessLookupError, OSError):
        return "dead"
    except Exception:
        return "unknown"


def _process_start_marker(pid: Any) -> Optional[int]:
    """Best-effort OS process creation marker used to reduce PID-reuse risk."""
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return None
    if process_id <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)
            if not handle:
                return None
            try:
                created = wintypes.FILETIME()
                exited = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                if not ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return None
                return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return None
    try:
        stat_path = f"/proc/{process_id}/stat"
        with open(stat_path, "r", encoding="ascii") as stream:
            # The second field can contain spaces and parentheses. Everything
            # after the final ')' begins with state; starttime is field 22.
            tail = stream.read().rsplit(")", 1)[1].strip().split()
        return int(tail[19])
    except Exception:
        return None


def _process_identity_matches(job: Dict[str, Any]) -> bool:
    if str(job.get("status") or "").lower() not in _ACTIVE_STATUSES:
        return False
    if not job.get("worker_token") or _process_state(job.get("pid")) != "alive":
        return False
    expected = job.get("process_start_marker")
    actual = _process_start_marker(job.get("pid"))
    if expected is None or actual is None:
        return False
    return int(expected) == int(actual)


def _address_is_unsafe(address: str, *, allow_local: bool) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if ip.is_unspecified or ip.is_multicast or ip.is_link_local:
        return True
    if not allow_local and (ip.is_loopback or ip.is_private or not ip.is_global):
        return True
    return False


def _validate_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        parsed.port  # force malformed-port validation
        if parsed.scheme != "https" and not ALLOW_LOCAL_MODEL_ENDPOINTS:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("base_url must be a credential-free HTTPS URL") from exc

    hostname = parsed.hostname.rstrip(".")
    try:
        literal = ipaddress.ip_address(hostname.split("%", 1)[0])
        addresses = {str(literal)}
    except ValueError:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except (socket.gaierror, OSError) as exc:
            raise ValueError("base_url hostname could not be safely resolved") from exc
    if not addresses or any(
        _address_is_unsafe(address, allow_local=ALLOW_LOCAL_MODEL_ENDPOINTS)
        for address in addresses
    ):
        if ALLOW_LOCAL_MODEL_ENDPOINTS:
            raise ValueError("base_url resolves to an unsafe link-local or special address")
        raise ValueError("base_url resolves to a private or unsafe address")
    return value.rstrip("/")


def _effective_model_endpoint(platform: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
    """Resolve and validate the endpoint that the worker will actually use.

    Request-only validation is insufficient because every provider BaseURL can
    also be overridden through the environment. Returning the exact validated
    value lets the child process use the same configuration rather than reading
    a different value later.
    """
    selected = (platform or os.getenv("LLM_PLATFORM") or Config.CURRENT_PLATFORM).strip()
    try:
        platform_config = ModelPlatform.get_platform_config(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported platform") from exc
    effective = base_url or os.getenv(platform_config["base_url_env"]) or platform_config["default_base_url"]
    return selected, _validate_base_url(effective)


def _safe_regex(pattern: str) -> bool:
    if len(pattern) > MAX_REGEX_LENGTH:
        return False
    # Backreferences, lookarounds, and nested quantified groups are the common
    # catastrophic cases for user-provided patterns in Python's backtracking RE.
    if re.search(r"\\[1-9]|\(\?[=!<]", pattern):
        return False
    if re.search(r"\((?:[^()\\]|\\.)*[*+{](?:[^()\\]|\\.)*\)[*+{]", pattern):
        return False
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def _parse_host_header(raw: str) -> Optional[tuple[str, Optional[int]]]:
    value = (raw or "").strip()
    if not value or any(ch in value for ch in "\\/@,\r\n\t"):
        return None
    try:
        parsed = urlsplit("//" + value)
        if not parsed.hostname or parsed.username or parsed.password:
            return None
        return parsed.hostname.rstrip(".").lower(), parsed.port
    except (TypeError, ValueError):
        return None


def _effective_port(scheme: str, port: Optional[int]) -> Optional[int]:
    if port is not None:
        return port
    return 443 if scheme.lower() == "https" else 80 if scheme.lower() == "http" else None


def _same_origin(request: Request, origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
        request_host = _parse_host_header(request.headers.get("host") or "")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or request_host is None
        ):
            return False
        host, port = request_host
        return (
            parsed.hostname.rstrip(".").lower() == host
            and parsed.scheme.lower() == request.url.scheme.lower()
            and _effective_port(parsed.scheme, parsed.port)
            == _effective_port(request.url.scheme, port)
        )
    except (TypeError, ValueError):
        return False

def _extract_api_token(request: Request) -> str:
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    key = (request.headers.get("x-api-key") or "").strip()
    return key

def _job_dir(job_id: str, *, create: bool = True) -> str:
    jid = _require_job_id(job_id)
    d = os.path.join(JOBS_DIR, jid)
    if create:
        os.makedirs(d, exist_ok=True)
    return d

@contextmanager
def _job_metadata_lock(job_id: str):
    """Serialize cross-process read/modify/write cycles for one job."""
    lock_path = os.path.join(_job_dir(job_id), ".metadata.lock")
    fd: Optional[int] = None
    deadline = time.monotonic() + 5.0
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(lock_path, "r", encoding="ascii") as stream:
                    owner_pid = stream.read().strip()
                if (
                    time.time() - os.path.getmtime(lock_path) > 30
                    and _process_state(owner_pid) == "dead"
                ):
                    os.remove(lock_path)
                    continue
            except (FileNotFoundError, ValueError):
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError("job metadata is busy")
            time.sleep(0.01)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        yield
    finally:
        try:
            os.close(fd)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def _save_job(job_id: str, job: Optional[Dict[str, Any]] = None) -> None:
    """Atomically replace job.json so readers never observe partial JSON."""
    snapshot = dict(job if job is not None else _JOBS[job_id])
    d = _job_dir(job_id)
    fp = os.path.join(d, "job.json")
    tmp = os.path.join(d, f".job.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, fp)
        _JOBS[job_id] = snapshot
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass

def _load_job(job_id: str) -> Dict[str, Any]:
    # 始终以磁盘为准，避免多进程更新不一致
    d = _job_dir(job_id, create=False)
    fp = os.path.join(d, "job.json")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            job = json.load(f)
        _JOBS[job_id] = job
        return job
    raise KeyError(job_id)

def _load_job_or_404(job_id: str) -> Dict[str, Any]:
    try:
        return _load_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

def _is_process_alive(pid: Any) -> bool:
    return _process_state(pid) == "alive"

def _job_start_lock_path(job_id: str) -> str:
    cache_dir = os.path.join(_job_dir(job_id), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "start.lock")

@contextmanager
def _job_start_lock(job_id: str):
    lock_path = _job_start_lock_path(job_id)
    fd: Optional[int] = None
    for attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            # 仅清理很短暂之外的陈旧锁，避免极端崩溃后永久阻塞启动。
            if attempt == 0:
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                    with open(lock_path, "r", encoding="utf-8") as stream:
                        owner_pid = stream.read().strip()
                    if age > 30 and _process_state(owner_pid) == "dead":
                        os.remove(lock_path)
                        continue
                except FileNotFoundError:
                    continue
                except Exception:
                    pass
            raise HTTPException(status_code=409, detail="job start is already in progress")
    if fd is None:
        raise HTTPException(status_code=409, detail="job start is already in progress")
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except Exception:
            pass

def _job_matches(
    job: Dict[str, Any],
    *,
    expected_generation: Optional[int] = None,
    expected_worker_token: Optional[str] = None,
    expected_operation: Optional[tuple[str, str]] = None,
) -> bool:
    if expected_generation is not None and int(job.get("generation") or 0) != expected_generation:
        return False
    if expected_worker_token is not None and job.get("worker_token") != expected_worker_token:
        return False
    if expected_operation is not None:
        field, token = expected_operation
        if job.get(field) != token:
            return False
    return True


def _update_job(
    job_id: str,
    *,
    expected_generation: Optional[int] = None,
    expected_worker_token: Optional[str] = None,
    expected_operation: Optional[tuple[str, str]] = None,
    **patch: Any,
) -> bool:
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if not _job_matches(
            job,
            expected_generation=expected_generation,
            expected_worker_token=expected_worker_token,
            expected_operation=expected_operation,
        ):
            return False
        job.update(patch)
        job["updated_at"] = _now_str()
        _save_job(job_id, job)
        return True


def _transition_job(
    job_id: str,
    target: str,
    allowed_from: Set[str],
    *,
    conflict: bool = False,
    expected_generation: Optional[int] = None,
    expected_worker_token: Optional[str] = None,
    **patch: Any,
) -> bool:
    """Compare-and-set a job state without overwriting concurrent cancellation."""
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if not _job_matches(
            job,
            expected_generation=expected_generation,
            expected_worker_token=expected_worker_token,
        ):
            return False
        current = str(job.get("status") or "").lower()
        if current not in allowed_from:
            if conflict:
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot transition job from {current or 'unknown'} to {target}",
                )
            return False
        job.update(patch)
        job["status"] = target
        if target in _TERMINAL_STATUSES:
            job["pid"] = None
            job["worker_token"] = None
            job["process_start_marker"] = None
        job["updated_at"] = _now_str()
        _save_job(job_id, job)
        return True

def _set_job_artifact(
    job_id: str,
    name: str,
    path: str,
    *,
    expected_generation: Optional[int] = None,
    expected_worker_token: Optional[str] = None,
    **patch: Any,
) -> Dict[str, Any]:
    if not _ARTIFACT_RE.fullmatch(name):
        raise ValueError("invalid artifact name")
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if not _job_matches(
            job,
            expected_generation=expected_generation,
            expected_worker_token=expected_worker_token,
        ):
            return dict(job.get("artifacts") or {})
        artifacts = dict(job.get("artifacts") or {})
        artifact_generations = dict(job.get("artifact_generations") or {})
        artifacts[name] = path
        artifact_generations[name] = int(job.get("generation") or 0)
        job.update(patch)
        job["artifacts"] = artifacts
        job["artifact_generations"] = artifact_generations
        job["updated_at"] = _now_str()
        _save_job(job_id, job)
        return artifacts


def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return the UI contract without filesystem paths, secrets, or process IDs."""
    job_id = str(job.get("id") or "")
    result: Dict[str, Any] = {
        "id": job_id,
        "status": job.get("status"),
        "generation": int(job.get("generation") or 0),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "message": job.get("message") or "",
        "progress": job.get("progress") or {},
        "artifacts": {
            str(name): f"/api/jobs/{job_id}/download?which={quote(str(name))}"
            for name, path in (job.get("artifacts") or {}).items()
            if (
                path
                and _ARTIFACT_RE.fullmatch(str(name))
                and (
                    not job.get("generation")
                    or int((job.get("artifact_generations") or {}).get(name, -1))
                    == int(job.get("generation") or 0)
                )
            )
        },
    }
    for key in (
        "stats", "quality_summary", "validation_summary", "pair_quality_summary",
        "validation_status", "pairs_status", "chatml_status",
    ):
        if key in job:
            result[key] = job[key]
    return result


def _public_progress(progress: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key in (
        "processed", "total", "processed_chunks", "total_chunks", "stage",
        "message", "elapsed_sec", "speed_cps", "eta_sec", "timestamp",
    ):
        if key in progress:
            value = progress[key]
            safe[key] = value[:1000] if isinstance(value, str) else value
    return safe


def _atomic_json_file(path: str, value: Any) -> None:
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def _remove_job_artifacts(
    job_id: str,
    names: Set[str],
    *,
    expected_generation: Optional[int] = None,
) -> bool:
    """Remove stale metadata first, then best-effort delete only job-scoped files."""
    paths: List[str] = []
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if not _job_matches(job, expected_generation=expected_generation):
            return False
        artifacts = dict(job.get("artifacts") or {})
        artifact_generations = dict(job.get("artifact_generations") or {})
        for name in names:
            raw = artifacts.pop(name, None)
            artifact_generations.pop(name, None)
            if raw:
                try:
                    paths.append(_resolve_job_scoped_path(job_id, str(raw), f"artifact:{name}"))
                except HTTPException:
                    pass
        job["artifacts"] = artifacts
        job["artifact_generations"] = artifact_generations
        if "extraction" in names:
            job.pop("stats", None)
            job.pop("quality_summary", None)
        if "validated" in names:
            job.pop("validation_summary", None)
        if names & {"pairs_zip", "pair_diagnostics"}:
            job.pop("pair_quality_summary", None)
            job.pop("pairs_source", None)
        job["updated_at"] = _now_str()
        _save_job(job_id, job)
    for path in paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except FileNotFoundError:
            pass
    return True


def _safe_remove_job_path(job_id: str, raw_path: Any) -> None:
    if not raw_path:
        return
    try:
        job_root = os.path.abspath(_job_dir(job_id, create=False))
        path = os.path.abspath(str(raw_path))
        if path == job_root or not _is_within(job_root, path):
            return
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        # Metadata is already invalidated. Cleanup is retried by retention;
        # never republish a stale path merely because Windows holds a handle.
        pass


def _begin_extraction(job_id: str) -> tuple[int, str, str, str]:
    stale_paths: List[Any] = []
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        status = str(job.get("status") or "").lower()
        if status in _ACTIVE_STATUSES:
            raise HTTPException(status_code=409, detail="job is already running")
        if status not in {"created", "cancelled", "succeeded", "failed"}:
            raise HTTPException(status_code=409, detail="job cannot be started from its current state")
        stale_paths.extend((job.get("artifacts") or {}).values())
        stale_paths.append(job.get("pairs_source"))
        generation = int(job.get("generation") or 0) + 1
        worker_token = uuid.uuid4().hex
        cache_dir = os.path.join(
            _job_dir(job_id),
            ".cache",
            f"generation-{generation}-{worker_token[:12]}",
        )
        os.makedirs(cache_dir, exist_ok=True)
        for field in (
            "stats", "quality_summary", "validation_summary", "pair_quality_summary",
            "pairs_source", "pairs_source_generation", "validation_token", "pairs_token",
            "chatml_token", "validation_status", "pairs_status", "chatml_status",
        ):
            job.pop(field, None)
        job.update(
            generation=generation,
            worker_token=worker_token,
            worker_started_epoch=time.time(),
            process_start_marker=None,
            pid=None,
            cache_dir=cache_dir,
            artifacts={},
            artifact_generations={},
            status="running",
            message="extraction started",
            updated_at=_now_str(),
        )
        _save_job(job_id, job)
        input_file = str(job["input_file"])
    for path in stale_paths:
        _safe_remove_job_path(job_id, path)
    return generation, worker_token, cache_dir, input_file


def _begin_derived_operation(
    job_id: str,
    *,
    kind: str,
    generation: int,
    invalidate: Set[str],
) -> str:
    if kind not in {"validation", "pairs", "chatml"}:
        raise ValueError("invalid derived operation")
    token_field = f"{kind}_token"
    status_field = f"{kind}_status"
    stale_paths: List[Any] = []
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if int(job.get("generation") or 0) != generation:
            raise HTTPException(status_code=409, detail="extraction generation changed")
        if str(job.get("status") or "").lower() != "succeeded":
            raise HTTPException(status_code=409, detail="extraction must succeed first")
        artifacts = dict(job.get("artifacts") or {})
        artifact_generations = dict(job.get("artifact_generations") or {})
        for name in invalidate:
            stale_paths.append(artifacts.pop(name, None))
            artifact_generations.pop(name, None)
        if invalidate & {"pairs_zip", "pair_diagnostics"}:
            stale_paths.append(job.pop("pairs_source", None))
            job.pop("pairs_source_generation", None)
            job.pop("pair_quality_summary", None)
            job.pop("pairs_status", None)
            job.pop("pairs_token", None)
        if "validated" in invalidate:
            job.pop("validation_summary", None)
            job.pop("validation_status", None)
            job.pop("validation_token", None)
        if "chatml" in invalidate:
            job.pop("chatml_status", None)
            job.pop("chatml_token", None)
        operation_token = uuid.uuid4().hex
        job[token_field] = operation_token
        job[status_field] = "running"
        job["artifacts"] = artifacts
        job["artifact_generations"] = artifact_generations
        job["updated_at"] = _now_str()
        _save_job(job_id, job)
    for path in stale_paths:
        _safe_remove_job_path(job_id, path)
    return operation_token


def _abort_derived_operation(
    job_id: str,
    kind: str,
    generation: int,
    operation_token: str,
    *staged_paths: Any,
) -> None:
    """Converge a failed derived operation and remove only its staging paths."""
    token_field = f"{kind}_token"
    status_field = f"{kind}_status"
    try:
        _update_job(
            job_id,
            expected_generation=generation,
            expected_operation=(token_field, operation_token),
            **{token_field: None, status_field: "failed"},
        )
    except (KeyError, OSError, RuntimeError):
        pass
    for path in staged_paths:
        _safe_remove_job_path(job_id, path)


def _reconcile_job(job_id: str) -> Dict[str, Any]:
    """Repair an active job only when its worker is definitely gone."""
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        status = str(job.get("status") or "").lower()
        if status not in _ACTIVE_STATUSES:
            if job.get("pid") is not None or job.get("worker_token") is not None:
                job["pid"] = None
                job["worker_token"] = None
                job["process_start_marker"] = None
                _save_job(job_id, job)
            return job
        try:
            age = time.time() - float(job.get("worker_started_epoch"))
        except (TypeError, ValueError):
            age = PROCESS_START_GRACE_SECONDS + 1
        pid = job.get("pid")
        if pid is None and age <= PROCESS_START_GRACE_SECONDS:
            return job
        if pid is not None:
            process_state = _process_state(pid)
            if process_state == "alive":
                return job
            # Access denial or another indeterminate result is not proof of a
            # live worker. Give a newly started child a bounded grace period,
            # then converge metadata without ever sending a signal to the PID.
            if process_state == "unknown" and age <= PROCESS_UNKNOWN_GRACE_SECONDS:
                return job
        target = "cancelled" if status == "cancelling" else "failed"
        job.update(
            status=target,
            message="worker exited before completion",
            pid=None,
            worker_token=None,
            process_start_marker=None,
            updated_at=_now_str(),
        )
        _save_job(job_id, job)
        return job


def _iter_job_records() -> List[tuple[str, Dict[str, Any], float]]:
    records: List[tuple[str, Dict[str, Any], float]] = []
    try:
        entries = list(os.scandir(JOBS_DIR))
    except FileNotFoundError:
        return records
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False) or not _JOB_ID_RE.fullmatch(entry.name):
            continue
        metadata = os.path.join(entry.path, "job.json")
        try:
            with open(metadata, "r", encoding="utf-8") as stream:
                job = json.load(stream)
            records.append((entry.name.lower(), job, os.path.getmtime(metadata)))
        except (OSError, ValueError, TypeError):
            continue
    return records


def _active_job_count(*, exclude: Optional[str] = None) -> int:
    return sum(
        1
        for job_id, job, _ in _iter_job_records()
        if job_id != exclude and str(job.get("status") or "").lower() in _ACTIVE_STATUSES
    )


def _claim_job_for_deletion(job_id: str) -> bool:
    """CAS a non-active job to deleting while excluding concurrent starts."""
    try:
        with _job_start_lock(job_id):
            with _job_metadata_lock(job_id):
                job = _load_job(job_id)
                if str(job.get("status") or "").lower() in _ACTIVE_STATUSES:
                    return False
                job.update(status="deleting", pid=None, worker_token=None, updated_at=_now_str())
                _save_job(job_id, job)
                return True
    except (HTTPException, KeyError, OSError, RuntimeError):
        return False


def _cleanup_stale_jobs() -> None:
    records = _iter_job_records()
    now = time.time()
    removable = [
        record
        for record in records
        if str(record[1].get("status") or "").lower() not in _ACTIVE_STATUSES
    ]
    selected = {
        job_id
        for job_id, _, modified in removable
        if JOB_RETENTION_SECONDS > 0 and now - modified > JOB_RETENTION_SECONDS
    }
    remaining = len(records) - len(selected)
    if MAX_STORED_JOBS > 0 and remaining >= MAX_STORED_JOBS:
        for job_id, _, _ in sorted(removable, key=lambda item: item[2]):
            if job_id in selected:
                continue
            selected.add(job_id)
            remaining -= 1
            if remaining < MAX_STORED_JOBS:
                break
    jobs_root = os.path.abspath(JOBS_DIR)
    for job_id in selected:
        path = os.path.abspath(os.path.join(jobs_root, job_id))
        if (
            path != jobs_root
            and _is_within(jobs_root, path)
            and _claim_job_for_deletion(job_id)
        ):
            shutil.rmtree(path, ignore_errors=True)
            _JOBS.pop(job_id, None)


def _job_disk_usage(job_id: str) -> int:
    total = 0
    root = _job_dir(job_id, create=False)
    for current, _, files in os.walk(root):
        for filename in files:
            try:
                total += os.path.getsize(os.path.join(current, filename))
            except OSError:
                continue
            if total > MAX_JOB_DISK_BYTES:
                return total
    return total


def _control_transition(
    job_id: str,
    target: str,
    allowed_from: Set[str],
    reason: Optional[str],
    default_message: str,
    *,
    expected_generation: Optional[int] = None,
    expected_worker_token: Optional[str] = None,
) -> Dict[str, Any]:
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if not _job_matches(
            job,
            expected_generation=expected_generation,
            expected_worker_token=expected_worker_token,
        ):
            raise HTTPException(status_code=409, detail="worker generation changed")
        current = str(job.get("status") or "").lower()
        if current not in allowed_from:
            raise HTTPException(
                status_code=409,
                detail=f"cannot {target} a job in {current or 'unknown'} state",
            )
        _write_control(
            job_id,
            target,
            reason,
            cache_dir=job.get("cache_dir"),
            generation=int(job.get("generation") or 0),
            worker_token=job.get("worker_token"),
        )
        job.update(
            status=target,
            message=reason or default_message,
            updated_at=_now_str(),
        )
        _save_job(job_id, job)
        return job

def _summarize_diagnostics(path: str) -> Dict[str, Any]:
    total = 0
    by_action: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                total += 1
                if obj.get("action"):
                    by_action[str(obj.get("action"))] += 1
                if obj.get("reason"):
                    by_reason[str(obj.get("reason"))] += 1
    except Exception:
        pass
    return {"events": total, "by_action": dict(by_action), "by_reason": dict(by_reason)}

def _read_progress(cache_dir: str) -> Dict[str, Any]:
    if not cache_dir:
        return {}
    try:
        fp = os.path.join(cache_dir, "progress.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _set_platform_env(platform: str, api_key: Optional[str], base_url: Optional[str], model_name: Optional[str]) -> None:
    if not platform:
        return
    os.environ["LLM_PLATFORM"] = platform
    # 获取该平台的变量名
    cfg = ModelPlatform.get_platform_config(platform)
    if api_key:
        os.environ[cfg["api_key_env"]] = api_key
    if base_url:
        os.environ[cfg["base_url_env"]] = base_url
    if model_name:
        os.environ[f"{platform.upper()}_MODEL_NAME"] = model_name

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
        if v is None or k not in mapping:
            continue
        attr, caster = mapping[k]
        try:
            setattr(Config, attr, caster(v))
        except Exception:
            setattr(Config, attr, v)

# ---- 控制文件（暂停/继续/取消）辅助函数 ----
def _control_path(job_id: str, cache_dir: Optional[str] = None) -> str:
    root = cache_dir or os.path.join(_job_dir(job_id), ".cache")
    return os.path.join(root, "control.json")

def _write_control(
    job_id: str,
    state: str,
    reason: Optional[str] = None,
    *,
    cache_dir: Optional[str] = None,
    generation: Optional[int] = None,
    worker_token: Optional[str] = None,
) -> Dict[str, Any]:
    ctrl = {
        "state": state,
        "reason": reason or "",
        "ts": _now_str(),
        "generation": generation,
        "worker_token": worker_token,
    }
    path = _control_path(job_id, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ctrl, f, ensure_ascii=False)
    os.replace(tmp, path)
    return ctrl

def _read_control(job_id: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    path = _control_path(job_id, cache_dir)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"state": "running", "reason": "", "ts": _now_str()}

# ---------- FastAPI ----------
app = FastAPI(title="Text2Dialog", version=PACKAGE_VERSION)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}


def _harden_response(response: Response) -> Response:
    for name, value in _SECURITY_HEADERS.items():
        response.headers[name] = value
    return response


def _guard_error(status_code: int, detail: str) -> Response:
    return _harden_response(JSONResponse(status_code=status_code, content={"detail": detail}))

@app.middleware("http")
async def access_guard(request: Request, call_next):
    peer_host = _client_host(request)
    is_local = _is_loopback(peer_host)
    parsed_host = _parse_host_header(request.headers.get("host") or "")
    if parsed_host is None:
        return _guard_error(400, "invalid Host header")

    # A loopback peer with a non-loopback Host is a DNS-rebinding request.
    if is_local and not _is_loopback(parsed_host[0]):
        return _guard_error(403, "Host is not allowed")

    fetch_site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if fetch_site in {"cross-site", "same-site"}:
        return _guard_error(403, "cross-origin request denied")
    origin = (request.headers.get("origin") or "").strip()
    if origin and not _same_origin(request, origin):
        return _guard_error(403, "cross-origin request denied")

    if request.method in {"POST", "PUT", "PATCH"} and request.url.path != "/api/jobs/create":
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                if int(raw_length) < 0:
                    raise ValueError
                if int(raw_length) > MAX_JSON_BODY_BYTES:
                    return _guard_error(413, "request body too large")
            except ValueError:
                return _guard_error(400, "invalid Content-Length")
        chunks: List[bytes] = []
        actual_length = 0
        async for chunk in request.stream():
            actual_length += len(chunk)
            if actual_length > MAX_JSON_BODY_BYTES:
                return _guard_error(413, "request body too large")
            chunks.append(chunk)
        # Starlette's wrapped receive replays Request._body to downstream
        # parsers, so checking a chunked body does not consume it.
        request._body = b"".join(chunks)  # type: ignore[attr-defined]

    if not is_local:
        if not ALLOW_REMOTE:
            return _guard_error(403, "remote access is disabled")
        public_shell = request.method == "GET" and (
            request.url.path == "/" or request.url.path.startswith("/static/")
        )
        if not public_shell:
            if not REMOTE_API_TOKEN:
                return _guard_error(403, "remote access requires TEXT2DIALOG_API_TOKEN")
            token = _extract_api_token(request)
            if not token or not hmac.compare_digest(token, REMOTE_API_TOKEN):
                return _guard_error(401, "invalid api token")
    response = await call_next(request)
    return _harden_response(response)

class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ExtractReq(_StrictRequest):
    platform: Optional[str] = Field(None, min_length=1, max_length=32)
    api_key: Optional[str] = Field(None, min_length=1, max_length=8192)
    base_url: Optional[str] = Field(None, min_length=1, max_length=2048)
    model_name: Optional[str] = Field(None, min_length=1, max_length=256)
    concurrent: bool = True
    threads: Optional[int] = Field(None, ge=1, le=64)
    save_chunk_text: Optional[bool] = None
    sort_output: Optional[bool] = None
    MAX_TOKEN_LEN: Optional[int] = Field(None, ge=64, le=1_000_000)
    COVER_CONTENT: Optional[int] = Field(None, ge=0, le=100_000)
    TEMPERATURE: Optional[float] = Field(None, ge=0.0, le=2.0, allow_inf_nan=False)
    REPLY_WINDOW: Optional[int] = Field(None, ge=1, le=1000)
    REPLY_CONFIDENCE_TH: Optional[float] = Field(None, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("platform")
    @classmethod
    def _known_platform(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in _KNOWN_PLATFORMS:
            raise ValueError("unsupported platform")
        return value

    @field_validator("base_url")
    @classmethod
    def _safe_base_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_base_url(value)

    @model_validator(mode="after")
    def _coherent_overrides(self):
        if self.base_url and not self.api_key:
            raise ValueError("api_key is required when overriding base_url")
        if (
            self.MAX_TOKEN_LEN is not None
            and self.COVER_CONTENT is not None
            and self.MAX_TOKEN_LEN <= self.COVER_CONTENT
        ):
            raise ValueError("MAX_TOKEN_LEN must be greater than COVER_CONTENT")
        return self

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/api/ping")
def ping():
    return {"ok": True, "time": _now_str()}

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
    }
    return {
        "platforms": platforms,
        "current_platform": Config.CURRENT_PLATFORM,
        "config": cfg,
        "schema_default": Config.DEFAULT_SCHEMA,
    }

@app.get("/api/capabilities")
def api_capabilities():
    """Return a compact integration contract for embedding systems."""
    _import_project_modules()
    return {
        "name": "Text2Dialog",
        "version": app.version,
        "job_id_pattern": _JOB_ID_RE.pattern,
        "paths": {"jobs": "/api/jobs/{job_id}", "static": "/static"},
        "features": {
            "remote_access": ALLOW_REMOTE,
            "external_paths": ALLOW_EXTERNAL_PATHS,
            "proxy_headers": TRUST_PROXY_HEADERS,
            "requires_remote_token": bool(REMOTE_API_TOKEN),
            "local_model_endpoints": ALLOW_LOCAL_MODEL_ENDPOINTS,
            "pause_resume_cancel": True,
            "quality_reports": True,
        },
        "limits": {
            "json_body_bytes": MAX_JSON_BODY_BYTES,
            "upload_bytes": MAX_UPLOAD_BYTES,
            "preview_default": 8,
            "preview_max": MAX_PREVIEW_LINES,
            "concurrent_jobs": MAX_CONCURRENT_JOBS,
            "stored_jobs": MAX_STORED_JOBS,
            "pair_roles": MAX_PAIR_ROLES,
            "pair_combinations": MAX_PAIR_COMBINATIONS,
        },
        "artifacts": [
            "extraction",
            "validated",
            "quality_report",
            "validation_report",
            "pairs_zip",
            "pair_diagnostics",
            "chatml",
        ],
        "platforms": ModelPlatform.list_platforms(),
        "endpoints": {
            "create_job": "POST /api/jobs/create",
            "start_extract": "POST /api/jobs/{job_id}/extract",
            "progress": "GET /api/jobs/{job_id}/progress",
            "preview": "GET /api/jobs/{job_id}/preview?which={artifact}&limit={count}",
            "control": "POST /api/jobs/{job_id}/control",
            "validate": "POST /api/validate",
            "pairs": "POST /api/pairs",
            "chatml": "POST /api/chatml",
            "download": "GET /api/jobs/{job_id}/download?which={artifact}",
        },
    }

@app.post("/api/jobs/create")
async def create_job(file: UploadFile = File(...)):
    _import_project_modules()
    _cleanup_stale_jobs()
    if MAX_STORED_JOBS > 0 and len(_iter_job_records()) >= MAX_STORED_JOBS:
        raise HTTPException(status_code=503, detail="job storage limit reached")
    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    infile = os.path.join(d, "input.txt")
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    total = 0
    try:
        with open(infile, "xb") as f:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="upload exceeds size limit")
                try:
                    decoder.decode(chunk, final=False)
                except UnicodeDecodeError:
                    raise HTTPException(status_code=415, detail="upload must be valid UTF-8 text")
                f.write(chunk)
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                raise HTTPException(status_code=415, detail="upload must be valid UTF-8 text")
            if total == 0:
                raise HTTPException(status_code=400, detail="upload is empty")
            f.flush()
            os.fsync(f.fileno())
        now = _now_str()
        _JOBS[job_id] = {
            "id": job_id,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "input_file": infile,
            "message": "created",
            "progress": {"processed": 0, "total": 0},
            "artifacts": {},
            "artifact_generations": {},
            "generation": 0,
        }
        _save_job(job_id)
        return {"job_id": job_id, "status": "created"}
    except HTTPException:
        _JOBS.pop(job_id, None)
        shutil.rmtree(d, ignore_errors=True)
        raise
    except Exception:
        _JOBS.pop(job_id, None)
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(status_code=500, detail="upload could not be stored")
    finally:
        await file.close()

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    _import_project_modules()
    job_id = _require_job_id(job_id)
    try:
        return _public_job(_reconcile_job(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

@app.get("/api/jobs/{job_id}/progress")
def poll_progress(job_id: str):
    _import_project_modules()
    job_id = _require_job_id(job_id)
    _load_job_or_404(job_id)
    job = _reconcile_job(job_id)
    prog = _public_progress(_read_progress(job.get("cache_dir")))
    return {"progress": prog, "status": job.get("status"), "message": job.get("message")}

@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, which: str):
    _import_project_modules()
    job_id = _require_job_id(job_id)
    if not _ARTIFACT_RE.fullmatch(which or ""):
        raise HTTPException(status_code=400, detail="invalid artifact")
    _load_job_or_404(job_id)
    job = _reconcile_job(job_id)
    art = job.get("artifacts", {})
    path = art.get(which)
    generation = int(job.get("generation") or 0)
    artifact_generation = (job.get("artifact_generations") or {}).get(which)
    if generation and int(artifact_generation if artifact_generation is not None else -1) != generation:
        path = None
    if not path:
        raise HTTPException(status_code=404, detail=f"{which} not found")
    try:
        safe_path = _resolve_job_scoped_path(job_id, str(path), f"artifact:{which}")
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"{which} not found")
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail=f"{which} not found")
    return FileResponse(safe_path, filename=os.path.basename(safe_path))


@app.get("/api/jobs/{job_id}/preview")
def preview_artifact(
    job_id: str,
    which: str = "extraction",
    limit: int = Query(8, ge=1, le=MAX_PREVIEW_LINES),
    lines: Optional[int] = Query(None, ge=1, le=MAX_PREVIEW_LINES),
):
    """Return a deliberately small text preview instead of the full artifact."""
    job_id = _require_job_id(job_id)
    if which not in _PREVIEW_ARTIFACTS:
        raise HTTPException(status_code=400, detail="artifact is not previewable")
    job = _load_job_or_404(job_id)
    generation = int(job.get("generation") or 0)
    artifact_generation = (job.get("artifact_generations") or {}).get(which)
    if generation and int(artifact_generation if artifact_generation is not None else -1) != generation:
        raise HTTPException(status_code=404, detail=f"{which} not found")
    raw_path = (job.get("artifacts") or {}).get(which)
    if not raw_path:
        raise HTTPException(status_code=404, detail=f"{which} not found")
    try:
        path = _resolve_job_scoped_path(job_id, str(raw_path), f"artifact:{which}")
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"{which} not found")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"{which} not found")
    with open(path, "rb") as f:
        payload = f.read(MAX_PREVIEW_BYTES + 1)
    byte_truncated = len(payload) > MAX_PREVIEW_BYTES
    text = payload[:MAX_PREVIEW_BYTES].decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    requested = lines if lines is not None else limit
    selected = all_lines[:requested]
    return {
        "which": which,
        "items": selected,
        "lines": selected,
        "text": "\n".join(selected),
        "truncated": byte_truncated or len(all_lines) > requested,
        "limit": requested,
    }

# ---- 子进程：执行抽取 ----
def _wait_for_worker_state(job_id: str, generation: int, worker_token: str) -> str:
    while True:
        job = _load_job(job_id)
        if not _job_matches(
            job,
            expected_generation=generation,
            expected_worker_token=worker_token,
        ):
            return "stale"
        status = str(job.get("status") or "").lower()
        if status == "paused":
            time.sleep(0.2)
            continue
        return status


def _publish_extraction_result(
    job_id: str,
    generation: int,
    worker_token: str,
    extraction_stage: str,
    quality_stage: Optional[str],
    stats: Dict[str, Any],
    quality_summary: Dict[str, Any],
) -> bool:
    final_extraction = os.path.join(_job_dir(job_id), f"extraction.g{generation}.jsonl")
    final_quality = os.path.join(_job_dir(job_id), f"quality.g{generation}.report.json")
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if (
            not _job_matches(
                job,
                expected_generation=generation,
                expected_worker_token=worker_token,
            )
            or str(job.get("status") or "").lower() != "running"
        ):
            return False
        os.replace(extraction_stage, final_extraction)
        artifacts = dict(job.get("artifacts") or {})
        artifact_generations = dict(job.get("artifact_generations") or {})
        artifacts["extraction"] = final_extraction
        artifact_generations["extraction"] = generation
        if quality_stage and os.path.isfile(quality_stage):
            os.replace(quality_stage, final_quality)
            artifacts["quality_report"] = final_quality
            artifact_generations["quality_report"] = generation
        job.update(
            artifacts=artifacts,
            artifact_generations=artifact_generations,
            stats=stats,
            quality_summary=quality_summary,
            status="succeeded",
            message="extraction completed",
            pid=None,
            worker_token=None,
            process_start_marker=None,
            updated_at=_now_str(),
        )
        _save_job(job_id, job)
        return True


def _worker_extract(
    job_id: str,
    req_body: Dict[str, Any],
    generation: int,
    worker_token: str,
    cache_dir: str,
    input_file: str,
) -> None:
    _import_project_modules()
    out_file: Optional[str] = None
    raw_out_file: Optional[str] = None
    quality_report: Optional[str] = None
    try:
        # 环境与配置仅影响子进程
        os.makedirs(cache_dir, exist_ok=True)

        current_job = _load_job(job_id)
        if (
            not _job_matches(
                current_job,
                expected_generation=generation,
                expected_worker_token=worker_token,
            )
            or str(current_job.get("status") or "").lower() != "running"
        ):
            return

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

        out_file = os.path.join(
            _job_dir(job_id),
            f".extraction.g{generation}.{worker_token}.stage.jsonl",
        )
        raw_out_file = out_file
        _update_job(
            job_id,
            expected_generation=generation,
            expected_worker_token=worker_token,
            message="正在提取对话…",
            cache_dir=cache_dir,
        )

        extractor = DialogueChain(
            schema=None,
            platform=req_body.get("platform") or os.getenv("LLM_PLATFORM"),
            max_workers=req_body.get("threads"),
            save_chunk_text=req_body.get("save_chunk_text"),
            cache_dir=cache_dir,
        )

        try:
            if req_body.get("concurrent", True):
                extractor.extract_dialogues_concurrent(input_file, out_file)
            else:
                extractor.extract_dialogues(input_file, out_file)
        except Exception as e:
            # 识别“取消”并退出
            if CancelledErrorCls and isinstance(e, CancelledErrorCls):
                _transition_job(
                    job_id,
                    "cancelled",
                    {"running", "paused", "cancelling"},
                    expected_generation=generation,
                    expected_worker_token=worker_token,
                    message="cancelled",
                )
                return
            # 其他异常继续抛出
            raise

        current = _wait_for_worker_state(job_id, generation, worker_token)
        control_state = str(_read_control(job_id, cache_dir).get("state") or "").lower()
        if current in {"cancelling", "cancelled"} or control_state in {"cancelling", "cancelled"}:
            _transition_job(
                job_id,
                "cancelled",
                {"running", "paused", "cancelling"},
                expected_generation=generation,
                expected_worker_token=worker_token,
                message="cancelled",
            )
            return
        if current != "running":
            return

        if req_body.get("sort_output") or getattr(Config, "DEFAULT_SORT_OUTPUT", False):
            out_file = extractor.sort_dialogues(out_file)

        quality_report = extractor.write_quality_report(out_file)
        stats = extractor.get_statistics(out_file)
        if _job_disk_usage(job_id) > MAX_JOB_DISK_BYTES:
            raise RuntimeError("job disk limit exceeded")
        final_state = _wait_for_worker_state(job_id, generation, worker_token)
        if final_state == "cancelling":
            _transition_job(
                job_id,
                "cancelled",
                {"cancelling"},
                expected_generation=generation,
                expected_worker_token=worker_token,
                message="cancelled",
            )
            return
        if final_state != "running":
            return
        _publish_extraction_result(
            job_id,
            generation,
            worker_token,
            out_file,
            quality_report,
            stats,
            extractor._quality_summary(),
        )
    except Exception:
        job = _load_job(job_id)
        if not _job_matches(
            job,
            expected_generation=generation,
            expected_worker_token=worker_token,
        ):
            return
        if str(job.get("status") or "").lower() == "cancelling":
            _transition_job(
                job_id,
                "cancelled",
                {"cancelling"},
                expected_generation=generation,
                expected_worker_token=worker_token,
                message="cancelled",
            )
        else:
            _transition_job(
                job_id,
                "failed",
                {"running", "paused"},
                expected_generation=generation,
                expected_worker_token=worker_token,
                message="extraction failed",
            )
    finally:
        # 清理进度文件
        try:
            pf = os.path.join(cache_dir, "progress.json")
            if os.path.exists(pf):
                os.remove(pf)
        except Exception:
            pass
        for staged in (raw_out_file, out_file, quality_report):
            if staged and ".stage" in os.path.basename(staged):
                _safe_remove_job_path(job_id, staged)

@app.post("/api/jobs/{job_id}/extract")
async def run_extract(job_id: str, req: ExtractReq):
    _import_project_modules()
    job_id = _require_job_id(job_id)
    _load_job_or_404(job_id)
    max_token_len = req.MAX_TOKEN_LEN if req.MAX_TOKEN_LEN is not None else Config.MAX_TOKEN_LEN
    cover_content = req.COVER_CONTENT if req.COVER_CONTENT is not None else Config.COVER_CONTENT
    if max_token_len <= cover_content:
        raise HTTPException(status_code=422, detail="MAX_TOKEN_LEN must be greater than COVER_CONTENT")
    try:
        effective_platform, effective_base_url = _effective_model_endpoint(
            req.platform,
            req.base_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request_body = req.model_dump()
    request_body["platform"] = effective_platform
    request_body["base_url"] = effective_base_url
    with _START_CAPACITY_LOCK, _job_start_lock(job_id):
        # 立即读取并验证
        job = _reconcile_job(job_id)
        status = str(job.get("status") or "").lower()
        if status in _ACTIVE_STATUSES:
            raise HTTPException(status_code=409, detail="job is already running")
        if status not in {"created", "cancelled", "succeeded", "failed"}:
            raise HTTPException(status_code=409, detail="job cannot be started from its current state")
        if MAX_CONCURRENT_JOBS > 0 and _active_job_count(exclude=job_id) >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="concurrent job limit reached")

        generation, worker_token, cache_dir, input_file = _begin_extraction(job_id)

        # 启动前将控制状态置为 running
        try:
            _write_control(
                job_id,
                "running",
                "starting",
                cache_dir=cache_dir,
                generation=generation,
                worker_token=worker_token,
            )
        except Exception:
            pass

        # 启动子进程（互不影响）
        p = Process(
            target=_worker_extract,
            args=(job_id, request_body, generation, worker_token, cache_dir, input_file),
        )
        p.daemon = True
        try:
            p.start()
        except Exception:
            _transition_job(
                job_id,
                "failed",
                {"running"},
                expected_generation=generation,
                expected_worker_token=worker_token,
                message="worker could not be started",
            )
            raise HTTPException(status_code=500, detail="worker could not be started")
        _update_job(
            job_id,
            expected_generation=generation,
            expected_worker_token=worker_token,
            pid=p.pid,
            process_start_marker=_process_start_marker(p.pid),
        )
        return {"ok": True, "status": "running", "generation": generation}

class ControlReq(_StrictRequest):
    action: Literal["pause", "resume", "cancel", "force-cancel"]
    reason: Optional[str] = Field(None, max_length=500)

@app.post("/api/jobs/{job_id}/control")
def control_job(job_id: str, req: ControlReq):
    """作业运行时控制：暂停/继续/取消/强制终止"""
    _import_project_modules()
    job_id = _require_job_id(job_id)
    _load_job_or_404(job_id)
    job = _reconcile_job(job_id)
    generation = int(job.get("generation") or 0)
    worker_token = job.get("worker_token")

    action = req.action
    if action == "pause":
        _control_transition(
            job_id, "paused", {"running"}, req.reason, "paused",
            expected_generation=generation, expected_worker_token=worker_token,
        )
        return {"ok": True, "status": "paused"}
    if action == "resume":
        _control_transition(
            job_id, "running", {"paused"}, req.reason, "running",
            expected_generation=generation, expected_worker_token=worker_token,
        )
        return {"ok": True, "status": "running"}
    if action == "cancel":
        _control_transition(
            job_id, "cancelling", {"running", "paused"}, req.reason, "cancelling",
            expected_generation=generation, expected_worker_token=worker_token,
        )
        return {"ok": True, "status": "cancelling"}

    # force-cancel
    job = _control_transition(
        job_id,
        "cancelling",
        {"running", "paused", "cancelling"},
        req.reason,
        "cancelling",
        expected_generation=generation,
        expected_worker_token=worker_token,
    )
    pid = job.get("pid")
    if pid:
        if not _process_identity_matches(job):
            if _process_state(pid) == "dead":
                _transition_job(
                    job_id,
                    "cancelled",
                    {"cancelling"},
                    expected_generation=generation,
                    expected_worker_token=worker_token,
                    message=req.reason or "cancelled",
                )
                return {"ok": True, "status": "cancelled"}
            raise HTTPException(status_code=409, detail="worker identity could not be verified")
        try:
            os.kill(int(pid), getattr(signal, "SIGTERM", signal.SIGINT))
        except ProcessLookupError:
            pass
        except PermissionError:
            raise HTTPException(status_code=409, detail="worker termination was denied")
        except (TypeError, ValueError, OSError):
            raise HTTPException(status_code=409, detail="worker could not be terminated")
    # 清理进度文件，声明作业已取消
    try:
        pf = os.path.join(str(job.get("cache_dir") or ""), "progress.json")
        if os.path.exists(pf):
            os.remove(pf)
    except Exception:
        pass
    _transition_job(
        job_id,
        "cancelled",
        {"cancelling"},
        expected_generation=generation,
        expected_worker_token=worker_token,
        message=req.reason or "cancelled",
    )
    return {"ok": True, "status": "cancelled"}

def _publish_validation_result(
    job_id: str,
    generation: int,
    operation_token: str,
    report: Dict[str, Any],
    report_stage: str,
    validated_stage: Optional[str],
) -> bool:
    report_final = os.path.join(_job_dir(job_id), f"validation.g{generation}.report.json")
    validated_final = os.path.join(_job_dir(job_id), f"extraction.g{generation}.validated.jsonl")
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if (
            not _job_matches(
                job,
                expected_generation=generation,
                expected_operation=("validation_token", operation_token),
            )
            or str(job.get("status") or "").lower() != "succeeded"
        ):
            return False
        os.replace(report_stage, report_final)
        artifacts = dict(job.get("artifacts") or {})
        artifact_generations = dict(job.get("artifact_generations") or {})
        artifacts["validation_report"] = report_final
        artifact_generations["validation_report"] = generation
        if validated_stage and bool(report.get("ok")):
            os.replace(validated_stage, validated_final)
            artifacts["validated"] = validated_final
            artifact_generations["validated"] = generation
        job.update(
            artifacts=artifacts,
            artifact_generations=artifact_generations,
            validation_summary=report,
            validation_status="succeeded" if bool(report.get("ok")) else "failed",
            validation_token=None,
            updated_at=_now_str(),
        )
        _save_job(job_id, job)
        return True


class ValidateReq(_StrictRequest):
    job_id: str = Field(pattern=_JOB_ID_RE.pattern)
    input_path: Optional[str] = Field(None, min_length=1, max_length=4096)

@app.post("/api/validate")
def run_validate(req: ValidateReq):
    _import_project_modules()
    job_id = _require_job_id(req.job_id)
    job = _load_job_or_404(job_id)
    if str(job.get("status") or "").lower() != "succeeded":
        raise HTTPException(status_code=409, detail="extraction must succeed before validation")
    generation = int(job.get("generation") or 0)
    canonical_raw = job.get("artifacts", {}).get("extraction")
    if generation and int((job.get("artifact_generations") or {}).get("extraction", -1)) != generation:
        raise HTTPException(status_code=409, detail="current extraction provenance is invalid")
    src_raw = req.input_path or canonical_raw
    if not src_raw:
        raise HTTPException(status_code=400, detail="no input file available for validate")
    src = _resolve_job_scoped_path(job_id, str(src_raw), "input_path")
    if req.input_path and canonical_raw:
        canonical = _resolve_job_scoped_path(job_id, str(canonical_raw), "extraction")
        if os.path.normcase(src) != os.path.normcase(canonical):
            raise HTTPException(status_code=409, detail="only the current extraction can be validated")
    if not os.path.exists(src):
        raise HTTPException(status_code=400, detail="no input file available for validate")
    operation_token = _begin_derived_operation(
        job_id,
        kind="validation",
        generation=generation,
        invalidate={
            "validated", "validation_report", "pairs_zip", "pair_diagnostics", "chatml",
        },
    )
    try:
        if hasattr(validator, "validate_with_report"):
            report = validator.validate_with_report(src)
        else:
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                code = validator.validate(src)
            report = {
                "ok": code == 0,
                "status": "通过" if code == 0 else "失败",
                "error_count": 0 if code == 0 else 1,
                "messages": [line for line in buf.getvalue().splitlines() if line.strip()],
            }
    except Exception:
        msg = "validation could not be completed"
        report = {"ok": False, "error_count": 1, "messages": [msg]}

    if not isinstance(report, dict):
        report = {"ok": False, "messages": ["validator returned an invalid report"]}
    messages = [str(item)[:1000] for item in (report.get("messages") or [])[:200]]
    report["messages"] = messages
    log = "\n".join(messages)[:100_000]
    report_path = os.path.join(
        _job_dir(job_id),
        f".validation.g{generation}.{operation_token}.report.stage.json",
    )
    try:
        _atomic_json_file(report_path, report)
    except Exception as exc:
        _abort_derived_operation(
            job_id,
            "validation",
            generation,
            operation_token,
            report_path,
        )
        raise HTTPException(status_code=500, detail="validation report could not be staged") from exc
    ok = bool(report.get("ok"))
    validated_stage: Optional[str] = None
    if ok:
        validated_stage = os.path.join(
            _job_dir(job_id),
            f".validation.g{generation}.{operation_token}.data.stage.jsonl",
        )
        try:
            with open(src, "rb") as fr, open(validated_stage, "xb") as fw:
                shutil.copyfileobj(fr, fw, length=UPLOAD_CHUNK_BYTES)
                fw.flush()
                os.fsync(fw.fileno())
        except Exception:
            _abort_derived_operation(
                job_id,
                "validation",
                generation,
                operation_token,
                validated_stage,
                report_path,
            )
            raise HTTPException(status_code=500, detail="validated copy could not be staged")
    try:
        published = _publish_validation_result(
            job_id,
            generation,
            operation_token,
            report,
            report_path,
            validated_stage,
        )
    except Exception as exc:
        _abort_derived_operation(
            job_id,
            "validation",
            generation,
            operation_token,
            report_path,
            validated_stage,
        )
        raise HTTPException(status_code=500, detail="validation result could not be published") from exc
    if not published:
        _safe_remove_job_path(job_id, report_path)
        _safe_remove_job_path(job_id, validated_stage)
        raise HTTPException(status_code=409, detail="extraction generation changed during validation")
    return {"ok": ok, "log": log, "report": report, "generation": generation}

class PairBuildReq(_StrictRequest):
    job_id: str = Field(pattern=_JOB_ID_RE.pattern)
    input_path: Optional[str] = Field(None, min_length=1, max_length=4096)
    out_dir: Optional[str] = Field(None, min_length=1, max_length=4096)
    merge_out: Optional[str] = Field(None, min_length=1, max_length=4096)
    pairs: Optional[List[str]] = Field(None, max_length=MAX_PAIR_COMBINATIONS)
    roles: Optional[List[str]] = Field(None, max_length=MAX_PAIR_ROLES)
    all_ordered_pairs: bool = False
    min_confidence: Optional[float] = Field(0.8, ge=0.0, le=1.0, allow_inf_nan=False)
    require_confidence: bool = False
    strict: bool = True
    min_src_chars: Optional[int] = Field(1, ge=0, le=1_000_000)
    min_reply_chars: Optional[int] = Field(1, ge=0, le=1_000_000)
    max_src_chars: Optional[int] = Field(None, ge=1, le=1_000_000)
    max_reply_chars: Optional[int] = Field(None, ge=1, le=1_000_000)
    deny_pattern: Optional[List[str]] = Field(None, max_length=MAX_REGEX_PATTERNS)
    list_roles: bool = False

    @field_validator("pairs", "roles")
    @classmethod
    def _bounded_string_lists(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return None
        cleaned: List[str] = []
        for value in values:
            value = value.strip()
            if not value or len(value) > 256:
                raise ValueError("list entries must contain 1 to 256 characters")
            cleaned.append(value)
        return cleaned

    @field_validator("deny_pattern")
    @classmethod
    def _valid_patterns(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        for value in values or []:
            if not value or not _safe_regex(value):
                raise ValueError("deny_pattern contains an unsafe regular expression")
        return values

    @model_validator(mode="after")
    def _coherent_lengths(self):
        if self.max_src_chars is not None and self.min_src_chars is not None and self.max_src_chars < self.min_src_chars:
            raise ValueError("max_src_chars must be >= min_src_chars")
        if self.max_reply_chars is not None and self.min_reply_chars is not None and self.max_reply_chars < self.min_reply_chars:
            raise ValueError("max_reply_chars must be >= min_reply_chars")
        return self

def _publish_pairs_result(
    job_id: str,
    generation: int,
    operation_token: str,
    *,
    stage_output: str,
    final_output: str,
    output_is_dir: bool,
    zip_stage: str,
    diagnostics_stage: str,
    diagnostics_summary: Dict[str, Any],
) -> bool:
    zip_final = os.path.join(_job_dir(job_id), f"pairs.g{generation}.zip")
    diagnostics_final = os.path.join(_job_dir(job_id), f"pairs.g{generation}.diagnostics.jsonl")
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if (
            not _job_matches(
                job,
                expected_generation=generation,
                expected_operation=("pairs_token", operation_token),
            )
            or str(job.get("status") or "").lower() != "succeeded"
            or not bool((job.get("validation_summary") or {}).get("ok"))
            or int((job.get("artifact_generations") or {}).get("validated", -1)) != generation
        ):
            return False
        if output_is_dir and not os.path.isdir(stage_output):
            raise RuntimeError("pair output directory was not staged")
        if not output_is_dir and not os.path.isfile(stage_output):
            raise RuntimeError("merged pair output was not staged")
        os.makedirs(os.path.dirname(final_output), exist_ok=True)
        _safe_remove_job_path(job_id, final_output)
        os.replace(stage_output, final_output)
        os.replace(zip_stage, zip_final)
        artifacts = dict(job.get("artifacts") or {})
        artifact_generations = dict(job.get("artifact_generations") or {})
        artifacts["pairs_zip"] = zip_final
        artifact_generations["pairs_zip"] = generation
        if os.path.isfile(diagnostics_stage):
            os.replace(diagnostics_stage, diagnostics_final)
            artifacts["pair_diagnostics"] = diagnostics_final
            artifact_generations["pair_diagnostics"] = generation
        job.update(
            artifacts=artifacts,
            artifact_generations=artifact_generations,
            pairs_source=final_output,
            pairs_source_generation=generation,
            pair_quality_summary=diagnostics_summary,
            pairs_status="succeeded",
            pairs_token=None,
            updated_at=_now_str(),
        )
        _save_job(job_id, job)
        return True


@app.post("/api/pairs")
def build_pairs(req: PairBuildReq):
    _import_project_modules()
    job_id = _require_job_id(req.job_id)
    job = _load_job_or_404(job_id)
    generation = int(job.get("generation") or 0)
    validated_raw = job.get("artifacts", {}).get("validated")
    if (
        not bool((job.get("validation_summary") or {}).get("ok"))
        or not validated_raw
        or int((job.get("artifact_generations") or {}).get("validated", -1)) != generation
    ):
        raise HTTPException(status_code=409, detail="validation must pass before building pairs")
    src_raw = req.input_path or validated_raw
    if not src_raw:
        raise HTTPException(status_code=400, detail="no usable JSONL input")
    src = _resolve_job_scoped_path(job_id, str(src_raw), "input_path")
    canonical_validated = _resolve_job_scoped_path(job_id, str(validated_raw), "validated")
    if os.path.normcase(src) != os.path.normcase(canonical_validated):
        raise HTTPException(status_code=409, detail="pairs must use the current validated extraction")
    if not os.path.exists(src):
        raise HTTPException(status_code=400, detail="no usable JSONL input")

    out_dir_raw = req.out_dir or os.path.join(
        _job_dir(job_id), "pair_datasets", f"generation-{generation}",
    )
    final_out_dir = _resolve_job_scoped_path(job_id, out_dir_raw, "out_dir")
    final_merge_out = (
        _resolve_job_scoped_path(job_id, req.merge_out, "merge_out")
        if req.merge_out
        else None
    )
    job_root = os.path.abspath(_job_dir(job_id))
    protected_paths = {
        os.path.abspath(str(path))
        for path in [job.get("input_file"), job.get("cache_dir"), *(job.get("artifacts") or {}).values()]
        if path
    }
    protected_paths.update(
        {
            os.path.join(job_root, "job.json"),
            os.path.join(job_root, f"pairs.g{generation}.zip"),
            os.path.join(job_root, f"pairs.g{generation}.diagnostics.jsonl"),
        }
    )
    for candidate in (final_out_dir, final_merge_out):
        if candidate and (
            os.path.abspath(candidate) == job_root
            or os.path.normcase(os.path.abspath(candidate)) == os.path.normcase(src)
            or any(
                os.path.normcase(os.path.abspath(candidate)) == os.path.normcase(protected)
                or (
                    candidate == final_out_dir
                    and _is_within(os.path.abspath(candidate), protected)
                )
                for protected in protected_paths
            )
        ):
            raise HTTPException(status_code=400, detail="pair output collides with protected job data")

    # support listing roles only
    if req.list_roles:
        try:
            counts = pair_builder.list_roles(Path(src))
            roles = [r for (r, _) in counts.most_common(MAX_PAIR_ROLES)]
        except Exception:
            roles = []
        return {"roles": roles}

    needs_all_roles = req.all_ordered_pairs or any(
        item.strip().upper() in {"ALL", "全部"}
        for item in (req.pairs or [])
    )
    preflight_roles: Optional[List[str]] = list(req.roles) if req.roles else None
    if needs_all_roles and preflight_roles is None:
        try:
            counts = pair_builder.list_roles(Path(src))
            preflight_roles = [role for role, _ in counts.most_common()]
        except Exception:
            preflight_roles = []
    if preflight_roles is not None and len(preflight_roles) > MAX_PAIR_ROLES:
        raise HTTPException(status_code=422, detail="role expansion exceeds the server limit")
    if (
        needs_all_roles
        and preflight_roles is not None
        and len(preflight_roles) * max(0, len(preflight_roles) - 1) > MAX_PAIR_COMBINATIONS
    ):
        raise HTTPException(status_code=422, detail="pair expansion exceeds the server limit")

    operation_token = _begin_derived_operation(
        job_id,
        kind="pairs",
        generation=generation,
        invalidate={"pairs_zip", "pair_diagnostics", "chatml"},
    )
    stage_base = os.path.join(
        _job_dir(job_id), f".pairs.g{generation}.{operation_token}",
    )
    out_dir = stage_base + ".dir.stage"
    merge_out = stage_base + ".merged.stage.jsonl" if final_merge_out else None
    diagnostics_out = stage_base + ".diagnostics.stage.jsonl"
    zip_path = stage_base + ".zip.stage"
    os.makedirs(out_dir, exist_ok=False)

    # expand ALL to explicit 'A,B'
    def _all_roles() -> List[str]:
        if preflight_roles is not None:
            return list(preflight_roles)
        try:
            counts = pair_builder.list_roles(Path(src))
            roles = [r for (r, _) in counts.most_common()]
        except Exception:
            return []
        if len(roles) > MAX_PAIR_ROLES:
            raise HTTPException(status_code=422, detail="role expansion exceeds the server limit")
        return roles

    if req.pairs:
        roles_for_expand = req.roles or _all_roles()
        expanded_pairs: List[str] = []
        for item in req.pairs:
            s = item.strip()
            if s.upper() in ("ALL", "全部"):
                for a in roles_for_expand:
                    for b in roles_for_expand:
                        if a != b:
                            expanded_pairs.append(f"{a},{b}")
            elif "," in s:
                expanded_pairs.append(s)
        # dedupe while preserving order
        seen = set()
        req.pairs = [p for p in expanded_pairs if (p not in seen and not seen.add(p))]
        if len(req.pairs) > MAX_PAIR_COMBINATIONS:
            raise HTTPException(status_code=422, detail="pair expansion exceeds the server limit")

    argv: List[str] = ["--input", src, "--out", out_dir, "--diagnostics-out", diagnostics_out]
    if merge_out:
        argv = ["--input", src, "--merge-out", merge_out, "--diagnostics-out", diagnostics_out]
    if req.pairs:
        argv.extend(["--pairs"] + req.pairs)

    # normalize roles args to avoid repeated --roles overriding
    roles_args: List[str] = []
    if req.all_ordered_pairs:
        roles_args = req.roles or _all_roles()
    elif req.roles:
        roles_args = req.roles
    if roles_args:
        argv.extend(["--roles"] + roles_args)
    if req.all_ordered_pairs:
        argv.append("--all-ordered-pairs")

    if req.min_confidence is not None:
        argv += ["--min-confidence", str(req.min_confidence)]
    if req.require_confidence:
        argv.append("--require-confidence")
    if req.strict:
        argv.append("--strict")
    else:
        argv.append("--no-strict")
    if req.min_src_chars is not None:
        argv += ["--min-src-chars", str(req.min_src_chars)]
    if req.min_reply_chars is not None:
        argv += ["--min-reply-chars", str(req.min_reply_chars)]
    if req.max_src_chars is not None:
        argv += ["--max-src-chars", str(req.max_src_chars)]
    if req.max_reply_chars is not None:
        argv += ["--max-reply-chars", str(req.max_reply_chars)]
    if req.deny_pattern:
        for pattern in req.deny_pattern:
            argv.extend(["--deny-pattern", pattern])
    if req.list_roles:
        argv.append("--list-roles")

    try:
        code = pair_builder.main(argv)
    except Exception as exc:
        _abort_derived_operation(
            job_id,
            "pairs",
            generation,
            operation_token,
            out_dir,
            merge_out,
            diagnostics_out,
            zip_path,
        )
        raise HTTPException(status_code=500, detail="pair dataset build failed") from exc
    if code != 0:
        _abort_derived_operation(
            job_id,
            "pairs",
            generation,
            operation_token,
            out_dir,
            merge_out,
            diagnostics_out,
            zip_path,
        )
        return {"ok": False, "log": "pair dataset build failed"}

    import zipfile
    archive_files: List[tuple[str, str]] = []
    if merge_out:
        if not os.path.isfile(merge_out):
            _abort_derived_operation(
                job_id,
                "pairs",
                generation,
                operation_token,
                out_dir,
                merge_out,
                diagnostics_out,
                zip_path,
            )
            raise HTTPException(status_code=500, detail="merged pair output was not produced")
        archive_files.append((merge_out, os.path.basename(final_merge_out or "pairs.jsonl")))
    else:
        for root, dirs, files in os.walk(out_dir):
            dirs[:] = [name for name in dirs if not os.path.islink(os.path.join(root, name))]
            for filename in files:
                path = os.path.join(root, filename)
                if os.path.islink(path):
                    _abort_derived_operation(
                        job_id,
                        "pairs",
                        generation,
                        operation_token,
                        out_dir,
                        merge_out,
                        diagnostics_out,
                        zip_path,
                    )
                    raise HTTPException(status_code=422, detail="pair output contains a symbolic link")
                archive_files.append((path, os.path.relpath(path, out_dir)))
    total_pair_bytes = sum(os.path.getsize(path) for path, _ in archive_files)
    if len(archive_files) > MAX_PAIR_FILES or total_pair_bytes > MAX_PAIR_ARTIFACT_BYTES:
        _abort_derived_operation(
            job_id,
            "pairs",
            generation,
            operation_token,
            out_dir,
            merge_out,
            diagnostics_out,
            zip_path,
        )
        raise HTTPException(status_code=413, detail="pair artifact exceeds the server limit")
    if _job_disk_usage(job_id) + total_pair_bytes > MAX_JOB_DISK_BYTES:
        _abort_derived_operation(
            job_id,
            "pairs",
            generation,
            operation_token,
            out_dir,
            merge_out,
            diagnostics_out,
            zip_path,
        )
        raise HTTPException(status_code=413, detail="job disk limit would be exceeded")
    try:
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, archive_name in archive_files:
                archive.write(path, arcname=archive_name)
    except Exception as exc:
        _abort_derived_operation(
            job_id,
            "pairs",
            generation,
            operation_token,
            out_dir,
            merge_out,
            diagnostics_out,
            zip_path,
        )
        raise HTTPException(status_code=500, detail="pair archive could not be staged") from exc
    pair_diag_summary = _summarize_diagnostics(diagnostics_out)
    stage_output = merge_out or out_dir
    final_output = final_merge_out or final_out_dir
    try:
        published = _publish_pairs_result(
            job_id,
            generation,
            operation_token,
            stage_output=stage_output,
            final_output=final_output,
            output_is_dir=merge_out is None,
            zip_stage=zip_path,
            diagnostics_stage=diagnostics_out,
            diagnostics_summary=pair_diag_summary,
        )
    except Exception as exc:
        _abort_derived_operation(
            job_id,
            "pairs",
            generation,
            operation_token,
            out_dir,
            merge_out,
            zip_path,
            diagnostics_out,
        )
        raise HTTPException(status_code=500, detail="pair result could not be published") from exc
    if not published:
        for staged in (out_dir, merge_out, zip_path, diagnostics_out):
            _safe_remove_job_path(job_id, staged)
        raise HTTPException(status_code=409, detail="extraction generation changed during pair build")
    if merge_out:
        _safe_remove_job_path(job_id, out_dir)
    return {
        "ok": True,
        "artifact": "/api/jobs/{}/download?which=pairs_zip".format(job_id),
        "diagnostics": pair_diag_summary,
        "generation": generation,
    }

class ChatMLReq(_StrictRequest):
    job_id: str = Field(pattern=_JOB_ID_RE.pattern)
    input: Optional[str] = Field(None, min_length=1, max_length=4096)
    mode: Literal["pair", "stitch"] = "pair"
    out: Optional[str] = Field(None, min_length=1, max_length=4096)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, allow_inf_nan=False)
    reverse: bool = False
    include_meta: bool = False
    max_turns: Optional[int] = Field(None, ge=1, le=1000)
    dedupe: bool = False
    system_text: Optional[str] = Field(None, max_length=100_000)
    system_template: Optional[str] = Field(None, max_length=4096)

    @model_validator(mode="after")
    def _coherent_role_direction(self):
        if self.mode == "pair" and self.reverse:
            raise ValueError("reverse is not defined for pair mode")
        return self


def _publish_chatml_result(
    job_id: str,
    generation: int,
    operation_token: str,
    stage_path: str,
    final_path: str,
) -> bool:
    with _job_metadata_lock(job_id):
        job = _load_job(job_id)
        if (
            not _job_matches(
                job,
                expected_generation=generation,
                expected_operation=("chatml_token", operation_token),
            )
            or int(job.get("pairs_source_generation") or -1) != generation
        ):
            return False
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        os.replace(stage_path, final_path)
        artifacts = dict(job.get("artifacts") or {})
        artifact_generations = dict(job.get("artifact_generations") or {})
        artifacts["chatml"] = final_path
        artifact_generations["chatml"] = generation
        job.update(
            artifacts=artifacts,
            artifact_generations=artifact_generations,
            chatml_status="succeeded",
            chatml_token=None,
            updated_at=_now_str(),
        )
        _save_job(job_id, job)
        return True

@app.post("/api/chatml")
def build_chatml(req: ChatMLReq):
    _import_project_modules()
    job_id = _require_job_id(req.job_id)
    job = _load_job_or_404(job_id)
    generation = int(job.get("generation") or 0)
    pairs_source = job.get("pairs_source")
    if (
        not pairs_source
        or int(job.get("pairs_source_generation") or -1) != generation
        or job.get("pairs_status") != "succeeded"
    ):
        raise HTTPException(status_code=409, detail="pair dataset must be built before ChatML export")
    inputs_raw = req.input or pairs_source
    out_raw = req.out or os.path.join(_job_dir(job_id), f"chatml.g{generation}.jsonl")
    inputs = _resolve_job_scoped_path(job_id, inputs_raw, "input")
    canonical_pairs = _resolve_job_scoped_path(job_id, str(pairs_source), "pairs_source")
    if os.path.normcase(inputs) != os.path.normcase(canonical_pairs):
        raise HTTPException(status_code=409, detail="ChatML must use the current pair dataset")
    out = _resolve_job_scoped_path(job_id, out_raw, "out")
    protected_outputs = {
        os.path.normcase(os.path.abspath(str(path)))
        for path in [
            job.get("input_file"),
            *(
                path
                for name, path in (job.get("artifacts") or {}).items()
                if name != "chatml"
            ),
        ]
        if path
    }
    if (
        os.path.normcase(out) == os.path.normcase(inputs)
        or os.path.normcase(os.path.abspath(out)) in protected_outputs
    ):
        raise HTTPException(status_code=400, detail="ChatML output collides with its input")

    operation_token = _begin_derived_operation(
        job_id,
        kind="chatml",
        generation=generation,
        invalidate={"chatml"},
    )
    stage_out = os.path.join(
        _job_dir(job_id), f".chatml.g{generation}.{operation_token}.stage.jsonl",
    )

    safe_system_template = _normalize_optional_at_file(job_id, req.system_template, "system_template")
    safe_system_text = _normalize_optional_at_file(job_id, req.system_text, "system_text")

    # use pair_to_chatml.py CLI entry
    argv = [
        "-i", inputs,
        "-o", stage_out,
        "--mode", req.mode or "pair",
    ]
    if req.min_confidence is not None:
        argv.extend(["--min-confidence", str(req.min_confidence)])
    if req.max_turns is not None:
        argv.extend(["--max-turns", str(req.max_turns)])
    if req.dedupe:
        argv.append("--dedupe")
    if req.reverse:
        argv.append("--reverse")
    if req.include_meta:
        argv.append("--include-meta")
    if safe_system_template:
        argv.extend(["--system-template", safe_system_template])
    elif safe_system_text:
        argv.extend(["--system", safe_system_text])

    try:
        code = p2c.main(argv)
    except Exception as exc:
        _abort_derived_operation(
            job_id,
            "chatml",
            generation,
            operation_token,
            stage_out,
        )
        raise HTTPException(status_code=500, detail="ChatML export failed") from exc
    if code != 0:
        _abort_derived_operation(
            job_id,
            "chatml",
            generation,
            operation_token,
            stage_out,
        )
        return {"ok": False, "log": "ChatML export failed"}
    try:
        published = _publish_chatml_result(
            job_id,
            generation,
            operation_token,
            stage_out,
            out,
        )
    except Exception as exc:
        _abort_derived_operation(
            job_id,
            "chatml",
            generation,
            operation_token,
            stage_out,
        )
        raise HTTPException(status_code=500, detail="ChatML result could not be published") from exc
    if not published:
        _safe_remove_job_path(job_id, stage_out)
        raise HTTPException(status_code=409, detail="extraction generation changed during ChatML export")
    return {
        "ok": True,
        "artifact": "/api/jobs/{}/download?which=chatml".format(job_id),
        "generation": generation,
    }

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
        roles.extend([obj.get("role"), obj.get("from_role"), obj.get("to_role")])
        return [r for r in roles if r]

    try:
        job_id = _require_job_id(job_id)
        job = _load_job(job_id)
        generation = int(job.get("generation") or 0)
        source_raw = job.get("pairs_source")
        if not source_raw or int(job.get("pairs_source_generation") or -1) != generation:
            artifacts = job.get("artifacts") or {}
            artifact_generations = job.get("artifact_generations") or {}
            if (
                bool((job.get("validation_summary") or {}).get("ok"))
                and int(artifact_generations.get("validated", -1)) == generation
            ):
                source_raw = artifacts.get("validated")
            else:
                return {}
        ds_dir = _resolve_job_scoped_path(job_id, str(source_raw), "pairs_source")

        cnt = Counter()
        walk = (
            [(os.path.dirname(ds_dir), [], [os.path.basename(ds_dir)])]
            if os.path.isfile(ds_dir)
            else os.walk(ds_dir)
        )
        files_seen = 0
        for root, _, files in walk:
            for fn in files:
                if not fn.endswith(".jsonl"):
                    continue
                files_seen += 1
                if files_seen > MAX_PAIR_FILES:
                    break
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
        return dict(cnt.most_common(MAX_PAIR_ROLES))
    except Exception:
        return {}

@app.get("/api/roles")
def list_roles(job_id: str):
    job_id = _require_job_id(job_id)
    _load_job_or_404(job_id)
    roles = list(_all_roles_from_pairs(job_id).keys())
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
    if path.endswith(".css"):
        media = "text/css"
    elif path.endswith(".js"):
        media = "application/javascript"
    elif path.endswith(".html"):
        media = "text/html"
    return Response(content=data, media_type=media)
