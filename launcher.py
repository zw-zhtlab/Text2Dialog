#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2Dialog Launcher（图形化一键启动器）
=================================================
为项目 Text2Dialog 提供：
- 一键自动配置 Python 虚拟环境并以可编辑模式安装项目
- 启动 / 关闭 FastAPI 服务（uvicorn）
- 查看 / 打开“控制台”（日志窗口，可复制）
- 一键打开前端页面（浏览器）
- 一键打开帮助文档（固定文件名：帮助文档.html，位于项目根目录）
- 配置 API Key 并写入 .env（适配 OpenAI / Moonshot / Gemini / DeepSeek / Qwen / Bedrock 等常见变量名）

使用方式
-------------------------------------------------
1) 电脑已安装 Python 3.10+（支持 3.10~3.13）。
2) 将本文件 `launcher.py` 放到与 `text2dialog/` 同级的目录（项目根目录）。
3) 直接运行：  `python launcher.py`  或在资源管理器中双击（Windows）。
"""

from __future__ import annotations

import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, DISABLED, END, NORMAL, X, Text, Tk, Toplevel
from tkinter import messagebox, ttk

PROJECT_DIR = Path(__file__).resolve().parent
APP_DIR = PROJECT_DIR / "text2dialog"
HELP_PATH = PROJECT_DIR / "帮助文档.html"


@dataclass(frozen=True)
class ProviderEnvSpec:
    """Environment-variable names consumed by one supported provider."""

    label: str
    api_key_env: str
    base_url_env: str
    model_env: str


PROVIDER_ENV_SPECS: dict[str, ProviderEnvSpec] = {
    "siliconflow": ProviderEnvSpec(
        "硅基流动 SiliconFlow",
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_BASE_URL",
        "SILICONFLOW_MODEL_NAME",
    ),
    "deepseek": ProviderEnvSpec(
        "DeepSeek",
        "DEEPSEEK_API",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL_NAME",
    ),
    "bailian": ProviderEnvSpec(
        "阿里云百炼 / 通义千问",
        "DASHSCOPE_API_KEY",
        "BAILIAN_BASE_URL",
        "BAILIAN_MODEL_NAME",
    ),
    "moonshot": ProviderEnvSpec(
        "月之暗面 Moonshot / Kimi",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL_NAME",
    ),
    "openai": ProviderEnvSpec(
        "OpenAI",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL_NAME",
    ),
    "gemini": ProviderEnvSpec(
        "Google Gemini (OpenAI 兼容接口)",
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "GEMINI_MODEL_NAME",
    ),
    "aws_bedrock": ProviderEnvSpec(
        "AWS Bedrock",
        "AWS_BEDROCK_API_KEY",
        "AWS_BEDROCK_BASE_URL",
        "AWS_BEDROCK_MODEL_NAME",
    ),
    "custom": ProviderEnvSpec(
        "自定义 OpenAI 兼容接口",
        "CUSTOM_API_KEY",
        "CUSTOM_BASE_URL",
        "CUSTOM_MODEL_NAME",
    ),
}

_ENV_ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<export>export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=.*$"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<key>[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)|DEEPSEEK_API)"
    r"\s*(?P<separator>[:=])\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SECRET_HEADER_RE = re.compile(
    r"(?i)\b(?P<header>Authorization|X-API-Key)\s*:\s*(?:Bearer\s+)?[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://)([^/@\s:]+):([^/@\s]+)@")
_TOKEN_SHAPE_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_SECRET_COMMAND_FLAGS = frozenset(
    {"--api-key", "--api_key", "--password", "--secret", "--token", "--x-api-key"}
)


def redact_sensitive_text(value: object) -> str:
    """Return log-safe text without common credential representations."""

    text = str(value)
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('key')}{match.group('separator')}[REDACTED]",
        text,
    )
    text = _SECRET_HEADER_RE.sub(
        lambda match: f"{match.group('header')}: [REDACTED]",
        text,
    )
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", text)
    return _TOKEN_SHAPE_RE.sub("[REDACTED]", text)


def redact_command_args(args: list[str]) -> list[str]:
    """Redact values following common credential-bearing command flags."""

    redacted: list[str] = []
    hide_next = False
    for arg in args:
        if hide_next:
            redacted.append("[REDACTED]")
            hide_next = False
            continue
        flag, separator, _ = arg.partition("=")
        normalized_flag = flag.casefold()
        if normalized_flag in _SECRET_COMMAND_FLAGS:
            redacted.append(
                f"{flag}=[REDACTED]" if separator else flag
            )
            hide_next = not separator
            continue
        redacted.append(redact_sensitive_text(arg))
    return redacted


def provider_env_updates(
    provider: str,
    *,
    api_key: str = "",
    base_url: str = "",
    model_name: str = "",
) -> dict[str, str]:
    """Build provider-specific updates without routing values through OpenAI keys."""

    try:
        spec = PROVIDER_ENV_SPECS[provider]
    except KeyError as exc:
        raise ValueError(f"不支持的平台：{provider}") from exc

    updates = {"LLM_PLATFORM": provider}
    for key, value in (
        (spec.api_key_env, api_key),
        (spec.base_url_env, base_url),
        (spec.model_env, model_name),
    ):
        normalized = value.strip()
        if normalized:
            updates[key] = normalized
    return updates


def _quote_dotenv_value(value: str) -> str:
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError("环境变量值不能包含 NUL 或换行符")
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _upsert_env_text(original: str, updates: dict[str, str]) -> str:
    """Replace each requested key once while retaining unrelated lines and comments."""

    if not updates:
        return original
    invalid_keys = [
        key for key in updates if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
    ]
    if invalid_keys:
        raise ValueError("环境变量名不合法")

    newline = "\r\n" if "\r\n" in original else "\n"
    output: list[str] = []
    replaced: set[str] = set()
    for line in original.splitlines():
        match = _ENV_ASSIGNMENT_RE.match(line)
        key = match.group("key") if match else None
        if key not in updates:
            output.append(line)
            continue
        if key in replaced:
            continue
        prefix = match.group("indent") + (match.group("export") or "")
        output.append(f"{prefix}{key}={_quote_dotenv_value(updates[key])}")
        replaced.add(key)

    if output and output[-1] != "" and any(key not in replaced for key in updates):
        output.append("")
    for key, value in updates.items():
        if key not in replaced:
            output.append(f"{key}={_quote_dotenv_value(value)}")
    return newline.join(output).rstrip("\r\n") + newline


def atomic_upsert_env(path: Path, updates: dict[str, str]) -> None:
    """Atomically upsert dotenv keys, preserving the old file on every failure."""

    path = Path(path)
    original = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    rendered = _upsert_env_text(original, updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        descriptor = -1
        with handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temporary_path, path.stat().st_mode)
        else:
            os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except BaseException:
        try:
            if descriptor >= 0:
                os.close(descriptor)
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


class Launcher:
    def __init__(self, master: Tk):
        self.master = master
        self.master.title("Text2Dialog 启动器")
        self.master.geometry("860x560")
        try:
            self.master.iconbitmap(default="")  # 安静处理无图标
        except Exception:
            pass

        self.proc: subprocess.Popen | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_window: Toplevel | None = None
        self.log_text: Text | None = None
        self.reader_thread: threading.Thread | None = None
        self.reader_stop = threading.Event()
        self._flush_scheduled = False  # 避免重复调度 flush

        # 顶部：操作区
        top = ttk.Frame(master, padding=(12, 12, 8, 8))
        top.pack(fill=X)

        self.btn_setup = ttk.Button(top, text="① 一键配置/修复环境", command=self.setup_env)
        self.btn_setup.pack(side="left", padx=(0, 8))

        self.btn_start = ttk.Button(top, text="② 启动服务", command=self.start_server, state="disabled")
        self.btn_start.pack(side="left", padx=8)

        self.btn_console = ttk.Button(top, text="打开控制台", command=self.open_console, state="disabled")
        self.btn_console.pack(side="left", padx=8)

        self.btn_stop = ttk.Button(top, text="停止服务", command=self.stop_server, state="disabled")
        self.btn_stop.pack(side="left", padx=8)

        self.btn_open_ui = ttk.Button(top, text="打开界面", command=self.open_ui, state="disabled")
        self.btn_open_ui.pack(side="left", padx=8)

        self.btn_help = ttk.Button(top, text="帮助文档", command=self.open_help)
        self.btn_help.pack(side="right")

        self.btn_save_env = ttk.Button(top, text="保存 API 配置(.env)", command=self.save_env_dialog)
        self.btn_save_env.pack(side="right", padx=8)

        # 中部：设置
        mid = ttk.LabelFrame(master, text="运行设置", padding=12)
        mid.pack(fill=X, padx=12)

        self.host_var = ttk.Entry(mid)
        self.port_var = ttk.Entry(mid, width=8)
        self.host_var.insert(0, "127.0.0.1")
        self.port_var.insert(0, "8000")
        ttk.Label(mid, text="Host：").pack(side="left")
        self.host_var.pack(side="left", padx=(0, 16))
        ttk.Label(mid, text="Port：").pack(side="left")
        self.port_var.pack(side="left")

        ttk.Label(mid, text="（启动后点击“打开界面”即可在浏览器访问）").pack(side="left", padx=16)

        # 底部：日志（默认折叠，仅显示摘要）
        bottom = ttk.LabelFrame(master, text="运行日志（自动滚动，可在“打开控制台”查看完整日志）", padding=8)
        bottom.pack(fill=BOTH, expand=True, padx=12, pady=(8, 12))

        self.log_summary = Text(bottom, height=16, wrap="word")
        self.log_summary.pack(fill=BOTH, expand=True)
        self.log_summary.insert(END, "提示：首次使用请点击“① 一键配置/修复环境”。\n")
        self.log_summary.config(state=DISABLED)

        # 预检查
        self.after_ms(100, self.post_init)

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------- 工具方法 --------------------
    def after_ms(self, ms, fn):
        self.master.after(ms, fn)

    def ui(self, fn):
        """将任意 UI 改动切回主线程执行。"""
        self.master.after(0, fn)

    def _schedule_flush(self):
        """安排一次 UI 刷新（把 log_queue 的内容刷入 Text）。"""
        if self._flush_scheduled:
            return
        self._flush_scheduled = True

        def _flush():
            try:
                batch: list[str] = []
                while True:
                    try:
                        batch.append(self.log_queue.get_nowait())
                    except queue.Empty:
                        break
                if batch:
                    text = "".join(batch)
                    # 摘要
                    self.log_summary.config(state=NORMAL)
                    self.log_summary.insert(END, text)
                    self.log_summary.see(END)
                    self.log_summary.config(state=DISABLED)
                    # 控制台
                    if self.log_text is not None:
                        self.log_text.config(state=NORMAL)
                        self.log_text.insert(END, text)
                        self.log_text.see(END)
                        self.log_text.config(state=DISABLED)
            finally:
                self._flush_scheduled = False

        self.master.after(0, _flush)

    def append_log(self, msg: str, to_console: bool = True):
        """线程安全：任何线程都可以调用。统一带时间戳，UTF-8 文本。"""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {redact_sensitive_text(msg)}\n"
        self.log_queue.put(line)
        self._schedule_flush()

    def append_raw(self, raw: str):
        """用于子进程块输出（已处理编码/换行），不加额外换行。"""
        self.log_queue.put(redact_sensitive_text(raw))
        self._schedule_flush()

    def detect_python(self) -> str:
        return sys.executable

    def venv_python(self) -> str:
        if platform.system() == "Windows":
            return str((PROJECT_DIR / ".venv" / "Scripts" / "python.exe").resolve())
        return str((PROJECT_DIR / ".venv" / "bin" / "python").resolve())

    def ensure_app_layout(self) -> bool:
        if not APP_DIR.exists():
            messagebox.showerror(
                "未找到项目",
                f"未在 {PROJECT_DIR} 下发现 text2dialog/ 目录。\n"
                "请将 launcher.py 放在项目根目录，与 text2dialog 同级。",
            )
            return False
        if not (PROJECT_DIR / "pyproject.toml").exists():
            messagebox.showerror("缺少项目元数据", "未找到 pyproject.toml")
            return False
        return True

    def _make_child_env(self, base: dict[str, str] | None = None, *, disable_tqdm: bool = False) -> dict[str, str]:
        """统一子进程环境，确保 UTF-8 与实时输出；必要时禁用 tqdm。"""
        env = dict(base or os.environ)
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        if disable_tqdm:
            env["TQDM_DISABLE"] = "1"
        return env

    # --------- 子进程输出读取（按块，抗 \r，无阻塞） ---------
    def _stream_output_chunks(self, p: subprocess.Popen, stop_event: threading.Event | None = None):
        """
        从子进程 p 读取 stdout（二进制），以块为单位解码为 UTF-8。
        将 `\r\n` 和 `\r` 规范化为 `\n`，并按行投递到日志。
        若 stop_event 置位则尽快结束（依赖终止进程关闭管道）。
        """
        assert p.stdout is not None, "process missing stdout"

        buf = ""
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            try:
                s = chunk.decode("utf-8", errors="replace")
            except Exception:
                s = chunk.decode(errors="replace")
            # 统一换行：\r\n -> \n，\r -> \n（避免 tqdm 覆盖刷新导致卡读/刷屏）
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            buf += s
            # 拆分完整行，尾部残片保留
            if "\n" in buf:
                parts = buf.split("\n")
                buf = parts.pop()  # 残片
                for line in parts:
                    # 保留空行输出与时间戳一致性
                    self.append_log(line)
        if buf:
            self.append_log(buf)

    def run_and_stream(self, args: list[str], cwd: Path | None = None, env: dict | None = None):
        """
        通用子进程执行：按块读取输出并写入日志（线程安全）。
        注意：此函数在调用线程内阻塞执行，适合安装/构建等一次性任务。
        """
        self.append_log(f"$ {' '.join(redact_command_args(args))}")
        p = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            env=(env or self._make_child_env()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,      # 二进制无缓冲
            text=False,     # 明确使用字节流
        )
        assert p.stdout is not None
        try:
            self._stream_output_chunks(p)
        finally:
            code = p.wait()
        return code

    # -------------------- 环境配置 --------------------
    def setup_env(self):
        if not self.ensure_app_layout():
            return

        self.btn_setup.config(state="disabled")
        self.append_log("开始配置虚拟环境与依赖…")
        t = threading.Thread(target=self._setup_env_thread, daemon=True)
        t.start()

    def _setup_env_thread(self):
        try:
            venv_dir = PROJECT_DIR / ".venv"
            py = self.detect_python()
            if not venv_dir.exists():
                self.append_log("创建虚拟环境 .venv …")
                code = self.run_and_stream([py, "-m", "venv", str(venv_dir)])
                if code != 0:
                    raise RuntimeError("创建虚拟环境失败")

            vpy = self.venv_python()
            self.append_log("升级 pip …")
            self.run_and_stream([vpy, "-m", "pip", "install", "--upgrade", "pip"])
            self.append_log("安装 Text2Dialog 项目与依赖…（可能需要数分钟）")
            code = self.run_and_stream([vpy, "-m", "pip", "install", "-e", str(PROJECT_DIR)])
            if code != 0:
                raise RuntimeError("安装依赖失败")

            self.append_log("环境就绪。现在可以“启动服务”。")
            self.ui(lambda: (
                self.btn_start.config(state="normal"),
                self.btn_console.config(state="normal"),
                self.btn_open_ui.config(state="normal")
            ))
        except Exception as e:
            error_message = f"环境配置失败：{redact_sensitive_text(e)}"
            self.append_log(error_message)
            self.ui(lambda message=error_message: messagebox.showerror("失败", message))
        finally:
            self.ui(lambda: self.btn_setup.config(state="normal"))

    # -------------------- 启动/停止服务 --------------------
    def start_server(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("提示", "服务已经在运行。")
            return

        if not self.ensure_app_layout():
            return

        vpy = self.venv_python()
        host = self.host_var.get().strip() or "127.0.0.1"
        port = self.port_var.get().strip() or "8000"

        cmd = [vpy, "-m", "uvicorn", "text2dialog.server:app", "--host", host, "--port", port]
        # 统一 UTF-8 + 实时输出；禁用 tqdm，避免后台 tqd m \r 刷新导致的管道堆积
        env = self._make_child_env(disable_tqdm=True)

        self.append_log("启动服务中…")
        self.proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, text=False  # 二进制无缓冲读取
        )

        self.reader_stop.clear()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_console.config(state="normal")
        self.btn_open_ui.config(state="normal")
        self.append_log(f"服务已启动： http://{host}:{port}")

    def _reader_loop(self):
        assert self.proc is not None and self.proc.stdout is not None
        try:
            self._stream_output_chunks(self.proc, stop_event=self.reader_stop)
            code = self.proc.wait()
            self.append_log(f"服务已退出，返回码 {code}")
        except Exception as e:
            self.append_log(f"日志读取异常：{e}")
        finally:
            # 归位按钮状态（切回主线程）
            self.ui(lambda: (
                self.btn_start.config(state="normal"),
                self.btn_stop.config(state="disabled")
            ))

    def stop_server(self):
        if self.proc is None or self.proc.poll() is not None:
            self.append_log("服务未在运行。")
            return
        try:
            self.append_log("正在请求服务停止…")
            self.reader_stop.set()
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.append_log("温和终止失败，强制杀死进程…")
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.append_log("服务已停止。")
        except Exception as e:
            self.append_log(f"停止失败：{e}")
        finally:
            self.proc = None
            if self.reader_thread and self.reader_thread.is_alive():
                try:
                    self.reader_thread.join(timeout=1.0)
                except Exception:
                    pass
            self.reader_thread = None
            self.reader_stop.clear()
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

    # -------------------- 控制台窗口 --------------------
    def open_console(self):
        if self.log_window and self.log_window.winfo_exists():
            try:
                self.log_window.lift()
                return
            except Exception:
                pass

        self.log_window = Toplevel(self.master)
        self.log_window.title("运行控制台（可复制）")
        self.log_window.geometry("900x580")

        container = ttk.Frame(self.log_window, padding=6)
        container.pack(fill=BOTH, expand=True)

        self.log_text = Text(container, wrap="word")
        self.log_text.pack(fill=BOTH, expand=True)
        self.log_text.config(state=DISABLED)

        # 将目前摘要区已有的文字复制一份到控制台
        self.log_text.config(state=NORMAL)
        content = self.log_summary.get("1.0", END)
        self.log_text.insert(END, content)
        self.log_text.config(state=DISABLED)

        ttk.Button(self.log_window, text="清空", command=self.clear_console).pack(side="left", padx=6, pady=6)
        ttk.Button(self.log_window, text="复制全部", command=self.copy_console).pack(side="left", padx=6, pady=6)

    def clear_console(self):
        if self.log_text:
            self.log_text.config(state=NORMAL)
            self.log_text.delete("1.0", END)
            self.log_text.config(state=DISABLED)

    def copy_console(self):
        if self.log_text:
            try:
                text = self.log_text.get("1.0", END)
                self.master.clipboard_clear()
                self.master.clipboard_append(text)
                messagebox.showinfo("已复制", "日志内容已复制到剪贴板")
            except Exception:
                pass

    # -------------------- 打开 UI & 帮助文档 --------------------
    def open_ui(self):
        host = self.host_var.get().strip() or "127.0.0.1"
        port = self.port_var.get().strip() or "8000"
        url = f"http://{host}:{port}"
        try:
            webbrowser.open(url)
        except Exception as e:
            self.append_log(f"无法打开浏览器：{e}")

    def open_help(self):
        # 固定文件名：帮助文档.html（位于项目根目录）
        if HELP_PATH.exists():
            self._open_file(HELP_PATH)
        else:
            messagebox.showinfo("未找到", "未在项目根目录发现：帮助文档.html")

    def _open_file(self, path: Path):
        if platform.system() == "Windows":
            os.startfile(str(path))  # type: ignore
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    # -------------------- 保存 .env --------------------
    def save_env_dialog(self):
        win = Toplevel(self.master)
        win.title("保存 API 配置到 .env")
        win.geometry("640x420")

        frm = ttk.Frame(win, padding=12)
        frm.pack(fill=BOTH, expand=True)

        ttk.Label(
            frm,
            text=(
                "选择一个平台并填写对应配置。空白字段会保留 .env 中的已有值；"
                "API Key 不会显示在日志中。"
            ),
            wraplength=600,
        ).pack(anchor="w")

        grid = ttk.Frame(frm)
        grid.pack(fill=X, pady=(12, 0))
        grid.columnconfigure(1, weight=1)

        labels = [spec.label for spec in PROVIDER_ENV_SPECS.values()]
        self.provider_ids_by_label = {
            spec.label: provider for provider, spec in PROVIDER_ENV_SPECS.items()
        }
        ttk.Label(grid, text="平台").grid(row=0, column=0, sticky="w", pady=6, padx=(0, 12))
        self.provider_entry = ttk.Combobox(grid, values=labels, state="readonly")
        self.provider_entry.grid(row=0, column=1, sticky="ew", pady=6)
        self.provider_entry.set(PROVIDER_ENV_SPECS["siliconflow"].label)

        ttk.Label(grid, text="API Key").grid(row=1, column=0, sticky="w", pady=6, padx=(0, 12))
        self.api_key_entry = ttk.Entry(grid, show="•")
        self.api_key_entry.grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(grid, text="Base URL（可选）").grid(
            row=2,
            column=0,
            sticky="w",
            pady=6,
            padx=(0, 12),
        )
        self.base_url_entry = ttk.Entry(grid)
        self.base_url_entry.grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(grid, text="模型名（可选）").grid(
            row=3,
            column=0,
            sticky="w",
            pady=6,
            padx=(0, 12),
        )
        self.model_name_entry = ttk.Entry(grid)
        self.model_name_entry.grid(row=3, column=1, sticky="ew", pady=6)

        ttk.Button(frm, text="保存 .env", command=self._write_env).pack(pady=10)

    def _write_env(self):
        env_path = PROJECT_DIR / ".env"
        try:
            provider = self.provider_ids_by_label[self.provider_entry.get()]
            updates = provider_env_updates(
                provider,
                api_key=self.api_key_entry.get(),
                base_url=self.base_url_entry.get(),
                model_name=self.model_name_entry.get(),
            )
            atomic_upsert_env(env_path, updates)
            messagebox.showinfo("已保存", f".env 已更新：{env_path}")
            self.append_log(f"已更新 .env 配置项：{', '.join(updates)}")
            self.api_key_entry.delete(0, END)
        except Exception as e:
            safe_error = redact_sensitive_text(e)
            self.append_log(f"写入 .env 失败：{safe_error}")
            messagebox.showerror("失败", f"写入 .env 失败：{safe_error}")

    # -------------------- 初始化 --------------------
    def post_init(self):
        ok = self.ensure_app_layout()
        # 如果有 env.example 且 .env 不存在，则先复制一份，降低上手成本
        try:
            example = APP_DIR / "env.example"
            target = PROJECT_DIR / ".env"
            if example.exists() and not target.exists():
                with open(example, "r", encoding="utf-8") as fsrc, open(target, "w", encoding="utf-8") as fdst:
                    fdst.write(fsrc.read())
                self.append_log("已从 env.example 生成 .env（可在“保存 API 配置”中继续补全）。")
        except Exception:
            pass

        if ok and (PROJECT_DIR / ".venv").exists():
            self.btn_start.config(state="normal")
            self.btn_console.config(state="normal")
            self.btn_open_ui.config(state="normal")
            self.append_log("检测到已存在的虚拟环境，可直接“启动服务”。")
        else:
            self.append_log("尚未创建虚拟环境，请先“一键配置/修复环境”。")

    def on_close(self):
        try:
            self.stop_server()
        except Exception:
            pass
        self.master.destroy()


def main():
    root = Tk()
    # 统一 ttk 样式
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
