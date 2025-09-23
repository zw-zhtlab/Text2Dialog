#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对话链：从小说等长文本中提取角色对话
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from config import Config

# 加载 .env 变量（如果存在）
load_dotenv(find_dotenv())

# ========== 日志设置 ==========
logging.basicConfig(
    level=getattr(logging, str(getattr(Config, "LOG_LEVEL", "INFO")).upper()),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(getattr(Config, "LOG_FILE", "dialogue_chain.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ========== 控制/取消 支持（新增） ==========
class CancelledError(RuntimeError):
    """用户请求取消作业时抛出的异常（用于优雅收尾）。"""


class ControlState:
    """
    通过 cache_dir/control.json 读取作业控制状态：
    {
      "state": "running" | "paused" | "cancelling",
      "reason": "...",
      "ts": "ISO8601"
    }
    缺省或异常时视为 "running"。
    """

    def __init__(self, cache_dir: str):
        self.ctrl_file = os.path.join(cache_dir, "control.json")
        self._state = "running"

    def _refresh(self) -> None:
        try:
            if os.path.exists(self.ctrl_file):
                with open(self.ctrl_file, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                    self._state = str(data.get("state", "running")) or "running"
            else:
                self._state = "running"
        except Exception:
            self._state = "running"

    def is_paused(self) -> bool:
        self._refresh()
        return self._state == "paused"

    def is_cancelling(self) -> bool:
        self._refresh()
        return self._state == "cancelling"

    def wait_if_paused(self, tick: float = 0.5) -> None:
        """若处于 paused，则循环等待；若处于 cancelling，抛出 CancelledError。"""
        while True:
            self._refresh()
            if self._state == "paused":
                time.sleep(tick)
                continue
            if self._state == "cancelling":
                raise CancelledError("用户取消")
            return

    def raise_if_cancelled(self) -> None:
        """若处于 cancelling，抛出 CancelledError。"""
        if self.is_cancelling():
            raise CancelledError("用户取消")


# ========== 数据结构 ==========
@dataclass
class DialogueItem:
    role: str
    dialogue: str
    reply: Optional[Dict[str, Any]] = field(default=None, compare=False)

    def __hash__(self) -> int:
        """用于去重的哈希（基于角色与对话文本，忽略大小写与首尾空白）。"""
        return hash((self.role.strip().lower(), self.dialogue.strip().lower()))


@dataclass
class ChunkDialogueItem:
    """带 chunk 信息的对话项。"""
    chunk_id: int
    dialogue_index: int
    role: str
    dialogue: str
    chunk_text: Optional[str] = None
    reply: Optional[Dict[str, Any]] = None

    def to_dict(self, include_chunk_text: bool = False) -> Dict[str, Any]:
        """转为字典格式。"""
        result = {
            "chunk_id": self.chunk_id,
            "dialogue_index": self.dialogue_index,
            "role": self.role,
            "dialogue": self.dialogue,
            "reply": self.reply,
        }
        if include_chunk_text and self.chunk_text:
            result["chunk_text"] = self.chunk_text
        return result

    def __hash__(self) -> int:
        """用于去重的哈希（基于角色与对话文本，忽略大小写与首尾空白）。"""
        return hash((self.role.strip().lower(), self.dialogue.strip().lower()))

    def __eq__(self, other: object) -> bool:
        """用于去重的等价比较。"""
        if isinstance(other, ChunkDialogueItem):
            return self.role == other.role and self.dialogue == other.dialogue
        if isinstance(other, DialogueItem):
            return self.role == other.role and self.dialogue == other.dialogue
        return False


@dataclass
class WorkItem:
    """工作单元（供并发处理）。"""
    index: int
    chunk_id: int
    chunk: str
    system_prompt: str


class ThreadSafeDialogueChain:
    """线程安全的对话链处理器（供并发提取使用）。"""

    def __init__(self, extractor: "DialogueChain"):
        self.extractor = extractor
        self.lock = threading.Lock()
        self.total_dialogues = 0
        self.processed_chunks = 0
        self.errors: List[str] = []

    def process_chunk(self, work_item: WorkItem) -> List[ChunkDialogueItem]:
        """处理单个文本块，返回标准输出项。"""
        try:
            # 新增：在进入处理前先响应暂停/取消
            self.extractor.control.wait_if_paused()
            self.extractor.control.raise_if_cancelled()

            # 调用 API 提取对话
            response = self.extractor._call_api_with_retry(
                work_item.system_prompt,
                work_item.chunk,
            )
            dialogues = self.extractor._parse_and_validate_response(response)

            # 线程安全地转换为标准输出（始终包含 chunk_id/dialogue_index）
            with self.lock:
                chunk_dialogues: List[ChunkDialogueItem] = []
                for dialogue_index, dialogue in enumerate(dialogues):
                    chunk_dialogue = ChunkDialogueItem(
                        chunk_id=work_item.chunk_id,
                        dialogue_index=dialogue_index,
                        role=dialogue.role,
                        dialogue=dialogue.dialogue,
                        reply=dialogue.reply,
                        chunk_text=work_item.chunk if self.extractor.save_chunk_text else None,
                    )
                    chunk_dialogues.append(chunk_dialogue)

                self.total_dialogues += len(chunk_dialogues)
                self.processed_chunks += 1
                return chunk_dialogues

        except CancelledError:
            # 让上层感知到取消（不要吞掉）
            raise
        except Exception as e:
            with self.lock:
                self.errors.append(f"处理第 {work_item.index + 1} 块时发生错误：{e}")
            logger.error(f"处理第 {work_item.index + 1} 块时发生错误：{e}")
            return []


# ========== 主类 ==========
class DialogueChain:
    """对话链主类"""

    # 并发写有序结果时使用
    _next_expected_chunk_id: int

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        platform: Optional[str] = None,
        max_workers: Optional[int] = None,
        save_chunk_text: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        初始化对话链工具

        Args:
            schema: 自定义提取模式，若为 None 则使用默认模式（Config.DEFAULT_SCHEMA）。
            platform: 指定使用的平台，若为 None 则使用环境变量/默认配置。
            max_workers: 并发线程数，若为 None 则使用配置的默认值。
            save_chunk_text: 是否保存原始 chunk 文本，若为 None 则使用配置默认值。
            cache_dir: 作业专属缓存目录（用于进度等）。优先使用此参数；若未提供则退回 Config.CACHE_DIR。
        """
        # 设置平台（如传入）
        if platform:
            Config.set_platform(platform)

        # 验证配置
        config_errors = Config.validate_config()
        if config_errors:
            raise ValueError(f"配置错误：{'；'.join(config_errors)}")

        # 获取当前平台配置
        platform_config = Config.get_current_platform_config()
        self.platform = platform_config["platform"]
        self.model_name = platform_config["model_name"]
        self.base_url: Optional[str] = platform_config.get("base_url")  # 保存 base_url 以便启发式判断

        self.schema = schema or Config.DEFAULT_SCHEMA
        self.client = OpenAI(
            api_key=platform_config["api_key"],
            base_url=platform_config["base_url"],
        )
        self.encoder = tiktoken.get_encoding(getattr(Config, "ENCODING", "cl100k_base"))

        # 并发设置
        self.max_workers = int(max_workers or getattr(Config, "MAX_WORKERS", 4))

        # 是否保存 chunk 原文
        self.save_chunk_text = (
            save_chunk_text
            if save_chunk_text is not None
            else bool(getattr(Config, "SAVE_CHUNK_TEXT", False))
        )

        # ===== 缓存目录（实例级，避免全局状态冲突） =====
        self.cache_dir = cache_dir or getattr(Config, "CACHE_DIR", ".cache")
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)

        # 新增：控制器（读取 .cache/control.json）
        self.control = ControlState(self.cache_dir)

        logger.info(f"对话链初始化完成 - 平台: {self.platform} ({platform_config['description']})")
        logger.info(f"使用模型: {self.model_name}")
        logger.info(f"并发线程数: {self.max_workers}")
        logger.info(f"缓存目录: {self.cache_dir}")

        # 并发写序 helper
        self._next_expected_chunk_id = 0

        # 进度计时起点（用于速度/ETA）
        self._progress_t0: Optional[float] = None

    # ---------- 提示词 ----------
    def _generate_system_prompt(self) -> str:
        """生成系统提示词（包含类型描述与示例）。"""
        attributes = self.schema["attributes"]
        attributes_str = ",\n    ".join(
            [f"{attr['name']}: {attr['type']} // {attr['description']}" for attr in attributes]
        )

        typescript = Config.get_typescript_template().format(
            task_description=self.schema["task_description"],
            attributes=attributes_str,
        )

        example_input = self.schema["example"][0]["text"]
        example_output = json.dumps(
            self.schema["example"][0]["script"],
            indent=4,
            ensure_ascii=False,
        )

        base_prompt = Config.get_system_prompt_template().format(
            TypeScript=typescript,
            Input=example_input,
            Output=example_output,
        )
        rules = f"""
IMPORTANT GUIDELINES:
- For any reply, target_index must point to an earlier utterance within this script (0-based).
- The gap between the current item and target_index must not exceed 6.
- Output strictly as a JSON array; each element must include role, dialogue, reply.
  reply is either null or {{target_index:int, target_role:string, confidence:number}}.
- Only set reply when the referenced utterance is from a DIFFERENT speaker:
  script[target_index].role MUST NOT equal the current item's role.
  If the most plausible reference is the same speaker or ambiguous, set reply = null.
- When reply is an object, you MUST include target_role and it MUST equal script[target_index].role,
  and MUST be different from the current item's role.
"""
        return base_prompt + "\n" + rules.strip()

    # ---------- 文件与分块 ----------
    def _read_text_file(self, file_path: str) -> str:
        """读取文本文件（UTF-8）。"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}：{e}")
            raise

    def _split_long_line(self, long_line: str) -> List[str]:
        """将长行按句子分割（中文句号/问号/感叹号），保证每段不超 token 上限。"""
        # 保留分隔符
        parts = re.split(r"([。！？?!])", long_line)
        sentences: List[str] = []
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
            else:
                if parts[i]:
                    sentences.append(parts[i])

        chunks: List[str] = []
        current = ""
        for sent in sentences:
            tokens = len(self.encoder.encode(current + sent))
            if tokens <= getattr(Config, "MAX_TOKEN_LEN", 2048):
                current += sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        文本分块（按行拼接，必要时对长行执行按句切分；块间加入 token 级重叠以保留上下文）。
        """
        chunks: List[str] = []
        lines = text.splitlines()

        # 预处理：清理空行与多余空白
        cleaned_lines = [ln.strip() for ln in lines if ln.strip()]

        current_chunk = ""
        current_tokens = 0
        max_tokens = int(getattr(Config, "MAX_TOKEN_LEN", 2048))
        cover_tokens = int(getattr(Config, "COVER_CONTENT", 0))

        i = 0
        while i < len(cleaned_lines):
            line = cleaned_lines[i]
            line_tokens = len(self.encoder.encode(line))

            # 如果单行超限，强制切分
            if line_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = ""
                    current_tokens = 0

                # 按句子分割长行
                for sentence in self._split_long_line(line):
                    chunks.append(sentence)
                i += 1
                continue

            # 放得下当前行：继续累积
            if current_tokens + line_tokens + 1 <= max_tokens:
                current_chunk += line + "\n"
                current_tokens += line_tokens + 1
                i += 1
            else:
                # 当前块已满，先落盘
                if current_chunk:
                    chunks.append(current_chunk.rstrip())

                # 形成重叠（向前回看若干行，累计不超过 cover_tokens）
                overlap_lines: List[str] = []
                temp_tokens = 0
                # 从 i-1 往前回看
                j = i - 1
                while j >= 0 and temp_tokens < cover_tokens:
                    line_j = cleaned_lines[j]
                    tokens_j = len(self.encoder.encode(line_j))
                    if temp_tokens + tokens_j <= cover_tokens:
                        # 从前往后重建
                        overlap_lines.insert(0, line_j)
                        temp_tokens += tokens_j
                    else:
                        break
                    j -= 1

                current_chunk = ("\n".join(overlap_lines) + "\n") if overlap_lines else ""
                current_tokens = temp_tokens

        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk.rstrip())

        logger.info(f"文本分块完成：共 {len(chunks)} 块")
        return chunks

    # ---------- LLM 调用 & 推理适配 ----------
    def _strip_reasoning_prefix(self, text: str) -> str:
        """
        移除可能出现的“思考/推理”前缀，确保只保留可展示答案。
        - Qwen Thinking：<think>...</think> 包裹的思考内容
        - 某些实现只输出闭合标签：在出现 </think> 时，截取其后的正文
        """
        if not text:
            return text
        # 去掉规范的 <think>...</think>（非贪婪，跨行）
        text = re.sub(r"(?is)<think>.*?</think>", "", text)
        # 仅有闭合标签时，截取其后部分
        if "</think>" in text:
            text = text.split("</think>", 1)[-1]
        return text.strip()

    def _extract_text_from_openai_responses(self, response: Any) -> Optional[str]:
        """
        从 OpenAI Responses API 响应中稳健提取纯文本。
        优先使用 response.output_text；否则回退遍历 output 内容。
        """
        # 1) 便捷属性
        try:
            output_text = getattr(response, "output_text", None)
            if output_text:
                return str(output_text).strip()
        except Exception:
            pass

        # 2) 遍历标准结构
        try:
            output = getattr(response, "output", None)
            if output:
                # output 是一组 "message" 等对象；逐层取 text
                for item in output:
                    content_list = getattr(item, "content", None)
                    if not content_list and isinstance(item, dict):
                        content_list = item.get("content")
                    if not content_list:
                        continue
                    for part in content_list:
                        # part 可能有 .text 或 dict["text"]
                        txt = getattr(part, "text", None)
                        if not txt and isinstance(part, dict):
                            # OpenAI SDK：{"type":"output_text","text":"..."}
                            txt = part.get("text")
                        if txt:
                            return str(txt).strip()
        except Exception:
            pass

        # 3) 某些 SDK 返回 dict
        if isinstance(response, dict):
            # 非严格，但尽量挖出 text
            try:
                if "output_text" in response and response["output_text"]:
                    return str(response["output_text"]).strip()
                out = response.get("output") or []
                for item in out:
                    for part in item.get("content", []):
                        if "text" in part and part["text"]:
                            return str(part["text"]).strip()
            except Exception:
                pass

        return None

    def _extract_text_from_chat_completion(self, resp: Any) -> str:
        """
        从 Chat Completions 响应中提取最终文本：
        - 忽略 reasoning_content（DeepSeek R1 等）
        - 兼容对象/字典两种访问方式
        """
        content: Optional[str] = None
        try:
            # SDK 对象风格
            choice = resp.choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                # 忽略 message.reasoning_content
                content = getattr(message, "content", None)
        except Exception:
            pass

        if content is None:
            try:
                # 字典风格
                choice0 = resp["choices"][0]
                msg = choice0.get("message", {})
                content = msg.get("content")
            except Exception:
                pass

        # 某些兼容实现可能用 choices[0].text
        if not content:
            try:
                content = getattr(resp.choices[0], "text", None)
            except Exception:
                pass
        if not content:
            try:
                content = resp["choices"][0].get("text")
            except Exception:
                pass

        return (content or "").strip()

    def _should_try_openai_responses(self) -> bool:
        """
        启发式判断是否优先尝试 OpenAI Responses API：
        - 平台为 openai，且模型名属于新版/推理家族
        - 或 base_url 指向 OpenAI 且模型名符合上述前缀
        """
        model = (self.model_name or "").lower()
        platform = (self.platform or "").lower()
        base = (self.base_url or "").lower()
        candidates = ("gpt-5", "o3", "o4", "gpt-4.1", "gpt-4o")
        looks_like_openai = (platform == "openai") or ("openai" in base)
        is_candidate = any(model.startswith(p) or p in model for p in candidates)
        return looks_like_openai and is_candidate and hasattr(self.client, "responses")

    def _call_api_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """带重试的 API 调用（适配推理模型，防止思考内容污染输出）。"""
        max_retries = int(getattr(Config, "MAX_RETRIES", 3))
        delay = float(getattr(Config, "RETRY_DELAY", 1.0))
        temperature = float(getattr(Config, "TEMPERATURE", 0.2))
        reasoning_effort = str(getattr(Config, "REASONING_EFFORT", os.getenv("REASONING_EFFORT", ""))).strip()

        last_err: Optional[BaseException] = None
        for attempt in range(max_retries):
            # 新增：每次尝试前响应暂停/取消
            self.control.wait_if_paused()
            self.control.raise_if_cancelled()

            try:
                content: Optional[str] = None

                # 1) 优先尝试 OpenAI Responses API（若命中）
                if self._should_try_openai_responses():
                    try:
                        kwargs: Dict[str, Any] = {
                            "model": self.model_name,
                            "input": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": temperature,
                        }
                        if reasoning_effort:
                            # OpenAI Responses 的 reasoning 控制（minimal / low / medium / high）
                            kwargs["reasoning"] = {"effort": reasoning_effort}

                        resp = self.client.responses.create(**kwargs)
                        content = self._extract_text_from_openai_responses(resp)
                        if content:
                            # Qwen Thinking 等不会走到这里，但为一致性仍清理一次
                            content = self._strip_reasoning_prefix(content)
                            return content
                    except Exception as e:
                        # 记录后回退到 Chat Completions
                        logger.debug(f"Responses API 不可用或提取失败，回退到 Chat Completions：{e}")

                # 2) 通用回退：OpenAI 兼容 Chat Completions（DeepSeek、Qwen、Moonshot、SiliconFlow 等）
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    stream=False,
                )
                content = self._extract_text_from_chat_completion(resp)

                # 统一剔除推理前缀（Qwen Thinking 的 <think> 等）
                content = self._strip_reasoning_prefix(content)
                return content

            except CancelledError:
                # 透传取消，让上层优雅收尾
                raise
            except Exception as e:
                last_err = e
                logger.warning(f"API 调用失败（尝试 {attempt + 1}/{max_retries}）：{e}")
                if attempt < max_retries - 1:
                    # 重试前也响应一次取消
                    self.control.raise_if_cancelled()
                    time.sleep(delay * (attempt + 1))

        logger.error("API 调用重试次数已用尽")
        if last_err:
            raise last_err
        raise RuntimeError("API 调用失败")

    # ---------- 解析与清洗 ----------
    def _clean_reply(self, current_index: int, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """清洗模型返回的 reply 字段。"""
        reply = item.get("reply")
        if not isinstance(reply, dict):
            return None

        target_index = reply.get("target_index")
        if not isinstance(target_index, int):
            return None
        if target_index < 0 or target_index >= current_index:
            return None
        if (current_index - target_index) > int(getattr(Config, "REPLY_WINDOW", 8)):
            return None

        confidence = reply.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)):
            return None
        confidence_value = float(confidence)
        if confidence_value < float(getattr(Config, "REPLY_CONFIDENCE_TH", 0.0)):
            return None

        cleaned: Dict[str, Any] = {"target_index": target_index}
        if "target_role" in reply and reply["target_role"] is not None:
            cleaned["target_role"] = str(reply["target_role"])
        cleaned["confidence"] = confidence_value
        return cleaned

    def _extract_first_json_array(self, text: str) -> Optional[str]:
        """
        从文本中提取第一个顶层 JSON 数组子串（容错：处理 ```json ... ```、多余前后文等情况）。
        """
        # 去掉 Markdown 代码围栏
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        # 尝试直接解析
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return text
            if isinstance(obj, dict) and "script" in obj and isinstance(obj["script"], list):
                return json.dumps(obj["script"], ensure_ascii=False)
        except Exception:
            pass

        # 扫描括号以找到数组起止
        start_positions = [m.start() for m in re.finditer(r"\[", text)]
        for start in start_positions:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start : i + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, list):
                                    return candidate
                            except Exception:
                                pass
                            break
        return None

    def _parse_and_validate_response(self, response: str) -> List[DialogueItem]:
        """解析并验证 API 响应，返回对话中间结构列表。"""
        text = response or ""
        # 尝试解析
        payload: Optional[str] = None
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                payload = text
            elif isinstance(obj, dict) and "script" in obj:
                payload = json.dumps(obj["script"], ensure_ascii=False)
        except json.JSONDecodeError:
            payload = self._extract_first_json_array(text)

        if payload is None:
            logger.error("无法从模型响应中提取 JSON 列表")
            logger.debug(f"原始响应：{text}")
            return []

        try:
            data = json.loads(payload)
        except Exception as e:
            logger.error(f"JSON 反序列化失败：{e}")
            logger.debug(f"候选 JSON：{payload}")
            return []

        if not isinstance(data, list):
            logger.warning("响应不是列表格式，已尝试纠正失败")
            return []

        dialogues: List[DialogueItem] = []
        for item in data:
            if isinstance(item, dict) and "role" in item and "dialogue" in item:
                role = str(item["role"]).strip()
                content = str(item["dialogue"]).strip()
                if role and content:
                    reply = self._clean_reply(len(dialogues), item)
                    dialogues.append(DialogueItem(role=role, dialogue=content, reply=reply))
                else:
                    logger.warning(f"跳过空对话项：{item}")
            else:
                logger.warning(f"跳过无效对话项：{item}")

        return dialogues

    # ---------- 去重（保持顺序） ----------
    def _remove_duplicates(self, dialogues: List[DialogueItem]) -> List[DialogueItem]:
        """保留原顺序以维持 reply 关联（当前默认不去重，直接返回）。"""
        return dialogues

    # ---------- 进度 ----------
    def _progress_file(self) -> str:
        return os.path.join(self.cache_dir, getattr(Config, "PROGRESS_FILE", "progress.json"))

    def _save_progress(
        self,
        file_path: str,
        processed_chunks: int,
        total_chunks: int,
        stage: str = "processing",
        message: Optional[str] = None,
    ) -> None:
        """保存进度信息到 progress.json（含阶段/速率/ETA）。"""
        pf = self._progress_file()

        # 初始化计时起点
        if self._progress_t0 is None:
            self._progress_t0 = time.time()

        now = time.time()
        elapsed = max(0.0, now - self._progress_t0)
        speed = (processed_chunks / elapsed) if elapsed > 0 else 0.0
        eta = None
        if total_chunks and speed > 0 and total_chunks >= processed_chunks:
            eta = max(0.0, (total_chunks - processed_chunks) / speed)

        progress_data = {
            "file_path": file_path,
            "processed_chunks": processed_chunks,
            "total_chunks": total_chunks,
            "stage": stage,                    
            "message": message or "",
            "elapsed_sec": elapsed,
            "speed_cps": speed,                
            "eta_sec": eta,
            "timestamp": now,
        }
        try:
            with open(pf, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存进度失败：{e}")

    def _load_progress(self, file_path: str) -> Optional[int]:
        """加载进度信息，返回已处理的块数。"""
        pf = self._progress_file()
        try:
            if os.path.exists(pf):
                with open(pf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("file_path") == file_path:
                        return int(data.get("processed_chunks", 0))
        except Exception as e:
            logger.warning(f"加载进度失败：{e}")
        return None

    # ---------- 已完成块扫描（用于续跑，新增） ----------
    def _scan_processed_chunk_ids(self, output_file: str) -> Set[int]:
        """
        扫描已有输出文件，返回其中出现过的 chunk_id 集合；
        若文件不存在或为空，返回空集。
        """
        processed: Set[int] = set()
        try:
            if not output_file or not os.path.exists(output_file):
                return processed
            with open(output_file, "r", encoding=getattr(Config, "OUTPUT_ENCODING", "utf-8")) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cid = obj.get("chunk_id")
                        if isinstance(cid, int):
                            processed.add(cid)
                    except Exception:
                        # 忽略脏行
                        continue
        except Exception as e:
            logger.warning(f"扫描已完成 chunk 失败：{e}")
        return processed

    def _first_missing_index(self, done_ids: Set[int]) -> int:
        """返回从 0 开始首个缺失的 chunk_id（用于确定按序写入起点）。"""
        i = 0
        while i in done_ids:
            i += 1
        return i

    # ---------- 顺序提取 ----------
    def extract_dialogues(self, file_path: str, output_file: Optional[str] = None) -> str:
        """
        从文本文件中提取对话（单线程；标准输出：包含 chunk_id/dialogue_index）。

        Args:
            file_path: 输入文本文件路径。
            output_file: 输出文件路径，若为 None 则自动生成。

        Returns:
            输出文件路径。
        """
        logger.info(f"开始处理文本：{file_path}")

        # 0) 进入初始化/分块阶段
        self._save_progress(file_path, 0, 0, stage="chunking", message="正在切分文本…")

        text = self._read_text_file(file_path)
        chunks = self._chunk_text(text)

        # 1) 分块完成后立即写入 0/N
        self._save_progress(file_path, 0, len(chunks), stage="processing", message="正在处理分块…")

        system_prompt = self._generate_system_prompt()

        if output_file is None:
            file_name = Path(file_path).stem
            output_file = f"{file_name}_dialogues.{getattr(Config, 'OUTPUT_FORMAT', 'jsonl')}"

        # 确保输出目录存在
        out_dir = Path(output_file).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # 进度恢复：优先使用 progress.json；如不可用，退回扫描输出文件（避免重复写）
        processed_chunks = self._load_progress(file_path) or 0
        processed_from_file = 0
        existing_ids = self._scan_processed_chunk_ids(output_file)
        if existing_ids:
            processed_from_file = self._first_missing_index(existing_ids)
        if processed_from_file > processed_chunks:
            processed_chunks = processed_from_file
            logger.info(f"从输出文件恢复进度：已处理 {processed_chunks}/{len(chunks)} 块")

        total_dialogues = 0
        was_cancelled = False

        with tqdm(total=len(chunks), desc="提取对话", initial=processed_chunks) as pbar:
            for i, chunk in enumerate(chunks):
                if i < processed_chunks:
                    continue

                # 新增：在进入新块前响应暂停/取消
                try:
                    self.control.wait_if_paused()
                    self.control.raise_if_cancelled()
                except CancelledError:
                    # 标记并优雅收尾
                    self._save_progress(
                        file_path, i, len(chunks),
                        stage="cancelled", message="用户取消"
                    )
                    was_cancelled = True
                    break

                try:
                    response = self._call_api_with_retry(system_prompt, chunk)
                    dialogues = self._parse_and_validate_response(response)

                    unique_dialogues = self._remove_duplicates(dialogues)

                    # 写入
                    with open(output_file, "a", encoding=getattr(Config, "OUTPUT_ENCODING", "utf-8")) as f:
                        for dialogue_index, dialogue in enumerate(unique_dialogues):
                            chunk_dialogue = ChunkDialogueItem(
                                chunk_id=i,
                                dialogue_index=dialogue_index,
                                role=dialogue.role,
                                dialogue=dialogue.dialogue,
                                reply=dialogue.reply,
                                chunk_text=chunk if self.save_chunk_text else None,
                            )
                            json.dump(
                                chunk_dialogue.to_dict(include_chunk_text=self.save_chunk_text),
                                f,
                                ensure_ascii=False,
                            )
                            f.write("\n")

                    total_dialogues += len(unique_dialogues)

                    # 保存进度
                    self._save_progress(
                        file_path,
                        i + 1,
                        len(chunks),
                        stage="processing",
                        message="正在处理分块…",
                    )

                    # 更新进度条
                    pbar.set_postfix({
                        "累计对话数": total_dialogues,
                        "本块有效": len(unique_dialogues),
                    })
                    pbar.update(1)

                except CancelledError:
                    # 取消发生在 API 调用或之后
                    self._save_progress(
                        file_path,
                        i,  # 当前块不计入
                        len(chunks),
                        stage="cancelled",
                        message="用户取消",
                    )
                    was_cancelled = True
                    break
                except Exception as e:
                    logger.error(f"处理第 {i + 1} 块时发生错误：{e}")
                    # 保持进度写入（不中断）
                    self._save_progress(
                        file_path,
                        i,  # 当前块失败不计入 processed
                        len(chunks),
                        stage="processing",
                        message=f"处理第 {i + 1} 块时发生错误：{e}",
                    )
                    continue

        # 收尾（不在此删除 progress.json；由 server 子进程统一清理）
        if was_cancelled:
            logger.info("处理被用户取消。")
        else:
            self._save_progress(file_path, len(chunks), len(chunks), stage="done", message="处理完成")
            logger.info(f"处理完成！共提取 {total_dialogues} 条对话，保存到：{output_file}")
        return output_file

    # ---------- 并发提取 ----------
    def extract_dialogues_concurrent(self, file_path: str, output_file: Optional[str] = None) -> str:
        """
        使用多线程并发从文本文件提取对话

        Args:
            file_path: 输入文本文件路径。
            output_file: 输出文件路径，若为 None 则自动生成。

        Returns:
            输出文件路径。
        """
        logger.info(f"开始并发处理文本：{file_path}")

        # 0) 初始化/分块提示
        self._save_progress(file_path, 0, 0, stage="chunking", message="正在切分文本…")

        text = self._read_text_file(file_path)
        chunks = self._chunk_text(text)
        logger.info(f"文本分块完成：共 {len(chunks)} 块，将使用 {self.max_workers} 个线程并发处理")

        # 1) 分块完成后立即写 0/N
        self._save_progress(file_path, 0, len(chunks), stage="processing", message="正在处理分块…")

        system_prompt = self._generate_system_prompt()

        if output_file is None:
            file_name = Path(file_path).stem
            output_file = f"{file_name}_dialogues_concurrent.{getattr(Config, 'OUTPUT_FORMAT', 'jsonl')}"

        # 确保输出目录存在
        Path(output_file).resolve().parent.mkdir(parents=True, exist_ok=True)

        # 续跑支持：扫描已写出的 chunk，过滤重复任务，并定位按序写入起点
        processed_ids: Set[int] = self._scan_processed_chunk_ids(output_file)
        next_expected = self._first_missing_index(processed_ids)
        if processed_ids:
            logger.info(f"检测到已有输出：已完成 {len(processed_ids)} 个 chunk，继续处理剩余部分（从 {next_expected} 起）。")

        thread_safe_extractor = ThreadSafeDialogueChain(self)

        work_items = [
            WorkItem(index=i, chunk_id=i, chunk=chunk, system_prompt=system_prompt)
            for i, chunk in enumerate(chunks) if i not in processed_ids
        ]

        total_dialogues = 0
        failed_chunks = 0
        was_cancelled = False

        # 有序写入的缓冲区：chunk_id -> List[ChunkDialogueItem]
        results_buffer: Dict[int, List[ChunkDialogueItem]] = {}
        completed_chunks: Set[int] = set(processed_ids)  # 已完成集合预置为扫描结果
        self._next_expected_chunk_id = next_expected

        def write_ready_results():
            """按序写入（从 _next_expected_chunk_id 起，连续可用的 chunk）。"""
            while self._next_expected_chunk_id in results_buffer:
                dialogues = results_buffer.pop(self._next_expected_chunk_id)
                with open(output_file, "a", encoding=getattr(Config, "OUTPUT_ENCODING", "utf-8")) as f:
                    for d in dialogues:
                        json.dump(d.to_dict(include_chunk_text=self.save_chunk_text), f, ensure_ascii=False)
                        f.write("\n")
                self._next_expected_chunk_id += 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item: Dict[Future, WorkItem] = {
                executor.submit(thread_safe_extractor.process_chunk, item): item
                for item in work_items
            }

            with tqdm(total=len(work_items), desc="并发提取对话") as pbar:
                for future in as_completed(future_to_item):
                    work_item = future_to_item[future]
                    try:
                        # 在获取结果前后也检查一次（可快速响应暂停/取消）
                        self.control.wait_if_paused()
                        self.control.raise_if_cancelled()

                        dialogues = future.result()

                        with thread_safe_extractor.lock:
                            results_buffer[work_item.chunk_id] = dialogues
                            completed_chunks.add(work_item.chunk_id)
                            total_dialogues += len(dialogues)

                        # 只要就绪就写入（保持有序）
                        write_ready_results()

                        # 保存并发进度（已完成块数，按总 chunks 计）
                        self._save_progress(
                            file_path,
                            len(completed_chunks),
                            len(chunks),
                            stage="processing",
                            message="正在处理分块…",
                        )

                        # 更新进度条
                        active_qsize = 0
                        try:
                            active_qsize = getattr(executor, "_work_queue").qsize()  # type: ignore[attr-defined]
                        except Exception:
                            pass

                        pbar.set_postfix({
                            "累计对话数": total_dialogues,
                            "失败块数": failed_chunks,
                            "队列待处理": active_qsize,
                        })
                        pbar.update(1)

                    except CancelledError:
                        # 用户取消：标记并继续等待其他线程尽快收束（软取消）
                        was_cancelled = True
                        self._save_progress(
                            file_path,
                            len(completed_chunks),
                            len(chunks),
                            stage="cancelled",
                            message="用户取消",
                        )
                        # 不 break：保持 with 块语义，等待已在执行的任务自然返回
                        pbar.set_postfix({
                            "累计对话数": total_dialogues,
                            "失败块数": failed_chunks,
                            "状态": "取消中",
                        })
                        pbar.update(1)
                    except Exception as e:
                        failed_chunks += 1
                        logger.error(f"处理第 {work_item.index + 1} 块时发生错误：{e}")
                        self._save_progress(
                            file_path,
                            len(completed_chunks),
                            len(chunks),
                            stage="processing",
                            message=f"部分块失败：{failed_chunks}；最近错误：{e}",
                        )
                        pbar.set_postfix({
                            "累计对话数": total_dialogues,
                            "失败块数": failed_chunks,
                        })
                        pbar.update(1)

        # 写入剩余结果（保险起见）
        for _ in range(3):
            if not results_buffer:
                break
            write_ready_results()

        # 输出错误汇总
        if thread_safe_extractor.errors:
            logger.warning(f"处理过程出现 {len(thread_safe_extractor.errors)} 个错误：")
            for error in thread_safe_extractor.errors[:5]:
                logger.warning(f"  - {error}")
            if len(thread_safe_extractor.errors) > 5:
                logger.warning(f"  - ... 还有 {len(thread_safe_extractor.errors) - 5} 个错误未逐条展示")

        # 收尾（不在此删除 progress.json；由 server 子进程统一清理）
        if was_cancelled:
            logger.info("并发处理被用户取消。")
        else:
            self._save_progress(file_path, len(chunks), len(chunks), stage="done", message="处理完成")
            logger.info(f"并发处理完成！共提取 {total_dialogues} 条对话，失败 {failed_chunks} 块，保存到：{output_file}")
        return output_file

    # ---------- 统计与工具 ----------
    def get_statistics(self, output_file: str) -> Dict[str, Any]:
        """获取输出文件的统计信息。"""
        try:
            dialogues = []
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        dialogues.append(data)

            role_counts: Dict[str, int] = {}
            for d in dialogues:
                role = d.get("role", "Unknown")
                role_counts[role] = role_counts.get(role, 0) + 1

            return {
                "total_dialogues": len(dialogues),
                "unique_roles": len(role_counts),
                "role_distribution": role_counts,
                "average_dialogue_length": (
                    sum(len(d.get("dialogue", "")) for d in dialogues) / len(dialogues) if dialogues else 0
                ),
            }
        except Exception as e:
            logger.error(f"统计信息生成失败：{e}")
            return {}

    def sort_dialogues(self, output_file: str, sorted_output_file: Optional[str] = None) -> str:
        """按 chunk_id 与 dialogue_index 排序对话并保存到新文件。"""
        try:
            self._save_progress(output_file, 0, 0, stage="sorting", message="正在排序输出…")
        except Exception:
            pass

        if sorted_output_file is None:
            base_name = Path(output_file).stem
            sorted_output_file = f"{base_name}_sorted.{getattr(Config, 'OUTPUT_FORMAT', 'jsonl')}"

        try:
            dialogues: List[Dict[str, Any]] = []
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        dialogues.append(json.loads(line.strip()))

            # 取消旧格式兼容：必须包含 chunk_id
            if not all("chunk_id" in d and "dialogue_index" in d for d in dialogues):
                raise ValueError("文件不包含 chunk_id/dialogue_index，无法排序。")

            dialogues.sort(key=lambda x: (x["chunk_id"], x["dialogue_index"]))

            with open(sorted_output_file, "w", encoding=getattr(Config, "OUTPUT_ENCODING", "utf-8")) as f:
                for d in dialogues:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")

            logger.info(f"对话排序完成，保存到：{sorted_output_file}")
            # 排序完成后打个完成标记（可选）
            try:
                self._save_progress(output_file, 0, 0, stage="done", message="排序完成")
            except Exception:
                pass
            return sorted_output_file

        except Exception as e:
            logger.error(f"对话排序失败：{e}")
            try:
                self._save_progress(output_file, 0, 0, stage="failed", message=f"排序失败：{e}")
            except Exception:
                pass
            return output_file

    def filter_by_chunk(self, output_file: str, chunk_ids: List[int], filtered_output_file: Optional[str] = None) -> str:
        """按 chunk_id 过滤对话并保存到新文件。"""
        if filtered_output_file is None:
            base_name = Path(output_file).stem
            chunk_str = "_".join(map(str, sorted(set(chunk_ids))))
            filtered_output_file = f"{base_name}_chunks_{chunk_str}.{getattr(Config, 'OUTPUT_FORMAT', 'jsonl')}"

        try:
            filtered_dialogues: List[Dict[str, Any]] = []
            chunk_id_set = set(chunk_ids)
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        if "chunk_id" not in data:
                            raise ValueError("检测到缺少 chunk_id 的记录。")
                        if data["chunk_id"] in chunk_id_set:
                            filtered_dialogues.append(data)

            with open(filtered_output_file, "w", encoding=getattr(Config, "OUTPUT_ENCODING", "utf-8")) as f:
                for d in filtered_dialogues:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")

            logger.info(f"按 chunk 过滤完成，保存到：{filtered_output_file}（共 {len(filtered_dialogues)} 条）")
            return filtered_output_file

        except Exception as e:
            logger.error(f"按 chunk 过滤失败：{e}")
            return output_file

    def get_chunk_statistics(self, output_file: str) -> Dict[str, Any]:
        """获取按 chunk 分组的统计信息。"""
        try:
            chunk_stats: Dict[int, Dict[str, Any]] = {}
            total_dialogues = 0

            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        chunk_id = data.get("chunk_id")
                        if chunk_id is None:
                            raise ValueError("检测到缺少 chunk_id 的记录，已取消对旧格式的兼容。")

                        if chunk_id not in chunk_stats:
                            chunk_stats[chunk_id] = {
                                "dialogue_count": 0,
                                "roles": {},
                                "total_length": 0,
                            }

                        chunk_stats[chunk_id]["dialogue_count"] += 1
                        role = data.get("role", "Unknown")
                        chunk_stats[chunk_id]["roles"][role] = chunk_stats[chunk_id]["roles"].get(role, 0) + 1
                        chunk_stats[chunk_id]["total_length"] += len(data.get("dialogue", ""))
                        total_dialogues += 1

            summary = {
                "total_chunks": len(chunk_stats),
                "total_dialogues": total_dialogues,
                "average_dialogues_per_chunk": (total_dialogues / len(chunk_stats) if chunk_stats else 0),
                "chunk_details": chunk_stats,
            }
            return summary

        except Exception as e:
            logger.error(f"Chunk 统计信息生成失败：{e}")
            return {}


# ========== CLI ==========
def main() -> int:
    """主函数 - 示例用法 / 命令行接口 """
    import argparse

    parser = argparse.ArgumentParser(description="从小说等长文本中提取角色对话（标准格式：含 chunk_id/dialogue_index）")
    parser.add_argument("input_file", nargs="?", help="输入文本文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（可选）")
    parser.add_argument(
        "--stats",
        action="store_true",
        default=bool(getattr(Config, "DEFAULT_SHOW_STATS", True)),
        help="显示统计信息（默认：开启）",
    )
    parser.add_argument(
        "--no-stats",
        action="store_false",
        dest="stats",
        help="不显示统计信息",
    )
    parser.add_argument(
        "-p",
        "--platform",
        help="指定平台（如 deepseek、openai、moonshot、siliconflow、custom 等）",
    )
    parser.add_argument(
        "-l",
        "--list-platforms",
        action="store_true",
        help="列出已支持的平台",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=int(getattr(Config, "MAX_WORKERS", 4)),
        help=f"并发线程数（默认：{getattr(Config, 'MAX_WORKERS', 4)}）",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        default=bool(getattr(Config, "DEFAULT_CONCURRENT", True)),
        help="使用多线程并发（默认：开启）",
    )
    parser.add_argument(
        "--no-concurrent",
        action="store_false",
        dest="concurrent",
        help="使用单线程处理",
    )
    parser.add_argument(
        "--save-chunk-text",
        action="store_true",
        default=bool(getattr(Config, "DEFAULT_SAVE_CHUNK_TEXT", False)),
        help="保存原始 chunk 文本（默认：关闭）",
    )
    parser.add_argument(
        "--no-save-chunk-text",
        action="store_false",
        dest="save_chunk_text",
        help="不保存原始 chunk 文本",
    )
    parser.add_argument(
        "--sort-output",
        action="store_true",
        default=bool(getattr(Config, "DEFAULT_SORT_OUTPUT", False)),
        help="完成后按 chunk_id 排序输出文件（默认：关闭）",
    )
    parser.add_argument(
        "--no-sort-output",
        action="store_false",
        dest="sort_output",
        help="不排序输出文件",
    )
    parser.add_argument(
        "--reply-window",
        type=int,
        default=int(getattr(Config, "REPLY_WINDOW", 8)),
        help=f"reply 可回溯窗口大小（默认：{getattr(Config, 'REPLY_WINDOW', 8)}）",
    )
    parser.add_argument(
        "--reply-confidence-th",
        type=float,
        default=float(getattr(Config, "REPLY_CONFIDENCE_TH", 0.0)),
        help=f"reply 置信度阈值（默认：{getattr(Config, 'REPLY_CONFIDENCE_TH', 0.0)}，低于该值将清除 reply）",
    )

    args = parser.parse_args()

    # 列出平台
    if args.list_platforms:
        from config import ModelPlatform

        print("=== 支持的平台 ===")
        for name, description in ModelPlatform.list_platforms().items():
            print(f"  {name}: {description}")
        print(f"\n当前默认平台: {getattr(Config, 'CURRENT_PLATFORM', '')}")
        return 0

    # 覆盖部分 Config 运行时参数（对当前进程有效）
    Config.REPLY_WINDOW = max(1, int(args.reply_window))
    Config.REPLY_CONFIDENCE_TH = max(0.0, min(1.0, float(args.reply_confidence_th)))

    # 必要参数校验
    if not args.input_file:
        parser.error("请提供输入文件路径")

    try:
        extractor = DialogueChain(
            platform=args.platform,
            max_workers=args.threads,
            save_chunk_text=args.save_chunk_text,
            cache_dir=getattr(Config, "CACHE_DIR", ".cache"),  # CLI 模式下仍可使用 Config.CACHE_DIR
        )

        # 提取对话
        if args.concurrent:
            print(f"🚀 使用多线程并发（{extractor.max_workers} 个线程）")
            output_file = extractor.extract_dialogues_concurrent(args.input_file, args.output)
        else:
            print("📝 使用单线程处理")
            output_file = extractor.extract_dialogues(args.input_file, args.output)

        # 后处理：排序
        if args.sort_output:
            print("🔄 按 chunk_id 排序输出文件 ...")
            sorted_file = extractor.sort_dialogues(output_file)
            print(f"✅ 排序完成：{sorted_file}")
            output_file = sorted_file

        # 统计信息
        if args.stats:
            stats = extractor.get_statistics(output_file)
            print("\n=== 统计信息 ===")
            print(f"使用平台：{extractor.platform}")
            print(f"使用模型：{extractor.model_name}")
            print(f"处理方式：{'多线程并发' if args.concurrent else '单线程'}"
                  f"{f'（{extractor.max_workers} 线程）' if args.concurrent else ''}")
            print(f"总对话数：{stats.get('total_dialogues', 0)}")
            print(f"角色数量：{stats.get('unique_roles', 0)}")
            print(f"平均对话长度：{stats.get('average_dialogue_length', 0):.1f} 字")

            # 显示 chunk 统计
            chunk_stats = extractor.get_chunk_statistics(output_file)
            if chunk_stats:
                print(f"总块数：{chunk_stats.get('total_chunks', 0)}")
                print(f"平均每块对话数：{chunk_stats.get('average_dialogues_per_chunk', 0):.1f}")

                details = chunk_stats.get("chunk_details", {})
                if details:
                    print("\n对话较密集的块（Top 5）：")
                    top = sorted(details.items(), key=lambda x: x[1]["dialogue_count"], reverse=True)[:5]
                    for chunk_id, info in top:
                        print(f"  块 {chunk_id}: {info['dialogue_count']} 条")

            print("\n角色分布：")
            for role, count in sorted(stats.get("role_distribution", {}).items(), key=lambda x: x[1], reverse=True):
                print(f"  {role}: {count} 条")

    except Exception as e:
        logger.error(f"程序执行失败：{e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
