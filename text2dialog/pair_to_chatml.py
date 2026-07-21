
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pair_to_chatml.py
=================

用途
----
将 `pair_dataset_builder.py` 生成的 **JSONL** 数据集（*合并文件*或*按角色对拆分的文件*）
转换为 **ChatML (messages 格式)** 的 SFT 训练数据。输出的每一行是：

    {"messages": [
        {"role": "system", "content": "<可选系统提示>"},
        {"role": "user", "content": "<源话语>"},
        {"role": "assistant", "content": "<回复话语>"},
        ... （可选多轮拼接） ...
    ]}

支持两种模式：
1) *pair*（默认）——每条有向样本（A→B）转换为**一轮**（user→assistant）。
2) *stitch*——在同一 chunk 内将**连续 A↔B 轮**拼接成一个多轮会话（可控最大轮数）。

为什么需要它
------------
`pair_dataset_builder.py` 的样本结构（单行 JSON）类似：

    {
      "source": {"chunk_id": 12, "dialogue_index": 5, "role": "A", "text": "......"},
      "reply":  {"chunk_id": 12, "dialogue_index": 6, "role": "B", "text": "......"},
      "pair": {"from": "A", "to": "B"},
      "confidence": 0.92
    }

而主流的 SFT 训练更推荐 ChatML 的消息数组格式。本脚本即做字段映射、
（可选）拼接多轮、（可选）注入系统提示，并输出到 JSONL。

主要特性
--------
- 输入：单个文件、目录（自动抓取 *.jsonl）、或 glob（如 data/pair_*.jsonl）。
- 模式：pair / stitch（同 chunk 连续轮拼接）。
- 过滤：最小置信度、去重。
- 角色：pair 固定 source→user、reply→assistant；仅 stitch 可交换说话者映射。
- 系统提示：默认“生成对输入内容的回复。”；也支持固定字符串或模板（可用 {from_role}/{to_role} 等变量）。
- 输出：写入 JSONL；可选附带 meta（pair、chunk、index、confidence 等）方便回溯。

使用示例
--------
# 1) 将目录下所有 pair_*.jsonl 转为 ChatML（每条样本一行，不拼接）
python pair_to_chatml.py \
  -i ./pair_datasets \
  -o ./sft_chatml.jsonl

# 2) 拼接多轮（同 chunk & 角色对连续），最多 6 轮，并注入系统模板
python pair_to_chatml.py \
  -i ./pair_datasets \
  -o ./sft_chatml_stitched.jsonl \
  --mode stitch --max-turns 6 \
  --system-template "你现在扮演 {to_role}，将与 {from_role} 进行对话，请准确、自然地回应。"

# 3) stitch 模式读取合并文件 + 最小置信度过滤 + 交换说话者映射
python pair_to_chatml.py \
  -i ./pair_datasets/all_pairs.jsonl \
  -o ./sft_chatml_rev.jsonl \
  --mode stitch \
  --min-confidence 0.85 \
  --reverse

兼容性说明
----------
- 输入必须来自当前仓库的 `pair_dataset_builder.py`；若字段名一致，也可用于同结构数据。
- 输出严格遵循 OpenAI ChatML「messages 列表」约定：role ∈ {"system","user","assistant"}，content 为字符串。

"""
import argparse
import dataclasses
import glob
import json
import math
import os
import string
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, DefaultDict
from collections import defaultdict


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _strict_positive_int(value: Any, field: str) -> int:
    result = _strict_nonnegative_int(value, field)
    if result == 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _strict_confidence(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not (0.0 <= result <= 1.0):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return result


def _validate_system_template(value: Optional[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError("system_template must be a string")
    allowed = {"from_role", "to_role", "src_role", "tgt_role"}
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(value) if field]
    except ValueError as exc:
        raise ValueError(f"invalid system_template: {exc}") from exc
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"unknown system_template fields: {sorted(unknown)}")

# -----------------------------
# 数据结构（与 pair_dataset_builder.py 对齐）
# -----------------------------

@dataclass
class Endpoint:
    chunk_id: int
    dialogue_index: int
    role: str
    text: str

@dataclass
class PairRecord:
    source: Endpoint
    reply: Endpoint
    pair_from: str
    pair_to: str
    confidence: Optional[float] = None

    @staticmethod
    def from_obj(obj: Dict[str, Any]) -> 'PairRecord':
        """将一行 JSON 转为 PairRecord；必要字段缺失会抛出 ValueError。"""
        try:
            src = obj["source"]
            tgt = obj["reply"]
        except Exception as e:
            raise ValueError(f"missing source/reply: {e}")

        def _endpoint(d: Dict[str, Any]) -> Endpoint:
            if not isinstance(d, dict):
                raise ValueError("endpoint must be an object")
            role = d.get("role")
            text = d.get("text")
            if not isinstance(role, str) or not role.strip():
                raise ValueError("endpoint role must be a non-empty string")
            if not isinstance(text, str):
                raise ValueError("endpoint text must be a string")
            return Endpoint(
                chunk_id=_strict_nonnegative_int(d.get("chunk_id"), "chunk_id"),
                dialogue_index=_strict_nonnegative_int(
                    d.get("dialogue_index"), "dialogue_index"
                ),
                role=role,
                text=text,
            )

        source = _endpoint(src)
        reply = _endpoint(tgt)
        if source.chunk_id != reply.chunk_id:
            raise ValueError("reply must reference the same chunk as source")
        if source.dialogue_index >= reply.dialogue_index:
            raise ValueError("reply must be later than source")
        if norm(source.role) == norm(reply.role):
            raise ValueError("source and reply roles must differ")
        raw_confidence = obj.get("confidence", None)
        confidence = (
            None
            if raw_confidence is None
            else _strict_confidence(raw_confidence, "confidence")
        )
        pair_obj = obj.get("pair", {})
        if not isinstance(pair_obj, dict):
            raise ValueError("pair must be an object")
        pair_from = pair_obj.get("from", source.role)
        pair_to = pair_obj.get("to", reply.role)
        if not isinstance(pair_from, str) or not isinstance(pair_to, str):
            raise ValueError("pair roles must be strings")
        if norm(pair_from) != norm(source.role) or norm(pair_to) != norm(reply.role):
            raise ValueError("pair roles must match source/reply roles")
        pr = PairRecord(
            source=source,
            reply=reply,
            pair_from=pair_from,
            pair_to=pair_to,
            confidence=confidence,
        )
        return pr

# -----------------------------
# I/O
# -----------------------------

def read_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] {p} line {ln}: skip invalid JSON ({e})", file=sys.stderr)

def write_jsonl(path: Union[str, Path], items: Iterable[Dict[str, Any]]) -> None:
    write_jsonl_atomic(path, items)


def write_jsonl_atomic(
    path: Union[str, Path],
    items: Iterable[Dict[str, Any]],
) -> int:
    """Write JSONL to a same-directory temp and atomically publish it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            for obj in items:
                json.dump(obj, handle, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, p)
        temp_name = None
        return count
    finally:
        if temp_name:
            try:
                os.remove(temp_name)
            except FileNotFoundError:
                pass


def paths_collide(left: Union[str, Path], right: Union[str, Path]) -> bool:
    a, b = Path(left).expanduser().resolve(), Path(right).expanduser().resolve()
    if os.path.normcase(str(a)) == os.path.normcase(str(b)):
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False

def discover_inputs(inputs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if any(ch in inp for ch in "*?[]"):
            matches = [Path(x) for x in glob.glob(inp)]
            out.extend([m for m in matches if m.is_file() and m.suffix.lower() == ".jsonl"])
        elif p.is_dir():
            out.extend([q for q in p.glob("*.jsonl") if q.is_file()])
        elif p.is_file():
            out.append(p)
        else:
            print(f"[WARN] input not found or unsupported: {inp}", file=sys.stderr)
    # 去重 & 排序
    uniq = sorted(set(out))
    if not uniq:
        print("[ERR] no input files found", file=sys.stderr)
    return uniq

# -----------------------------
# 清理/工具
# -----------------------------

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

def content_ok(s: str, min_len: int = 1) -> bool:
    return isinstance(s, str) and len(norm(s)) >= min_len

def passes_confidence(c: Optional[float], th: Optional[float]) -> bool:
    try:
        if c is None:
            return th is None
        value = _strict_confidence(c, "confidence")
        if th is None:
            return True
        threshold = _strict_confidence(th, "min_confidence")
        return value >= threshold
    except ValueError:
        return False

def default_system() -> str:
    return "生成对输入内容的回复。"

# -----------------------------
# ChatML 构建
# -----------------------------

@dataclass
class ChatMessage:
    role: str
    content: str

    def to_json(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

@dataclass
class ChatRecord:
    messages: List[ChatMessage]
    meta: Optional[Dict[str, Any]] = None

    def to_json(self, include_meta: bool = False) -> Dict[str, Any]:
        obj = {"messages": [m.to_json() for m in self.messages]}
        if include_meta and self.meta:
            obj["meta"] = self.meta
        return obj

# -----------------------------
# 拼接器（可选）
# -----------------------------

@dataclass(frozen=True)
class StitchKey:
    chunk_id: int
    from_role: str
    to_role: str

@dataclass
class PairKey:
    chunk_id: int
    src_idx: int
    tgt_idx: int

def group_for_stitch(pairs: Iterable[PairRecord]) -> Dict[StitchKey, List[PairRecord]]:
    """
    将样本按 (chunk_id, from_role, to_role) 分桶，并在每桶内按 src/tgt index 排序。
    约束：仅在**同一个 chunk**内进行拼接（跨 chunk 对小说来说其上下文连续性不可保证）。
    """
    buckets: DefaultDict[StitchKey, List[PairRecord]] = defaultdict(list)
    for pr in pairs:
        if pr.source.chunk_id != pr.reply.chunk_id:
            # 保守起见：跨 chunk 的样本放到各自桶，但 stitch 时不会跨样本拼接
            key = StitchKey(chunk_id=pr.source.chunk_id, from_role=pr.pair_from, to_role=pr.pair_to)
        else:
            key = StitchKey(chunk_id=pr.source.chunk_id, from_role=pr.pair_from, to_role=pr.pair_to)
        buckets[key].append(pr)
    # 每桶内排序：先按源 index，再按目标 index
    for key, lst in buckets.items():
        lst.sort(key=lambda r: (r.source.dialogue_index, r.reply.dialogue_index))
    return buckets

def stitch_sequences(records: List[PairRecord], max_turns: int) -> List[List[PairRecord]]:
    """
    给定同一 (chunk_id, A→B) 桶内按 index 排序后的样本，
    以“紧邻接续”为准则拼接为若干多轮会话：
      A(i) -> B(i+1) ；下一轮必须从 B(i+1) -> A(i+2) 开始，如此交替。
    注意：pair_dataset_builder.py 的样本是“有向边”，这里通过 index 连续性简易串联。
    """
    sessions: List[List[PairRecord]] = []
    i = 0
    n = len(records)
    while i < n:
        cur = [records[i]]
        i += 1
        # 尝试向后拼接，直到达上限或不再连续
        while len(cur) < max_turns and i < n:
            prev = cur[-1]
            cand = records[i]
            # 连续判据：后一条的 source 必须是前一条的 reply（同 chunk & index 相邻）
            consecutive = (
                (cand.source.chunk_id == prev.reply.chunk_id) and
                (cand.source.dialogue_index == prev.reply.dialogue_index)
            )
            if consecutive:
                cur.append(cand)
                i += 1
            else:
                break
        sessions.append(cur)
    return sessions

@dataclass(frozen=True)
class BiStitchKey:
    chunk_id: int
    role_a: str
    role_b: str

def _bi_key(from_role: str, to_role: str) -> Tuple[str, str]:
    """无序角色对（按字典序排序后作为 key）"""
    a, b = sorted([from_role, to_role])
    return a, b

def group_for_stitch_bidirectional(pairs: Iterable[PairRecord]) -> Dict[BiStitchKey, List[PairRecord]]:
    """
    将样本按 (chunk_id, {role_a, role_b}) 分桶；桶内包含 A→B 与 B→A 两个方向的样本，
    供后续按“上一轮的 reply 即下一轮的 source”进行双向拼接。
    """
    buckets: DefaultDict[BiStitchKey, List[PairRecord]] = defaultdict(list)
    for pr in pairs:
        # Cross-chunk and backwards edges cannot form a chronological ChatML
        # conversation and must never be stitched together.
        if pr.source.chunk_id != pr.reply.chunk_id:
            continue
        if pr.source.dialogue_index >= pr.reply.dialogue_index:
            continue
        if norm(pr.source.role) == norm(pr.reply.role):
            continue
        a, b = _bi_key(pr.source.role, pr.reply.role)
        key = BiStitchKey(chunk_id=pr.source.chunk_id, role_a=a, role_b=b)
        buckets[key].append(pr)
    # 桶内按 source.index 再 reply.index 排序，利于稳定构链
    for key, lst in buckets.items():
        lst.sort(key=lambda r: (r.source.dialogue_index, r.reply.dialogue_index))
    return buckets

def stitch_sequences_bidirectional(records: List[PairRecord], max_turns: int) -> List[List[PairRecord]]:
    """
    在同一 (chunk_id, {A,B}) 桶内构造多轮会话：
      规则：后一条的 source 必须与前一条的 reply 完全一致（chunk/index/role）。
    采用贪心策略：从最早的样本开始，尽量延长，避免交叉复用（消费掉已用边）。
    """
    # 按 source 起点建立索引： (role, index) -> [record_index...]
    by_source: DefaultDict[Tuple[int, str, int], List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_source[(r.source.chunk_id, r.source.role, r.source.dialogue_index)].append(i)
    # 每个列表按 reply.index 升序，尽量选择最近的下一轮
    for k in by_source:
        by_source[k].sort(key=lambda i: records[i].reply.dialogue_index)

    used: set = set()
    # used 存放的是 records 中的下标索引
    sessions: List[List[PairRecord]] = []

    # 以 source.index/ reply.index 为起点排序，稳定生成
    seeds = sorted(range(len(records)), key=lambda i: (records[i].source.dialogue_index, records[i].reply.dialogue_index))

    for si in seeds:
        if si in used:
            continue
        start = records[si]
        # 开始一条新会话
        cur: List[PairRecord] = [start]
        used.add(si)

        # 向后扩展
        while len(cur) < max_turns:
            last = cur[-1]
            expect_role = last.reply.role
            expect_idx = last.reply.dialogue_index
            cands = by_source.get((last.reply.chunk_id, expect_role, expect_idx), [])
            # 选择首个尚未使用的候选
            next_idx = None
            for ci in cands:
                if ci in used:
                    continue
                next_idx = ci
                used.add(ci)
                break
            if next_idx is None:
                break
            cur.append(records[next_idx])
        sessions.append(cur)
    return sessions

# -----------------------------
# 主转换逻辑
# -----------------------------

def convert_pair_to_chatml(
    inputs: Sequence[Path],
    mode: str = "pair",
    min_confidence: Optional[float] = None,
    reverse_roles: bool = False,
    system_text: Optional[str] = None,
    system_template: Optional[str] = None,
    max_turns: int = 1,
    include_meta: bool = False,
    dedupe: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Validate conversion options eagerly, then return a lazy record iterator."""
    if mode not in {"pair", "stitch"}:
        raise ValueError(f"unknown mode: {mode}")
    _strict_positive_int(max_turns, "max_turns")
    if min_confidence is not None:
        _strict_confidence(min_confidence, "min_confidence")
    if mode == "pair" and reverse_roles:
        raise ValueError("reverse_roles is not supported in pair mode")
    if system_text is not None and not isinstance(system_text, str):
        raise ValueError("system_text must be a string")
    _validate_system_template(system_template)
    return _convert_pair_to_chatml_iter(
        inputs=inputs,
        mode=mode,
        min_confidence=min_confidence,
        reverse_roles=reverse_roles,
        system_text=system_text,
        system_template=system_template,
        max_turns=max_turns,
        include_meta=include_meta,
        dedupe=dedupe,
    )


def _convert_pair_to_chatml_iter(
    inputs: Sequence[Path],
    mode: str = "pair",
    min_confidence: Optional[float] = None,
    reverse_roles: bool = False,
    system_text: Optional[str] = None,
    system_template: Optional[str] = None,
    max_turns: int = 1,
    include_meta: bool = False,
    dedupe: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    将 pair JSONL 转为 ChatML JSONL（逐条 yield）。

    - mode="pair": 一条 PairRecord -> 一行 ChatML（可含系统），共 1 轮。
    - mode="stitch": 同 chunk & 同角色对 的连续 PairRecord 拼接为一个多轮 ChatML，
      最多 max_turns 轮。

    角色映射：pair 固定 source → user、reply → assistant；stitch 可通过
    reverse_roles 交换说话者与 ChatML user/assistant 的映射，但不倒置时间。

    系统提示：
      优先使用 system_template（可含 {from_role}/{to_role}/{src_role}/{tgt_role} 占位符）；
      否则使用 system_text；两者都缺省则不注入系统消息。
    """
    if mode not in {"pair", "stitch"}:
        raise ValueError(f"unknown mode: {mode}")
    _strict_positive_int(max_turns, "max_turns")
    if min_confidence is not None:
        _strict_confidence(min_confidence, "min_confidence")
    if mode == "pair" and reverse_roles:
        raise ValueError("reverse_roles is not supported in pair mode")

    # 1) 读取 & 过滤 & 归并
    all_pairs: List[PairRecord] = []
    seen_sig: set = set()  # 用于去重
    for path in inputs:
        for obj in read_jsonl(path):
            try:
                pr = PairRecord.from_obj(obj)
            except Exception as e:
                print(f"[WARN] {path} skip line (bad fields): {e}", file=sys.stderr)
                continue
            if not passes_confidence(pr.confidence, min_confidence):
                continue
            if not content_ok(pr.source.text) or not content_ok(pr.reply.text):
                continue
            if dedupe:
                sig = (norm(pr.source.text), norm(pr.reply.text))
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
            all_pairs.append(pr)

    if not all_pairs:
        return  # 空

    def build_system(pr: PairRecord) -> Optional[str]:
        if system_template:
            from_role, to_role = pr.pair_from, pr.pair_to
            if mode == "stitch" and reverse_roles:
                from_role, to_role = to_role, from_role
            fmt_vars = dict(
                from_role=from_role, to_role=to_role,
                src_role=pr.source.role, tgt_role=pr.reply.role,
            )
            try:
                return system_template.format(**fmt_vars)
            except Exception as e:
                raise ValueError(f"system-template format error: {e}") from e
        elif system_text:
            return system_text
        else:
            return default_system()

    # 2) 转换
    if mode == "pair":
        for pr in all_pairs:
            u_txt, a_txt = (pr.source.text, pr.reply.text)
            msgs: List[ChatMessage] = []
            sys_msg = build_system(pr)
            if sys_msg:
                msgs.append(ChatMessage(role="system", content=sys_msg))
            msgs.append(ChatMessage(role="user", content=norm(u_txt)))
            msgs.append(ChatMessage(role="assistant", content=norm(a_txt)))
            meta = None
            if include_meta:
                meta = {
                    "pair": {"from": pr.pair_from, "to": pr.pair_to},
                    "source": dataclasses.asdict(pr.source),
                    "reply": dataclasses.asdict(pr.reply),
                    "confidence": pr.confidence,
                }
            yield ChatRecord(messages=msgs, meta=meta).to_json(include_meta=include_meta)

    elif mode == "stitch":
        # 分桶（chunk_id, {A,B} 无序角色对）
        buckets = group_for_stitch_bidirectional(all_pairs)
        for key, recs in buckets.items():
            # 在同一角色对内做双向连续拼接
            sessions = stitch_sequences_bidirectional(recs, max_turns=max_turns)
            for sess in sessions:
                if not sess:
                    continue
                msgs: List[ChatMessage] = []
                # 系统消息按首条样本生成
                sys_msg = build_system(sess[0])
                if sys_msg:
                    msgs.append(ChatMessage(role="system", content=sys_msg))
                # Expand the directed edge chain into one chronological list
                # of utterances, then map speakers to ChatML roles. Reversal
                # changes the speaker mapping; it never reverses time.
                first = sess[0]
                user_speaker = first.source.role
                assistant_speaker = first.reply.role
                if reverse_roles:
                    user_speaker, assistant_speaker = assistant_speaker, user_speaker

                utterances = [first.source] + [edge.reply for edge in sess]
                mapped: List[ChatMessage] = []
                for utterance in utterances:
                    if utterance.role == user_speaker:
                        mapped.append(ChatMessage(role="user", content=norm(utterance.text)))
                    elif utterance.role == assistant_speaker:
                        mapped.append(ChatMessage(role="assistant", content=norm(utterance.text)))
                    else:
                        mapped = []
                        break

                while mapped and mapped[0].role != "user":
                    mapped.pop(0)
                while mapped and mapped[-1].role != "assistant":
                    mapped.pop()
                if len(mapped) < 2 or any(
                    message.role != ("user" if i % 2 == 0 else "assistant")
                    for i, message in enumerate(mapped)
                ):
                    continue
                msgs.extend(mapped)
                meta = None
                if include_meta:
                    meta = {
                        "pair": {"from": sess[0].pair_from, "to": sess[0].pair_to},
                        "chunk_id": key.chunk_id,
                        "turns": len(sess),
                        "indices": [(p.source.dialogue_index, p.reply.dialogue_index) for p in sess],
                        "confidences": [p.confidence for p in sess],
                    }
                yield ChatRecord(messages=msgs, meta=meta).to_json(include_meta=include_meta)

# -----------------------------
# CLI
# -----------------------------

def load_text_maybe_from_file(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("@"):
        path = Path(s[1:])
        return path.read_text(encoding="utf-8")
    return s

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert pair_dataset_builder JSONL to ChatML SFT JSONL.")
    ap.add_argument("-i", "--input", nargs="+", required=True,
                    help="输入：文件/目录/glob（可多值）。目录会匹配 *.jsonl；glob 例：'data/pair_*.jsonl'")
    ap.add_argument("-o", "--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--mode", choices=["pair", "stitch"], default="pair",
                    help="pair: 每条样本 1 轮；stitch: 同 chunk 连续拼接为多轮")
    ap.add_argument("--max-turns", type=int, default=4, help="stitch 模式下，每个会话最大轮数（默认 4）")
    ap.add_argument("--min-confidence", type=float, default=None, help="过滤最小置信度（默认不过滤）")
    ap.add_argument("--dedupe", action="store_true", help="按 (source.text, reply.text) 去重")
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="仅 stitch 模式：交换说话者到 user/assistant 的映射",
    )
    ap.add_argument("--system", dest="system_text", default=None,
                    help="系统消息文本；若以 @ 开头，视为从文件读取内容")
    ap.add_argument("--system-template", default=None,
                    help="系统消息模板（优先级更高），可用 {from_role}/{to_role}/{src_role}/{tgt_role}")
    ap.add_argument("--include-meta", action="store_true", help="在每行附加 meta 字段，便于回溯调试")

    args = ap.parse_args(argv)

    inputs = discover_inputs(args.input)
    if not inputs:
        return 2

    out_path = Path(args.out)
    if any(paths_collide(path, out_path) for path in inputs):
        print("[ERR] output path must not overwrite an input JSONL", file=sys.stderr)
        return 2

    system_text = load_text_maybe_from_file(args.system_text)
    system_template = load_text_maybe_from_file(args.system_template)

    try:
        items = convert_pair_to_chatml(
            inputs=inputs,
            mode=args.mode,
            min_confidence=args.min_confidence,
            reverse_roles=args.reverse,
            system_text=system_text,
            system_template=system_template,
            max_turns=args.max_turns,
            include_meta=args.include_meta,
            dedupe=args.dedupe,
        )
        count = write_jsonl_atomic(out_path, items)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[ERR] ChatML conversion failed: {exc}", file=sys.stderr)
        return 2
    print(f"[OK] wrote {count} ChatML records -> {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
