
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
- 角色：默认将 source→user、reply→assistant；可反转。
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

# 3) 读取合并文件 + 最小置信度过滤 + 反转角色（把 reply 当作 user）
python pair_to_chatml.py \
  -i ./pair_datasets/all_pairs.jsonl \
  -o ./sft_chatml_rev.jsonl \
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
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, DefaultDict
from collections import defaultdict

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
            return Endpoint(
                chunk_id=int(d["chunk_id"]),
                dialogue_index=int(d["dialogue_index"]),
                role=str(d["role"]),
                text=str(d["text"]),
            )

        pr = PairRecord(
            source=_endpoint(src),
            reply=_endpoint(tgt),
            pair_from=str(obj.get("pair", {}).get("from", src.get("role", ""))),
            pair_to=str(obj.get("pair", {}).get("to", tgt.get("role", ""))),
            confidence=(obj.get("confidence", None)),
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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in items:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

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
    if th is None:  # 不过滤
        return True
    try:
        return (c is not None) and (float(c) >= float(th))
    except Exception:
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
        a, b = _bi_key(pr.pair_from, pr.pair_to)
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
    by_source: DefaultDict[Tuple[str, int], List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_source[(r.source.role, r.source.dialogue_index)].append(i)
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
            cands = by_source.get((expect_role, expect_idx), [])
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
    """
    将 pair JSONL 转为 ChatML JSONL（逐条 yield）。

    - mode="pair": 一条 PairRecord -> 一行 ChatML（可含系统），共 1 轮。
    - mode="stitch": 同 chunk & 同角色对 的连续 PairRecord 拼接为一个多轮 ChatML，
      最多 max_turns 轮。

    角色映射：
      默认：source → user, reply → assistant；
      若 reverse_roles=True：source → assistant, reply → user。

    系统提示：
      优先使用 system_template（可含 {from_role}/{to_role}/{src_role}/{tgt_role} 占位符）；
      否则使用 system_text；两者都缺省则不注入系统消息。
    """
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
            fmt_vars = dict(
                from_role=pr.pair_from, to_role=pr.pair_to,
                src_role=pr.source.role, tgt_role=pr.reply.role,
            )
            try:
                return system_template.format(**fmt_vars)
            except Exception as e:
                print(f"[WARN] system-template format error: {e}", file=sys.stderr)
                return None
        elif system_text:
            return system_text
        else:
            return default_system()

    # 2) 转换
    if mode == "pair":
        for pr in all_pairs:
            u_txt, a_txt = (pr.source.text, pr.reply.text)
            u_role, a_role = ("user", "assistant")
            if reverse_roles:
                u_txt, a_txt = a_txt, u_txt
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
                # 依次展开为 user/assistant 轮；避免重复加入“重叠”的上一轮回复
                # 会话的 user/assistant 对应到具体说话人：以首条样本的方向为准
                first = sess[0]
                user_speaker = first.pair_from
                assistant_speaker = first.pair_to
                if reverse_roles:
                    user_speaker, assistant_speaker = assistant_speaker, user_speaker

                # 先放入首轮完整的 user→assistant
                if not reverse_roles:
                    msgs.append(ChatMessage(role="user", content=norm(first.source.text)))
                    msgs.append(ChatMessage(role="assistant", content=norm(first.reply.text)))
                else:
                    # 反转：以 reply 作为 user 的第一句
                    msgs.append(ChatMessage(role="user", content=norm(first.reply.text)))
                    msgs.append(ChatMessage(role="assistant", content=norm(first.source.text)))

                # 后续各轮只追加“新出现”的一边
                for pr in sess[1:]:
                    # pr.pair_from 表示当前这条边的说话人（源）
                    # 如果源是 assistant_speaker，说明上一轮我们已经加入了它的内容（assistant）；
                    # 此时只需追加用户的新一句（reply）。反之亦然。
                    if pr.pair_from == assistant_speaker:
                        # assistant -> user，只追加用户的回复
                        msgs.append(ChatMessage(role="user", content=norm(pr.reply.text)))
                    elif pr.pair_from == user_speaker:
                        # user -> assistant，只追加助手的回复
                        msgs.append(ChatMessage(role="assistant", content=norm(pr.reply.text)))
                    else:
                        # 理论上不会发生，保险起见忽略之
                        continue
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
    else:
        raise ValueError(f"unknown mode: {mode}")

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
    ap.add_argument("--reverse", action="store_true", help="反转角色：把 reply 当作 user")
    ap.add_argument("--system", dest="system_text", default=None,
                    help="系统消息文本；若以 @ 开头，视为从文件读取内容")
    ap.add_argument("--system-template", default=None,
                    help="系统消息模板（优先级更高），可用 {from_role}/{to_role}/{src_role}/{tgt_role}")
    ap.add_argument("--include-meta", action="store_true", help="在每行附加 meta 字段，便于回溯调试")

    args = ap.parse_args(argv)

    inputs = discover_inputs(args.input)
    if not inputs:
        return 2

    system_text = load_text_maybe_from_file(args.system_text)
    system_template = load_text_maybe_from_file(args.system_template)

    items = convert_pair_to_chatml(
        inputs=inputs,
        mode=args.mode,
        min_confidence=args.min_confidence,
        reverse_roles=args.reverse,
        system_text=system_text,
        system_template=system_template,
        max_turns=max(1, int(args.max_turns or 1)),
        include_meta=args.include_meta,
        dedupe=args.dedupe,
    )

    # 写出
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        count = 0
        for obj in items or []:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    print(f"[OK] wrote {count} ChatML records -> {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
