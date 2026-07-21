#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_output.py - 检查 dialogue_chain JSONL 格式
"""
import json
import math
import sys
import contextlib
import io
from typing import Any, Dict, List

ChunkEntry = Dict[str, Any]


def _error(message: str) -> None:
    print(message, file=sys.stderr)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate(path: str) -> int:
    by_chunk: Dict[int, List[ChunkEntry]] = {}
    ok = True

    with open(path, 'r', encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                _error(f"[line {line_no}] 无效的 JSON：{exc}")
                ok = False
                continue
            if not isinstance(record, dict):
                _error(f"[line {line_no}] 记录不是 JSON 对象")
                ok = False
                continue

            chunk_id = record.get('chunk_id')
            if not _is_int(chunk_id) or chunk_id < 0:
                _error(f"[line {line_no}] 非法的 chunk_id: {chunk_id!r}")
                ok = False
                continue

            by_chunk.setdefault(chunk_id, []).append({'line': line_no, 'record': record})

    if not by_chunk:
        _error('未找到分块记录；请确保该文件由当前流水线生成')
        print('失败', file=sys.stderr)
        return 1

    chunk_index_maps: Dict[int, Dict[int, ChunkEntry]] = {}

    for cid, entries in sorted(by_chunk.items()):
        seen_indices = set()
        valid_entries: List[ChunkEntry] = []
        for entry in entries:
            record = entry['record']
            line_no = entry['line']
            dialogue_index = record.get('dialogue_index')
            valid = True

            if not _is_int(dialogue_index) or dialogue_index < 0:
                _error(f"[chunk {cid}] line {line_no} di={dialogue_index!r}: dialogue_index 必须是非负整数")
                ok = False
                valid = False
            elif dialogue_index in seen_indices:
                _error(f"[chunk {cid}] line {line_no} di={dialogue_index}: 在同一 chunk 内出现重复的 dialogue_index")
                ok = False
                valid = False
            else:
                seen_indices.add(dialogue_index)

            if valid:
                role = record.get('role')
                if not isinstance(role, str) or not role.strip():
                    _error(f"[chunk {cid}] line {line_no} di={dialogue_index}: 缺失或为空的 role")
                    ok = False
                    valid = False

            if valid:
                dialogue = record.get('dialogue')
                if not isinstance(dialogue, str) or not dialogue.strip():
                    _error(f"[chunk {cid}] line {line_no} di={dialogue_index}: 缺失或为空的 dialogue 文本")
                    ok = False
                    valid = False

            if valid:
                reply = record.get('reply')
                if reply is not None and not isinstance(reply, dict):
                    _error(f"[chunk {cid}] line {line_no} di={dialogue_index}: reply 必须为 null 或对象，实际为 {type(reply).__name__}")
                    ok = False
                    valid = False

            if valid:
                valid_entries.append(entry)

        entries[:] = valid_entries
        entries.sort(key=lambda entry: entry['record']['dialogue_index'])
        index_map: Dict[int, ChunkEntry] = {}
        for expected_idx, entry in enumerate(entries):
            record = entry['record']
            line_no = entry['line']
            dialogue_index = record['dialogue_index']
            index_map[dialogue_index] = entry
            if dialogue_index != expected_idx:
                _error(f"[chunk {cid}] line {line_no} di={dialogue_index}: 预期 dialogue_index 为 {expected_idx}")
                ok = False
        chunk_index_maps[cid] = index_map

    for cid, entries in sorted(by_chunk.items()):
        for entry in entries:
            record = entry['record']
            line_no = entry['line']
            dialogue_index = record['dialogue_index']
            reply = record.get('reply')
            if not isinstance(reply, dict):
                continue

            target_index = reply.get('target_index')
            if not _is_int(target_index) or target_index < 0:
                _error(f"[chunk {cid}] line {line_no} di={dialogue_index}: 无效的 target_index {target_index!r}")
                ok = False
                continue

            target_chunk_id = reply.get('target_chunk_id', cid)
            if not _is_int(target_chunk_id) or target_chunk_id < 0:
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: "
                    f"invalid target_chunk_id {target_chunk_id!r}"
                )
                ok = False
                continue
            if target_chunk_id != cid:
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: "
                    "reply must reference an earlier utterance in the same chunk"
                )
                ok = False
                continue

            target_chunk = chunk_index_maps.get(cid)
            if target_chunk is None:
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: "
                    "current chunk is missing"
                )
                ok = False
                continue
            if not (0 <= target_index < dialogue_index):
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: "
                    f"reply must reference an earlier local index, got {target_index}"
                )
                ok = False
                continue
            if target_index not in target_chunk:
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: "
                    f"target_index {target_index} is absent from chunk {cid}"
                )
                ok = False
                continue

            confidence = reply.get('confidence')
            if confidence is None or not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not (0.0 <= float(confidence) <= 1.0):
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: confidence 必须是 [0, 1] 区间内的有限数值"
                )
                ok = False

            target_record = target_chunk[target_index]['record']
            actual_target_role = target_record['role'].strip()
            current_role = record['role'].strip()
            target_role = reply.get('target_role')
            if not isinstance(target_role, str) or not target_role.strip():
                _error(f"[chunk {cid}] line {line_no} di={dialogue_index}: target_role 必须是非空字符串")
                ok = False
            elif target_role.strip() != actual_target_role:
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: target_role "
                    f"{target_role!r} 与被引用发言的 role {actual_target_role!r} 不一致"
                )
                ok = False

            if current_role == actual_target_role:
                _error(
                    f"[chunk {cid}] line {line_no} di={dialogue_index}: reply 不能指向同一 speaker"
                )
                ok = False

    print('通过' if ok else '失败', file=sys.stderr)
    return 0 if ok else 1


def validate_with_report(path: str, max_messages: int = 200) -> Dict[str, Any]:
    """运行校验并返回可给 API/前端消费的结构化报告。"""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        code = validate(path)
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    messages = [line for line in lines if line not in {"通过", "失败"}]
    return {
        "ok": code == 0,
        "status": "通过" if code == 0 else "失败",
        "error_count": len(messages),
        "messages": messages[:max_messages],
        "truncated": len(messages) > max_messages,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('用法：python validate_output.py <file.jsonl>', file=sys.stderr)
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))
