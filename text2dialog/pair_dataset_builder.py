
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pair_dataset_builder.py
=======================

用途
----
从 `dialogue_chain.py` 生成并经 `validate_output.py` 检查通过的 **JSONL** 输出里，
抽取**两个角色之间的有向对话样本**：`A -> B` 与 `B -> A` 互不相同。

要点
--------
1) **默认置信度阈值 0.8**（可用 `--min-confidence` 调整）。若缺失置信度，将**丢弃**该样本，以避免污染。
2) **严格模式 --strict（默认启用）**：进一步只保留能**确定**为 A→B 的样本：
   - 仅以解析到的**真实被指向发言**的 `role` 与当前发言 `role` 判断方向；不做回退猜测。
   - 若 `reply.target_role` 存在但与解析得到的源 `role` 不一致，视为**不确定**，丢弃。
   - 要求存在 `confidence`，且 >= `--min-confidence`（默认 0.8，可调）。
3) **鲁棒跳过不合法数据**：
   - 无法解析的 JSON 行、非 dict、关键字段缺失/类型错误 → **跳过并告警**，不中止。
   - 源/回复文本若为空、包含非法控制字符（除 `\\t\\n\\r`）、或包含 Unicode 替换字符 `\\uFFFD` → **跳过**。
   - 可用 `--deny-pattern` 指定正则黑名单；命中则跳过（对源与回复文本都生效）。
   - 可用 `--min-src-chars/--min-reply-chars/--max-src-chars/--max-reply-chars` 设置字符长度阈值。
4) **多对输出**：`--pairs "A,B" "B,C" "B,D"` 会分别生成三个文件；方向不同各自独立。

快速示例
--------
# 仅列出角色与频次
python pair_dataset_builder.py -i output.jsonl --list-roles

# 严格抽取 A->B、B->C、B->D 三个方向（各一个文件），阈值默认 0.8
python pair_dataset_builder.py -i output.jsonl --pairs "A,B" "B,C" "B,D" --strict -o out_dir

# 同上，但把阈值调到 0.9，并过滤过短回复
python pair_dataset_builder.py -i output.jsonl --pairs "A,B" "B,C" "B,D" --strict --min-confidence 0.9 --min-reply-chars 2 -o out_dir

# 若希望在抽取前显式调用校验器
python pair_dataset_builder.py -i output.jsonl --validate-path path/to/validate_output.py --pairs "A,B" --strict

输出格式
--------
每行一条 JSON：
{
  "source": {"chunk_id": 1, "dialogue_index": 0, "role": "A", "text": "…"},
  "reply":  {"chunk_id": 1, "dialogue_index": 1, "role": "B", "text": "…"},
  "pair":   {"from": "A", "to": "B"},
  "confidence": 0.93
}
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterable


# -----------------------------
# I/O 与工具
# -----------------------------

def _norm_role(s: str) -> str:
    """统一规范角色名：去首尾空白，并保持原大小写（只做 trim）。"""
    return (s or "").strip()


def _safe_pair_name(src: str, tgt: str) -> str:
    """把角色对转换为安全的文件名片段。"""
    def safe(s: str) -> str:
        s = s.strip()
        s = re.sub(r'[\\/:*?"<>|]+', '_', s)  # 替换非法路径字符
        s = re.sub(r'\s+', '_', s)           # 空白压缩为 '_'
        return s or "EMPTY"
    return f"pair_{safe(src)}__to__{safe(tgt)}.jsonl"


def _read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    逐行读取 JSONL，返回 (line_no, record)。对异常行**告警并跳过**。
    """
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            line = (raw or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[WARN] 跳过第 {i} 行：非法 JSON（{exc}）", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                print(f"[WARN] 跳过第 {i} 行：期望对象(dict)，得到 {type(obj).__name__}", file=sys.stderr)
                continue
            yield i, obj


def _try_import_validator(validator_path: Optional[Path]) -> Optional[Any]:
    """尝试导入 validate_output.py 并返回模块；失败时返回 None。"""
    if not validator_path:
        return None
    vp = validator_path.resolve()
    if not vp.exists():
        print(f"[WARN] 无法找到校验脚本: {vp}", file=sys.stderr)
        return None
    sys.path.insert(0, str(vp.parent))
    try:
        import validate_output  # type: ignore
    except Exception as e:
        print(f"[WARN] 无法导入 validate_output.py: {e}", file=sys.stderr)
        return None
    return validate_output


# -----------------------------
# 内容合法性检查
# -----------------------------

def _has_illegal_control_chars(s: str) -> bool:
    """存在除 \\t\\n\\r 之外的其它控制字符视为非法。"""
    for ch in s:
        if ch in ('\t', '\n', '\r'):
            continue
        if ord(ch) < 32 or unicodedata.category(ch) == 'Cc':
            return True
    return False


def _is_text_legal(text: str,
                   min_chars: int = 1,
                   max_chars: Optional[int] = None,
                   deny_res: Optional[List[re.Pattern]] = None) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < int(min_chars):
        return False
    if max_chars is not None and len(t) > int(max_chars):
        return False
    if "\uFFFD" in t:
        return False
    if _has_illegal_control_chars(t):
        return False
    if deny_res:
        for rx in deny_res:
            try:
                if rx.search(t):
                    return False
            except Exception:
                # 若正则异常，忽略该条正则
                continue
    return True


# -----------------------------
# 数据结构
# -----------------------------

@dataclass(frozen=True)
class UtteranceKey:
    chunk_id: int
    index: int


@dataclass
class Utterance:
    key: UtteranceKey
    role: str
    text: str
    # 保留原始行号与记录，便于溯源/调试
    line_no: int
    raw: Dict[str, Any]


@dataclass
class PairSample:
    # 源（被回复）
    src_chunk_id: int
    src_index: int
    src_role: str
    src_text: str
    # 目标（回复）
    tgt_chunk_id: int
    tgt_index: int
    tgt_role: str
    tgt_text: str
    # 其他
    pair: Tuple[str, str]
    confidence: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "source": {
                "chunk_id": self.src_chunk_id,
                "dialogue_index": self.src_index,
                "role": self.src_role,
                "text": self.src_text,
            },
            "reply": {
                "chunk_id": self.tgt_chunk_id,
                "dialogue_index": self.tgt_index,
                "role": self.tgt_role,
                "text": self.tgt_text,
            },
            "pair": {"from": self.src_role, "to": self.tgt_role},
            "confidence": self.confidence,
        }


# -----------------------------
# 抽取主流程
# -----------------------------

def build_index(records: Iterable[Tuple[int, Dict[str, Any]]]):
    """
    建立 (chunk_id, dialogue_index) -> Utterance 的索引；并为每个 chunk 保留顺序列表。
    对异常记录**跳过**并告警；不中止全局处理。
    """
    idx: Dict[UtteranceKey, Utterance] = {}
    by_chunk: Dict[int, List[Utterance]] = defaultdict(list)

    bad = 0
    for line_no, rec in records:
        try:
            cid = int(rec["chunk_id"])
            di = int(rec["dialogue_index"])
            role = _norm_role(str(rec["role"]))
            text = str(rec.get("dialogue") or rec.get("text") or "")
            if not role:
                raise ValueError("空 role")
        except Exception as e:
            bad += 1
            print(f"[WARN] 跳过第 {line_no} 行：关键字段缺失或类型错误（{e}）；记录={rec}", file=sys.stderr)
            continue

        u = Utterance(
            key=UtteranceKey(chunk_id=cid, index=di),
            role=role,
            text=text,
            line_no=line_no,
            raw=rec,
        )
        idx[u.key] = u
        by_chunk[cid].append(u)

    # 按 index 排序每个 chunk
    for cid, L in by_chunk.items():
        L.sort(key=lambda u: u.key.index)

    if bad > 0:
        print(f"[INFO] 构建索引时跳过了 {bad} 条异常记录。", file=sys.stderr)

    return idx, by_chunk


def _resolve_reply_target(cur: Utterance, idx: Dict[UtteranceKey, Utterance]):
    """
    给定当前发言，解析其 reply 指向的**目标发言**。
    返回: (source_utterance, confidence, target_role_field) 或 None。
    """
    rec = cur.raw
    reply = rec.get("reply", None)
    if reply in (None, {}, []):
        return None

    if not isinstance(reply, dict):
        return None

    try:
        target_index = reply.get("target_index", None)
        target_chunk_id = reply.get("target_chunk_id", cur.key.chunk_id)
        if target_index is None:
            return None
        src_key = UtteranceKey(int(target_chunk_id), int(target_index))
    except Exception:
        return None

    src = idx.get(src_key, None)
    if src is None:
        return None

    conf = reply.get("confidence", None)
    if conf is not None:
        try:
            conf = float(conf)
        except Exception:
            conf = None

    t_role = reply.get("target_role", None)
    if t_role is not None:
        t_role = _norm_role(str(t_role))

    return (src, conf, t_role)


def extract_pairs(jsonl_path: Path,
                  role_pairs: List[Tuple[str, str]],
                  min_confidence: Optional[float] = 0.8,
                  require_confidence: bool = False,
                  use_target_role_fallback: bool = True,
                  drop_if_target_role_inconsistent: bool = False,
                  min_src_chars: int = 1,
                  min_reply_chars: int = 1,
                  max_src_chars: Optional[int] = None,
                  max_reply_chars: Optional[int] = None,
                  deny_patterns: Optional[List[str]] = None
                  ) -> Dict[Tuple[str, str], List[PairSample]]:
    """
    从 JSONL 抽取指定**有序角色对**的数据集。

    参数（仅列出关键项）
    ----
    min_confidence: 置信度阈值，默认 0.8；若缺失置信度将丢弃。
    require_confidence: 若 True，则必须存在 confidence（严格模式将启用）。
    use_target_role_fallback: 若 True，当 (src.role, tgt.role) 不匹配时，允许用 reply.target_role 做一次匹配尝试。
    drop_if_target_role_inconsistent: 若 True，reply.target_role 存在但与解析得到的 src.role 不一致时丢弃。
    * 文本合法性过滤项见函数签名。
    """
    # 预处理：规范化角色对
    norm_pairs = [(_norm_role(a), _norm_role(b)) for a, b in role_pairs]
    want = set(norm_pairs)

    # 将整个 JSONL 读入一次，供之后的索引并复用
    records: List[Tuple[int, Dict[str, Any]]] = list(_read_jsonl(jsonl_path))

    # 建立索引以便动态查找
    idx, _ = build_index(records)

    deny_res: List[re.Pattern] = []
    if deny_patterns:
        for p in deny_patterns:
            try:
                deny_res.append(re.compile(p))
            except Exception as e:
                print(f"[WARN] 忽略非法正则（{p}）：{e}", file=sys.stderr)

    buckets: Dict[Tuple[str, str], List[PairSample]] = defaultdict(list)

    for line_no, rec in records:
        cid = rec.get("chunk_id", None)
        di = rec.get("dialogue_index", None)
        if cid is None or di is None:
            # 这类错误在 build_index 已经告警；这里直接跳过
            continue

        cur = idx.get(UtteranceKey(int(cid), int(di)))
        if cur is None:
            continue

        resolved = _resolve_reply_target(cur, idx)
        if not resolved:
            continue
        src, conf, target_role_field = resolved

        # 自回复或同条异常，跳过
        if src.key.chunk_id == cur.key.chunk_id and src.key.index == cur.key.index:
            continue

        src_role = _norm_role(src.role)
        tgt_role = _norm_role(cur.role)

        # 若要求一致性，且 target_role 存在但与解析到的 src.role 不同，则丢弃
        if drop_if_target_role_inconsistent and target_role_field is not None:
            if _norm_role(target_role_field) != src_role:
                continue

        # 方向匹配
        pair_src_role = src_role
        pair_tgt_role = tgt_role
        pair = (pair_src_role, pair_tgt_role)
        matched = pair in want

        if not matched and use_target_role_fallback and target_role_field is not None:
            alt_src_role = _norm_role(target_role_field)
            alt_pair = (alt_src_role, tgt_role)
            if alt_pair in want:
                pair_src_role = alt_src_role
                pair = alt_pair
                matched = True

        if not matched:
            continue

        # 置信度过滤（缺失置信度也视为不通过，以避免污染）
        if require_confidence and conf is None:
            continue
        if min_confidence is not None:
            if conf is None:
                continue
            if float(conf) < float(min_confidence):
                continue

        # 文本合法性过滤
        if not _is_text_legal(src.text, min_chars=min_src_chars, max_chars=max_src_chars, deny_res=deny_res):
            continue
        if not _is_text_legal(cur.text, min_chars=min_reply_chars, max_chars=max_reply_chars, deny_res=deny_res):
            continue

        sample = PairSample(
            src_chunk_id=src.key.chunk_id,
            src_index=src.key.index,
            src_role=pair_src_role,
            src_text=src.text,
            tgt_chunk_id=cur.key.chunk_id,
            tgt_index=cur.key.index,
            tgt_role=pair_tgt_role,
            tgt_text=cur.text,
            pair=pair,
            confidence=(float(conf) if conf is not None else None),
        )
        buckets[pair].append(sample)

    return buckets


def list_roles(jsonl_path: Path) -> Counter:
    """统计角色出现频次（按发言条数计）。"""
    c = Counter()
    for _, rec in _read_jsonl(jsonl_path):
        role = _norm_role(str(rec.get("role", "")))
        if role:
            c[role] += 1
    return c


def parse_pairs_arg(pairs: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for p in pairs:
        if "," not in p:
            raise ValueError(f"--pairs 元素应形如 'A,B'，得到: {p}")
        a, b = p.split(",", 1)
        out.append((_norm_role(a), _norm_role(b)))
    return out




def expand_pairs_with_wildcards(pairs: List[str], roles: Iterable[str]) -> List[Tuple[str, str]]:
    # 支持通配：ALL/全部
    R = [_norm_role(r) for r in roles]
    out: List[Tuple[str, str]] = []
    for pair in pairs:
        if "," not in pair:
            raise ValueError(f"--pairs 元素应形如 'A,B'，得到: {pair}")
        a, b = [_norm_role(s) for s in pair.split(",", 1)]
        is_all_a = (a.upper() == "ALL") or (a == "全部")
        is_all_b = (b.upper() == "ALL") or (b == "全部")
        if is_all_a and is_all_b:
            for x in R:
                for y in R:
                    if x != y:
                        out.append((x, y))
        elif is_all_a:
            for x in R:
                if x != b:
                    out.append((x, b))
        elif is_all_b:
            for y in R:
                if a != y:
                    out.append((a, y))
        else:
            out.append((a, b))
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for pr in out:
        if pr not in seen:
            uniq.append(pr)
            seen.add(pr)
    return uniq

def all_ordered_pairs(roles: List[str]) -> List[Tuple[str, str]]:
    R = [_norm_role(r) for r in roles]
    out: List[Tuple[str, str]] = []
    for i, a in enumerate(R):
        for j, b in enumerate(R):
            if i == j:
                continue
            out.append((a, b))
    # 去重 & 保序
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def write_outputs(buckets: Dict[Tuple[str, str], List[PairSample]], out_dir: Path = None, merge_file: Path = None) -> None:
    if merge_file and out_dir:
        raise ValueError("二选一：只能指定 --merge-out 或 --out")
    if merge_file:
        merge_file.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with merge_file.open("w", encoding="utf-8") as f:
            for pair, samples in buckets.items():
                for s in samples:
                    json.dump(s.to_json(), f, ensure_ascii=False)
                    f.write("\n")
                    total += 1
        print(f"[OK] 写入合并文件: {merge_file}（共 {total} 条）")
        return

    if not out_dir:
        out_dir = Path(".") / "pair_datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for (src, tgt), samples in buckets.items():
        out_path = out_dir / _safe_pair_name(src, tgt)
        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                json.dump(s.to_json(), f, ensure_ascii=False)
                f.write("\n")
                total += 1
        print(f"[OK] {src} -> {tgt}: {len(samples)} 条 -> {out_path}")
    print(f"[DONE] 共写入 {len(buckets)} 个有序对，合计 {total} 条样本；输出目录: {out_dir}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="从已验证 JSONL 中，抽取任意两个角色(有向)的对话数据集；支持严格模式与非法文本过滤")
    ap.add_argument("-i", "--input", required=True, help="validate_output.py 通过校验的 JSONL 文件路径")
    ap.add_argument("-o", "--out", default=None, help="输出目录（为每个 pair 单独一个 JSONL）；默认 ./pair_datasets")
    ap.add_argument("--merge-out", default=None, help="把所有 pair 合并到一个 JSONL 文件（与 --out 互斥）")

    grp = ap.add_argument_group("角色选择")
    grp.add_argument("--pairs", nargs="*", default=[], help="显式指定有序对，如：--pairs 'A,B' 'B,C' 'B,D'")
    grp.add_argument("--roles", nargs="+", action="append", default=None, help="给定一批角色名，与 --all-ordered-pairs 配合使用")
    grp.add_argument("--all-ordered-pairs", action="store_true", help="对 --roles 的集合生成所有有序对")

    flt = ap.add_argument_group("过滤/阈值（默认严谨以避免污染）")
    flt.add_argument("--min-confidence", type=float, default=0.8, help="仅保留置信度 >= 阈值（默认 0.8；若置信度缺失将丢弃）")
    flt.add_argument("--require-confidence", action="store_true", help="必须存在 confidence 才保留；--strict 自动启用")
    flt.add_argument("--strict", action="store_true", default=True, help="严格模式：仅保留能**确定**为所选 A→B 的样本（详见脚本头注，默认启用）")
    flt.add_argument("--no-strict", dest="strict", action="store_false", help="关闭严格模式，允许使用回退逻辑并放宽置信度要求。")

    textflt = ap.add_argument_group("文本合法性（源与回复各自生效）")
    textflt.add_argument("--min-src-chars", type=int, default=1, help="源文本最小字符数（默认 1）")
    textflt.add_argument("--min-reply-chars", type=int, default=1, help="回复文本最小字符数（默认 1）")
    textflt.add_argument("--max-src-chars", type=int, default=None, help="源文本最大字符数（默认不限制）")
    textflt.add_argument("--max-reply-chars", type=int, default=None, help="回复文本最大字符数（默认不限制）")
    textflt.add_argument("--deny-pattern", action="append", default=None, help="正则黑名单（可多次提供）；命中则丢弃（对源与回复都生效）")

    misc = ap.add_argument_group("其他")
    misc.add_argument("--list-roles", action="store_true", help="仅统计并输出角色列表与频次，不做抽取")
    misc.add_argument("--validate-path", default=None, help="可选：validate_output.py 的路径；提供则先做校验")

    args = ap.parse_args(argv)

    # 兼容 action='append'：将 roles 扁平化
    if isinstance(args.roles, list) and args.roles and isinstance(args.roles[0], list):
        _flat = []
        [ _flat.extend(g) for g in args.roles ]
        args.roles = _flat
    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        print(f"[ERR] 输入文件不存在: {jsonl_path}", file=sys.stderr)
        return 2

    # 可选：校验
    if args.validate_path:
        mod = _try_import_validator(Path(args.validate_path))
        if mod and hasattr(mod, "validate"):
            print("[INFO] 正在调用 validate_output.validate() ...", file=sys.stderr)
            try:
                rc = int(mod.validate(str(jsonl_path)))  # type: ignore
            except Exception as e:
                print(f"[WARN] 调用 validate() 失败，跳过显式校验：{e}", file=sys.stderr)
                rc = 0
            if rc != 0:
                print("[ERR] 校验未通过，终止。", file=sys.stderr)
                return 3
        else:
            print("[WARN] 未能导入 validate_output.validate()，跳过显式校验。", file=sys.stderr)

    # 仅列出角色
    if args.list_roles:
        counts = list_roles(jsonl_path)
        total = sum(counts.values())
        print("角色频次：")
        for role, cnt in counts.most_common():
            print(f"  {role}: {cnt}")
        print(f"总发言条数: {total}")
        return 0

    # 决定有序对
    role_pairs: List[Tuple[str, str]] = []
    if args.pairs:
        has_wildcard = any(("," in p) and (("ALL" in p.upper()) or ("全部" in p)) for p in args.pairs)
        if has_wildcard:
            roles_for_expand: List[str] = args.roles or list(list_roles(jsonl_path).keys())
            role_pairs.extend(expand_pairs_with_wildcards(args.pairs, roles_for_expand))
        else:
            role_pairs.extend(parse_pairs_arg(args.pairs))

    if args.all_ordered_pairs:
        roles_for_all = args.roles or list(list_roles(jsonl_path).keys())
        if not roles_for_all:
            print("[ERR] 未在输入中发现任何角色，无法生成有序对。", file=sys.stderr)
            return 2
        role_pairs.extend(all_ordered_pairs(roles_for_all))

    if not role_pairs:
        print("[ERR] 请使用 --pairs 或 --roles + --all-ordered-pairs 指定有序对。", file=sys.stderr)
        return 2

# 去重 & 保序
    seen = set()
    uniq_pairs: List[Tuple[str, str]] = []
    for p in role_pairs:
        if p not in seen:
            uniq_pairs.append(p)
            seen.add(p)
    role_pairs = uniq_pairs

    # 严格模式设置
    require_confidence = bool(args.require_confidence or args.strict)
    min_confidence = args.min_confidence  # 默认 0.8

    # 严格模式：禁用 target_role 回退；开启不一致即丢弃
    use_target_role_fallback = (not args.strict)
    drop_if_target_role_inconsistent = bool(args.strict)

    # 抽取
    buckets = extract_pairs(
        jsonl_path=jsonl_path,
        role_pairs=role_pairs,
        min_confidence=min_confidence,
        require_confidence=require_confidence,
        use_target_role_fallback=use_target_role_fallback,
        drop_if_target_role_inconsistent=drop_if_target_role_inconsistent,
        min_src_chars=args.min_src_chars,
        min_reply_chars=args.min_reply_chars,
        max_src_chars=args.max_src_chars,
        max_reply_chars=args.max_reply_chars,
        deny_patterns=args.deny_pattern,
    )

    # 写文件
    if args.merge_out:
        write_outputs(buckets, out_dir=None, merge_file=Path(args.merge_out))
    else:
        out_dir = Path(args.out) if args.out else None
        write_outputs(buckets, out_dir=out_dir, merge_file=None)

    # 简要统计
    print("\n统计：")
    for pair, L in buckets.items():
        print(f"  {pair[0]} -> {pair[1]}: {len(L)} 条")
    print(f"  合计样本: {sum(len(v) for v in buckets.values())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
