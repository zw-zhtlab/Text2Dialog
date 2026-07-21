#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Programmatic pipeline helpers for embedding Text2Dialog.

This module wraps the existing CLI-oriented modules with small dataclass-based
APIs. It is intended for local, in-process integration. Runtime overrides touch
process-wide environment variables and ``Config`` class attributes while the
operation is running, so callers should avoid running multiple pipelines with
different LLM credentials in the same Python process at the same time.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config import Config, ModelPlatform
from . import pair_dataset_builder as pairs_mod
from . import pair_to_chatml as chatml_mod
from . import validate_output


PathLike = Union[os.PathLike, str]


@dataclass
class ExtractOptions:
    platform: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    concurrent: bool = True
    threads: Optional[int] = None
    save_chunk_text: Optional[bool] = None
    sort_output: bool = False
    cache_dir: Optional[PathLike] = None
    max_token_len: Optional[int] = None
    cover_content: Optional[int] = None
    temperature: Optional[float] = None
    reply_window: Optional[int] = None
    reply_confidence_th: Optional[float] = None
    write_quality_report: bool = True
    collect_stats: bool = True
    schema: Optional[Dict[str, Any]] = None


@dataclass
class ExtractionResult:
    extraction_path: Path
    quality_report_path: Optional[Path] = None
    stats: Optional[Dict[str, Any]] = None


@dataclass
class PairBuildOptions:
    pairs: Optional[Sequence[Any]] = None
    roles: Optional[Sequence[str]] = None
    all_ordered_pairs: bool = False
    out_dir: Optional[PathLike] = None
    merge_out: Optional[PathLike] = None
    min_confidence: Optional[float] = 0.8
    require_confidence: bool = False
    strict: bool = True
    min_src_chars: int = 1
    min_reply_chars: int = 1
    max_src_chars: Optional[int] = None
    max_reply_chars: Optional[int] = None
    deny_patterns: Optional[Sequence[str]] = None
    diagnostics_out: Optional[PathLike] = None


@dataclass
class PairBuildResult:
    out_dir: Optional[Path] = None
    merge_out: Optional[Path] = None
    diagnostics_path: Optional[Path] = None
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    total_samples: int = 0


@dataclass
class ChatMLOptions:
    mode: str = "pair"
    max_turns: int = 4
    min_confidence: Optional[float] = None
    reverse: bool = False
    include_meta: bool = False
    dedupe: bool = False
    system_text: Optional[str] = None
    system_template: Optional[str] = None


@dataclass
class ChatMLResult:
    output_path: Path
    count: int


@dataclass
class PipelineResult:
    extraction: ExtractionResult
    validation_report: Dict[str, Any]
    pairs: PairBuildResult
    chatml: ChatMLResult


def _as_path(value: PathLike, *, create_parent: bool = False) -> Path:
    path = Path(value).expanduser()
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_distinct_paths(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    for output in outputs:
        for source in inputs:
            if chatml_mod.paths_collide(source, output):
                raise ValueError(f"output path must not overwrite an input: {output}")
    for index, output in enumerate(outputs):
        for other in outputs[index + 1:]:
            if chatml_mod.paths_collide(output, other):
                raise ValueError(f"output paths must be distinct: {output}")


def _coerce_pair(item: Any) -> Tuple[str, str]:
    if isinstance(item, str):
        if "," not in item:
            raise ValueError(f"pair string must look like 'A,B', got: {item!r}")
        left, right = item.split(",", 1)
        return left.strip(), right.strip()
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return str(item[0]).strip(), str(item[1]).strip()
    raise ValueError(f"unsupported pair value: {item!r}")


def _resolve_role_pairs(input_path: Path, options: PairBuildOptions) -> List[Tuple[str, str]]:
    role_pairs: List[Tuple[str, str]] = []
    requested_pairs = list(options.pairs or [])
    roles = [str(r).strip() for r in (options.roles or []) if str(r).strip()]

    if requested_pairs:
        pair_strings: List[str] = []
        direct_pairs: List[Tuple[str, str]] = []
        for item in requested_pairs:
            if isinstance(item, str):
                pair_strings.append(item)
            else:
                direct_pairs.append(_coerce_pair(item))
        if pair_strings:
            roles_for_expand = roles or list(pairs_mod.list_roles(input_path).keys())
            role_pairs.extend(pairs_mod.expand_pairs_with_wildcards(pair_strings, roles_for_expand))
        role_pairs.extend(direct_pairs)

    if options.all_ordered_pairs:
        roles_for_all = roles or list(pairs_mod.list_roles(input_path).keys())
        role_pairs.extend(pairs_mod.all_ordered_pairs(list(roles_for_all)))

    seen = set()
    deduped: List[Tuple[str, str]] = []
    for pair in role_pairs:
        if pair not in seen:
            deduped.append(pair)
            seen.add(pair)
    if not deduped:
        raise ValueError("no role pairs selected; set pairs or all_ordered_pairs")
    return deduped


@contextmanager
def _runtime_overrides(options: ExtractOptions):
    platform = options.platform
    old_platform = Config.CURRENT_PLATFORM
    old_env: Dict[str, Optional[str]] = {}
    old_attrs: Dict[str, Any] = {}

    def set_env(name: str, value: Optional[str]) -> None:
        if not name or value is None:
            return
        if name not in old_env:
            old_env[name] = os.environ.get(name)
        os.environ[name] = value

    def set_attr(name: str, value: Any) -> None:
        if value is None:
            return
        if name not in old_attrs:
            old_attrs[name] = getattr(Config, name)
        setattr(Config, name, value)

    try:
        if platform:
            Config.set_platform(platform)
        effective_platform = Config.CURRENT_PLATFORM
        platform_config = ModelPlatform.get_platform_config(effective_platform)
        set_env("LLM_PLATFORM", effective_platform)
        set_env(platform_config["api_key_env"], options.api_key)
        set_env(platform_config["base_url_env"], options.base_url)
        set_env(f"{effective_platform.upper()}_MODEL_NAME", options.model_name)

        set_attr("MAX_TOKEN_LEN", options.max_token_len)
        set_attr("COVER_CONTENT", options.cover_content)
        set_attr("TEMPERATURE", options.temperature)
        set_attr("REPLY_WINDOW", options.reply_window)
        set_attr("REPLY_CONFIDENCE_TH", options.reply_confidence_th)
        if options.save_chunk_text is not None:
            set_attr("SAVE_CHUNK_TEXT", bool(options.save_chunk_text))
        if options.cache_dir is not None:
            set_attr("CACHE_DIR", str(Path(options.cache_dir)))

        yield
    finally:
        Config.CURRENT_PLATFORM = old_platform
        for name, value in old_attrs.items():
            setattr(Config, name, value)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_extraction(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    options: Optional[ExtractOptions] = None,
) -> ExtractionResult:
    """Run DialogueChain extraction and return registered local artifacts."""
    opts = options or ExtractOptions()
    source = _as_path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    output = _as_path(output_path, create_parent=True) if output_path else None

    with _runtime_overrides(opts):
        from .dialogue_chain import DialogueChain

        extractor = DialogueChain(
            schema=opts.schema,
            platform=opts.platform,
            max_workers=opts.threads,
            save_chunk_text=opts.save_chunk_text,
            cache_dir=str(opts.cache_dir) if opts.cache_dir is not None else None,
        )
        if opts.concurrent:
            produced = extractor.extract_dialogues_concurrent(str(source), str(output) if output else None)
        else:
            produced = extractor.extract_dialogues(str(source), str(output) if output else None)
        if opts.sort_output:
            produced = extractor.sort_dialogues(produced)

        quality_report = None
        if opts.write_quality_report:
            quality_report = Path(extractor.write_quality_report(produced))
        stats = extractor.get_statistics(produced) if opts.collect_stats else None
        return ExtractionResult(extraction_path=Path(produced), quality_report_path=quality_report, stats=stats)


def validate_extraction(path: PathLike, *, max_messages: int = 200) -> Dict[str, Any]:
    """Validate an extraction JSONL and return the structured report."""
    return validate_output.validate_with_report(str(_as_path(path)), max_messages=max_messages)


def build_pair_dataset(input_path: PathLike, options: Optional[PairBuildOptions] = None) -> PairBuildResult:
    """Build directed role-pair samples from a validated extraction JSONL."""
    opts = options or PairBuildOptions()
    source = _as_path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    role_pairs = _resolve_role_pairs(source, opts)
    diagnostics: Optional[List[Dict[str, Any]]] = [] if opts.diagnostics_out else None

    out_dir = Path(opts.out_dir) if opts.out_dir else None
    merge_out = Path(opts.merge_out) if opts.merge_out else None
    prospective_outputs: List[Path] = []
    if merge_out:
        prospective_outputs.append(merge_out)
    else:
        target_dir = out_dir or Path("pair_datasets")
        prospective_outputs.extend(
            target_dir / pairs_mod._safe_pair_name(src, tgt) for src, tgt in role_pairs
        )
    if opts.diagnostics_out:
        prospective_outputs.append(Path(opts.diagnostics_out))
    _ensure_distinct_paths([source], prospective_outputs)

    buckets = pairs_mod.extract_pairs(
        jsonl_path=source,
        role_pairs=role_pairs,
        min_confidence=opts.min_confidence,
        require_confidence=bool(opts.require_confidence or opts.strict),
        use_target_role_fallback=not opts.strict,
        drop_if_target_role_inconsistent=bool(opts.strict),
        min_src_chars=opts.min_src_chars,
        min_reply_chars=opts.min_reply_chars,
        max_src_chars=opts.max_src_chars,
        max_reply_chars=opts.max_reply_chars,
        deny_patterns=list(opts.deny_patterns or []),
        diagnostics=diagnostics,
    )

    pairs_mod.write_outputs(buckets, out_dir=out_dir, merge_file=merge_out)

    diagnostics_path = None
    if opts.diagnostics_out and diagnostics is not None:
        diagnostics_path = _as_path(opts.diagnostics_out, create_parent=True)
        with diagnostics_path.open("w", encoding="utf-8") as f:
            for event in diagnostics:
                json.dump(event, f, ensure_ascii=False)
                f.write("\n")

    counts = {pair: len(samples) for pair, samples in buckets.items()}
    return PairBuildResult(
        out_dir=out_dir if out_dir else (None if merge_out else Path("pair_datasets")),
        merge_out=merge_out,
        diagnostics_path=diagnostics_path,
        counts=counts,
        total_samples=sum(counts.values()),
    )


def convert_pairs_to_chatml(
    inputs: Sequence[PathLike],
    output_path: PathLike,
    options: Optional[ChatMLOptions] = None,
) -> ChatMLResult:
    """Convert pair JSONL files/directories/globs into ChatML JSONL."""
    opts = options or ChatMLOptions()
    input_strings = [str(Path(item)) for item in inputs]
    discovered = chatml_mod.discover_inputs(input_strings)
    if not discovered:
        raise FileNotFoundError(f"no pair JSONL inputs found: {input_strings}")

    system_text = chatml_mod.load_text_maybe_from_file(opts.system_text)
    system_template = chatml_mod.load_text_maybe_from_file(opts.system_template)
    output = _as_path(output_path)
    _ensure_distinct_paths(discovered, [output])
    records = chatml_mod.convert_pair_to_chatml(
        inputs=discovered,
        mode=opts.mode,
        min_confidence=opts.min_confidence,
        reverse_roles=opts.reverse,
        system_text=system_text,
        system_template=system_template,
        max_turns=opts.max_turns,
        include_meta=opts.include_meta,
        dedupe=opts.dedupe,
    )

    count = chatml_mod.write_jsonl_atomic(output, records)
    return ChatMLResult(output_path=output, count=count)


def run_dataset_pipeline(
    input_path: PathLike,
    work_dir: PathLike,
    *,
    extract_options: Optional[ExtractOptions] = None,
    pair_options: Optional[PairBuildOptions] = None,
    chatml_options: Optional[ChatMLOptions] = None,
) -> PipelineResult:
    """Run extract -> validate -> pair -> ChatML as an embeddable local workflow."""
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)

    extraction = run_extraction(
        input_path,
        root / "extraction.jsonl",
        options=extract_options or ExtractOptions(cache_dir=root / ".cache"),
    )
    validation_report = validate_extraction(extraction.extraction_path)
    if not validation_report.get("ok"):
        raise ValueError(f"validation failed: {validation_report.get('messages', [])}")

    pair_opts = pair_options or PairBuildOptions(all_ordered_pairs=True)
    if pair_opts.out_dir is None and pair_opts.merge_out is None:
        pair_opts = PairBuildOptions(**{**pair_opts.__dict__, "out_dir": root / "pair_datasets"})
    pairs = build_pair_dataset(extraction.extraction_path, pair_opts)

    pair_inputs: List[PathLike]
    if pairs.merge_out:
        pair_inputs = [pairs.merge_out]
    elif pairs.out_dir:
        pair_inputs = [pairs.out_dir]
    else:
        pair_inputs = [root / "pair_datasets"]
    chatml = convert_pairs_to_chatml(pair_inputs, root / "chatml.jsonl", chatml_options)
    return PipelineResult(extraction=extraction, validation_report=validation_report, pairs=pairs, chatml=chatml)
