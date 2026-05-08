"""Public integration surface for Text2Dialog.

The package keeps imports lazy so embedding applications can inspect package
metadata without initializing LLM clients or logging handlers.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "ModelPlatform",
    "DialogueChain",
    "CancelledError",
    "DialogueItem",
    "ChunkDialogueItem",
    "ExtractOptions",
    "ExtractionResult",
    "PairBuildOptions",
    "PairBuildResult",
    "ChatMLOptions",
    "ChatMLResult",
    "PipelineResult",
    "run_extraction",
    "validate_extraction",
    "build_pair_dataset",
    "convert_pairs_to_chatml",
    "run_dataset_pipeline",
]


def __getattr__(name: str):
    if name in {"Config", "ModelPlatform"}:
        from .config import Config, ModelPlatform

        return {"Config": Config, "ModelPlatform": ModelPlatform}[name]

    if name in {"DialogueChain", "CancelledError", "DialogueItem", "ChunkDialogueItem"}:
        from .dialogue_chain import CancelledError, ChunkDialogueItem, DialogueChain, DialogueItem

        return {
            "DialogueChain": DialogueChain,
            "CancelledError": CancelledError,
            "DialogueItem": DialogueItem,
            "ChunkDialogueItem": ChunkDialogueItem,
        }[name]

    if name in {
        "ExtractOptions",
        "ExtractionResult",
        "PairBuildOptions",
        "PairBuildResult",
        "ChatMLOptions",
        "ChatMLResult",
        "PipelineResult",
        "run_extraction",
        "validate_extraction",
        "build_pair_dataset",
        "convert_pairs_to_chatml",
        "run_dataset_pipeline",
    }:
        from . import pipeline

        return getattr(pipeline, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
