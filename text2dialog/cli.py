"""Stable console entry points for Text2Dialog."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


def extract_main() -> int:
    """Run the dialogue extraction command."""
    from .dialogue_chain import main

    return main()


def server_main(argv: Sequence[str] | None = None) -> int:
    """Run the local web console with conservative network defaults."""
    parser = argparse.ArgumentParser(description="Run the Text2Dialog web console.")
    parser.add_argument("--host", default=os.getenv("TEXT2DIALOG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TEXT2DIALOG_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Reload when source files change (development only).")
    parser.add_argument("--log-level", default=os.getenv("TEXT2DIALOG_LOG_LEVEL", "info"))
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "text2dialog.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def validate_main(argv: Sequence[str] | None = None) -> int:
    """Validate an extraction JSONL file."""
    parser = argparse.ArgumentParser(description="Validate Text2Dialog extraction JSONL.")
    parser.add_argument("input", help="Extraction JSONL path")
    args = parser.parse_args(argv)

    from .validate_output import validate

    return validate(args.input)


def pairs_main() -> int:
    """Build directed role-pair datasets."""
    from .pair_dataset_builder import main

    return main()


def chatml_main() -> int:
    """Convert pair datasets to ChatML."""
    from .pair_to_chatml import main

    return main()
