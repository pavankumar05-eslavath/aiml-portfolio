"""Shared setup for the command-line scripts.

Puts ``backend/`` on ``sys.path`` so the scripts import the very same application
code the API uses. The ingestion, evaluation and experiment scripts are thin CLI
wrappers over the service layer, not parallel implementations - if a script
disagreed with the API about how a product is embedded, evaluation numbers would
not describe the running system.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
DATA_ROOT = REPO_ROOT / "data"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def configure_script_logging(level: str = "INFO") -> None:
    """Configure logging for CLI use."""
    from app.core.logging import configure_logging

    configure_logging(level, "console")


def banner(title: str, width: int = 78) -> None:
    """Print a section header."""
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def kv(label: str, value: object, indent: int = 2) -> None:
    """Print an aligned label/value line."""
    print(f"{' ' * indent}{label:.<34} {value}")


def display_path(path: Path) -> str:
    """Render a path relative to the repository root when possible.

    Paths supplied on the command line may be relative or absolute, and may sit
    outside the repository; this never raises for any of those cases.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
