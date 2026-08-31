"""Configuration loading and path resolution.

Paths in configs/config.yaml are relative to the project root, which is derived
from this file's location, so nothing in the codebase hardcodes an absolute path
and the project runs from any checkout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Parsed configuration with helpers for path resolution."""

    raw: dict[str, Any] = field(repr=False)
    path: Path = DEFAULT_CONFIG

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- frequently used values -------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def target(self) -> str:
        return str(self.raw["data"]["target"])

    @property
    def series_key(self) -> str:
        return str(self.raw["data"]["series_key"])

    @property
    def horizons(self) -> list[int]:
        return [int(h) for h in self.raw["horizons"]]

    @property
    def lags(self) -> list[int]:
        return [int(x) for x in self.raw["features"]["lags"]]

    @property
    def rolling_windows(self) -> list[int]:
        return [int(x) for x in self.raw["features"]["rolling_windows"]]

    def resolve(self, *keys: str) -> Path:
        """Resolve a nested config value into an absolute path."""
        node: Any = self.raw
        for k in keys:
            node = node[k]
        return (PROJECT_ROOT / str(node)).resolve()

    def raw_file(self, name: str) -> Path:
        """Absolute path of one downloaded source file.

        Named by the CONFIG KEY, not by the remote basename: the remote paths are
        ``train/target.parquet`` and ``test/target.parquet``, which both reduce to
        ``target.parquet`` and would silently overwrite each other locally.
        """
        if name not in self.raw["data"]["files"]:
            raise KeyError(f"unknown source file {name!r}")
        suffix = Path(str(self.raw["data"]["files"][name])).suffix
        return (self.resolve("data", "raw_dir") / f"{name}{suffix}").resolve()

    def ensure_dirs(self) -> None:
        for keys in (("paths", "models_dir"), ("paths", "reports_dir"),
                     ("paths", "figures_dir"), ("data", "processed_dir"),
                     ("data", "raw_dir")):
            self.resolve(*keys).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    with p.open(encoding="utf-8") as fh:
        return Config(raw=yaml.safe_load(fh), path=p)
