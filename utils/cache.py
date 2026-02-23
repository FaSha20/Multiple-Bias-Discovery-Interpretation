# utils/cache.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class JsonCache:
    cache_path: Path
    _data: Dict[str, Any]

    @classmethod
    def load(cls, cache_path: Path) -> "JsonCache":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        return cls(cache_path=cache_path, _data=data)

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def save(self) -> None:
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.cache_path)


class CacheKeys:
    """
    Standardized cache keys (hashed) to avoid collisions and keep cache compact.
    """

    @staticmethod
    def propose_features(dim_id: int, grid_a_path: str, grid_b_paths: list[str], prompt_version: str) -> str:
        return stable_hash(
            {
                "type": "propose_features",
                "dim_id": dim_id,
                "grid_a": grid_a_path,
                "grid_b": grid_b_paths,
                "prompt_version": prompt_version,
            }
        )

    @staticmethod
    def batch_severity(feature: str, grid_path: str, tile_ids: list[str], prompt_version: str) -> str:
        return stable_hash(
            {
                "type": "batch_severity",
                "feature": feature,
                "grid": grid_path,
                "tile_ids": tile_ids,
                "prompt_version": prompt_version,
            }
        )

    @staticmethod
    def single_severity(feature: str, image_id: str, prompt_version: str) -> str:
        return stable_hash(
            {
                "type": "single_severity",
                "feature": feature,
                "image_id": image_id,
                "prompt_version": prompt_version,
            }
        )