"""Yardımcı bileşenler: konfigürasyon yükleme, seed yönetimi, geometri.

Bu modül paketin altyapı katmanını oluşturur. ConfigLoader, YAML dosyasından
gelen parametreleri noktalı erişimle (config.environment.width gibi) sunan
hafif bir okuyucu sağlar. Geometri yardımcıları LiDAR ışın atışları ve
çarpışma denetimi gibi farklı modüllerce ortak şekilde kullanılır.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# -----------------------------------------------------------------------------
# Konfigürasyon yükleme
# -----------------------------------------------------------------------------

class _AttrDict(dict):
    """Dictionary'yi attribute erişimine açan hafif sarmalayıcı.

    Bu sınıf, ``config["environment"]["width"]`` yerine
    ``config.environment.width`` yazımına izin verir. İç içe sözlükler de
    özyinelemeli olarak sarmalanır.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        if data:
            for key, value in data.items():
                self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def to_dict(self) -> Dict[str, Any]:
        def _unwrap(v: Any) -> Any:
            if isinstance(v, _AttrDict):
                return {k: _unwrap(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_unwrap(x) for x in v]
            return v
        return _unwrap(self)


class ConfigLoader:
    """YAML konfigürasyon dosyalarını yükler ve attribute erişimi sağlar.

    Bir konfigürasyon dosyası ``extends: <path>`` anahtarı içeriyorsa,
    önce belirtilen taban dosya yüklenir; ardından mevcut dosya bu
    tabanın üzerine derinlemesine (recursive) merge edilir. Bu sayede
    ``experiments/`` altındaki türev senaryolar yalnızca farkları içerir
    ve baz konfigürasyon tek bir yerden sürdürülür.
    """

    @classmethod
    def load(cls, path: str | os.PathLike) -> _AttrDict:
        raw = cls._load_with_extends(Path(path))
        return _AttrDict(raw)

    @classmethod
    def _load_with_extends(cls, path: Path) -> Dict[str, Any]:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        extends = data.pop("extends", None)
        if extends:
            base_ref = Path(extends)
            base_path = base_ref if base_ref.is_absolute() else (path.parent / base_ref)
            base = cls._load_with_extends(base_path)
            cls._deep_merge(base, data)
            return base
        if not data:
            raise ValueError(f"Boş konfigürasyon dosyası: {path}")
        return data

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                ConfigLoader._deep_merge(base[key], value)
            else:
                base[key] = value


# -----------------------------------------------------------------------------
# Tekrarlanabilirlik
# -----------------------------------------------------------------------------

@dataclass
class RandomContext:
    """Tüm modüllerin paylaşacağı rastgelelik bağlamı.

    Tek bir master seed üzerinden, alt bileşenler (engel yerleşimi, lidar,
    imu, encoder) için bağımsız ``np.random.Generator`` örnekleri türetir.
    Bu sayede bir modülde değişiklik diğer modüllerin sayı akışını bozmaz.
    """

    seed: int
    _master: np.random.SeedSequence | None = None

    def __post_init__(self) -> None:
        self._master = np.random.SeedSequence(self.seed)
        # Python'un random modülü ve numpy global state de deterministik olsun
        random.seed(self.seed)
        np.random.seed(self.seed)

    def child(self, label: str) -> np.random.Generator:
        """Etiketten türetilen kararlı bir alt RNG döndürür.

        Python'un yerleşik ``hash()`` fonksiyonu PYTHONHASHSEED rastgelelemesi
        nedeniyle process'ler arasında tutarlı değildir; bu nedenle etiket
        SHA-256 ile sabit bir 32-bit tohuma indirgenir. Böylece aynı
        master seed + aynı etiket her zaman aynı RNG akışını üretir.
        """
        assert self._master is not None
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        spawn_key = (int.from_bytes(digest[:4], "big"),)
        child_seq = np.random.SeedSequence(
            entropy=self._master.entropy,
            spawn_key=spawn_key,
        )
        return np.random.default_rng(child_seq)


# -----------------------------------------------------------------------------
# Geometri yardımcıları
# -----------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Açıyı (-pi, pi] aralığına indirger."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if a <= -math.pi:
        a += 2.0 * math.pi
    return a


def normalize_angle_array(angles: np.ndarray) -> np.ndarray:
    """Vektörel açı normalizasyonu."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "ConfigLoader",
    "RandomContext",
    "normalize_angle",
    "normalize_angle_array",
    "euclidean",
    "clamp",
    "ensure_dir",
]
