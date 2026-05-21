"""Hata metrikleri: RMSE / MAE ve toplu analiz.

Ödevin 6.5 maddesi gereği gerçek yol ile tahmini yol arasındaki hata,
konum ve yönelim ekseninde ayrı ayrı raporlanır. Bu modül; ham yörünge
matrislerinden ($N\\times 3$ — $x, y, \\theta$) RMSE ve MAE değerlerini
hesaplar ve bunları tek bir özet sözlüğünde toplar. Yönelim hatasında
açı sarmalaması (wrap-around) ele alınmıştır.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from .utils import normalize_angle_array


# -----------------------------------------------------------------------------
# Temel istatistikler
# -----------------------------------------------------------------------------

def rmse(errors: np.ndarray) -> float:
    """Hata vektöründen RMSE."""
    if errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(errors: np.ndarray) -> float:
    """Hata vektöründen MAE."""
    if errors.size == 0:
        return float("nan")
    return float(np.mean(np.abs(errors)))


def position_errors(true_xy: np.ndarray, est_xy: np.ndarray) -> np.ndarray:
    """Her zaman adımı için 2B konum hatasının büyüklüğü (m)."""
    diff = true_xy - est_xy
    return np.linalg.norm(diff, axis=1)


def heading_errors(true_theta: np.ndarray, est_theta: np.ndarray) -> np.ndarray:
    """Sarmalama düzeltilmiş yönelim hataları (rad)."""
    return normalize_angle_array(true_theta - est_theta)


# -----------------------------------------------------------------------------
# Toplu analiz
# -----------------------------------------------------------------------------

@dataclass
class LocalizationMetrics:
    """Tek bir tahminci için (DR veya EKF) hata özeti."""

    name: str
    position_rmse: float
    position_mae: float
    heading_rmse: float
    heading_mae: float
    final_position_error: float
    final_heading_error: float

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.pop("name")
        return d


class MetricsAnalyzer:
    """Yörünge dizilerinden lokalizasyon metriklerini üretir."""

    @staticmethod
    def evaluate(
        true_traj: np.ndarray,
        est_traj: np.ndarray,
        name: str,
    ) -> LocalizationMetrics:
        """``true_traj`` ve ``est_traj`` aynı uzunlukta (N,3) dizilerdir."""
        n = min(true_traj.shape[0], est_traj.shape[0])
        true = true_traj[:n]
        est = est_traj[:n]

        pos_err = position_errors(true[:, :2], est[:, :2])
        head_err = heading_errors(true[:, 2], est[:, 2])

        return LocalizationMetrics(
            name=name,
            position_rmse=rmse(pos_err),
            position_mae=mae(pos_err),
            heading_rmse=rmse(head_err),
            heading_mae=mae(head_err),
            final_position_error=float(pos_err[-1]) if pos_err.size else float("nan"),
            final_heading_error=float(abs(head_err[-1])) if head_err.size else float("nan"),
        )


__all__ = [
    "rmse",
    "mae",
    "position_errors",
    "heading_errors",
    "LocalizationMetrics",
    "MetricsAnalyzer",
]
