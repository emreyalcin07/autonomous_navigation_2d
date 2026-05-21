"""LiDAR taraması üzerinde mesafe eşikleme, filtreleme ve kümeleme.

Sensör katmanından gelen ham ``LidarScan`` bu modülde işlenip navigasyon
için kullanılabilir bir formda dışarı verilir. Akış üç aşamadan oluşur:

1. **Mesafe eşikleme** — max_range yakınındaki ölçümler "isabet yok"
   sayılır (ham veri akışında gürültü zirvelerini elemek için emniyet
   payı bırakılır).
2. **Medyan filtre** — açı boyunca tek boyutlu küçük pencereli medyan,
   nokta gürültüsünü bastırır ve uçuş süresi tarzı LiDAR çıktıları için
   pratik bir önişlemdir.
3. **Mesafe tabanlı kümeleme** — açıya göre sıralı kartezyen noktalar
   arasında ardışık Öklid mesafesi belirli bir eşiği aşıyorsa yeni küme
   başlatılır. 360° tarama için dizinin başı/sonu da bitişik kabul edilir.
   Sonuç ``Cluster`` listesi, navigasyonun engel merkez ve sınırlarını
   tahmin etmesine olanak verir.

Bu sade yaklaşım, DBSCAN benzeri davranışı dış bağımlılık olmadan üretir
ve gerçek zamanlı çalışacak hafiflikte tutulur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .sensors import LidarScan


@dataclass
class Cluster:
    """Tek bir engel kümesi (global karteziyen)."""

    points: np.ndarray              # (k, 2)
    indices: np.ndarray             # ışın indeksleri (k,)

    @property
    def size(self) -> int:
        return int(self.points.shape[0])

    @property
    def centroid(self) -> Tuple[float, float]:
        if self.size == 0:
            return float("nan"), float("nan")
        return float(self.points[:, 0].mean()), float(self.points[:, 1].mean())


@dataclass
class LidarProcessingResult:
    """LiDAR işleme çıktısı.

    raw_points : (N,2) — eşiğin altına düşen tüm geçerli noktalar (filtrelenmemiş)
    filtered_points : (M,2) — medyan filtre sonrası noktalar
    clusters : küme listesi (her biri global karteziyen)
    """

    raw_points: np.ndarray
    filtered_points: np.ndarray
    clusters: List[Cluster] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def cluster_centroids(self) -> np.ndarray:
        if not self.clusters:
            return np.zeros((0, 2))
        return np.array([c.centroid for c in self.clusters], dtype=float)


class LidarProcessor:
    """Tarama bazlı LiDAR işleme: eşikleme + medyan filtre + kümeleme."""

    def __init__(
        self,
        median_window: int = 3,
        cluster_eps: float = 0.3,
        min_cluster_size: int = 3,
        range_threshold_margin: float = 0.05,
    ) -> None:
        if median_window < 1 or median_window % 2 == 0:
            raise ValueError("median_window pozitif tek tam sayı olmalı.")
        self.median_window = int(median_window)
        self.cluster_eps = float(cluster_eps)
        self.min_cluster_size = int(min_cluster_size)
        self.range_threshold_margin = float(range_threshold_margin)

    @classmethod
    def from_config(cls, cfg) -> "LidarProcessor":
        c = cfg.lidar_processing
        return cls(
            median_window=int(c.median_window),
            cluster_eps=float(c.cluster_eps),
            min_cluster_size=int(c.min_cluster_size),
            range_threshold_margin=float(c.range_threshold_margin),
        )

    # ------------------------------------------------------------------ pipeline

    def process(
        self,
        scan: LidarScan,
        pose: Tuple[float, float, float],
    ) -> LidarProcessingResult:
        """Ham taramayı işleyip kümeleri global karteziyen olarak döner."""
        threshold = scan.max_range - self.range_threshold_margin
        valid_mask = scan.ranges < threshold

        # 1) Ham geçerli noktaları global karteziyene çevir
        raw_pts = self._scan_to_cartesian(scan, pose, valid_mask)

        # 2) Medyan filtre — açı boyunca menzil üzerinde uygulanır
        filtered_ranges = self._median_filter(scan.ranges, self.median_window)
        filtered_mask = filtered_ranges < threshold
        filtered_pts = self._ranges_to_cartesian(
            scan.angles, filtered_ranges, pose, filtered_mask
        )

        # 3) Kümeleme — filtre sonrası geçerli noktalar üzerinde
        valid_indices = np.where(filtered_mask)[0]
        clusters = self._cluster_consecutive(
            filtered_pts, valid_indices, len(scan.ranges)
        )

        return LidarProcessingResult(
            raw_points=raw_pts,
            filtered_points=filtered_pts,
            clusters=clusters,
        )

    # ------------------------------------------------------------------ iç araçlar

    @staticmethod
    def _scan_to_cartesian(
        scan: LidarScan,
        pose: Tuple[float, float, float],
        mask: np.ndarray,
    ) -> np.ndarray:
        if not mask.any():
            return np.zeros((0, 2))
        x, y, theta = pose
        ga = scan.angles[mask] + theta
        r = scan.ranges[mask]
        return np.column_stack([x + r * np.cos(ga), y + r * np.sin(ga)])

    @staticmethod
    def _ranges_to_cartesian(
        angles: np.ndarray,
        ranges: np.ndarray,
        pose: Tuple[float, float, float],
        mask: np.ndarray,
    ) -> np.ndarray:
        if not mask.any():
            return np.zeros((0, 2))
        x, y, theta = pose
        ga = angles[mask] + theta
        r = ranges[mask]
        return np.column_stack([x + r * np.cos(ga), y + r * np.sin(ga)])

    @staticmethod
    def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
        """Dairesel (wrap) 1B medyan filtre."""
        if window == 1:
            return values.copy()
        n = values.size
        half = window // 2
        out = np.empty_like(values)
        padded = np.concatenate([values[-half:], values, values[:half]])
        for i in range(n):
            out[i] = np.median(padded[i : i + window])
        return out

    def _cluster_consecutive(
        self,
        points: np.ndarray,
        beam_indices: np.ndarray,
        total_beams: int,
    ) -> List[Cluster]:
        """Ardışık noktaları mesafe eşiğine göre gruplar.

        Tam dönüş (360°) taramada başı ve sonu birleştirir.
        """
        if points.shape[0] == 0:
            return []

        eps = self.cluster_eps
        groups: List[List[int]] = []
        current: List[int] = [0]

        for i in range(1, points.shape[0]):
            # Hem geometrik bitişiklik hem de ışın komşuluğu istenir;
            # küçük açı sıçramalarına da izin verilir.
            beam_gap = beam_indices[i] - beam_indices[i - 1]
            spatial_gap = np.linalg.norm(points[i] - points[i - 1])
            if spatial_gap <= eps and beam_gap <= 3:
                current.append(i)
            else:
                groups.append(current)
                current = [i]
        groups.append(current)

        # 360° wrap: ilk ve son grup gerçekten bitişikse birleştir
        if (
            len(groups) >= 2
            and beam_indices[0] == 0
            and beam_indices[-1] == total_beams - 1
        ):
            first_pt = points[groups[0][0]]
            last_pt = points[groups[-1][-1]]
            if np.linalg.norm(first_pt - last_pt) <= eps:
                groups[0] = groups[-1] + groups[0]
                groups.pop()

        clusters: List[Cluster] = []
        for g in groups:
            if len(g) < self.min_cluster_size:
                continue
            idx = np.array(g, dtype=int)
            clusters.append(
                Cluster(points=points[idx], indices=beam_indices[idx].copy())
            )
        return clusters


__all__ = ["LidarProcessor", "LidarProcessingResult", "Cluster"]
