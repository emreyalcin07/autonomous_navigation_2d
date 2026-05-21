"""2B simülasyon ortamı: engeller, sınırlar ve ışın atışı.

Ortam, robotun içinde hareket ettiği dünya modelidir. Engeller polimorfik
bir sınıf hiyerarşisiyle ifade edilir (Obstacle taban sınıfı, ardından
RectObstacle ve CircleObstacle). Her engel kendi geometrisi için
``contains_point``, ``distance_to_point`` ve ``ray_intersection`` arayüzünü
sağlar. Bu sayede LiDAR sensörü engelin tipini bilmek zorunda kalmaz;
sadece arayüzü çağırır.

Environment2D ise engellerin toplamı, dünya sınırları ve başlangıç/hedef
noktalarını kapsar; çarpışma denetimi ve ışın atış servislerini sunar.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Engel sınıfları
# =============================================================================

class Obstacle(ABC):
    """Engellerin soyut taban sınıfı."""

    @abstractmethod
    def contains_point(self, x: float, y: float, inflate: float = 0.0) -> bool:
        """Verilen nokta engelin (şişirilmiş) içinde mi?"""

    @abstractmethod
    def distance_to_point(self, x: float, y: float) -> float:
        """Engele en yakın yüzeye olan öklid mesafesi (içerideyse 0)."""

    @abstractmethod
    def ray_intersection(
        self,
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        max_range: float,
    ) -> Optional[float]:
        """Yarı doğru-engel kesişimine en yakın mesafe.

        ``(ox, oy)`` ışının başlangıcı, ``(dx, dy)`` birim yön vektörüdür.
        Kesişim yoksa ``None``, varsa ``[0, max_range]`` aralığında mesafe.
        """

    @abstractmethod
    def bbox(self) -> Tuple[float, float, float, float]:
        """Eksen hizalı sınır kutusu: (xmin, ymin, xmax, ymax)."""


@dataclass
class RectObstacle(Obstacle):
    """Eksen hizalı dikdörtgen engel."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x_max(self) -> float:
        return self.x + self.width

    @property
    def y_max(self) -> float:
        return self.y + self.height

    def contains_point(self, px: float, py: float, inflate: float = 0.0) -> bool:
        return (
            self.x - inflate <= px <= self.x_max + inflate
            and self.y - inflate <= py <= self.y_max + inflate
        )

    def distance_to_point(self, px: float, py: float) -> float:
        dx = max(self.x - px, 0.0, px - self.x_max)
        dy = max(self.y - py, 0.0, py - self.y_max)
        return math.hypot(dx, dy)

    def ray_intersection(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float
    ) -> Optional[float]:
        # AABB için "slab" yöntemi
        t_min = 0.0
        t_max = max_range

        for origin, direction, lo, hi in (
            (ox, dx, self.x, self.x_max),
            (oy, dy, self.y, self.y_max),
        ):
            if abs(direction) < 1e-12:
                if origin < lo or origin > hi:
                    return None
                continue
            inv = 1.0 / direction
            t1 = (lo - origin) * inv
            t2 = (hi - origin) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None

        if t_min < 0.0 or t_min > max_range:
            return None
        return t_min

    def bbox(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.x_max, self.y_max


@dataclass
class CircleObstacle(Obstacle):
    """Dairesel engel."""

    cx: float
    cy: float
    radius: float

    def contains_point(self, px: float, py: float, inflate: float = 0.0) -> bool:
        return math.hypot(px - self.cx, py - self.cy) <= self.radius + inflate

    def distance_to_point(self, px: float, py: float) -> float:
        return max(0.0, math.hypot(px - self.cx, py - self.cy) - self.radius)

    def ray_intersection(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float
    ) -> Optional[float]:
        # |o + t*d - c|^2 = r^2  ikinci derece denklem
        fx = ox - self.cx
        fy = oy - self.cy
        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - self.radius * self.radius
        disc = b * b - 4 * a * c
        if disc < 0.0:
            return None
        sq = math.sqrt(disc)
        t1 = (-b - sq) / (2 * a)
        t2 = (-b + sq) / (2 * a)
        for t in (t1, t2):
            if 0.0 <= t <= max_range:
                return t
        return None

    def bbox(self) -> Tuple[float, float, float, float]:
        return (
            self.cx - self.radius,
            self.cy - self.radius,
            self.cx + self.radius,
            self.cy + self.radius,
        )


# =============================================================================
# 2B ortam
# =============================================================================

@dataclass
class Environment2D:
    """Robotun içinde hareket ettiği 2B dünya.

    Ortam, dünyanın eksen hizalı sınırlarını, engel listesini, başlangıç ve
    hedef noktalarını kapsar. ``cast_ray`` LiDAR ve VFH benzeri reaktif
    yöntemler için ortak ışın atış servisi sağlar; ``is_collision`` ise
    robot yarıçapı dahil çarpışma denetimi yapar.
    """

    width: float
    height: float
    start: Tuple[float, float]
    goal: Tuple[float, float]
    goal_tolerance: float = 0.5
    obstacles: List[Obstacle] = field(default_factory=list)

    # ------------------------------------------------------------------ kurulum

    @classmethod
    def build(
        cls,
        cfg,
        rng: np.random.Generator,
    ) -> "Environment2D":
        """Konfigürasyon ve RNG'den deterministik ortam üretir."""
        env = cls(
            width=float(cfg.environment.width),
            height=float(cfg.environment.height),
            start=tuple(cfg.environment.start),
            goal=tuple(cfg.environment.goal),
            goal_tolerance=float(cfg.environment.goal_tolerance),
        )
        env._populate_obstacles(cfg, rng)
        return env

    def _populate_obstacles(self, cfg, rng: np.random.Generator) -> None:
        n = int(cfg.environment.n_obstacles)
        obs_cfg = cfg.environment.obstacle
        kind = str(obs_cfg.type).lower()
        s_min = float(obs_cfg.min_size)
        s_max = float(obs_cfg.max_size)
        clearance = float(obs_cfg.min_clearance)
        margin = float(cfg.environment.boundary_margin)
        robot_r = float(cfg.robot.radius)

        max_attempts = n * 200
        attempts = 0
        while len(self.obstacles) < n and attempts < max_attempts:
            attempts += 1
            size_w = float(rng.uniform(s_min, s_max))
            size_h = float(rng.uniform(s_min, s_max))
            shape = kind
            if kind == "mixed":
                shape = "rectangle" if rng.random() < 0.7 else "circle"

            if shape == "rectangle":
                x = float(rng.uniform(margin, self.width - size_w - margin))
                y = float(rng.uniform(margin, self.height - size_h - margin))
                candidate: Obstacle = RectObstacle(x, y, size_w, size_h)
            else:
                r = 0.5 * min(size_w, size_h)
                cx = float(rng.uniform(margin + r, self.width - margin - r))
                cy = float(rng.uniform(margin + r, self.height - margin - r))
                candidate = CircleObstacle(cx, cy, r)

            if not self._is_valid_obstacle(candidate, robot_r, clearance):
                continue
            self.obstacles.append(candidate)

        if len(self.obstacles) < n:
            raise RuntimeError(
                f"İstenen {n} engel yerleştirilemedi (yalnız "
                f"{len(self.obstacles)} adet). Boyutları/clearance'ı gevşetin."
            )

    def _is_valid_obstacle(
        self, candidate: Obstacle, robot_r: float, clearance: float
    ) -> bool:
        # Başlangıç ve hedefin etrafında güvenlik bandı bırak
        safety = robot_r + clearance
        if candidate.distance_to_point(*self.start) < safety:
            return False
        if candidate.distance_to_point(*self.goal) < safety:
            return False
        # Diğer engellerle çakışma yok
        cx0, cy0, cx1, cy1 = candidate.bbox()
        for other in self.obstacles:
            ox0, oy0, ox1, oy1 = other.bbox()
            # Şişirilmiş AABB örtüşmesi: hızlı eleme
            if (
                cx0 - clearance <= ox1
                and cx1 + clearance >= ox0
                and cy0 - clearance <= oy1
                and cy1 + clearance >= oy0
            ):
                return False
        return True

    # ------------------------------------------------------------------ servisler

    def in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            margin <= x <= self.width - margin
            and margin <= y <= self.height - margin
        )

    def is_collision(self, x: float, y: float, robot_radius: float) -> bool:
        """Robot merkezi (x,y) ve verilen yarıçapla çarpışma var mı?"""
        if not self.in_bounds(x, y, margin=robot_radius):
            return True
        for ob in self.obstacles:
            if ob.distance_to_point(x, y) <= robot_radius:
                return True
        return False

    def cast_ray(
        self,
        ox: float,
        oy: float,
        angle: float,
        max_range: float,
    ) -> float:
        """Verilen pozisyondan verilen global açıda ışın atar.

        En yakın engele veya dünya sınırına olan mesafeyi döndürür.
        Hiçbir engel bulunmazsa ``max_range`` döndürülür.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Dünya sınırları için kesişim (rect 0..w, 0..h gibi davran)
        t_best = self._ray_world_bounds(ox, oy, dx, dy, max_range)

        for ob in self.obstacles:
            t = ob.ray_intersection(ox, oy, dx, dy, max_range)
            if t is not None and t < t_best:
                t_best = t

        return min(t_best, max_range)

    def _ray_world_bounds(
        self, ox: float, oy: float, dx: float, dy: float, max_range: float
    ) -> float:
        """Işının dünya kenarlarıyla kesişimi (sonsuz bir kutu içinde)."""
        t_max = max_range
        for origin, direction, lo, hi in (
            (ox, dx, 0.0, self.width),
            (oy, dy, 0.0, self.height),
        ):
            if abs(direction) < 1e-12:
                continue
            if direction > 0:
                t = (hi - origin) / direction
            else:
                t = (lo - origin) / direction
            if 0.0 < t < t_max:
                t_max = t
        return t_max

    def goal_reached(self, x: float, y: float) -> bool:
        return math.hypot(x - self.goal[0], y - self.goal[1]) <= self.goal_tolerance


__all__ = [
    "Obstacle",
    "RectObstacle",
    "CircleObstacle",
    "Environment2D",
]
