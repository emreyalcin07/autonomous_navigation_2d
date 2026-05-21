"""Hibrit navigasyon: RRT global yol + APF lokal engelden kaçınma.

Sistem iki katmanlı çalışır. **RRT** (Rapidly-exploring Random Tree)
katmanı, ortam haritası üzerinden başlangıçtan hedefe çarpışmasız bir
düğüm zinciri üretir. Üretilen ham yol kısayol-tabanlı düzeltme
(shortcut smoothing) ile sadeleştirilir. **APF** (yapay potansiyel
alanlar) katmanı ise her zaman adımında, robot ile mevcut waypoint
arasındaki çekici kuvveti ve LiDAR'dan gelen filtrelenmiş engel
noktalarından türeyen itici kuvveti toplayıp anlık bir yön referansı
üretir. Bu yön referansı, non-holonomic diferansiyel sürüş robotuna
uygun :math:`v, \\omega` komutlarına çevrilir.

Bu yaklaşım, ödevin "öncelikli tercih" olarak işaret ettiği RRT + APF
hibrit yapısını birebir karşılar. Global yol uzun-erim öngörü, APF ise
sensör tabanlı tepkisellik sağlar; ikisi bir arada lokal minimum riskini
azaltır ve sensör gürültüsüne karşı dayanıklılık verir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .environment import Environment2D
from .robot import DifferentialDriveRobot
from .utils import normalize_angle


# =============================================================================
# RRT
# =============================================================================

@dataclass
class _RRTNode:
    pos: np.ndarray
    parent_idx: Optional[int]


class RRTPlanner:
    """Çift uçlu olmayan klasik RRT — goal-bias ve kısayol yumuşatmalı."""

    def __init__(
        self,
        env: Environment2D,
        robot_radius: float,
        step_size: float,
        max_iters: int,
        goal_bias: float,
        goal_threshold: float,
        safety_margin: float,
        collision_check_resolution: float,
        smoothing_iters: int,
        rng: np.random.Generator,
    ) -> None:
        self.env = env
        self.inflate = robot_radius + safety_margin
        self.step = float(step_size)
        self.max_iters = int(max_iters)
        self.goal_bias = float(goal_bias)
        self.goal_threshold = float(goal_threshold)
        self.collision_resolution = float(collision_check_resolution)
        self.smoothing_iters = int(smoothing_iters)
        self.rng = rng
        self.iterations_used: int = 0
        self.last_tree_size: int = 0

    @classmethod
    def from_config(cls, cfg, env: Environment2D, rng: np.random.Generator) -> "RRTPlanner":
        c = cfg.navigation.rrt
        return cls(
            env=env,
            robot_radius=float(cfg.robot.radius),
            step_size=float(c.step_size),
            max_iters=int(c.max_iters),
            goal_bias=float(c.goal_bias),
            goal_threshold=float(c.goal_threshold),
            safety_margin=float(c.safety_margin),
            collision_check_resolution=float(c.collision_check_resolution),
            smoothing_iters=int(c.smoothing_iters),
            rng=rng,
        )

    # ------------------------------------------------------------------ planlama

    def plan(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> List[np.ndarray]:
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)

        if self._point_blocked(start_arr):
            raise RuntimeError("Başlangıç noktası şişirilmiş çarpışma içinde.")
        if self._point_blocked(goal_arr):
            raise RuntimeError("Hedef noktası şişirilmiş çarpışma içinde.")

        nodes: List[_RRTNode] = [_RRTNode(pos=start_arr.copy(), parent_idx=None)]
        positions: List[np.ndarray] = [start_arr.copy()]
        goal_idx: Optional[int] = None

        for it in range(self.max_iters):
            self.iterations_used = it + 1

            # Örnek
            if self.rng.random() < self.goal_bias:
                target = goal_arr
            else:
                target = np.array([
                    self.rng.uniform(0.0, self.env.width),
                    self.rng.uniform(0.0, self.env.height),
                ])

            # En yakın düğüm (vektörize)
            arr = np.asarray(positions)
            dists = np.linalg.norm(arr - target, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest = positions[nearest_idx]

            direction = target - nearest
            d = float(np.linalg.norm(direction))
            if d < 1e-9:
                continue
            step = self.step if d > self.step else d
            new_pt = nearest + (step / d) * direction

            if not self._segment_free(nearest, new_pt):
                continue

            nodes.append(_RRTNode(pos=new_pt, parent_idx=nearest_idx))
            positions.append(new_pt)

            # Hedefe ulaşıldı mı?
            if np.linalg.norm(new_pt - goal_arr) <= self.goal_threshold:
                if self._segment_free(new_pt, goal_arr):
                    nodes.append(_RRTNode(pos=goal_arr.copy(), parent_idx=len(nodes) - 1))
                    positions.append(goal_arr.copy())
                    goal_idx = len(nodes) - 1
                    break

        self.last_tree_size = len(nodes)
        if goal_idx is None:
            return []

        # Yolu oluştur (goal'dan geriye)
        path: List[np.ndarray] = []
        cur: Optional[int] = goal_idx
        while cur is not None:
            path.append(nodes[cur].pos.copy())
            cur = nodes[cur].parent_idx
        path.reverse()
        return path

    def smooth(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Rastlantısal kısayol algoritması.

        Yolu kısaltmak için iki rastgele indeks arasında çarpışmasız
        düz çizgi varsa aradaki düğümler atılır. Tipik olarak yolu
        keskin zigzaglardan kurtarır.
        """
        if len(path) < 3:
            return list(path)
        smoothed: List[np.ndarray] = [np.asarray(p, dtype=float) for p in path]
        for _ in range(self.smoothing_iters):
            n = len(smoothed)
            if n < 3:
                break
            i = int(self.rng.integers(0, n - 2))
            j = int(self.rng.integers(i + 2, n))
            if self._segment_free(smoothed[i], smoothed[j]):
                smoothed = smoothed[: i + 1] + smoothed[j:]
        return smoothed

    # ------------------------------------------------------------------ iç

    def _point_blocked(self, pt: np.ndarray) -> bool:
        return self.env.is_collision(float(pt[0]), float(pt[1]), self.inflate)

    def _segment_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        d = float(np.linalg.norm(p2 - p1))
        if d < 1e-9:
            return not self._point_blocked(p1)
        n = max(2, int(math.ceil(d / self.collision_resolution)) + 1)
        ts = np.linspace(0.0, 1.0, n)
        for t in ts:
            pt = p1 + t * (p2 - p1)
            if self._point_blocked(pt):
                return False
        return True


# =============================================================================
# APF (Artificial Potential Field)
# =============================================================================

@dataclass
class APFForces:
    total: np.ndarray
    attractive: np.ndarray
    repulsive: np.ndarray


class APFController:
    """Çekici (waypoint'e doğru) + LiDAR-tabanlı itici kuvvet."""

    def __init__(
        self,
        k_attr: float,
        k_rep: float,
        rep_distance: float,
        max_force_norm: float,
    ) -> None:
        self.k_attr = float(k_attr)
        self.k_rep = float(k_rep)
        self.rep_distance = float(rep_distance)
        self.max_force_norm = float(max_force_norm)

    @classmethod
    def from_config(cls, cfg) -> "APFController":
        c = cfg.navigation.apf
        return cls(
            k_attr=float(c.k_attr),
            k_rep=float(c.k_rep),
            rep_distance=float(c.rep_distance),
            max_force_norm=float(c.max_force_norm),
        )

    def compute(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        obstacle_points: Optional[np.ndarray] = None,
    ) -> APFForces:
        # F_attr: birim vektör * k_attr (uzakta da sınırlı kalır)
        diff = target_pos - robot_pos
        d = float(np.linalg.norm(diff))
        if d > 1e-9:
            f_attr = self.k_attr * (diff / d)
        else:
            f_attr = np.zeros(2)

        # F_rep: yakın noktalardan robot'a doğru, mesafe küçüldükçe büyür
        f_rep = np.zeros(2)
        if obstacle_points is not None and len(obstacle_points) > 0:
            v = robot_pos - obstacle_points              # (M, 2)
            dists = np.linalg.norm(v, axis=1)            # (M,)
            mask = (dists < self.rep_distance) & (dists > 1e-3)
            if mask.any():
                d_sel = dists[mask]
                v_sel = v[mask]
                mags = (
                    self.k_rep
                    * (1.0 / d_sel - 1.0 / self.rep_distance)
                    * (1.0 / (d_sel * d_sel))
                )
                units = v_sel / d_sel[:, None]
                f_rep = (mags[:, None] * units).sum(axis=0)

        f_total = f_attr + f_rep
        norm = float(np.linalg.norm(f_total))
        if norm > self.max_force_norm:
            f_total = f_total * (self.max_force_norm / norm)

        return APFForces(total=f_total, attractive=f_attr, repulsive=f_rep)


# =============================================================================
# Hibrit navigator (kontrolcü)
# =============================================================================

class HybridNavigator:
    """RRT yolu üzerinde APF tabanlı waypoint takipçi.

    ``compute_control(robot, env, runner)`` imzası SimulationRunner'ın
    bekledigi `ControlFn`'e doğrudan uyar. Runner üzerinde tutulan son
    LiDAR işleme sonucu ``runner.last_lidar_result`` reaktif itme
    bileşeni için doğrudan beslenir.
    """

    def __init__(self, env: Environment2D, cfg, rng: np.random.Generator) -> None:
        self.env = env
        self.cfg = cfg
        self.planner = RRTPlanner.from_config(cfg, env, rng)
        self.apf = APFController.from_config(cfg)

        ctrl = cfg.navigation.controller
        self.k_v = float(ctrl.k_v)
        self.k_omega = float(ctrl.k_omega)
        self.waypoint_threshold = float(ctrl.waypoint_threshold)
        self.heading_slowdown_rad = float(ctrl.heading_slowdown_rad)
        self.slowdown_factor = float(ctrl.slowdown_factor)
        self.stuck_window_seconds = float(ctrl.stuck_window_seconds)
        self.stuck_threshold = float(ctrl.stuck_threshold)

        self.v_max = float(cfg.robot.limits.v_max)

        self.path: List[np.ndarray] = []
        self.waypoint_idx: int = 0
        self._position_window: List[Tuple[float, np.ndarray]] = []
        self.stuck: bool = False

    # ------------------------------------------------------------------ planlama

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> List[np.ndarray]:
        """RRT yolunu hesapla ve düzelt."""
        raw = self.planner.plan(start, goal)
        if not raw:
            raise RuntimeError(
                f"RRT yol bulamadı ({self.planner.iterations_used} iter, "
                f"{self.planner.last_tree_size} düğüm)."
            )
        self.path = self.planner.smooth(raw)
        self.waypoint_idx = 1 if len(self.path) > 1 else 0
        return self.path

    def path_as_array(self) -> np.ndarray:
        if not self.path:
            return np.zeros((0, 2))
        return np.asarray(self.path, dtype=float)

    # ------------------------------------------------------------------ kontrolcü

    def _advance_waypoint(self, pos: np.ndarray) -> np.ndarray:
        if not self.path:
            return pos
        while self.waypoint_idx < len(self.path) - 1:
            wp = self.path[self.waypoint_idx]
            if np.linalg.norm(wp - pos) < self.waypoint_threshold:
                self.waypoint_idx += 1
            else:
                break
        return self.path[self.waypoint_idx]

    def _track_stuck(self, t_now: float, pos: np.ndarray) -> None:
        self._position_window.append((t_now, pos.copy()))
        cutoff = t_now - self.stuck_window_seconds
        while self._position_window and self._position_window[0][0] < cutoff:
            self._position_window.pop(0)
        if len(self._position_window) >= 5:
            disp = float(
                np.linalg.norm(self._position_window[-1][1] - self._position_window[0][1])
            )
            self.stuck = disp < self.stuck_threshold

    def compute_control(
        self,
        robot: DifferentialDriveRobot,
        env: Environment2D,
        runner,
    ) -> Tuple[float, float]:
        x, y, theta = robot.pose
        pos = np.array([x, y], dtype=float)

        target = self._advance_waypoint(pos)

        # LiDAR engel noktaları (filtrelenmiş)
        hits: Optional[np.ndarray] = None
        if runner.last_lidar_result is not None:
            pts = runner.last_lidar_result.filtered_points
            if pts.shape[0] > 0:
                hits = pts

        forces = self.apf.compute(pos, target, hits)
        F = forces.total

        # Yön referansı: kuvvet yönü; çok küçükse doğrudan hedefe doğru
        if float(np.linalg.norm(F)) > 1e-3:
            phi_star = math.atan2(F[1], F[0])
        else:
            d = target - pos
            phi_star = math.atan2(d[1], d[0])

        alpha = normalize_angle(phi_star - theta)

        # Komut üretimi
        v_cmd = self.k_v * self.v_max * max(0.0, math.cos(alpha))
        if abs(alpha) > self.heading_slowdown_rad:
            v_cmd *= self.slowdown_factor

        omega_cmd = self.k_omega * alpha

        # Sıkışma izleme
        t_now = (len(runner.records) + 1) * runner.dt
        self._track_stuck(t_now, pos)

        return v_cmd, omega_cmd


__all__ = [
    "RRTPlanner",
    "APFController",
    "APFForces",
    "HybridNavigator",
]
