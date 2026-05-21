"""Diferansiyel sürüşlü mobil robot (non-holonomic).

Robot durumu :math:`(x, y, \\theta)` ve gövde hızı :math:`(v, \\omega)` ile
modellenir. Kinematik denklemler:

.. math::
   \\dot{x} = v\\cos\\theta,\\quad
   \\dot{y} = v\\sin\\theta,\\quad
   \\dot{\\theta} = \\omega.

Komut girişleri ``v_cmd, omega_cmd`` doğrudan uygulanmaz; önce ivme
limitleri (a_max, alpha_max) üzerinden filtrelenir, sonra hız limitlerine
clamp edilir. Bu Adım 2-3'te navigasyon kontrolcülerinin gerçekçi bir
robot modeliyle çalışmasını sağlar. Çarpışma denetimi ortam üzerinden
yapılır; çarpışma anında robot durdurulur ve durum işaretlenir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .environment import Environment2D
from .utils import clamp, normalize_angle


@dataclass
class RobotLimits:
    v_max: float = 1.2
    v_min: float = -0.4
    omega_max: float = 1.5
    a_max: float = 1.0
    alpha_max: float = 2.5

    @classmethod
    def from_config(cls, cfg) -> "RobotLimits":
        L = cfg.robot.limits
        return cls(
            v_max=float(L.v_max),
            v_min=float(L.v_min),
            omega_max=float(L.omega_max),
            a_max=float(L.a_max),
            alpha_max=float(L.alpha_max),
        )


@dataclass
class RobotState:
    """Sürekli zamanlı durum: poz + gövde hızları."""

    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0

    def as_pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta


class DifferentialDriveRobot:
    """Non-holonomic diferansiyel sürüşlü robot modeli.

    Robot dünyayla yalnızca ``Environment2D`` üzerinden etkileşir. Gerçek
    (ground-truth) yörünge zaman serisi olarak tutulur; lokalizasyon
    modülleri kendi tahminlerini ayrı tutar (Adım 2).
    """

    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        radius: float,
        wheel_base: float,
        limits: RobotLimits,
    ) -> None:
        self.state = RobotState(x=x, y=y, theta=normalize_angle(theta))
        self.radius = float(radius)
        self.wheel_base = float(wheel_base)
        self.limits = limits

        # Geçmiş izler (analiz/görselleştirme için)
        self._traj: List[Tuple[float, float, float]] = [self.state.as_pose()]
        self._cmd_history: List[Tuple[float, float]] = []
        self._collided: bool = False

    # ------------------------------------------------------------------ fabrika

    @classmethod
    def from_config(cls, cfg) -> "DifferentialDriveRobot":
        start = cfg.environment.start
        theta0 = float(cfg.robot.get("initial_theta", 0.0))
        return cls(
            x=float(start[0]),
            y=float(start[1]),
            theta=theta0,
            radius=float(cfg.robot.radius),
            wheel_base=float(cfg.robot.wheel_base),
            limits=RobotLimits.from_config(cfg),
        )

    # ------------------------------------------------------------------ geometrik özellikler

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self.state.as_pose()

    @property
    def has_collided(self) -> bool:
        return self._collided

    @property
    def trajectory(self) -> np.ndarray:
        """Şimdiye kadarki ground-truth yörünge: (N,3) [x, y, theta]."""
        return np.asarray(self._traj, dtype=float)

    @property
    def command_history(self) -> np.ndarray:
        """Uygulanan (v, omega) komut tarihi: (N,2)."""
        return np.asarray(self._cmd_history, dtype=float)

    # ------------------------------------------------------------------ kinematik

    def step(
        self,
        v_cmd: float,
        omega_cmd: float,
        dt: float,
        environment: Environment2D,
    ) -> RobotState:
        """Bir adım ileri al.

        Çarpışma denetimi: yeni pozda çarpışma varsa hareket geri alınır,
        hızlar sıfırlanır ve ``has_collided`` bayrağı ayarlanır.
        """
        if self._collided:
            # Bir kez çarpıştıysa, üst katman karar verene kadar dur
            self._traj.append(self.state.as_pose())
            self._cmd_history.append((0.0, 0.0))
            return self.state

        v_target = clamp(v_cmd, self.limits.v_min, self.limits.v_max)
        w_target = clamp(omega_cmd, -self.limits.omega_max, self.limits.omega_max)

        # İvme limitleri
        dv_max = self.limits.a_max * dt
        dw_max = self.limits.alpha_max * dt
        v = self.state.v + clamp(v_target - self.state.v, -dv_max, dv_max)
        w = self.state.omega + clamp(w_target - self.state.omega, -dw_max, dw_max)

        # Diferansiyel sürüş kinematiği (basit Euler; küçük dt için yeterli)
        new_x = self.state.x + v * math.cos(self.state.theta) * dt
        new_y = self.state.y + v * math.sin(self.state.theta) * dt
        new_theta = normalize_angle(self.state.theta + w * dt)

        if environment.is_collision(new_x, new_y, self.radius):
            # Çarpışma: hareketi reddet, hızı sıfırla, bayrağı kaldır
            self._collided = True
            self.state.v = 0.0
            self.state.omega = 0.0
            self._traj.append(self.state.as_pose())
            self._cmd_history.append((v_cmd, omega_cmd))
            return self.state

        self.state.x = new_x
        self.state.y = new_y
        self.state.theta = new_theta
        self.state.v = v
        self.state.omega = w

        self._traj.append(self.state.as_pose())
        self._cmd_history.append((v_cmd, omega_cmd))
        return self.state

    # ------------------------------------------------------------------ teker hızları

    def wheel_speeds(self) -> Tuple[float, float]:
        """Mevcut gövde hızından (v, omega) -> (v_left, v_right) türetir.

        :math:`v_r = v + \\omega \\cdot L/2`,
        :math:`v_l = v - \\omega \\cdot L/2`.
        """
        v = self.state.v
        w = self.state.omega
        L = self.wheel_base
        v_r = v + w * L / 2.0
        v_l = v - w * L / 2.0
        return v_l, v_r


__all__ = ["DifferentialDriveRobot", "RobotState", "RobotLimits"]
