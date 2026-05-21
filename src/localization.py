"""Lokalizasyon modülleri: Dead Reckoning ve Extended Kalman Filter.

İki ayrı tahminci sağlanır:

- ``DeadReckoning``: yalnızca enkoderden okunan gövde hızlarını (v, ω)
  entegre eder. Gürültü doğrudan kümülatif şekilde duruma karışır;
  zamanla hata büyür. Karşılaştırma temeli olarak rapor edilir.

- ``ExtendedKalmanFilter``: aynı kontrol modelini kullanır ancak her
  adımda IMU yaw ölçümünü düzeltme adımında değerlendirir. Bu sayede
  yönelim sürüklenmesi (drift) bastırılır ve buna bağlı konum hatası da
  azalır.

Durum vektörü :math:`\\mathbf{x}=[x,y,\\theta]^{\\top}`, kontrol
:math:`\\mathbf{u}=[v,\\omega]^{\\top}`. Predict:

.. math::
   \\mathbf{x}^{-}_k = f(\\mathbf{x}_{k-1}, \\mathbf{u}_k, \\Delta t),\\quad
   P^{-}_k = F_k P_{k-1} F_k^{\\top} + G_k Q G_k^{\\top}.

IMU yaw güncellemesi (lineer ölçüm modeli) ile:

.. math::
   H = [0\\;0\\;1],\\quad
   K = P^{-}H^{\\top}\\,(H P^{-}H^{\\top} + R)^{-1}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .utils import normalize_angle


# =============================================================================
# Dead Reckoning
# =============================================================================

class DeadReckoning:
    """Enkoderden gelen (v, ω)'yi entegre eden saf odometri tahmini."""

    def __init__(self, x: float, y: float, theta: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.theta = normalize_angle(float(theta))
        self._history: List[Tuple[float, float, float]] = [self.pose]

    @classmethod
    def from_config(cls, cfg) -> "DeadReckoning":
        start = cfg.environment.start
        theta0 = float(cfg.robot.get("initial_theta", 0.0))
        return cls(x=float(start[0]), y=float(start[1]), theta=theta0)

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta

    @property
    def trajectory(self) -> np.ndarray:
        return np.asarray(self._history, dtype=float)

    def update(self, v: float, omega: float, dt: float) -> None:
        """(v, ω) komutunu/ölçümünü dt boyunca entegre eder."""
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta = normalize_angle(self.theta + omega * dt)
        self._history.append(self.pose)


# =============================================================================
# Extended Kalman Filter
# =============================================================================

@dataclass
class EKFConfig:
    """EKF için kovaryans parametreleri."""

    sigma_x0: float = 0.05
    sigma_y0: float = 0.05
    sigma_theta0: float = 0.02
    sigma_v: float = 0.06        # kontrol gürültüsü, m/s
    sigma_omega: float = 0.04    # kontrol gürültüsü, rad/s
    sigma_imu_yaw: float = 0.05  # ölçüm gürültüsü, rad
    use_imu_update: bool = True

    @classmethod
    def from_config(cls, cfg) -> "EKFConfig":
        ekf = cfg.localization.ekf
        ips = ekf.initial_pose_sigma
        pn = ekf.process_noise
        mn = ekf.measurement_noise
        return cls(
            sigma_x0=float(ips.x),
            sigma_y0=float(ips.y),
            sigma_theta0=float(ips.theta),
            sigma_v=float(pn.v),
            sigma_omega=float(pn.omega),
            sigma_imu_yaw=float(mn.imu_yaw_sigma),
            use_imu_update=bool(ekf.get("use_imu_update", True)),
        )


class ExtendedKalmanFilter:
    """3 durumlu EKF: encoder ile predict, IMU yaw ile update."""

    def __init__(
        self,
        x0: float,
        y0: float,
        theta0: float,
        ekf_cfg: EKFConfig,
    ) -> None:
        self.x = np.array([x0, y0, normalize_angle(theta0)], dtype=float)
        self.P = np.diag([
            ekf_cfg.sigma_x0 ** 2,
            ekf_cfg.sigma_y0 ** 2,
            ekf_cfg.sigma_theta0 ** 2,
        ])
        self.Q = np.diag([ekf_cfg.sigma_v ** 2, ekf_cfg.sigma_omega ** 2])
        self.R_yaw = ekf_cfg.sigma_imu_yaw ** 2
        self.use_imu_update = ekf_cfg.use_imu_update

        self._history: List[np.ndarray] = [self.x.copy()]
        self._cov_history: List[np.ndarray] = [np.diag(self.P).copy()]

    @classmethod
    def from_config(cls, cfg) -> "ExtendedKalmanFilter":
        start = cfg.environment.start
        theta0 = float(cfg.robot.get("initial_theta", 0.0))
        return cls(
            x0=float(start[0]),
            y0=float(start[1]),
            theta0=theta0,
            ekf_cfg=EKFConfig.from_config(cfg),
        )

    # ------------------------------------------------------------------ erişim

    @property
    def pose(self) -> Tuple[float, float, float]:
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    @property
    def trajectory(self) -> np.ndarray:
        return np.asarray(self._history, dtype=float)

    @property
    def covariance_diag(self) -> np.ndarray:
        """Her adımdaki P matrisinin köşegeni: (N, 3)."""
        return np.asarray(self._cov_history, dtype=float)

    # ------------------------------------------------------------------ predict

    def predict(self, v: float, omega: float, dt: float) -> None:
        """Encoder ölçümünden (v, ω) ile durum ve kovaryansı ilerlet."""
        x, y, theta = self.x
        c, s = math.cos(theta), math.sin(theta)

        # f(x, u)
        x_new = x + v * c * dt
        y_new = y + v * s * dt
        theta_new = normalize_angle(theta + omega * dt)

        # Jakobiyenler
        F = np.array([
            [1.0, 0.0, -v * s * dt],
            [0.0, 1.0,  v * c * dt],
            [0.0, 0.0,  1.0],
        ])
        G = np.array([
            [c * dt, 0.0],
            [s * dt, 0.0],
            [0.0,    dt],
        ])

        self.x = np.array([x_new, y_new, theta_new])
        self.P = F @ self.P @ F.T + G @ self.Q @ G.T

    # ------------------------------------------------------------------ update

    def update_yaw(self, z_yaw: float) -> None:
        """IMU yaw ölçümüyle yönelimi düzelt."""
        if not self.use_imu_update:
            return

        H = np.array([[0.0, 0.0, 1.0]])
        # Açı farkı yumuşatılarak alınmalı (wrap-around)
        innovation = normalize_angle(z_yaw - self.x[2])
        S = float(H @ self.P @ H.T) + self.R_yaw   # skaler
        K = (self.P @ H.T).flatten() / S           # (3,)

        self.x = self.x + K * innovation
        self.x[2] = normalize_angle(self.x[2])
        self.P = (np.eye(3) - np.outer(K, H.flatten())) @ self.P

    # ------------------------------------------------------------------ kayıt

    def commit(self) -> None:
        """Mevcut tahmini tarihçeye işle (her sim adımının sonunda)."""
        self._history.append(self.x.copy())
        self._cov_history.append(np.diag(self.P).copy())


__all__ = ["DeadReckoning", "ExtendedKalmanFilter", "EKFConfig"]
