"""Sensör modelleri: LiDAR, IMU, tekerlek enkoderi.

Tüm sensörler ortak bir arayüze (BaseSensor) uyar; ``measure`` çağrısı
robotun anlık (gerçek) durumunu ve ortamı alıp gürültülü bir ölçüm üretir.
Gauss gürültü standart sapmaları config'den okunur. Her sensör kendi
``np.random.Generator`` örneğine sahiptir; bu sayede deneyler arasında
gürültü akışları izole tutulur ve aynı seedle birebir tekrarlanır.

Adım 2'de ``LidarProcessor`` (mesafe eşikleme + kümeleme) ve
``ExtendedKalmanFilter`` bu sensörlerin çıktısını tüketecektir.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .environment import Environment2D
from .robot import DifferentialDriveRobot
from .utils import normalize_angle


# =============================================================================
# Taban sınıf
# =============================================================================

class BaseSensor(ABC):
    """Tüm sensörler için ortak arayüz."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    @abstractmethod
    def measure(
        self,
        robot: DifferentialDriveRobot,
        environment: Environment2D,
        dt: float,
    ):
        """Anlık ölçümü üret."""


# =============================================================================
# LiDAR
# =============================================================================

@dataclass
class LidarScan:
    """Tek bir LiDAR taraması.

    ``angles`` robot gövde çerçevesindeki ışın açılarıdır (radian).
    ``ranges`` her ışın için (gürültülü) mesafe ölçümüdür. ``hits`` boolean
    maskesi ise gerçek bir engele isabet ettik mi yoksa max_range'e mi
    ulaştık ayrımı için kullanılır.
    """

    angles: np.ndarray
    ranges: np.ndarray
    hits: np.ndarray
    max_range: float
    min_range: float

    def to_cartesian(self, pose: Tuple[float, float, float]) -> np.ndarray:
        """Taramayı global karteziyen noktalara çevirir: (N, 2)."""
        x, y, theta = pose
        global_angles = self.angles + theta
        xs = x + self.ranges * np.cos(global_angles)
        ys = y + self.ranges * np.sin(global_angles)
        return np.column_stack([xs, ys])


class LiDARSensor(BaseSensor):
    """2B LiDAR sensörü, Gauss menzil gürültüsü ve dropout destekli."""

    def __init__(
        self,
        rng: np.random.Generator,
        n_beams: int,
        fov_deg: float,
        max_range: float,
        min_range: float,
        range_noise_std: float,
        angle_noise_std: float,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__(rng)
        self.n_beams = int(n_beams)
        self.fov = math.radians(float(fov_deg))
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.range_noise_std = float(range_noise_std)
        self.angle_noise_std = float(angle_noise_std)
        self.dropout_prob = float(dropout_prob)

        # Gövde çerçevesindeki ışın açıları (sabit)
        if self.fov >= 2 * math.pi - 1e-6:
            # Tam dönüş: 0..2pi, son nokta tekrar etmesin
            self._beam_angles = np.linspace(
                -math.pi, math.pi, self.n_beams, endpoint=False
            )
        else:
            self._beam_angles = np.linspace(
                -self.fov / 2, self.fov / 2, self.n_beams
            )

    @classmethod
    def from_config(cls, cfg, rng: np.random.Generator) -> "LiDARSensor":
        c = cfg.sensors.lidar
        return cls(
            rng=rng,
            n_beams=int(c.n_beams),
            fov_deg=float(c.fov_deg),
            max_range=float(c.max_range),
            min_range=float(c.min_range),
            range_noise_std=float(c.range_noise_std),
            angle_noise_std=float(c.angle_noise_std),
            dropout_prob=float(c.get("dropout_prob", 0.0)),
        )

    @property
    def beam_angles(self) -> np.ndarray:
        return self._beam_angles.copy()

    def measure(
        self,
        robot: DifferentialDriveRobot,
        environment: Environment2D,
        dt: float,
    ) -> LidarScan:
        x, y, theta = robot.pose
        ranges = np.empty(self.n_beams, dtype=float)
        hits = np.zeros(self.n_beams, dtype=bool)

        # Açı gürültüsü: ışın başı bağımsız
        angle_noise = self._rng.normal(0.0, self.angle_noise_std, size=self.n_beams)

        for i, beam in enumerate(self._beam_angles):
            global_angle = normalize_angle(theta + beam + angle_noise[i])
            r = environment.cast_ray(x, y, global_angle, self.max_range)
            hits[i] = r < self.max_range - 1e-6
            ranges[i] = r

        # Gauss menzil gürültüsü
        ranges += self._rng.normal(0.0, self.range_noise_std, size=self.n_beams)

        # Dropout: bazı ölçümleri max_range'e ata (kayıp)
        if self.dropout_prob > 0.0:
            mask = self._rng.random(self.n_beams) < self.dropout_prob
            ranges[mask] = self.max_range
            hits[mask] = False

        # Min-max kırpma
        ranges = np.clip(ranges, self.min_range, self.max_range)

        return LidarScan(
            angles=self._beam_angles.copy(),
            ranges=ranges,
            hits=hits,
            max_range=self.max_range,
            min_range=self.min_range,
        )


# =============================================================================
# IMU
# =============================================================================

@dataclass
class IMUMeasurement:
    """IMU okuması: yaw (rad) ve yaw_rate (rad/s)."""

    yaw: float
    yaw_rate: float


class IMUSensor(BaseSensor):
    """Yön ve açısal hız okuyan, sabit biaslı gürültülü IMU."""

    def __init__(
        self,
        rng: np.random.Generator,
        gyro_noise_std: float,
        gyro_bias: float,
        yaw_noise_std: float,
    ) -> None:
        super().__init__(rng)
        self.gyro_noise_std = float(gyro_noise_std)
        self.gyro_bias = float(gyro_bias)
        self.yaw_noise_std = float(yaw_noise_std)

    @classmethod
    def from_config(cls, cfg, rng: np.random.Generator) -> "IMUSensor":
        c = cfg.sensors.imu
        return cls(
            rng=rng,
            gyro_noise_std=float(c.gyro_noise_std),
            gyro_bias=float(c.gyro_bias),
            yaw_noise_std=float(c.yaw_noise_std),
        )

    def measure(
        self,
        robot: DifferentialDriveRobot,
        environment: Environment2D,
        dt: float,
    ) -> IMUMeasurement:
        true_yaw = robot.state.theta
        true_yaw_rate = robot.state.omega

        yaw_meas = normalize_angle(
            true_yaw + self._rng.normal(0.0, self.yaw_noise_std)
        )
        gyro_meas = (
            true_yaw_rate
            + self.gyro_bias
            + self._rng.normal(0.0, self.gyro_noise_std)
        )
        return IMUMeasurement(yaw=yaw_meas, yaw_rate=gyro_meas)


# =============================================================================
# Tekerlek enkoderi
# =============================================================================

@dataclass
class EncoderMeasurement:
    """Tekerlek enkoder okuması.

    Hem ham teker hızları (v_left, v_right) hem de bunlardan türetilen
    gövde-çerçevesi hızları (v, omega) sağlanır. Dead reckoning ve EKF
    aşamasında her ikisi de kullanılabilir.
    """

    v_left: float
    v_right: float
    v: float
    omega: float


class WheelEncoder(BaseSensor):
    """Sol/sağ tekerin doğrusal hızını ölçer, Gauss gürültüsü uygular."""

    def __init__(
        self,
        rng: np.random.Generator,
        velocity_noise_std: float,
        omega_noise_std: float,
        quantization: float = 0.0,
    ) -> None:
        super().__init__(rng)
        self.velocity_noise_std = float(velocity_noise_std)
        self.omega_noise_std = float(omega_noise_std)
        self.quantization = float(quantization)

    @classmethod
    def from_config(cls, cfg, rng: np.random.Generator) -> "WheelEncoder":
        c = cfg.sensors.encoder
        return cls(
            rng=rng,
            velocity_noise_std=float(c.velocity_noise_std),
            omega_noise_std=float(c.omega_noise_std),
            quantization=float(c.get("quantization", 0.0)),
        )

    def measure(
        self,
        robot: DifferentialDriveRobot,
        environment: Environment2D,
        dt: float,
    ) -> EncoderMeasurement:
        true_vl, true_vr = robot.wheel_speeds()
        vl = true_vl + self._rng.normal(0.0, self.velocity_noise_std)
        vr = true_vr + self._rng.normal(0.0, self.velocity_noise_std)

        if self.quantization > 0.0:
            vl = round(vl / self.quantization) * self.quantization
            vr = round(vr / self.quantization) * self.quantization

        # Gövde hızlarını teker hızlarından geri çıkar; üzerine
        # bağımsız bir küçük omega gürültüsü ekle (yaprak yumuşatma)
        L = robot.wheel_base
        v_body = 0.5 * (vr + vl)
        omega_body = (vr - vl) / L + self._rng.normal(0.0, self.omega_noise_std)

        return EncoderMeasurement(
            v_left=vl, v_right=vr, v=v_body, omega=omega_body
        )


__all__ = [
    "BaseSensor",
    "LiDARSensor",
    "LidarScan",
    "IMUSensor",
    "IMUMeasurement",
    "WheelEncoder",
    "EncoderMeasurement",
]
