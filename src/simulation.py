"""Simülasyon orkestratörü.

``SimulationRunner`` tüm modülleri (ortam, robot, sensörler, lokalizasyon,
LiDAR işleme) bir araya getirip zaman adımlarını yönetir. Bu adımda
(Adım 2) navigasyon henüz yoktur; bunun yerine konfigürasyondan okunan
açık-döngü komutlar (sabit hız + hafif yay) uygulanır. Adım 3'te aynı
arayüzü kullanarak gerçek bir navigasyon kontrolcüsü yerine geçecektir.

Çıktı kayıtları üç eksende üretilir:

- Yörünge tablosu (zaman, gerçek poz, dead reckoning pozu, EKF pozu,
  EKF kovaryans köşegeni)
- LiDAR özet bilgileri (her adımda küme sayısı, en yakın engel mesafesi)
- Lokalizasyon metrikleri (RMSE/MAE — sonunda hesaplanır)
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .environment import Environment2D
from .lidar_processing import LidarProcessor, LidarProcessingResult
from .localization import DeadReckoning, ExtendedKalmanFilter
from .logger import ExperimentLogger
from .metrics import MetricsAnalyzer
from .robot import DifferentialDriveRobot
from .sensors import IMUSensor, LiDARSensor, WheelEncoder
from .utils import RandomContext, ensure_dir


ControlFn = Callable[[DifferentialDriveRobot, Environment2D, "SimulationRunner"], Tuple[float, float]]


# -----------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Tek bir simülasyon adımının düz kaydı."""

    t: float
    true_x: float
    true_y: float
    true_theta: float
    dr_x: float
    dr_y: float
    dr_theta: float
    ekf_x: float
    ekf_y: float
    ekf_theta: float
    ekf_var_x: float
    ekf_var_y: float
    ekf_var_theta: float
    enc_v: float
    enc_omega: float
    imu_yaw: float
    n_lidar_clusters: int
    min_obstacle_range: float


# -----------------------------------------------------------------------------

class SimulationRunner:
    """Tüm modülleri orkestre eden simülasyon koşucusu."""

    def __init__(
        self,
        cfg,
        logger: Optional[ExperimentLogger] = None,
        control_fn: Optional[ControlFn] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or ExperimentLogger(
            name="autonav.sim", log_file=cfg.output.log_file
        )
        self.control_fn = control_fn or self._default_open_loop_control

        # Rastgelelik
        seed = int(cfg.experiment.random_seed)
        self.rng_ctx = RandomContext(seed=seed)

        # Modüller
        self.env = Environment2D.build(cfg, self.rng_ctx.child("environment"))
        self.robot = DifferentialDriveRobot.from_config(cfg)
        self.lidar = LiDARSensor.from_config(cfg, self.rng_ctx.child("lidar"))
        self.imu = IMUSensor.from_config(cfg, self.rng_ctx.child("imu"))
        self.encoder = WheelEncoder.from_config(cfg, self.rng_ctx.child("encoder"))
        self.lidar_proc = LidarProcessor.from_config(cfg)
        self.dr = DeadReckoning.from_config(cfg)
        self.ekf = ExtendedKalmanFilter.from_config(cfg)

        self.dt = float(cfg.simulation.dt)
        self.max_steps = int(math.ceil(float(cfg.simulation.max_time) / self.dt))
        self.stop_on_collision = bool(cfg.simulation.get("stop_on_collision", True))

        # Açık-döngü komutları
        ol = cfg.simulation.get("open_loop", {})
        self._ol_v = float(ol.get("v", 0.6)) if ol else 0.6
        self._ol_omega = float(ol.get("omega", 0.1)) if ol else 0.1

        # Kayıtlar
        self.records: List[StepRecord] = []
        self.last_lidar_result: Optional[LidarProcessingResult] = None

        # LiDAR snapshot (görselleştirme için)
        self.lidar_snapshot_t = float(cfg.simulation.get("lidar_snapshot_t", 5.0))
        self.lidar_snapshot_step = max(0, int(round(self.lidar_snapshot_t / self.dt)))
        self.lidar_snapshot: Optional[dict] = None

        # Çalışma durumu
        self.terminated_reason: str = "max_time"
        self.elapsed_seconds: float = 0.0

        # Navigator (varsa) — main.py tarafından bağlanır
        self.navigator = None

    # ------------------------------------------------------------------ varsayılan kontrolcü

    def _default_open_loop_control(
        self, robot: DifferentialDriveRobot, env: Environment2D, _runner
    ) -> Tuple[float, float]:
        return self._ol_v, self._ol_omega

    # ------------------------------------------------------------------ ana döngü

    def run(self) -> List[StepRecord]:
        self.logger.section(f"Simülasyon koşusu: {self.cfg.experiment.name}")
        self.logger.info(
            "dt=%.3fs, max_steps=%d, sensörler=[LiDAR, IMU, Encoder]",
            self.dt, self.max_steps,
        )

        t0 = time.time()
        for k in range(self.max_steps):
            t = k * self.dt

            # 1) Kontrol
            v_cmd, w_cmd = self.control_fn(self.robot, self.env, self)

            # 2) Gerçek dinamik
            self.robot.step(v_cmd, w_cmd, self.dt, self.env)
            if self.robot.has_collided and self.stop_on_collision:
                self.terminated_reason = "collision"
                self.logger.warning("Çarpışma — simülasyon t=%.2fs'de durduruldu.", t)
                break

            # 3) Sensör ölçümleri
            enc_meas = self.encoder.measure(self.robot, self.env, self.dt)
            imu_meas = self.imu.measure(self.robot, self.env, self.dt)
            scan = self.lidar.measure(self.robot, self.env, self.dt)

            # 4) Dead reckoning (encoder ile)
            self.dr.update(enc_meas.v, enc_meas.omega, self.dt)

            # 5) EKF: predict (encoder) + update (IMU yaw)
            self.ekf.predict(enc_meas.v, enc_meas.omega, self.dt)
            self.ekf.update_yaw(imu_meas.yaw)
            self.ekf.commit()

            # 6) LiDAR işleme — şimdilik tahmin için kullanılmıyor;
            # özet metriklerle kaydedilir, Adım 3'te navigasyona beslenecek
            lidar_result = self.lidar_proc.process(scan, self.robot.pose)
            self.last_lidar_result = lidar_result

            # Snapshot
            if self.lidar_snapshot is None and k >= self.lidar_snapshot_step:
                self.lidar_snapshot = {
                    "step": k,
                    "t": t,
                    "pose": self.robot.pose,
                    "ekf_pose": self.ekf.pose,
                    "scan": scan,
                    "processing": lidar_result,
                }

            min_obs_range = (
                float(np.min(scan.ranges[scan.hits])) if scan.hits.any() else float("nan")
            )

            # 7) Kayıt
            true_pose = self.robot.pose
            dr_pose = self.dr.pose
            ekf_pose = self.ekf.pose
            cov = np.diag(self.ekf.P)
            self.records.append(StepRecord(
                t=t,
                true_x=true_pose[0], true_y=true_pose[1], true_theta=true_pose[2],
                dr_x=dr_pose[0], dr_y=dr_pose[1], dr_theta=dr_pose[2],
                ekf_x=ekf_pose[0], ekf_y=ekf_pose[1], ekf_theta=ekf_pose[2],
                ekf_var_x=float(cov[0]), ekf_var_y=float(cov[1]), ekf_var_theta=float(cov[2]),
                enc_v=enc_meas.v, enc_omega=enc_meas.omega,
                imu_yaw=imu_meas.yaw,
                n_lidar_clusters=lidar_result.n_clusters,
                min_obstacle_range=min_obs_range,
            ))

            # 8) Başarı koşulu
            if self.env.goal_reached(true_pose[0], true_pose[1]):
                self.terminated_reason = "goal_reached"
                self.logger.info("Hedefe ulaşıldı, t=%.2fs.", t)
                break
        else:
            self.terminated_reason = "max_time"

        self.elapsed_seconds = time.time() - t0
        self.logger.info(
            "Koşum tamamlandı: %d adım, %.3fs, sebep=%s",
            len(self.records), self.elapsed_seconds, self.terminated_reason,
        )
        return self.records

    # ------------------------------------------------------------------ analiz

    def trajectories(self) -> Dict[str, np.ndarray]:
        """Gerçek, DR ve EKF yörüngelerini eş uzunlukta diziler halinde döner."""
        if not self.records:
            empty = np.zeros((0, 3))
            return {"true": empty, "dr": empty, "ekf": empty}
        true_t = np.array([[r.true_x, r.true_y, r.true_theta] for r in self.records])
        dr_t = np.array([[r.dr_x, r.dr_y, r.dr_theta] for r in self.records])
        ekf_t = np.array([[r.ekf_x, r.ekf_y, r.ekf_theta] for r in self.records])
        return {"true": true_t, "dr": dr_t, "ekf": ekf_t}

    def compute_metrics(self) -> Dict[str, dict]:
        traj = self.trajectories()
        if traj["true"].shape[0] == 0:
            return {}
        dr_m = MetricsAnalyzer.evaluate(traj["true"], traj["dr"], name="dead_reckoning")
        ekf_m = MetricsAnalyzer.evaluate(traj["true"], traj["ekf"], name="ekf")
        return {
            "dead_reckoning": dr_m.to_dict(),
            "ekf": ekf_m.to_dict(),
        }

    # ------------------------------------------------------------------ kayıt

    def save_outputs(self) -> Dict[str, str]:
        results_dir = ensure_dir(self.cfg.output.results_dir)
        traj_csv = results_dir / "trajectory_data.csv"
        metrics_csv = results_dir / "metrics.csv"
        summary_json = results_dir / "experiment_summary.json"

        # trajectory_data.csv
        with traj_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "t", "true_x", "true_y", "true_theta",
                "dr_x", "dr_y", "dr_theta",
                "ekf_x", "ekf_y", "ekf_theta",
                "ekf_var_x", "ekf_var_y", "ekf_var_theta",
                "enc_v", "enc_omega", "imu_yaw",
                "n_lidar_clusters", "min_obstacle_range",
            ])
            for r in self.records:
                w.writerow([
                    f"{r.t:.4f}",
                    f"{r.true_x:.5f}", f"{r.true_y:.5f}", f"{r.true_theta:.5f}",
                    f"{r.dr_x:.5f}", f"{r.dr_y:.5f}", f"{r.dr_theta:.5f}",
                    f"{r.ekf_x:.5f}", f"{r.ekf_y:.5f}", f"{r.ekf_theta:.5f}",
                    f"{r.ekf_var_x:.6e}", f"{r.ekf_var_y:.6e}", f"{r.ekf_var_theta:.6e}",
                    f"{r.enc_v:.5f}", f"{r.enc_omega:.5f}", f"{r.imu_yaw:.5f}",
                    r.n_lidar_clusters,
                    f"{r.min_obstacle_range:.5f}" if not math.isnan(r.min_obstacle_range) else "",
                ])

        # metrics.csv
        metrics = self.compute_metrics()
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "estimator",
                "position_rmse_m", "position_mae_m",
                "heading_rmse_rad", "heading_mae_rad",
                "final_position_error_m", "final_heading_error_rad",
            ])
            for name, m in metrics.items():
                w.writerow([
                    name,
                    f"{m['position_rmse']:.5f}", f"{m['position_mae']:.5f}",
                    f"{m['heading_rmse']:.5f}", f"{m['heading_mae']:.5f}",
                    f"{m['final_position_error']:.5f}", f"{m['final_heading_error']:.5f}",
                ])

        # planned_path.csv (varsa)
        nav_info = {}
        if self.navigator is not None and self.navigator.path:
            path_csv = results_dir / "planned_path.csv"
            with path_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["idx", "x", "y"])
                for i, p in enumerate(self.navigator.path):
                    w.writerow([i, f"{float(p[0]):.5f}", f"{float(p[1]):.5f}"])
            self.logger.info("Planlanan yol: %s (%d waypoint)", path_csv, len(self.navigator.path))
            nav_info = {
                "planner": "RRT",
                "waypoint_count": len(self.navigator.path),
                "rrt_iterations": int(self.navigator.planner.iterations_used),
                "rrt_tree_size": int(self.navigator.planner.last_tree_size),
                "stuck_detected": bool(self.navigator.stuck),
            }

        # experiment_summary.json
        true_pose = self.robot.pose
        summary = {
            "experiment": self.cfg.experiment.name,
            "random_seed": int(self.cfg.experiment.random_seed),
            "config_path": str(self.cfg.get("__path__", "config/config.yaml")),
            "n_obstacles": len(self.env.obstacles),
            "env_size": [self.env.width, self.env.height],
            "start": list(self.env.start),
            "goal": list(self.env.goal),
            "executed_steps": len(self.records),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "terminated_reason": self.terminated_reason,
            "collision_detected": bool(self.robot.has_collided),
            "goal_reached": self.terminated_reason == "goal_reached",
            "final_true_pose": {
                "x": round(true_pose[0], 4),
                "y": round(true_pose[1], 4),
                "theta_rad": round(true_pose[2], 4),
            },
            "metrics": metrics,
            "navigation": nav_info,
            "sensor_noise": {
                "lidar_range_sigma": float(self.cfg.sensors.lidar.range_noise_std),
                "imu_yaw_sigma": float(self.cfg.sensors.imu.yaw_noise_std),
                "encoder_velocity_sigma": float(self.cfg.sensors.encoder.velocity_noise_std),
            },
        }
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info("Yörünge kaydı: %s", traj_csv)
        self.logger.info("Metrikler:    %s", metrics_csv)
        self.logger.info("Özet:         %s", summary_json)
        return {
            "trajectory_csv": str(traj_csv),
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
        }


__all__ = ["SimulationRunner", "StepRecord", "ControlFn"]
