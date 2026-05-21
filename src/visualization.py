"""Yayın kalitesinde simülasyon görselleştirmeleri.

Bu modül ödevin 6. bölümünde istenen tüm grafikleri üretir. Her grafikte
başlık, eksen etiketleri, birimler ve legend bulunur; renk paleti tüm
çıktılar boyunca tutarlıdır. Bütün dosyalar ``outputs/figures/`` altına
PNG olarak kaydedilir.

Üretilen görseller:

- ``environment_map.png`` — 2B ortam, engeller, başlangıç ve hedef
- ``trajectory_comparison.png`` — planlanan, gerçek, dead reckoning ve EKF
- ``lidar_raw_filtered.png`` — bir taramanın ham ve filtrelenmiş hali
- ``lidar_clusters.png`` — aynı tarama için kümeler ve centroidler
- ``localization_errors.png`` — x, y, θ hatalarının zaman serisi
- ``rmse_mae_summary.png`` — DR ve EKF için RMSE/MAE karşılaştırması
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .environment import CircleObstacle, Environment2D, RectObstacle
from .utils import normalize_angle_array


# Renk paleti (tüm grafiklerde aynı)
COLOR_TRUE = "black"
COLOR_DR = "#ff7f0e"
COLOR_EKF = "#2ca02c"
COLOR_PATH = "#1f77b4"
COLOR_START = "#2ca02c"
COLOR_GOAL = "#d62728"
COLOR_OBSTACLE = "#5a5a5a"


class SimulationVisualizer:
    """Tüm görselleri tek bir yerden üreten yardımcı sınıf."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.figures_dir = Path(cfg.output.figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
        })

    # ------------------------------------------------------------------ ortak

    @staticmethod
    def _draw_obstacles(ax, env: Environment2D) -> None:
        for ob in env.obstacles:
            if isinstance(ob, RectObstacle):
                ax.add_patch(mpatches.Rectangle(
                    (ob.x, ob.y), ob.width, ob.height,
                    facecolor=COLOR_OBSTACLE, edgecolor="black",
                    linewidth=0.8, alpha=0.85,
                ))
            elif isinstance(ob, CircleObstacle):
                ax.add_patch(mpatches.Circle(
                    (ob.cx, ob.cy), ob.radius,
                    facecolor=COLOR_OBSTACLE, edgecolor="black",
                    linewidth=0.8, alpha=0.85,
                ))

    @staticmethod
    def _setup_world_ax(ax, env: Environment2D) -> None:
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3, linestyle=":")

    @staticmethod
    def _draw_robot_marker(ax, pose, radius: float, color: str = COLOR_START) -> None:
        x, y, theta = pose
        ax.plot(x, y, "o", color=color, markersize=8, zorder=4)
        ax.add_patch(mpatches.Circle(
            (x, y), radius, fill=False, edgecolor=color, linewidth=1.2, zorder=4,
        ))
        ax.arrow(
            x, y, 0.9 * np.cos(theta), 0.9 * np.sin(theta),
            head_width=0.35, head_length=0.35, fc=color, ec=color,
            length_includes_head=True, zorder=5,
        )

    def _save(self, fig, name: str) -> Path:
        out = self.figures_dir / name
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ------------------------------------------------------------------ 1) ortam

    def environment_map(self, env: Environment2D, robot_radius: float) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_obstacles(ax, env)

        sx, sy = env.start
        gx, gy = env.goal
        ax.plot(sx, sy, "o", color=COLOR_START, markersize=12, label="Başlangıç", zorder=5)
        ax.plot(gx, gy, "*", color=COLOR_GOAL, markersize=18, label="Hedef", zorder=5)

        ax.add_patch(mpatches.Circle(
            (gx, gy), env.goal_tolerance, fill=False,
            edgecolor=COLOR_GOAL, linestyle="--", linewidth=1.0,
        ))
        ax.add_patch(mpatches.Circle(
            (sx, sy), robot_radius, fill=False,
            edgecolor=COLOR_START, linestyle=":", linewidth=1.2,
        ))

        self._setup_world_ax(ax, env)
        ax.set_title("Ortam Haritası — 2B Yerleşim, Engeller, Başlangıç ve Hedef")
        ax.legend(loc="upper left", framealpha=0.92)
        return self._save(fig, "environment_map.png")

    # ------------------------------------------------------------------ 2) yörünge

    def trajectory_comparison(
        self,
        env: Environment2D,
        runner,
        navigator,
    ) -> Path:
        trajs = runner.trajectories()
        true_t = trajs["true"]
        dr_t = trajs["dr"]
        ekf_t = trajs["ekf"]
        path_arr = navigator.path_as_array() if navigator is not None else np.zeros((0, 2))

        fig, ax = plt.subplots(figsize=(11, 6.5))
        self._draw_obstacles(ax, env)

        if path_arr.shape[0] > 0:
            ax.plot(
                path_arr[:, 0], path_arr[:, 1],
                color=COLOR_PATH, linestyle="--", linewidth=2.0, alpha=0.85,
                label=f"Planlanan yol (RRT, {path_arr.shape[0]} waypoint)",
                zorder=2,
            )
            ax.scatter(
                path_arr[:, 0], path_arr[:, 1],
                marker="s", color=COLOR_PATH, s=40, zorder=3,
            )

        if true_t.shape[0] > 0:
            ax.plot(
                true_t[:, 0], true_t[:, 1],
                color=COLOR_TRUE, linewidth=2.0, label="Gerçek yol", zorder=3,
            )
        if dr_t.shape[0] > 0:
            ax.plot(
                dr_t[:, 0], dr_t[:, 1],
                color=COLOR_DR, linestyle="--", linewidth=1.6,
                label="Dead Reckoning", zorder=3,
            )
        if ekf_t.shape[0] > 0:
            ax.plot(
                ekf_t[:, 0], ekf_t[:, 1],
                color=COLOR_EKF, linestyle=":", linewidth=1.8,
                label="EKF füzyonu", zorder=3,
            )

        sx, sy = env.start
        gx, gy = env.goal
        ax.plot(sx, sy, "o", color=COLOR_START, markersize=12, label="Başlangıç", zorder=5)
        ax.plot(gx, gy, "*", color=COLOR_GOAL, markersize=18, label="Hedef", zorder=5)

        self._setup_world_ax(ax, env)
        ax.set_title("Yörünge Karşılaştırması — Planlanan, Gerçek, Dead Reckoning ve EKF")
        ax.legend(loc="lower right", framealpha=0.92)
        return self._save(fig, "trajectory_comparison.png")

    # ------------------------------------------------------------------ 3) LiDAR ham/filtre

    def lidar_raw_filtered(
        self, env: Environment2D, snapshot: Optional[dict], robot_radius: float
    ) -> Optional[Path]:
        if snapshot is None:
            return None

        pose = snapshot["pose"]
        result = snapshot["processing"]
        scan = snapshot["scan"]
        t = snapshot["t"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax in axes:
            self._draw_obstacles(ax, env)
            self._setup_world_ax(ax, env)
            self._draw_robot_marker(ax, pose, robot_radius)

        raw_pts = result.raw_points
        if raw_pts.shape[0] > 0:
            axes[0].scatter(
                raw_pts[:, 0], raw_pts[:, 1],
                s=10, color=COLOR_GOAL, alpha=0.75,
                label=f"Ham LiDAR (n={raw_pts.shape[0]})",
            )
        axes[0].set_title(
            f"Ham LiDAR Taraması — t = {t:.2f} s, max menzil = {scan.max_range:.1f} m"
        )
        axes[0].legend(loc="upper right", framealpha=0.92)

        filt_pts = result.filtered_points
        if filt_pts.shape[0] > 0:
            axes[1].scatter(
                filt_pts[:, 0], filt_pts[:, 1],
                s=10, color=COLOR_PATH, alpha=0.75,
                label=f"Filtrelenmiş (n={filt_pts.shape[0]})",
            )
        win = int(self.cfg.lidar_processing.median_window)
        axes[1].set_title(f"Filtrelenmiş LiDAR — medyan pencere = {win}")
        axes[1].legend(loc="upper right", framealpha=0.92)

        return self._save(fig, "lidar_raw_filtered.png")

    # ------------------------------------------------------------------ 4) LiDAR kümeleri

    def lidar_clusters(
        self, env: Environment2D, snapshot: Optional[dict], robot_radius: float
    ) -> Optional[Path]:
        if snapshot is None:
            return None

        pose = snapshot["pose"]
        result = snapshot["processing"]
        t = snapshot["t"]

        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_obstacles(ax, env)
        self._setup_world_ax(ax, env)
        self._draw_robot_marker(ax, pose, robot_radius)

        cmap = plt.get_cmap("tab10")
        for i, cluster in enumerate(result.clusters):
            color = cmap(i % 10)
            ax.scatter(
                cluster.points[:, 0], cluster.points[:, 1],
                s=18, color=color, alpha=0.9, edgecolors="none",
                label=f"Küme {i + 1} (n={cluster.size})",
            )
            cx, cy = cluster.centroid
            ax.plot(cx, cy, "x", color=color, markersize=11, markeredgewidth=2.0)

        ax.set_title(
            f"LiDAR Kümeleme — {result.n_clusters} küme  (t = {t:.2f} s, "
            f"ε = {self.cfg.lidar_processing.cluster_eps} m, "
            f"n_min = {self.cfg.lidar_processing.min_cluster_size})"
        )
        if result.n_clusters > 0:
            ax.legend(
                loc="upper right", framealpha=0.92,
                ncol=2 if result.n_clusters > 4 else 1,
                fontsize=9,
            )
        return self._save(fig, "lidar_clusters.png")

    # ------------------------------------------------------------------ 5) hata zaman serisi

    def localization_errors(self, runner) -> Optional[Path]:
        recs = runner.records
        if not recs:
            return None

        t = np.array([r.t for r in recs])
        true = np.array([[r.true_x, r.true_y, r.true_theta] for r in recs])
        dr = np.array([[r.dr_x, r.dr_y, r.dr_theta] for r in recs])
        ekf = np.array([[r.ekf_x, r.ekf_y, r.ekf_theta] for r in recs])

        ex_dr = dr[:, 0] - true[:, 0]
        ey_dr = dr[:, 1] - true[:, 1]
        et_dr = normalize_angle_array(dr[:, 2] - true[:, 2])

        ex_ekf = ekf[:, 0] - true[:, 0]
        ey_ekf = ekf[:, 1] - true[:, 1]
        et_ekf = normalize_angle_array(ekf[:, 2] - true[:, 2])

        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

        for ax, dr_e, ekf_e, ylabel in (
            (axes[0], ex_dr, ex_ekf, "x hatası [m]"),
            (axes[1], ey_dr, ey_ekf, "y hatası [m]"),
            (axes[2], et_dr, et_ekf, r"$\theta$ hatası [rad]"),
        ):
            ax.plot(t, dr_e, color=COLOR_DR, linewidth=1.0, label="Dead Reckoning")
            ax.plot(t, ekf_e, color=COLOR_EKF, linewidth=1.4, label="EKF Füzyonu")
            ax.axhline(0.0, color="k", linewidth=0.6)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, linestyle=":")

        axes[0].set_title("Zaman Boyunca Lokalizasyon Hataları (Gerçek − Tahmin)")
        axes[0].legend(loc="upper left", framealpha=0.92)
        axes[2].set_xlabel("Zaman [s]")

        return self._save(fig, "localization_errors.png")

    # ------------------------------------------------------------------ 6) RMSE/MAE özet

    def rmse_mae_summary(self, metrics: Dict[str, dict]) -> Optional[Path]:
        if not metrics:
            return None

        order = ["dead_reckoning", "ekf"]
        order = [k for k in order if k in metrics]
        display_names = {"dead_reckoning": "Dead Reckoning", "ekf": "EKF Füzyonu"}
        labels = [display_names[k] for k in order]

        pos_rmse = [metrics[k]["position_rmse"] for k in order]
        pos_mae = [metrics[k]["position_mae"] for k in order]
        head_rmse = [metrics[k]["heading_rmse"] for k in order]
        head_mae = [metrics[k]["heading_mae"] for k in order]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(labels), dtype=float)
        w = 0.36

        def _annotate(ax, xs, ys, fmt="{:.3f}"):
            for xi, yi in zip(xs, ys):
                ax.annotate(
                    fmt.format(yi), (xi, yi),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points",
                )

        axes[0].bar(x - w / 2, pos_rmse, w, label="RMSE", color=COLOR_PATH)
        axes[0].bar(x + w / 2, pos_mae, w, label="MAE", color=COLOR_DR)
        _annotate(axes[0], x - w / 2, pos_rmse)
        _annotate(axes[0], x + w / 2, pos_mae)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Konum hatası [m]")
        axes[0].set_title("Konum Hatası — RMSE ve MAE")
        axes[0].legend(framealpha=0.92)
        axes[0].grid(True, axis="y", alpha=0.3, linestyle=":")

        axes[1].bar(x - w / 2, head_rmse, w, label="RMSE", color=COLOR_PATH)
        axes[1].bar(x + w / 2, head_mae, w, label="MAE", color=COLOR_DR)
        _annotate(axes[1], x - w / 2, head_rmse, fmt="{:.4f}")
        _annotate(axes[1], x + w / 2, head_mae, fmt="{:.4f}")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Yön hatası [rad]")
        axes[1].set_title("Yön Hatası — RMSE ve MAE")
        axes[1].legend(framealpha=0.92)
        axes[1].grid(True, axis="y", alpha=0.3, linestyle=":")

        return self._save(fig, "rmse_mae_summary.png")

    # ------------------------------------------------------------------ orkestratör

    def render_all(
        self,
        env: Environment2D,
        runner,
        navigator,
        metrics: Dict[str, dict],
        robot_radius: float,
    ) -> Dict[str, str]:
        outputs: Dict[str, Optional[Path]] = {
            "environment_map": self.environment_map(env, robot_radius),
            "trajectory_comparison": self.trajectory_comparison(env, runner, navigator),
            "lidar_raw_filtered": self.lidar_raw_filtered(env, runner.lidar_snapshot, robot_radius),
            "lidar_clusters": self.lidar_clusters(env, runner.lidar_snapshot, robot_radius),
            "localization_errors": self.localization_errors(runner),
            "rmse_mae_summary": self.rmse_mae_summary(metrics),
        }
        return {k: str(v) for k, v in outputs.items() if v is not None}


__all__ = ["SimulationVisualizer"]
