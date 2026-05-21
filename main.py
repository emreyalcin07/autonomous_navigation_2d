"""Simülasyon giriş noktası.

Bu betik konfigürasyon dosyasını okur, ``SimulationRunner`` ile tüm
modülleri ayağa kaldırır, RRT yolunu hesaplar, simülasyonu çalıştırır
ve son olarak ``SimulationVisualizer`` ile tüm yayın kalitesi
görselleri üretir. Pipeline tek bir komutla baştan sona koşar:

    python main.py --config config/config.yaml
"""

from __future__ import annotations

import argparse

from src.logger import ExperimentLogger
from src.navigation import HybridNavigator
from src.simulation import SimulationRunner
from src.utils import ConfigLoader, ensure_dir
from src.visualization import SimulationVisualizer


def run(config_path: str) -> dict:
    cfg = ConfigLoader.load(config_path)
    ensure_dir(cfg.output.figures_dir)
    ensure_dir(cfg.output.results_dir)

    logger = ExperimentLogger(name="autonav.main", log_file=cfg.output.log_file)
    logger.run_header(cfg.experiment.name, int(cfg.experiment.random_seed))

    # 1) Simülasyon çekirdeği (ortam + robot + sensörler + lokalizasyon)
    runner = SimulationRunner(cfg=cfg, logger=logger)

    # 2) Hibrit navigator: RRT global + APF lokal
    logger.section("Yol planlama (RRT)")
    navigator = HybridNavigator(
        env=runner.env, cfg=cfg, rng=runner.rng_ctx.child("rrt"),
    )
    path = navigator.plan(runner.env.start, runner.env.goal)
    logger.info(
        "RRT yolu hazır: ağaç=%d düğüm, %d iter., %d waypoint (smoothing sonrası)",
        navigator.planner.last_tree_size,
        navigator.planner.iterations_used,
        len(path),
    )
    runner.navigator = navigator
    runner.control_fn = navigator.compute_control

    # 3) Simülasyon koşumu
    runner.run()
    runner.save_outputs()

    metrics = runner.compute_metrics()
    if metrics:
        logger.section("Lokalizasyon metrikleri")
        for est_name, m in metrics.items():
            logger.info(
                "%-15s | konum RMSE=%.4f m  MAE=%.4f m  | yön RMSE=%.4f rad  MAE=%.4f rad",
                est_name, m["position_rmse"], m["position_mae"],
                m["heading_rmse"], m["heading_mae"],
            )

    # 4) Görselleştirme
    logger.section("Görselleştirme")
    viz = SimulationVisualizer(cfg)
    figures = viz.render_all(
        env=runner.env,
        runner=runner,
        navigator=navigator,
        metrics=metrics,
        robot_radius=runner.robot.radius,
    )
    for name, path_str in figures.items():
        logger.info("  %-22s -> %s", name, path_str)

    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="2B Otonom Navigasyon Simülasyonu")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
