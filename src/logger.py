"""Deneysel loglama altyapısı.

Hem konsola hem de dosyaya yazan bir logger sağlar. Her simülasyon
koşusunun başlangıç zamanı, kullanılan seed, sensör gürültü parametreleri,
hedefe ulaşma durumu ve nihai hata metrikleri tek bir yerden takip edilir.
Bu sayede rapor metnindeki her sayı, çıktı dosyalarından kanıtlanabilir
biçimde geri izlenebilir.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ExperimentLogger:
    """Konsol + dosya hibrit logger.

    Aynı süreçte birden fazla kez kurulduğunda handler tekrarına karşı
    korumalıdır. Log formatı zaman damgalı ve seviyelidir; debug seviyesi
    sadece dosyaya, info ve üstü hem konsola hem dosyaya yazılır.
    """

    _instance: Optional["ExperimentLogger"] = None

    def __init__(self, name: str = "autonav", log_file: Optional[str] = None) -> None:
        self.name = name
        self.log_file = log_file
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        if not self._logger.handlers:
            self._attach_handlers(log_file)

    def _attach_handlers(self, log_file: Optional[str]) -> None:
        fmt = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Konsol
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)

        # Dosya
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

    # ------------------------------------------------------------------ API

    def info(self, msg: str, *args) -> None:
        self._logger.info(msg, *args)

    def debug(self, msg: str, *args) -> None:
        self._logger.debug(msg, *args)

    def warning(self, msg: str, *args) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        self._logger.error(msg, *args)

    def section(self, title: str) -> None:
        """Görsel olarak ayrılmış bir bölüm başlığı yazar."""
        bar = "=" * max(8, len(title) + 4)
        self._logger.info(bar)
        self._logger.info("  %s", title)
        self._logger.info(bar)

    def run_header(self, experiment_name: str, seed: int) -> None:
        self.section(f"Simülasyon başlatılıyor: {experiment_name}")
        self.info("Zaman: %s", datetime.now().isoformat(timespec="seconds"))
        self.info("Random seed: %d", seed)


__all__ = ["ExperimentLogger"]
