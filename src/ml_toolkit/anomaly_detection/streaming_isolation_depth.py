"""
流式异常检测模块 - 基于参考集的在线孤立度计算
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.distance import euclidean


class StreamingIsolationDepth:
    """流式异常检测（参考集孤立度）。"""

    def __init__(
        self,
        n_reference_samples: int = 500,
        update_freq: int = 100,
        distance_metric: str = "euclidean",
        contamination: float = 0.2,
        decay_strategy: str = "FIFO",
    ):
        self.n_reference_samples = n_reference_samples
        self.update_freq = update_freq
        self.distance_metric = distance_metric
        self.contamination = contamination
        self.decay_strategy = decay_strategy

        self.reference_buffer = deque(maxlen=n_reference_samples)
        self.reference_array = None

        self.short_buffer = deque(maxlen=100)
        self.medium_buffer = deque(maxlen=1440)
        self.long_buffer = deque(maxlen=10080)

        self.distance_stats = {
            "short": {"mean": 0.0, "std": 1.0},
            "medium": {"mean": 0.0, "std": 1.0},
            "long": {"mean": 0.0, "std": 1.0},
        }

        self.sample_count = 0
        self.covariance_matrix = None

    def _initialize_reference_set(self, initial_data: np.ndarray) -> None:
        indices = np.random.choice(
            len(initial_data),
            size=min(self.n_reference_samples, len(initial_data)),
            replace=False,
        )
        for idx in indices:
            self.reference_buffer.append(initial_data[idx])

        self.reference_array = np.array(list(self.reference_buffer))
        self._update_distance_statistics()

    def _update_reference_set(self, sample: np.ndarray) -> None:
        self.reference_buffer.append(sample)

        if self.sample_count % self.update_freq == 0:
            self.reference_array = np.array(list(self.reference_buffer))
            self._update_distance_statistics()

    def _update_distance_statistics(self) -> None:
        if len(self.reference_buffer) < 2:
            return

        ref_array = np.array(list(self.reference_buffer))
        distances = []

        for i in range(min(100, len(ref_array))):
            for j in range(i + 1, min(100, len(ref_array))):
                distances.append(euclidean(ref_array[i], ref_array[j]))

        if distances:
            for window_name, buffer in [
                ("short", self.short_buffer),
                ("medium", self.medium_buffer),
                ("long", self.long_buffer),
            ]:
                if len(buffer) > 0:
                    window_dists = np.array(list(buffer))
                    self.distance_stats[window_name]["mean"] = float(np.mean(window_dists))
                    self.distance_stats[window_name]["std"] = float(np.std(window_dists)) + 1e-8

    def _compute_isolation_distance(self, sample: np.ndarray, window: str = "long") -> float:
        if self.reference_array is None or len(self.reference_array) == 0:
            return 0.0

        if self.distance_metric in {"euclidean", "mahalanobis"}:
            distances = np.linalg.norm(self.reference_array - sample, axis=1)
        else:
            distances = np.linalg.norm(self.reference_array - sample, axis=1)

        return float(np.mean(distances))

    def update(self, sample: np.ndarray) -> None:
        self.sample_count += 1
        self._update_reference_set(sample)

        self.short_buffer.append(sample)
        self.medium_buffer.append(sample)
        self.long_buffer.append(sample)

    def score(self, sample: np.ndarray, window: str = "long") -> float:
        if self.reference_array is None or len(self.reference_array) == 0:
            return 0.0

        distance = self._compute_isolation_distance(sample, window)

        stats = self.distance_stats[window]
        if stats["std"] > 1e-8:
            normalized_score = (distance - stats["mean"]) / stats["std"]
            anomaly_score = 1.0 / (1.0 + np.exp(-normalized_score))
        else:
            anomaly_score = 0.5

        return float(np.clip(anomaly_score, 0.0, 1.0))

    def score_batch(self, samples: np.ndarray, window: str = "long") -> np.ndarray:
        return np.array([self.score(sample, window) for sample in samples])

    def get_statistics(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "reference_set_size": len(self.reference_buffer),
            "distance_stats": self.distance_stats,
            "short_buffer_size": len(self.short_buffer),
            "medium_buffer_size": len(self.medium_buffer),
            "long_buffer_size": len(self.long_buffer),
        }
