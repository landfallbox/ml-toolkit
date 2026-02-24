"""
流式阈值优化模块 - 在线动态调整异常判断阈值
"""

from __future__ import annotations

from collections import deque

import numpy as np


class StreamingThresholdOptimizer:
    """在线流式阈值优化。"""

    def __init__(
        self,
        local_window_size: int = 100,
        global_ema_decay: float = 0.01,
        alpha: float = 0.7,
        min_samples_for_optimization: int = 50,
    ):
        self.local_window_size = local_window_size
        self.global_ema_decay = global_ema_decay
        self.alpha = alpha
        self.min_samples_for_optimization = min_samples_for_optimization

        self.local_score_buffer = deque(maxlen=local_window_size)
        self.local_decision_buffer = deque(maxlen=local_window_size)

        self.global_threshold = 0.5
        self.local_threshold = 0.5
        self.adaptive_threshold = 0.5

        self.sample_count = 0
        self.f1_history = deque(maxlen=1000)

    def _compute_f1_score(self, scores: np.ndarray, decisions: np.ndarray, threshold: float) -> float:
        if len(scores) == 0:
            return 0.0

        predictions = (scores >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (decisions == 1))
        fp = np.sum((predictions == 1) & (decisions == 0))
        fn = np.sum((predictions == 0) & (decisions == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return float(f1)

    def _optimize_threshold_binary_search(self) -> float:
        if len(self.local_score_buffer) < self.min_samples_for_optimization:
            return self.local_threshold

        scores = np.array(list(self.local_score_buffer))
        decisions = np.array(list(self.local_decision_buffer))

        candidates = np.percentile(scores, np.linspace(10, 90, 20))
        best_f1 = 0.0
        best_threshold = float(np.median(scores))

        for threshold in candidates:
            f1 = self._compute_f1_score(scores, decisions, threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

        self.f1_history.append(best_f1)
        return best_threshold

    def update(self, score: float, decision: int, perform_optimization: bool = True) -> None:
        self.sample_count += 1

        self.local_score_buffer.append(score)
        self.local_decision_buffer.append(decision)

        if perform_optimization and self.sample_count % 10 == 0:
            self.local_threshold = self._optimize_threshold_binary_search()

        if self.sample_count == 1:
            self.global_threshold = self.local_threshold
        else:
            self.global_threshold = (
                (1 - self.global_ema_decay) * self.global_threshold
                + self.global_ema_decay * self.local_threshold
            )

        self.adaptive_threshold = (
            self.alpha * self.local_threshold + (1 - self.alpha) * self.global_threshold
        )

    def get_adaptive_threshold(self) -> float:
        return self.adaptive_threshold

    def get_local_threshold(self) -> float:
        return self.local_threshold

    def get_global_threshold(self) -> float:
        return self.global_threshold

    def get_statistics(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "adaptive_threshold": self.adaptive_threshold,
            "local_threshold": self.local_threshold,
            "global_threshold": self.global_threshold,
            "local_buffer_size": len(self.local_score_buffer),
            "recent_f1": float(self.f1_history[-1]) if len(self.f1_history) > 0 else 0.0,
            "avg_f1": float(np.mean(list(self.f1_history))) if len(self.f1_history) > 0 else 0.0,
        }
