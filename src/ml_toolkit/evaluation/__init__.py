"""
评估框架库
包含通用的评估器基类和评估流程
"""

from .evaluator import Evaluator
from .lstm_evaluator import LSTMEvaluator
from .metrics import (
    calculate_accuracy,
    calculate_f1,
    calculate_loss,
    calculate_mae,
    calculate_mape,
    calculate_precision,
    calculate_r2_score,
    calculate_recall,
    calculate_rmse,
)
from .event_gate_metrics import (
    check_trigger_rate_constraint,
    compute_delta_violation_time_pct,
    compute_ppr,
    summarize_event_gate_constraints,
)

__all__ = [
    "Evaluator",
    "LSTMEvaluator",
    "calculate_accuracy",
    "calculate_f1",
    "calculate_loss",
    "calculate_mae",
    "calculate_mape",
    "calculate_precision",
    "calculate_r2_score",
    "calculate_recall",
    "calculate_rmse",
    "compute_ppr",
    "compute_delta_violation_time_pct",
    "check_trigger_rate_constraint",
    "summarize_event_gate_constraints",
]
