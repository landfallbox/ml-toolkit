"""
训练框架库
包含通用的训练器基类和训练流程
"""

from .trainer import Trainer
from .lstm_trainer import LSTMTrainer

__all__ = ["Trainer", "LSTMTrainer"]
