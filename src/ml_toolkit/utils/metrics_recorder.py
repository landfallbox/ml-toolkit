"""
@Author      : landfallbox
@Date        : 2026/02/04 星期二
@Description : 指标记录器
"""
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class MetricsRecorder:
    """
    指标记录器

    职责：
    - 记录训练和验证指标
    - 保存指标到 JSON 文件
    - 保存训练历史到 CSV 文件
    """

    def __init__(
        self,
        experiment_dir: Path,
        metrics_filename: Optional[str] = None,
        history_filename: Optional[str] = None
    ):
        """
        初始化指标记录器

        参数：
            experiment_dir: 实验目录
            metrics_filename: 指标文件名（可选，如果不提供则使用默认值）
            history_filename: 训练历史文件名（可选，如果不提供则使用默认值）
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 设置文件名
        if metrics_filename is None or history_filename is None:
            from config.common_config import CommonConfig
            if metrics_filename is None:
                metrics_filename = CommonConfig.METRICS_FILENAME
            if history_filename is None:
                history_filename = CommonConfig.TRAINING_HISTORY_FILENAME

        self.metrics_filename = metrics_filename
        self.history_filename = history_filename
        self.metrics = []

    def save_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        保存指标到 metrics.json

        参数：
            metrics: 指标字典
            step: 步骤编号（可选）
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **self._to_native(metrics)
        }

        self.metrics.append(entry)
        metrics_file = self.experiment_dir / self.metrics_filename
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

    def load_metrics(self) -> Dict[str, Any]:
        """
        从 metrics.json 加载指标

        返回：
            指标字典

        异常：
            FileNotFoundError: 当文件不存在时抛出
        """
        metrics_file = self.experiment_dir / self.metrics_filename

        if not metrics_file.exists():
            raise FileNotFoundError(f"指标文件不存在: {metrics_file}")

        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 如果是列表格式，返回最后一项；如果是字典，直接返回
        if isinstance(data, list) and len(data) > 0:
            return data[-1]
        elif isinstance(data, dict):
            return data
        else:
            return {}

    def save_training_history(self, history: Dict[str, Any]):
        """
        保存训练历史到 CSV 文件

        参数：
            history: 训练历史字典，包含 train_loss, val_loss, train_metrics, val_metrics 等
        """
        import pandas as pd

        records = []
        num_epochs = len(history.get("train_loss", []))

        for i in range(num_epochs):
            record = {"epoch": i + 1}

            # 添加 loss
            if "train_loss" in history:
                record["train_loss"] = history["train_loss"][i]
            if "val_loss" in history:
                record["val_loss"] = history["val_loss"][i]

            # 添加其他训练指标
            if "train_metrics" in history and i < len(history["train_metrics"]):
                train_m = history["train_metrics"][i]
                for key, value in train_m.items():
                    if key != "loss":
                        record[f"train_{key}"] = value

            # 添加其他验证指标
            if "val_metrics" in history and i < len(history["val_metrics"]):
                val_m = history["val_metrics"][i]
                for key, value in val_m.items():
                    if key != "loss":
                        record[f"val_{key}"] = value

            records.append(record)

        df = pd.DataFrame(records)
        history_file = self.experiment_dir / self.history_filename
        df.to_csv(history_file, index=False)

    def _to_native(self, obj):
        """
        转换为 Python 原生类型

        参数：
            obj: 待转换对象

        返回：
            Python 原生类型对象
        """
        import numpy as np
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_native(item) for item in obj]
        else:
            return obj
