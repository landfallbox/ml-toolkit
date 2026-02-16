# ml-toolkit

一个跨项目的**机器学习算法库**，提供可复用的深度学习和机器学习组件。

## 🚀 快速开始

### 安装

```bash
# 使用 uv 同步依赖（推荐）
uv sync
```

## 🏗️ 项目结构

```
ml-toolkit/
├── src/ml_toolkit/
│   ├── models/              # 神经网络模型
│   ├── training/            # 训练框架
│   ├── evaluation/          # 评估框架
│   ├── data_processing/     # 数据处理工具
│   ├── anomaly_detection/   # 异常检测
│   ├── calibrated_classifier/  # 校准分类器
│   ├── data_augmentation/   # 数据增强
│   ├── rl/                  # 强化学习
│   └── utils/               # 通用工具
├── pyproject.toml           # 项目配置
├── uv.lock                  # 依赖锁文件
└── README.md                # 本文件
```

## 💡 核心设计理念

1. **模块化**：每个功能独立成模块，易于维护和扩展
2. **可复用性**：设计为跨项目的通用工具库，支持多种使用场景
3. **灵活配置**：提供参数配置选项，适应不同需求

## 📄 许可证

MIT License

---

欢迎提交 Issue 和 Pull Request！



