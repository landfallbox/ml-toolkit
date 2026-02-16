# Contributing to ml-toolkit

首先，感谢你考虑为 ml-toolkit 做出贡献！👋

## 贡献指南

### 报告 Bug

- 在 [Issues](https://github.com/landfallbox/ml-toolkit/issues) 中搜索是否已有相同问题
- 创建新 Issue 时，请提供：
  - 明确的问题描述
  - 复现步骤
  - 预期行为 vs 实际行为
  - Python 版本、PyTorch 版本等环境信息
  - 相关的代码或错误堆栈追踪

### 提交功能建议

- 在 [Issues](https://github.com/landfallbox/ml-toolkit/issues) 中提出你的想法
- 清晰描述功能的用途和实现思路
- 提供使用示例

### 代码贡献

#### 准备开发环境

```bash
# 克隆仓库
git clone https://github.com/yourusername/ml-toolkit.git
cd ml-toolkit

# 使用 uv 同步依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

#### 代码风格

本项目使用以下工具保证代码质量：

- **black** - 代码格式化
- **pylint** - 代码检查
- **pytest** - 单元测试

在提交前，请运行：

```bash
# 格式化代码
uv run black src/ tests/

# 检查代码风格
uv run pylint src/

# 运行测试
uv run pytest tests/
```

#### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

##### PR 要求

- ✅ 所有测试通过
- ✅ 新增代码需要有单元测试覆盖
- ✅ 遵循代码风格指南（black、pylint）
- ✅ 更新相关文档
- ✅ 在 PR 描述中说明改动的原因和影响

### 编码规范

- 遵循 PEP 8 规范
- 使用有意义的变量名和函数名
- 添加类型注解
- 为公开接口添加 docstring
- 代码注释应使用中文或英文，保持一致

### 目录结构说明

```
ml-toolkit/
├── src/ml_toolkit/          # 源代码
│   ├── models/              # 模型实现
│   ├── training/            # 训练框架
│   ├── evaluation/          # 评估框架
│   ├── data_processing/     # 数据处理
│   └── ...
├── tests/                   # 单元测试
├── docs/                    # 文档
├── pyproject.toml          # 项目配置
└── README.md               # 项目说明
```

### 提交信息规范

使用清晰的、描述性的提交信息：

- ✅ `feat: 添加 LSTM 模型支持`
- ✅ `fix: 修复数据归一化中的 NaN 处理`
- ✅ `docs: 更新 API 文档`
- ✅ `test: 为异常检测器添加单元测试`
- ❌ `update` / `fix bug` / `change`

### 许可证

通过提交代码，你同意你的贡献将在 MIT License 下发布。

## 问题反馈

- 遇到问题？在 [Issues](https://github.com/landfallbox/ml-toolkit/issues) 中提问
- 有好主意？欢迎开启 Discussion

---

感谢你的贡献！🎉

