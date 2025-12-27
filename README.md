# 决策式AI模型开发与优化技能训练框架

本框架提供了一个完整的技能训练案例，涵盖了数据选择、模型开发、参数配置、代码编辑及优化到模型发布的完整流程。

## 支持模型
- VGG16
- ResNet50
- Inception V3

## 核心功能
1. **数据选择与加载**: 默认使用 CIFAR-10 数据集。
2. **模型参数配置**: 支持通过环境变量或命令行参数配置学习率、Batch Size、Epochs 等。
3. **开发优化模式**: 提供 `train`, `optimize`, `develop` 三种模式，用户可以通过修改输入参数或代码进行模型优化。
4. **SSE 实时反馈**: 训练过程中的进度、指标、日志均以 SSE (Server-Sent Events) 格式输出。
5. **容器化部署**: 提供 Dockerfile 和测试脚本，支持在隔离环境中运行。

## 目录结构
- `main.py`: 框架入口，负责解析参数和组织流程。
- `models/`: 模型定义与工厂类。
- `train/`: 包含数据集加载 (`dataset.py`) 和训练逻辑 (`trainer.py`)。
- `utils/`: SSE 通信工具与 YAML 配置文件读写。
- `input/`: 输入数据和模型权重存放路径（需挂载）。
- `output/`: 训练结果和最佳模型存放路径（需挂载）。

## 快速开始

### 1. 准备工作
确保 `/input` 目录下包含以下内容：
- `CIFAR-10/`: 包含 `train` 和 `test` 子目录的数据集。
- `model/`: 包含预训练权重文件 (`vgg16.pth`, `resnet50.pth`, `inception_v3.pth`)。

### 2. 构建镜像
由于 Docker 是通过 snap 安装的，请确保在用户 Home 路径下的项目根目录执行：
```bash
docker build -t decision-ai-training:v1.0 .
```

### 3. 运行测试
使用提供的测试脚本或手动运行容器：
```bash
chmod +x test_all.sh
./test_all.sh
```

### 4. 优化示例
通过修改环境变量进行参数优化：
```bash
docker run --rm \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    -e MODEL_NAME=resnet50 \
    -e LR=0.00001 \
    -e PROCESS=optimize \
    decision-ai-training:v1.0
```

## SSE 输出格式说明
输出信息严格遵循 SSE 格式，例如：
```
data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2025/12/27-10:00:00:123", "data": {"event": "training_progress", "progress": 50, "message": "Epoch 1/5", "log": "[50%] Epoch 1/5\n"}}
```

