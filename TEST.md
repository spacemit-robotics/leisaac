# LeIsaac 工作流指南

本文档提供 LeIsaac 框架完整工作流的技术实现指南，涵盖从仿真环境配置、数据采集、数据转换、模型训练到分布式推理评测的全流程。

## 快速导航

1. [系统架构](#系统架构)
2. [环境配置](#环境配置)
3. [网络数据采集](#网络数据采集)
4. [数据集转换](#数据集转换)
5. [模型训练](#模型训练)
6. [分布式推理](#分布式推理)

## 系统架构

LeIsaac 采用**仿真器 + 推理器**的分布式架构。各组件职责分工如下：

| 组件 | 职责 | 部署位置 |
|------|------|---------|
| **仿真器** | Isaac Sim/Lab 环境管理；数据采集；评测执行；可选Lab训练 | NVIDIA GPU + CUDA 12.8 |
| **推理器** | SO101 Leader 连接；关节状态采样；推理服务（gRPC） | PC 或开发板 |

**通信协议**：采用 gRPC 进行双向通信，支持客户端流式观测上传和单向动作下发。

## 环境配置

### 仿真环境

#### 克隆仓库并初始化 Conda 环境

```bash
git clone git@github.com:spacemit-robotics/leisaac.git --recursive
cd leisaac

conda create -n leisaac python=3.11
conda activate leisaac
```

#### 安装依赖

```bash
# PyTorch 与 CUDA
pip install --upgrade pip
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# IsaacSim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 系统依赖
sudo apt install cmake build-essential

# IsaacLab
cd dependencies/IsaacLab
./isaaclab.sh --install
cd ../..

# LeIsaac
pip install -e source/leisaac

# LeRobot，数据集转换需要
pip install lerobot==0.4.2
pip install numpy==1.26.0
```

#### 下载仿真资产

下载 [GitHub Releases v0.1.0](https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/assets.tar.gz) 中的资产包，解压至`assets`目录。

预期目录结构：

```
assets/
├── robots/
│   └── so101_follower.usd
└── scenes/
    ├── kitchen_with_orange/
    ├── table_with_cube/
    └── custom_scene/
```

如果已有场景满足需求，按照 [add custm task](https://lightwheelai.github.io/leisaac/docs/tutorials/custom_task) 添加一个客制化仿真场景。

#### 环境验证

```bash
python scripts/environments/list_envs.py
```

若顺利输出可用任务列表，表示服务器环境配置正确。

### 推理环境

本地需分别配置数据采集和分布式推理环境。

#### 克隆仓库并初始化 Conda 环境

```bash
git clone git@github.com:spacemit-robotics/leisaac.git --recursive
cd leisaac

python -m venv local-venv
source local-venv/bin/activate
```

#### 数据采集环境

```bash
pip install pyserial deepdiff tqdm feetech-servo-sdk
```

**串口权限配置**：SO101 Leader 臂通常使用 `/dev/ttyACM0`。若提示权限不足，执行：

```bash
sudo usermod -aG dialout $USER
# 重新登录或执行 newgrp dialout生效
```

#### 分布式推理环境

```bash
pip install "lerobot>=0.4.2"
pip install grpcio grpcio-tools protobuf
```

若使用本地开发版 LeRobot：

```bash
cd /path/to/lerobot

pip install -e .
pip install -e ".[async]"
```

## 网络数据采集

远程采集分两步：本地启动 Leader 数据发送服务，服务器启动仿真并录制。

### 本地：启动 Leader 发送服务

```bash
python scripts/tools/leader_sender.py \
    --port /dev/ttyACM0 \
    --listen-port 5050
```

**首次使用**将自动触发 SO101 Leader 臂校准。若需重新校准，添加 `--recalibrate`：

```bash
python scripts/tools/leader_sender.py \
    --port /dev/ttyACM0 \
    --listen-port 5050 \
    --recalibrate
```

**参数说明**：

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyACM0` | SO101 Leader 臂的串口设备路径 |
| `--listen-port` | `5050` | 接收服务器连接的监听端口 |

### 服务器：启动远程 Teleop 采集

```bash
python scripts/environments/teleoperation/teleop_network.py \
    --task LeIsaac-SO101-CustomTask-v0 \
    --leader-host 10.0.91.83 \
    --leader-port 5050 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/custom_task.hdf5
```

**参数说明**：

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--task` | `LeIsaac-SO101-CustomTask-v0` | 任务标识符，决定环境配置、观测和动作空间 |
| `--leader-host` | `10.0.91.83` | 本地客户端 的 IP 地址 |
| `--leader-port` | `5050` | 与 `leader_sender.py` 监听端口一致 |
| `--dataset_file` | `./datasets/custom_task.hdf5` | 数据集保存路径（HDF5 格式） |
| `--enable_cameras` | — | 启用摄像头数据采集 |
| `--record` | — | 启用数据录制 |

### 采集时的控制器操作

| 按键 | 功能 |
|------|------|
| `B` | 开始/恢复机械臂控制 |
| `R` | 标记失败轨迹，重置环境 |
| `N` | 标记成功轨迹，重置环境 |
| `Ctrl + C` | 结束采集并保存数据 |

## 数据集转换

此步骤将 HDF5 格式的录制文件转换为 LeRobot 数据集格式，在**服务器**执行。

### 安装转换依赖

```bash
conda activate leisaac

# 前面已安装则忽略
pip install lerobot==0.4.2
pip install numpy==1.26.0
```

### 执行转换

```bash
python scripts/convert/isaaclab2lerobotv3.py \
    --task_name=LeIsaac-SO101-CustomTask-v0 \
    --repo_id=EverNorif/so101_test_custom_task \
    --hdf5_root=./datasets \
    --hdf5_files=custom_task.hdf5
```

**参数说明**：

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--task_name` | `LeIsaac-SO101-CustomTask-v0` | 采集使用的任务名，用于读取环境配置 |
| `--repo_id` | `EverNorif/so101_test_custom_task` | 数据集在 Hugging Face Hub 的标识 |
| `--hdf5_root` | `./datasets` | HDF5 文件所在目录 |
| `--hdf5_files` | `custom_task.hdf5` | 待转换的文件名（支持逗号分隔多文件） |

转换完成后将生成 LeRobot 格式数据集，可选推送至 Hugging Face Hub。

## 模型训练

LeIsaac 重点聚焦数据采集和推理评测。模型训练由外部框架承担，支持：

- [LeRobot](https://github.com/huggingface/lerobot)（推荐）
- [GR00T](https://github.com/NVIDIA/Isaac-GR00T)

### LeRobot ACT 模型训练示例

```bash
cd /path/to/lerobot
conda activate leisaac

lerobot-train \
  --policy.type=act \
  --policy.repo_id=EverNorif/act_test_custom_task \
  --dataset.repo_id=EverNorif/so101_test_custom_task \
  --output_dir=outputs/train/act_test_custom_task \
  --job_name=act_test_custom_task \
  --batch_size=4 \
  --steps=100000 \
  --policy.device=cuda
```

**关键参数**：

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--dataset.repo_id` | `EverNorif/so101_test_custom_task` | 第 4 节转换的数据集标识 |
| `--policy.repo_id` | `EverNorif/act_test_custom_task` | 训练完成后的模型标识（Hugging Face Hub） |
| `--output_dir` | `outputs/train/act_custom_task` | 本地 checkpoint 保存目录 |
| `--steps` | `100000` | 训练步数 |

### Checkpoint 结构

训练完成后的模型目录结构：

```text
outputs/train/act_test_custom_task/checkpoints/last/pretrained_model/
├── config.json                                          # 模型配置文件，包含 chunk_size、层数、隐层维度等超参数
├── model.safetensors                                    # 模型权重（SafeTensors 格式）
├── policy_preprocessor.json                             # 观测预处理配置（图像归一化、关节缩放等）
├── policy_preprocessor_step_3_normalizer_processor.safetensors  # 预处理器权重（观测数据的均值和标准差）
├── policy_postprocessor.json                            # 动作后处理配置（幅度限制、插值方式等）
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors  # 后处理器权重（动作反归一化参数）
└── train_config.json                                    # 训练配置记录（所有训练超参数）
```

## 分布式推理

### 前置检查清单

- [ ] 服务器可正常启动仿真环境（`list_envs.py` 输出无误）
- [ ] 本地已完整安装 LeRobot 推理环境（含 `async` 扩展）
- [ ] 本地可访问训练好的 checkpoint 目录
- [ ] 服务器和本地网络互通

### 本地：启动推理服务

```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=5555
```

**参数说明**：

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听所有网卡，允许远程连接 |
| `--port` | `5555` | 监听端口，需与服务器端 `--policy_port` 一致 |

### 服务器：启动仿真推理评测

```bash
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-CustomTask-v0 \
    --eval_rounds=1 \
    --policy_type=lerobot-act \
    --policy_host=10.0.91.83 \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=100 \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path models/act_test_custom_task/checkpoints/020000/pretrained_model/
```

**参数说明**：

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--task` | `LeIsaac-SO101-CustomTask-v0` | 评测任务标识 |
| `--policy_host` | `10.0.91.83` | 推理服务端 的 IP 地址；本地同机填 `127.0.0.1` |
| `--policy_port` | `5555` | 推理服务监听端口 |
| `--policy_type` | `lerobot-act` | 策略类型，支持 `lerobot-act`、`lerobot-smolvla` 等 |
| `--policy_checkpoint_path` | `models/act_test_custom_task/checkpoints/020000/pretrained_model/` | Checkpoint 的绝对路径 |
| `--policy_timeout_ms` | `5000` | 等待推理服务返回动作的超时时间（毫秒）；若推理设备较慢（如纯 CPU），需适当调大，否则每帧均超时导致无动作输出 |
| `--policy_action_horizon` | `100` | **必须与模型 config.json 的 chunk_size 相同** |

### 关键参数说明

**⚠️ `--policy_action_horizon` 与 `chunk_size` 必须相同**

- 模型训练时使用 `chunk_size=100` 生成 100 步预测
- 若推理时设 `--policy_action_horizon=8`，则仅执行前 8 步，之后从接近相同的观测状态重新预测
- 结果：机械臂在原地反复晃动，无法完成任务
- **解决方案**：推理时设 `--policy_action_horizon=100`

示例验证：查看 checkpoint 的 `config.json`：

```bash
cat models/act_test_custom_task/checkpoints/020000/pretrained_model/config.json | grep chunk_size
```

例如若输出 `"chunk_size": 100`，则推理时必须设 `--policy_action_horizon=100`。

## 参考资源

- IsaacLab: https://github.com/isaac-sim/IsaacLab
- LeRobot: https://github.com/huggingface/lerobot
- ACT: https://arxiv.org/abs/2304.13399