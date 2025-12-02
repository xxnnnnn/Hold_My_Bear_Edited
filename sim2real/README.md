# Hold My Beer - Sim2Real Deployment

本目录提供 Hold My Beer 任务的 sim2sim/sim2real 部署脚本，支持 Unitree G1 机器人。

## 目录

- [安装](#安装)
- [配置](#配置)
- [部署](#部署)
- [键盘控制](#键盘控制)
- [配置说明](#配置说明)
- [训练与部署一致性检查](#训练与部署一致性检查)
- [常见问题](#常见问题)
- [文件结构](#文件结构)

## 安装

### 环境要求
- Ubuntu 22.04 LTS
- Python 3.10
- CUDA (如果使用 GPU 训练)

### 创建 conda 环境
```bash
conda create -n hmbgym python=3.10
conda activate hmbgym
```

### 安装依赖
```bash
# 安装 unitree_sdk2_python
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
cd ..

# 安装其他依赖
cd sim2real
pip install -r requirements.txt
```

## 配置

### 配置文件位置
主要配置文件：`config/g1/g1_27dof_ee.yaml`

### 关键配置项

```yaml
ROBOT_TYPE: 'g1_27dof'
ROBOT_SCENE: "../humanoidverse/data/robots/g1/scene_g1_27dof_fakehand_freebase.xml"
DOMAIN_ID: 0
INTERFACE: "lo"  # Sim2Sim 用 "lo" (Linux) 或 "lo0" (Mac), Sim2Real 用实际网络接口如 "en0"
SDK_TYPE: "unitree"
USE_JOYSTICK: 0  # 0: 键盘控制, 1: 手柄控制
RL_RATE: 100     # RL 策略控制频率 (Hz)，应与训练时的控制频率一致
```

### 重要配置检查

1. **关节顺序 (`dof_names`)**: 必须与训练配置完全一致
2. **PD 增益 (`JOINT_KP`, `JOINT_KD`)**: 应与训练配置一致
3. **默认关节角度 (`DEFAULT_DOF_ANGLES`)**: 应与训练配置一致
4. **控制频率 (`RL_RATE`)**: 应与训练时的控制频率一致（训练时 `control_decimation=2`, `sim_dt=0.005`，所以 `dt=0.01`，控制频率=100Hz）

## 部署

### Sim2Sim (模拟器测试)

**步骤 1: 启动 MuJoCo 模拟器**
```bash
cd sim2real
python sim_env/base_sim.py --config=config/g1/g1_27dof_ee.yaml
```

**步骤 2: 启动策略** (新开一个终端)
```bash
cd sim2real
python rl_policy/hold_my_beer/s2s_eval.py \
  --config=config/g1/g1_27dof_ee.yaml \
  --model_path=models/hold_my_beer/baseline_8000.onnx
```

> [!NOTE]
> 机器人会浮在空中，这是模拟被绳子吊起的状态。

#### MuJoCo 窗口快捷键
- `7/8`: 调整弹性带高度
- `9`: 切换弹性带是否启用
- `backspace`: 重置模拟

### Sim2Real (真实机器人)

**直接启动策略** (不需要 MuJoCo)
```bash
cd sim2real
python rl_policy/hold_my_beer/s2s_eval.py \
  --config=config/g1/g1_27dof_ee.yaml \
  --model_path=models/hold_my_beer/baseline_8000.onnx
```

> [!IMPORTANT]
> - **Sim2Sim**: 需要先启动 MuJoCo，再启动策略
> - **Sim2Real**: 只需启动策略
> - **键盘控制**: 确保在运行策略的终端中操作键盘（不是 MuJoCo 窗口）
> - **安全**: 部署到真实机器人前，务必先在 Sim2Sim 中充分测试

## 键盘控制

### 基础控制
| 按键 | 功能 | 说明 |
|------|------|------|
| `]` | 启动策略 | 开始使用 RL 策略控制机器人 |
| `o` | 停止策略 | 停止策略，动作设为 0 |
| `i` | 初始化状态 | 机器人回到默认姿态 |
| `=` | 切换模式 | 在站立模式（0）和行走模式（1）之间切换 |

### 运动控制
| 按键 | 功能 | 说明 |
|------|------|------|
| `W/S` | 前进/后退 | 增加/减少前进速度（仅在行走模式） |
| `A/D` | 左移/右移 | 增加左侧移/右侧移速度（仅在行走模式） |
| `Q/E` | 左转/右转 | 逆时针/顺时针旋转 |
| `Z` | 速度归零 | 将所有速度命令设为 0 |

### 末端执行器（EE）控制
| 按键 | 功能 | 说明 |
|------|------|------|
| `X/C` | X 轴控制 | X 增加（前进）/ X 减少（后退） |
| `V/B` | Y 轴控制 | Y 增加（左侧）/ Y 减少（右侧） |
| `N/M` | Z 轴控制 | Z 增加（向上）/ Z 减少（向下） |

> [!NOTE]
> EE 控制步长为 0.03m，每次按键调整一次。

### 步态控制
| 按键 | 功能 | 说明 |
|------|------|------|
| `1/2` | 步态周期 | 增加/减少步态周期（Gait Period） |

### PD 增益控制
| 按键 | 功能 | 说明 |
|------|------|------|
| `4/7` | KP 调整（粗调） | 减少/增加 KP 缩放（0.1 步长） |
| `5/6` | KP 调整（细调） | 减少/增加 KP 缩放（0.01 步长） |
| `0` | KP 重置 | 重置 KP 缩放为 1.0 |

## 配置说明

### 观察值结构

策略启动时会自动打印观察值结构，包括：
- 每个观察组的维度
- 历史长度
- 总堆叠维度
- 命令值（速度、步态周期、EE 位置等）

### 命令初始化

默认命令值：
- `lin_vel_command`: [0.0, 0.0] (前进速度, 侧移速度)
- `ang_vel_command`: 0.0 (旋转速度)
- `stand_command`: 1.0 (行走模式)
- `gait_command`: 0.65 (步态周期，秒)
- `ee_command`: active=1.0, pos=[0.3, -0.15, 0.05], tolerance=0.15




## 文件结构

```
sim2real/
├── config/
│   └── g1/
│       └── g1_27dof_ee.yaml          # 主配置文件
├── models/
│   └── hold_my_beer/
│       └── baseline_8000.onnx        # 训练好的 ONNX 模型
├── rl_policy/
│   ├── base_policy.py                # 基础策略类
│   ├── dec_loco/
│   │   └── dec_loco.py               # 解耦运动策略
│   └── hold_my_beer/
│       └── s2s_eval.py               # Hold My Beer 策略脚本
├── sim_env/
│   └── base_sim.py                   # MuJoCo 模拟器
├── utils/
│   ├── comm/                         # 通信模块（命令发送、状态处理）
│   ├── sdk2py_bridge/                # SDK 桥接
│   └── math.py                       # 数学工具
├── requirements.txt                  # Python 依赖
└── README.md                         # 本文档
```