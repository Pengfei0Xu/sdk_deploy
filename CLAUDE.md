# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

基于 ROS2 的机器人控制 SDK，适用于绝影（DeepRobotics）四足机器人（Lite3 和 M20）。实现了 Sim-to-sim 和 Sim-to-real 的 RL 策略部署流程，采用有限状态机（FSM）架构。

## 构建命令

### Sim-to-sim（x86，本地开发）

```bash
# 安装依赖
pip install "numpy < 2.0" mujoco

# 先 source ROS2 环境
source /opt/ros/<ros-distro>/setup.bash  # Ubuntu 22.04 用 humble，20.04 用 foxy

# 构建 Lite3
colcon build --packages-up-to lite3_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86

# 构建 M20
colcon build --packages-up-to m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86

# 构建传输服务包
colcon build --packages-up-to lite3_sdk_service
```

### Sim-to-real（ARM 交叉编译，部署到机器人）

```bash
# 为 Lite3 机载电脑交叉编译（aarch64）
colcon build --packages-up-to lite3_sdk_deploy --cmake-args -DBUILD_PLATFORM=arm

# 为 M20 交叉编译
colcon build --packages-select m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=arm
```

`BUILD_PLATFORM=arm` 会将编译器切换为 `aarch64-linux-gnu-gcc/g++`。

## 运行

### Sim-to-sim

```bash
# 终端 1：RL 部署节点
source install/setup.bash
ros2 run lite3_sdk_deploy rl_deploy      # Lite3
export ROS_DOMAIN_ID=1 && source install/setup.bash && ros2 run m20_sdk_deploy rl_deploy        # M20（还需设置 export ROS_DOMAIN_ID=1）

# 终端 2：MuJoCo 仿真
source install/setup.bash
python3 src/Lite3_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
export ROS_DOMAIN_ID=1 && source install/setup.bash && python3 src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py
```

### Sim-to-real（在机器人上运行）

```bash
source install/setup.bash
ros2 run lite3_sdk_deploy rl_deploy
```

### 传输/服务节点（仅 Lite3）

```bash
# UDP 控制服务
source install/setup.bash
ros2 run lite3_sdk_service sdk_service
```

## 架构

### 功能包结构

| 功能包 | 用途 |
|---|---|
| `src/drdds` | ROS2 消息/服务定义（消息类型：ImuData、JointsData、JointsDataCmd、GamepadData、BatteryData 等） |
| `src/Lite3_sdk_deploy` | Lite3 RL 策略部署（主 FSM 可执行文件 `rl_deploy`） |
| `src/M20_sdk_deploy` | M20 RL 策略部署（与 Lite3 结构相同） |
| `src/lite3_transfer` | Lite3 机载电脑上的 UDP↔ROS2 桥接；发布 `/IMU_DATA`、`/JOINTS_DATA`；订阅 `/JOINTS_CMD`；提供 `/SDK_MODE` 服务 |
| `src/lite3_sdk_service` | 独立的 UDP 控制服务，用于远程启停 `lite3_transfer` 并调整发布频率 |

### FSM 架构（位于 `Lite3_sdk_deploy` / `M20_sdk_deploy`）

```
main.cpp
  └─ QStateMachine (state_machine/quadruped/q_state_machine.hpp)
       继承 StateMachineBase (state_machine/state_machine_base.h)
         ├─ RobotInterface (interface/robot/robot_interface.h)
         │    ├─ 硬件: lite3_interface.hpp / dds_interface.hpp
         │    └─ 仿真: mujoco_simulation_ros2.py（Python，独立进程）
         ├─ UserCommandInterface (interface/user_command/user_command_interface.h)
         │    ├─ KeyboardInterface / KeyboardInterfaceSim
         │    └─ RetroidGamepadInterface
         ├─ ControlParameters (state_machine/parameters/control_parameters.h)
         └─ 状态 (state_machine/quadruped/*.hpp):
              ├─ IdleState       (kIdle)
              ├─ StandUpState    (kStandUp)
              ├─ JointDampingState (kJointDamping)
              └─ RLControlState  (kRLControl)
```

**控制循环**：`StateMachineBase::RunThread()` 以 5ms（200 Hz）频率运行，使用 `timerfd`/`epoll`。每个 tick 调用 `RefreshRobotData()`、`current_controller_->Run()`，并检查状态转换。

**状态枚举**（`custom_types.h`）：
- `kIdle=0` → `kStandUp=1` → `kRLControl=6`；失控或急停时回退到 `kJointDamping=2`。

**策略推理**：ONNX Runtime（`third_party/onnxruntime/`）—— 通过 `BUILD_PLATFORM` 选择 `x86` 或 `arm` 目录。通过 `run_policy/lite3_policy_runner.hpp` / `policy_runner_base.hpp` 运行。

### ROS2 话题 / 服务

| 话题/服务 | 类型 | 方向 |
|---|---|---|
| `/JOINTS_CMD` | `drdds/msg/JointsDataCmd` | rl_deploy → 机器人 |
| `/JOINTS_DATA` | `drdds/msg/JointsData` | 机器人 → rl_deploy |
| `/IMU_DATA` | `drdds/msg/ImuData` | 机器人 → rl_deploy |
| `/GAMEPAD_DATA` | `drdds/msg/GamepadData` | 手柄 → rl_deploy |
| `/BATTERY_DATA` | `drdds/msg/BatteryData` | 机器人 → rl_deploy |
| `/SDK_MODE` | `drdds/srv/StdSrvInt32` | 服务调用，用于设置模式/频率 |
| `/EMERGENCY_STOP_SIGNAL` | `drdds/msg/StdMsgInt32` | 触发立即进入 JointDamping 状态 |

### 切换控制模式

在 `main.cpp` 中修改 `RemoteCommandType`：
- `RemoteCommandType::kKeyBoard` —— 键盘（默认）
- `RemoteCommandType::kRetroidGamepad` —— 手柄（仅 Sim-to-real）

## 机器人连接

| 机器人 | WiFi | SSH | 密码 |
|---|---|---|---|
| Lite3 | `lite3********`（密码：`12345678`） | `ssh ysc@192.168.2.1` | `'`（单引号） |
| M20 | `M20********`（密码：`12345678`） | `ssh user@10.21.31.103` | `'`（单引号） |

传输文件到 Lite3：
```bash
ssh ysc@192.168.2.1 'mkdir -p ~/Lite3_sdk_deploy/src' && \
scp -r ~/sdk_deploy/src/drdds ~/sdk_deploy/src/Lite3_sdk_deploy ysc@192.168.2.1:~/Lite3_sdk_deploy/src
```

## 关键实现说明

- **ONNX Runtime 兼容性**：如果跨主机编译时出现 `vector::operator[]` 断言错误，说明捆绑的 ONNX Runtime 库版本不兼容。请将 `third_party/onnxruntime/arm/` 替换为适合你架构的正确版本。
- **M20 需要 SDK 授权码**：在 M20 上启用 SDK 模式前，需联系技术支持获取每台机器人的授权码。
- **频率调整**（`/SDK_MODE` 服务）：`command: 200` 将关节/IMU 话题发布频率设置为 200 Hz（默认值）。
- **`ROS_DOMAIN_ID`**：M20 的 sim-to-sim 使用 `ROS_DOMAIN_ID=1`；Lite3 默认为 `0`。
