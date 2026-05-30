# 冗余机械臂运动-力控制仿真

本项目用于复现一套冗余机械臂运动-力控制优化算法。当前版本先从
**4-DOF 平面机械臂的纯运动学仿真**开始验证算法核心，不直接复现论文中
KUKA iiwa + ROS 的实验系统。

本项目重点关注速度层的冗余解析和约束优化，而不是完整刚体动力学仿真。
当前已实现：

- 4-DOF 平面机械臂正运动学。
- 末端雅可比矩阵。
- 接触面 `y = 0`。
- 基于线性刚度模型的接触力。
- 控制输出为关节速度 `theta_dot`。
- 使用欧拉积分更新关节角。
- 关节角、关节速度、准静态力矩约束。
- 速度层 QP 形式重构。
- 投影 RNN 迭代求解约束速度优化问题。
- 固定点位置-力控制仿真。
- 直线轨迹位置-力控制仿真。
- 有无 torque optimization 的对比。

## 目录结构

```text
.
├── main_sim_fixed_point.py
├── main_sim_line_tracking.py
├── robot_kinematics.py
├── rnn_controller.py
├── constraints.py
├── plotting.py
├── config.py
└── README.md
```

## 文件说明

### `config.py`

项目的参数配置文件，使用 dataclass 管理仿真参数。

主要内容包括：

- 4-DOF 平面机械臂连杆长度。
- 关节角上下限。
- 关节速度上限。
- 准静态力矩上限。
- 关节阻尼系数。
- 接触面刚度。
- 控制器增益。
- 投影 RNN 的步长和迭代次数。
- 初始关节角。
- 结果输出目录。

如果需要调整机器人尺寸、约束范围、接触刚度、控制增益或仿真时间，优先修改
这个文件。

### `robot_kinematics.py`

机器人运动学模块，定义 `PlanarArm4DOF` 类。

主要函数：

- `forward_kinematics(q)`：计算末端位置 `[x, y]`。
- `link_points(q)`：计算各关节和连杆端点位置，可用于动画或调试。
- `jacobian(q)`：计算 2x4 的末端雅可比矩阵。
- `contact_force(q, surface)`：根据末端相对接触面的穿透量计算接触力。
- `quasistatic_torque(...)`：根据接触力和关节阻尼估计准静态关节力矩。

这是整个项目的运动学核心。

### `constraints.py`

约束构造与检查模块。

主要内容：

- 构造关节速度约束。
- 构造单步欧拉积分下的关节角约束。
- 构造准静态力矩约束。
- 将速度向量投影到线性半空间约束内。
- 计算各类约束的 margin。
- 输出所有约束中的最小 margin。

投影 RNN 控制器在每个控制周期都会调用该模块。

### `rnn_controller.py`

速度层控制器模块。

主要内容：

- 将运动-力跟踪任务重构为二次优化问题。
- 可选加入 torque optimization 项。
- 使用连续时间投影 RNN 神经动力学，并用显式欧拉法积分求解带约束的关节速度。
- 提供 `make_task_velocity(...)`，用于根据位置误差和力误差生成末端任务速度。

控制器的输出是关节速度 `theta_dot`。

当前 RNN 不是需要训练的深度学习网络，而是在线优化求解器。对速度层 QP

```text
min 1/2 * omega^T H omega + p^T omega
s.t. A omega <= b
```

控制器采用投影神经动力学：

```text
dot(omega) = gamma * (P_Omega(omega - alpha * (H omega + p)) - omega)
```

其中：

- `omega` 是 RNN 状态，也就是待求的关节速度。
- `P_Omega` 是到约束集合 `A omega <= b` 的投影。
- `alpha` 对应代码中的 `rnn_step`。
- `gamma` 对应代码中的 `rnn_ode_gain`。
- `rnn_ode_dt` 是连续时间 RNN 的数值积分步长。

每个控制周期内，代码对上述微分方程执行若干步显式欧拉积分，得到当前周期的
关节速度命令。

### `plotting.py`

绘图和指标输出模块。

主要绘制：

- 末端轨迹。
- 接触力跟踪曲线。
- 关节角曲线。
- 关节速度曲线。
- 准静态力矩曲线。
- 最小约束 margin 曲线。

仿真结束后，结果图会保存到 `results/` 目录。

### `main_sim_fixed_point.py`

固定点位置-力控制仿真入口。

仿真目标：

- 末端 `x` 方向跟踪固定目标位置。
- 末端在 `y = 0` 接触面上维持期望接触力。

脚本会自动运行两组对比：

- 不使用 torque optimization。
- 使用 torque optimization。

输出图像：

```text
results/fixed_point_no_torque_opt.png
results/fixed_point_torque_opt.png
```

### `main_sim_line_tracking.py`

直线轨迹位置-力控制仿真入口。

仿真目标：

- 末端沿接触面方向跟踪一条直线轨迹。
- 同时维持期望法向接触力。

脚本会自动运行两组对比：

- 不使用 torque optimization。
- 使用 torque optimization。

输出图像：

```text
results/line_tracking_no_torque_opt.png
results/line_tracking_torque_opt.png
```

## 使用方法

进入项目根目录：

```powershell
cd D:\gitproject\RedundantManipulator
```

运行固定点位置-力控制仿真：

```powershell
$env:PYTHONNOUSERSITE='1'; conda run python -B main_sim_fixed_point.py
```

运行直线轨迹位置-力控制仿真：

```powershell
$env:PYTHONNOUSERSITE='1'; conda run python -B main_sim_line_tracking.py
```

说明：当前机器上的用户级 Python 包可能会干扰 conda 环境中的
NumPy/Matplotlib，因此建议加上：

```powershell
$env:PYTHONNOUSERSITE='1'
```

## 输出结果

仿真运行后会在终端打印指标字典，包括：

- `rms_x_error_m`：末端 `x` 方向 RMS 跟踪误差。
- `rms_force_error_n`：接触力 RMS 跟踪误差。
- `max_abs_tau_nm`：最大绝对准静态力矩。
- `max_abs_dq_deg_s`：最大绝对关节速度。
- `worst_constraint_margin`：所有约束中的最小 margin。

如果 `worst_constraint_margin >= 0`，说明关节角、关节速度和准静态力矩约束
均未被违反。

所有图像结果保存在：

```text
results/
```

## 当前建模假设

当前版本是用于验证算法主体的简化模型，主要假设如下：

- 机器人是 4-DOF 平面机械臂。
- 接触面固定为 `y = 0`。
- 接触力由线性刚度模型计算。
- 控制器工作在速度层。
- 关节角通过欧拉积分更新。
- 力矩为接触力和关节阻尼产生的准静态估计。
- torque optimization 是速度层的力矩代理惩罚项，不是完整刚体动力学意义下
  的力矩最小化。

这些简化使代码更适合先验证冗余解析、QP 形式重构、投影 RNN 求解和约束处理。

## 后续可扩展方向

可以在当前基础上继续扩展：

- 增加平面机械臂动画。
- 增加更多轨迹类型。
- 保存仿真历史数据为 CSV 或 NumPy 文件。
- 增加运动学、雅可比矩阵和约束投影的单元测试。
- 将准静态力矩代理替换为完整动力学模型。
- 在速度层算法验证稳定后，再扩展到 7-DOF KUKA iiwa 和 ROS 实验系统。

## 7-DOF KUKA iiwa 三维仿真

项目现在也包含一套 7-DOF KUKA LBR iiwa 的三维速度层运动-力混合控制仿真。
该版本仍然是纯运动学/速度层仿真，不包含完整刚体动力学和 ROS。

新增文件：

```text
config_iiwa.py
robot_kinematics_iiwa.py
constraints_iiwa.py
rnn_controller_iiwa.py
plotting_iiwa.py
main_sim_iiwa_tasks.py
```

主要内容：

- 使用 7R DH 近似模型描述 KUKA LBR iiwa 构型。
- 末端三维正运动学和 3x7 位置 Jacobian。
- 接触平面为 `z = 0`。
- 末端允许轻微穿透平面以产生法向接触力。
- 中间关节/连杆点必须保持在接触平面上方，避免非物理穿模。
- `x-y` 平面执行位置轨迹跟踪。
- `z` 方向通过接触力误差生成法向速度，实现力控制。
- 支持固定点、直线和圆弧三类任务。
- 每类任务都运行有/无 torque optimization 两组对比。

运行三维 KUKA iiwa 仿真：

```powershell
python main_sim_iiwa_tasks.py
```

输出目录：

```text
results_iiwa/
```

输出图像包括：

- `fixed_point_no_torque_opt.png`
- `fixed_point_torque_opt.png`
- `fixed_point_arm_no_torque_opt.png`
- `fixed_point_arm_torque_opt.png`
- `fixed_point_optimization_comparison.png`
- `line_tracking_*`
- `arc_tracking_*`

需要注意：当前 7-DOF 版本的 torque optimization 是速度层零空间力矩代理优化。
在同时加入接触力控制、关节约束、力矩约束和连杆避障约束后，零空间可调余量会被压缩。
因此某些任务中力矩指标改善可能不如 4-DOF 平面示例明显。后续如果要更贴近论文实验，
可以进一步引入更精确的 KUKA iiwa 运动学参数、动力学模型和任务优先级 QP。
