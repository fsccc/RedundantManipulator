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
- 使用投影 RNN 迭代求解带约束的关节速度。
- 提供 `make_task_velocity(...)`，用于根据位置误差和力误差生成末端任务速度。

控制器的输出是关节速度 `theta_dot`。

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
