import numpy as np

from config_iiwa import default_iiwa_config
from constraints_iiwa import constraint_margins_3d, worst_margin
from plotting_iiwa import (
    plot_arm_snapshots_3d,
    plot_optimization_comparison_3d,
    plot_summary_3d,
    print_metrics_3d,
)
from rnn_controller_iiwa import ProjectedRNNController3D, make_task_velocity_3d
from robot_kinematics_iiwa import KukaIiwaKinematics


def fixed_reference(time, t_final):
    return np.array([0.31, 0.06]), np.zeros(2)


def line_reference(time, t_final):
    start = np.array([0.26, -0.12])
    stop = np.array([0.42, 0.18])
    rate = (stop - start) / t_final
    return start + rate * time, rate


def arc_reference(time, t_final):
    center = np.array([0.32, 0.03])
    radius = 0.15
    theta0 = -0.85
    theta1 = 0.95
    omega = (theta1 - theta0) / t_final
    theta = theta0 + omega * time
    target = center + radius * np.array([np.cos(theta), np.sin(theta)])
    feedforward = radius * omega * np.array([-np.sin(theta), np.cos(theta)])
    return target, feedforward


TASKS = {
    "fixed_point": (fixed_reference, 6.0),
    "line_tracking": (line_reference, 7.0),
    "arc_tracking": (arc_reference, 7.0),
}


def run_task(task_name, use_torque_optimization=True, make_plots=True):
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    reference, t_final = TASKS[task_name]
    cfg = default_iiwa_config()
    cfg.t_final = t_final
    arm = KukaIiwaKinematics(cfg.robot)
    controller = ProjectedRNNController3D(cfg, use_torque_optimization)

    steps = int(cfg.t_final / cfg.controller.dt)
    q = cfg.initial_q.copy()
    dq = np.zeros(arm.n)
    measured_force = arm.contact_force(q, cfg.surface)
    history = {key: [] for key in ["t", "q", "dq", "tau", "pos", "force", "target_xy", "desired_force", "worst_margin"]}

    for step in range(steps):
        time = step * cfg.controller.dt
        target_xy, feedforward_xy = reference(time, cfg.t_final)
        pos = arm.forward_kinematics(q)
        raw_force = arm.contact_force(q, cfg.surface)
        measured_force = cfg.surface.force_filter * raw_force + (1.0 - cfg.surface.force_filter) * measured_force
        j_xyz = arm.jacobian(q)
        task_velocity = make_task_velocity_3d(
            pos,
            target_xy,
            cfg.desired_force,
            measured_force,
            cfg,
            feedforward_xy=feedforward_xy,
        )
        dq = controller.solve(q, j_xyz, task_velocity, measured_force, dq, arm=arm)
        tau = arm.quasistatic_torque(q, measured_force, dq=dq, damping=cfg.robot.joint_damping)
        q = q + cfg.controller.dt * dq
        margins = constraint_margins_3d(q, dq, tau, cfg, arm=arm)

        history["t"].append(time)
        history["q"].append(q.copy())
        history["dq"].append(dq.copy())
        history["tau"].append(tau.copy())
        history["pos"].append(pos.copy())
        history["force"].append(measured_force)
        history["target_xy"].append(target_xy.copy())
        history["desired_force"].append(cfg.desired_force)
        history["worst_margin"].append(worst_margin(margins))

    for key, value in history.items():
        history[key] = np.asarray(value)

    suffix = "torque_opt" if use_torque_optimization else "no_torque_opt"
    if make_plots:
        plot_summary_3d(
            history,
            cfg,
            f"KUKA iiwa {task_name} ({suffix})",
            cfg.output_dir / f"{task_name}_{suffix}.png",
        )
        plot_arm_snapshots_3d(
            history,
            cfg,
            arm,
            f"KUKA iiwa snapshots {task_name} ({suffix})",
            cfg.output_dir / f"{task_name}_arm_{suffix}.png",
        )
    print_metrics_3d(f"iiwa_{task_name}_{suffix}", history, cfg)
    return history, cfg


def run_all_tasks():
    for task_name in TASKS:
        no_opt, cfg = run_task(task_name, use_torque_optimization=False)
        opt, _ = run_task(task_name, use_torque_optimization=True)
        plot_optimization_comparison_3d(
            no_opt,
            opt,
            cfg,
            f"KUKA iiwa torque optimization comparison ({task_name})",
            cfg.output_dir / f"{task_name}_optimization_comparison.png",
        )


if __name__ == "__main__":
    run_all_tasks()
