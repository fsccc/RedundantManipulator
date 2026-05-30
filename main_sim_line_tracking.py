import numpy as np

from config import default_config
from constraints import constraint_margins, worst_margin
from plotting import (
    plot_arm_snapshots,
    plot_optimization_comparison,
    plot_summary,
    print_metrics,
    print_optimization_comparison,
    save_arm_animation,
)
from rnn_controller import ProjectedRNNController, make_task_velocity
from robot_kinematics import PlanarArm4DOF


def line_reference(time, t_final):
    start = 0.38
    stop = 0.72
    rate = (stop - start) / t_final
    return start + rate * time, rate


def run(use_torque_optimization=True):
    cfg = default_config()
    cfg.t_final = 7.0
    arm = PlanarArm4DOF(cfg.robot.link_lengths)
    controller = ProjectedRNNController(cfg, use_torque_optimization)

    steps = int(cfg.t_final / cfg.controller.dt)
    q = cfg.initial_q.copy()
    dq = np.zeros(arm.n)
    measured_force = arm.contact_force(q, cfg.surface)

    history = {key: [] for key in ["t", "q", "dq", "tau", "pos", "force", "target_x", "desired_force", "worst_margin"]}
    for step in range(steps):
        time = step * cfg.controller.dt
        target_x, feedforward_x = line_reference(time, cfg.t_final)
        pos = arm.forward_kinematics(q)
        raw_force = arm.contact_force(q, cfg.surface)
        measured_force = (
            cfg.surface.force_filter * raw_force
            + (1.0 - cfg.surface.force_filter) * measured_force
        )
        j_xy = arm.jacobian(q)
        task_velocity = make_task_velocity(
            pos,
            target_x,
            cfg.desired_force,
            measured_force,
            cfg,
            feedforward_x=feedforward_x,
        )
        dq = controller.solve(q, j_xy, task_velocity, measured_force, dq, arm=arm)
        tau = arm.quasistatic_torque(
            q, measured_force, dq=dq, damping=cfg.robot.joint_damping
        )

        q = q + cfg.controller.dt * dq
        margins = constraint_margins(q, dq, tau, cfg, arm=arm)

        history["t"].append(time)
        history["q"].append(q.copy())
        history["dq"].append(dq.copy())
        history["tau"].append(tau.copy())
        history["pos"].append(pos.copy())
        history["force"].append(measured_force)
        history["target_x"].append(target_x)
        history["desired_force"].append(cfg.desired_force)
        history["worst_margin"].append(worst_margin(margins))

    for key, value in history.items():
        history[key] = np.asarray(value)

    suffix = "torque_opt" if use_torque_optimization else "no_torque_opt"
    plot_summary(
        history,
        cfg,
        f"Line force-position tracking ({suffix})",
        cfg.output_dir / f"line_tracking_{suffix}.png",
    )
    plot_arm_snapshots(
        history,
        cfg,
        arm,
        f"Line tracking arm snapshots ({suffix})",
        cfg.output_dir / f"line_tracking_arm_{suffix}.png",
    )
    save_arm_animation(
        history,
        cfg,
        arm,
        f"Line tracking arm animation ({suffix})",
        cfg.output_dir / f"line_tracking_arm_{suffix}.gif",
    )
    print_metrics(f"line_tracking_{suffix}", history)
    return history


if __name__ == "__main__":
    no_opt = run(use_torque_optimization=False)
    opt = run(use_torque_optimization=True)
    cfg = default_config()
    cfg.t_final = 7.0
    plot_optimization_comparison(
        no_opt,
        opt,
        cfg,
        "Line tracking torque optimization comparison",
        cfg.output_dir / "line_tracking_optimization_comparison.png",
    )
    print_optimization_comparison("line_tracking_comparison", no_opt, opt, cfg)
