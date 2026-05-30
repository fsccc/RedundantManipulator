'''
Author: fsccc 271812746@qq.com
Date: 2026-05-28 20:03:47
LastEditors: fsccc 271812746@qq.com
LastEditTime: 2026-05-29 15:40:30
FilePath: \RedundantManipulator\main_sim_fixed_point.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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


def run(use_torque_optimization=True):
    cfg = default_config()
    arm = PlanarArm4DOF(cfg.robot.link_lengths)
    controller = ProjectedRNNController(cfg, use_torque_optimization)

    steps = int(cfg.t_final / cfg.controller.dt)
    q = cfg.initial_q.copy()
    dq = np.zeros(arm.n)
    target_x = 0.55
    measured_force = arm.contact_force(q, cfg.surface)

    history = {key: [] for key in ["t", "q", "dq", "tau", "pos", "force", "target_x", "desired_force", "worst_margin"]}
    for step in range(steps):
        time = step * cfg.controller.dt
        pos = arm.forward_kinematics(q)
        raw_force = arm.contact_force(q, cfg.surface)
        measured_force = (
            cfg.surface.force_filter * raw_force
            + (1.0 - cfg.surface.force_filter) * measured_force
        )
        j_xy = arm.jacobian(q)
        task_velocity = make_task_velocity(
            pos, target_x, cfg.desired_force, measured_force, cfg
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
        f"Fixed point force-position control ({suffix})",
        cfg.output_dir / f"fixed_point_{suffix}.png",
    )
    plot_arm_snapshots(
        history,
        cfg,
        arm,
        f"Fixed point arm snapshots ({suffix})",
        cfg.output_dir / f"fixed_point_arm_{suffix}.png",
    )
    save_arm_animation(
        history,
        cfg,
        arm,
        f"Fixed point arm animation ({suffix})",
        cfg.output_dir / f"fixed_point_arm_{suffix}.gif",
    )
    print_metrics(f"fixed_point_{suffix}", history)
    return history


if __name__ == "__main__":
    no_opt = run(use_torque_optimization=False)
    opt = run(use_torque_optimization=True)
    cfg = default_config()
    plot_optimization_comparison(
        no_opt,
        opt,
        cfg,
        "Fixed point torque optimization comparison",
        cfg.output_dir / "fixed_point_optimization_comparison.png",
    )
    print_optimization_comparison("fixed_point_comparison", no_opt, opt, cfg)
