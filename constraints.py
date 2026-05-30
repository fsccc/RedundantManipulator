import numpy as np


def build_velocity_constraints(q, j_xy, force, cfg, arm=None):
    robot = cfg.robot
    dt = cfg.controller.dt
    n = q.size

    rows = []
    rhs = []

    eye = np.eye(n)
    rows.extend(eye)
    rhs.extend(robot.dq_max)
    rows.extend(-eye)
    rhs.extend(robot.dq_max)

    rows.extend(eye)
    rhs.extend((robot.q_max - q) / dt)
    rows.extend(-eye)
    rhs.extend((q - robot.q_min) / dt)

    damping = np.diag(robot.joint_damping)
    tau_contact = j_xy.T @ np.array([0.0, force], dtype=float)
    rows.extend(damping)
    rhs.extend(robot.tau_max - tau_contact)
    rows.extend(-damping)
    rhs.extend(robot.tau_max + tau_contact)

    if arm is not None:
        for row, bound in build_link_avoidance_constraints(q, arm, cfg):
            rows.append(row)
            rhs.append(bound)

    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def build_link_avoidance_constraints(q, arm, cfg):
    clearance = cfg.surface.link_clearance
    alpha = cfg.surface.avoidance_gain
    sample_count = cfg.surface.avoidance_samples_per_link
    constraints = []

    # The end-effector is allowed to penetrate slightly to generate contact force.
    # Internal joints and link samples must stay above the contact plane.
    for link_index in range(arm.n - 1):
        for sample in range(1, sample_count + 1):
            fraction = sample / sample_count
            pos, point_j = arm.point_position_and_jacobian(q, link_index, fraction)
            height_margin = pos[1] - (cfg.surface.y_contact + clearance)
            constraints.append((-point_j[1], alpha * height_margin))
    return constraints


def project_halfspaces(x, a_mat, b_vec, passes=3):
    projected = np.asarray(x, dtype=float).copy()
    for _ in range(passes):
        for a, b in zip(a_mat, b_vec):
            violation = float(a @ projected - b)
            if violation > 0.0:
                denom = float(a @ a)
                if denom > 1.0e-12:
                    projected = projected - (violation / denom) * a
    return projected


def link_clearance_margins(q, arm, cfg):
    clearance = cfg.surface.link_clearance
    margins = []
    for link_index in range(arm.n - 1):
        for sample in range(1, cfg.surface.avoidance_samples_per_link + 1):
            fraction = sample / cfg.surface.avoidance_samples_per_link
            pos, _ = arm.point_position_and_jacobian(q, link_index, fraction)
            margins.append(pos[1] - (cfg.surface.y_contact + clearance))
    return np.asarray(margins, dtype=float)


def constraint_margins(q, dq, tau, cfg, arm=None):
    margins = {
        "q_min": q - cfg.robot.q_min,
        "q_max": cfg.robot.q_max - q,
        "dq": cfg.robot.dq_max - np.abs(dq),
        "tau": cfg.robot.tau_max - np.abs(tau),
    }
    if arm is not None:
        margins["link_clearance"] = link_clearance_margins(q, arm, cfg)
    return margins


def worst_margin(margins):
    return min(float(np.min(value)) for value in margins.values())
