import numpy as np


def build_velocity_constraints_3d(q, j_xyz, force, cfg, arm=None):
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
    tau_contact = j_xyz.T @ np.array([0.0, 0.0, force], dtype=float)
    rows.extend(damping)
    rhs.extend(robot.tau_max - tau_contact)
    rows.extend(-damping)
    rhs.extend(robot.tau_max + tau_contact)

    if arm is not None:
        for row, bound in build_link_avoidance_constraints_3d(q, arm, cfg):
            rows.append(row)
            rhs.append(bound)

    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def build_link_avoidance_constraints_3d(q, arm, cfg):
    constraints = []
    clearance_z = cfg.surface.z_contact + cfg.surface.link_clearance
    alpha = cfg.surface.avoidance_gain
    # Joint frames before the wrist/tool must stay above the table clearance.
    # The end effector is allowed to penetrate the plane to generate force.
    for point_index in range(1, arm.n):
        point, point_j = arm.point_jacobian(q, point_index)
        margin = point[2] - clearance_z
        constraints.append((-point_j[2], alpha * margin))
    return constraints


def project_halfspaces(x, a_mat, b_vec, passes=4):
    projected = np.asarray(x, dtype=float).copy()
    for _ in range(passes):
        for a, b in zip(a_mat, b_vec):
            violation = float(a @ projected - b)
            if violation > 0.0:
                denom = float(a @ a)
                if denom > 1.0e-12:
                    projected = projected - (violation / denom) * a
    return projected


def link_clearance_margins_3d(q, arm, cfg):
    clearance_z = cfg.surface.z_contact + cfg.surface.link_clearance
    margins = []
    for point_index in range(1, arm.n):
        point, _ = arm.point_jacobian(q, point_index)
        margins.append(point[2] - clearance_z)
    return np.asarray(margins, dtype=float)


def constraint_margins_3d(q, dq, tau, cfg, arm=None):
    margins = {
        "q_min": q - cfg.robot.q_min,
        "q_max": cfg.robot.q_max - q,
        "dq": cfg.robot.dq_max - np.abs(dq),
        "tau": cfg.robot.tau_max - np.abs(tau),
    }
    if arm is not None:
        margins["link_clearance"] = link_clearance_margins_3d(q, arm, cfg)
    return margins


def worst_margin(margins):
    return min(float(np.min(value)) for value in margins.values())
