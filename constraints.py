import numpy as np


def build_velocity_constraints(q, j_xy, force, cfg):
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

    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


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


def constraint_margins(q, dq, tau, cfg):
    return {
        "q_min": q - cfg.robot.q_min,
        "q_max": cfg.robot.q_max - q,
        "dq": cfg.robot.dq_max - np.abs(dq),
        "tau": cfg.robot.tau_max - np.abs(tau),
    }


def worst_margin(margins):
    return min(float(np.min(value)) for value in margins.values())
