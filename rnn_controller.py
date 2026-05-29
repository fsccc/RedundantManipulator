import numpy as np

from constraints import build_velocity_constraints, project_halfspaces


class ProjectedRNNController:
    def __init__(self, cfg, use_torque_optimization=True):
        self.cfg = cfg
        self.use_torque_optimization = use_torque_optimization

    def solve(self, q, j_xy, task_velocity, force, previous_dq=None):
        n = q.size
        reg = self.cfg.controller.velocity_regularization
        hessian = j_xy.T @ j_xy + reg * np.eye(n)
        gradient = -(j_xy.T @ task_velocity)

        if self.use_torque_optimization:
            damping = np.diag(self.cfg.robot.joint_damping)
            tau_bias = j_xy.T @ np.array([0.0, force], dtype=float)
            beta = self.cfg.controller.torque_regularization
            j_pinv = np.linalg.pinv(j_xy)
            nullspace = np.eye(n) - j_pinv @ j_xy
            hessian = hessian + beta * (nullspace.T @ damping.T @ damping @ nullspace)
            gradient = gradient + beta * (nullspace.T @ damping.T @ tau_bias)

        a_mat, b_vec = build_velocity_constraints(q, j_xy, force, self.cfg)
        if previous_dq is None:
            state = np.zeros(n, dtype=float)
        else:
            state = np.asarray(previous_dq, dtype=float).copy()
        state = project_halfspaces(
            state, a_mat, b_vec, passes=self.cfg.controller.projection_passes
        )

        step = self.cfg.controller.rnn_step
        for _ in range(self.cfg.controller.rnn_iters):
            descent = state - step * (hessian @ state + gradient)
            state = project_halfspaces(
                descent, a_mat, b_vec, passes=self.cfg.controller.projection_passes
            )
        return state


def make_task_velocity(pos, target_x, desired_force, measured_force, cfg, feedforward_x=0.0):
    vx = feedforward_x + cfg.controller.task_gain_xy[0] * (target_x - pos[0])
    force_error = desired_force - measured_force
    vy = -cfg.controller.force_gain * force_error
    return np.array([vx, vy], dtype=float)
