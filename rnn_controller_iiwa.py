import numpy as np

from constraints_iiwa import build_velocity_constraints_3d, project_halfspaces


class ProjectedRNNController3D:
    def __init__(self, cfg, use_torque_optimization=True):
        self.cfg = cfg
        self.use_torque_optimization = use_torque_optimization

    def solve(self, q, j_xyz, task_velocity, force, previous_dq=None, arm=None):
        n = q.size
        reg = self.cfg.controller.velocity_regularization
        hessian = j_xyz.T @ j_xyz + reg * np.eye(n)
        gradient = -(j_xyz.T @ task_velocity)

        if self.use_torque_optimization:
            damping = np.diag(self.cfg.robot.joint_damping)
            tau_bias = j_xyz.T @ np.array([0.0, 0.0, force], dtype=float)
            beta = self.cfg.controller.torque_regularization
            j_pinv = np.linalg.pinv(j_xyz)
            nullspace = np.eye(n) - j_pinv @ j_xyz
            hessian = hessian + beta * (nullspace.T @ damping.T @ damping @ nullspace)
            gradient = gradient + beta * (nullspace.T @ damping.T @ tau_bias)

        a_mat, b_vec = build_velocity_constraints_3d(q, j_xyz, force, self.cfg, arm=arm)
        state = np.zeros(n, dtype=float) if previous_dq is None else np.asarray(previous_dq, dtype=float).copy()
        state = project_halfspaces(state, a_mat, b_vec, passes=self.cfg.controller.projection_passes)

        gradient_step = self.cfg.controller.rnn_step
        ode_gain = self.cfg.controller.rnn_ode_gain
        ode_dt = self.cfg.controller.rnn_ode_dt
        for _ in range(self.cfg.controller.rnn_iters):
            projected_equilibrium = project_halfspaces(
                state - gradient_step * (hessian @ state + gradient),
                a_mat,
                b_vec,
                passes=self.cfg.controller.projection_passes,
            )
            state_dot = ode_gain * (projected_equilibrium - state)
            state = state + ode_dt * state_dot
            state = project_halfspaces(
                state, a_mat, b_vec, passes=self.cfg.controller.projection_passes
            )
        return state


def make_task_velocity_3d(pos, target_xy, desired_force, measured_force, cfg, feedforward_xy=None):
    if feedforward_xy is None:
        feedforward_xy = np.zeros(2)
    v_xy = feedforward_xy + cfg.controller.position_gain * (np.asarray(target_xy) - pos[:2])
    force_error = desired_force - measured_force
    v_z = -cfg.controller.force_gain * force_error
    return np.array([v_xy[0], v_xy[1], v_z], dtype=float)
