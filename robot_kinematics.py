import numpy as np


class PlanarArm4DOF:
    def __init__(self, link_lengths):
        self.link_lengths = np.asarray(link_lengths, dtype=float)
        self.n = self.link_lengths.size

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=float)
        angles = np.cumsum(q)
        x = np.sum(self.link_lengths * np.cos(angles))
        y = np.sum(self.link_lengths * np.sin(angles))
        return np.array([x, y], dtype=float)

    def link_points(self, q):
        q = np.asarray(q, dtype=float)
        angles = np.cumsum(q)
        points = [np.zeros(2)]
        pos = np.zeros(2)
        for length, angle in zip(self.link_lengths, angles):
            pos = pos + length * np.array([np.cos(angle), np.sin(angle)])
            points.append(pos.copy())
        return np.vstack(points)

    def jacobian(self, q):
        q = np.asarray(q, dtype=float)
        angles = np.cumsum(q)
        j = np.zeros((2, self.n), dtype=float)
        for col in range(self.n):
            active = slice(col, self.n)
            j[0, col] = -np.sum(self.link_lengths[active] * np.sin(angles[active]))
            j[1, col] = np.sum(self.link_lengths[active] * np.cos(angles[active]))
        return j

    def contact_force(self, q, surface):
        y = self.forward_kinematics(q)[1]
        penetration = max(0.0, surface.y_contact - y)
        return surface.stiffness * penetration

    def quasistatic_torque(self, q, contact_force, dq=None, damping=None):
        wrench = np.array([0.0, contact_force], dtype=float)
        tau = self.jacobian(q).T @ wrench
        if dq is not None and damping is not None:
            tau = tau + np.asarray(damping, dtype=float) * np.asarray(dq, dtype=float)
        return tau
