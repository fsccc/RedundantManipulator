import numpy as np


def _dh_transform(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class KukaIiwaKinematics:
    def __init__(self, robot_config):
        self.a = np.asarray(robot_config.a, dtype=float)
        self.alpha = np.asarray(robot_config.alpha, dtype=float)
        self.d = np.asarray(robot_config.d, dtype=float)
        self.n = 7

    def transforms(self, q):
        q = np.asarray(q, dtype=float)
        transforms = [np.eye(4)]
        t = np.eye(4)
        for idx in range(self.n):
            t = t @ _dh_transform(self.a[idx], self.alpha[idx], self.d[idx], q[idx])
            transforms.append(t.copy())
        return transforms

    def forward_kinematics(self, q):
        return self.transforms(q)[-1][:3, 3].copy()

    def link_points(self, q):
        return np.vstack([t[:3, 3] for t in self.transforms(q)])

    def jacobian(self, q):
        transforms = self.transforms(q)
        ee = transforms[-1][:3, 3]
        j = np.zeros((3, self.n), dtype=float)
        for idx in range(self.n):
            origin = transforms[idx][:3, 3]
            axis = transforms[idx][:3, 2]
            j[:, idx] = np.cross(axis, ee - origin)
        return j

    def point_jacobian(self, q, point_index):
        transforms = self.transforms(q)
        point = transforms[point_index][:3, 3]
        j = np.zeros((3, self.n), dtype=float)
        for idx in range(min(point_index, self.n)):
            origin = transforms[idx][:3, 3]
            axis = transforms[idx][:3, 2]
            j[:, idx] = np.cross(axis, point - origin)
        return point, j

    def contact_force(self, q, surface):
        z = self.forward_kinematics(q)[2]
        penetration = max(0.0, surface.z_contact - z)
        return surface.stiffness * penetration

    def quasistatic_torque(self, q, contact_force, dq=None, damping=None):
        wrench = np.array([0.0, 0.0, contact_force], dtype=float)
        tau = self.jacobian(q).T @ wrench
        if dq is not None and damping is not None:
            tau = tau + np.asarray(damping, dtype=float) * np.asarray(dq, dtype=float)
        return tau
