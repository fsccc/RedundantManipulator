from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class IiwaRobotConfig:
    d: np.ndarray = field(
        default_factory=lambda: np.array([0.34, 0.0, 0.40, 0.0, 0.40, 0.0, 0.126])
    )
    alpha: np.ndarray = field(
        default_factory=lambda: np.array(
            [-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0.0]
        )
    )
    a: np.ndarray = field(default_factory=lambda: np.zeros(7))
    q_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([-170, -120, -170, -120, -170, -120, -175]))
    )
    q_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([170, 120, 170, 120, 170, 120, 175]))
    )
    dq_max: np.ndarray = field(default_factory=lambda: np.deg2rad(np.array([85] * 7)))
    tau_max: np.ndarray = field(default_factory=lambda: np.array([70, 70, 70, 70, 40, 40, 25], dtype=float))
    joint_damping: np.ndarray = field(
        default_factory=lambda: np.array([0.35, 0.32, 0.28, 0.24, 0.18, 0.14, 0.10])
    )


@dataclass
class IiwaSurfaceConfig:
    z_contact: float = 0.0
    stiffness: float = 900.0
    force_filter: float = 0.35
    link_clearance: float = 0.035
    avoidance_gain: float = 16.0


@dataclass
class IiwaControllerConfig:
    dt: float = 0.004
    position_gain: np.ndarray = field(default_factory=lambda: np.array([3.8, 3.8]))
    force_gain: float = 0.006
    velocity_regularization: float = 2.0e-3
    torque_regularization: float = 2.0e-3
    rnn_step: float = 0.055
    rnn_ode_gain: float = 1.0
    rnn_ode_dt: float = 0.65
    rnn_iters: int = 90
    projection_passes: int = 4


@dataclass
class IiwaSimulationConfig:
    robot: IiwaRobotConfig = field(default_factory=IiwaRobotConfig)
    surface: IiwaSurfaceConfig = field(default_factory=IiwaSurfaceConfig)
    controller: IiwaControllerConfig = field(default_factory=IiwaControllerConfig)
    desired_force: float = 10.0
    t_final: float = 6.0
    initial_q: np.ndarray = field(
        default_factory=lambda: np.deg2rad(
            np.array([-35.24867117, 104.94635610, 58.16662524, -103.82147147, -83.25875666, 108.93638768, 85.29379524])
        )
    )
    output_dir: Path = Path("results_iiwa")


def default_iiwa_config() -> IiwaSimulationConfig:
    return IiwaSimulationConfig()
