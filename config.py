from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RobotConfig:
    link_lengths: np.ndarray = field(
        default_factory=lambda: np.array([0.36, 0.30, 0.24, 0.18], dtype=float)
    )
    q_min: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([-160.0, -145.0, -145.0, -160.0]))
    )
    q_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([160.0, 145.0, 145.0, 160.0]))
    )
    dq_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([95.0, 110.0, 125.0, 140.0]))
    )
    tau_max: np.ndarray = field(
        default_factory=lambda: np.array([18.0, 14.0, 10.0, 7.0], dtype=float)
    )
    joint_damping: np.ndarray = field(
        default_factory=lambda: np.array([0.20, 0.16, 0.12, 0.10], dtype=float)
    )


@dataclass
class SurfaceConfig:
    y_contact: float = 0.0
    stiffness: float = 650.0
    force_filter: float = 0.35


@dataclass
class ControllerConfig:
    dt: float = 0.004
    task_gain_xy: np.ndarray = field(
        default_factory=lambda: np.array([4.5, 0.0], dtype=float)
    )
    force_gain: float = 0.010
    velocity_regularization: float = 2.0e-3
    torque_regularization: float = 4.0e-3
    rnn_step: float = 0.08
    rnn_iters: int = 80
    projection_passes: int = 3


@dataclass
class SimulationConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    t_final: float = 6.0
    desired_force: float = 8.0
    initial_q: np.ndarray = field(
        default_factory=lambda: np.deg2rad(
            np.array([86.25940566, -143.35582717, 60.38086835, -51.28727993])
        )
    )
    output_dir: Path = Path("results")


def default_config() -> SimulationConfig:
    return SimulationConfig()
