from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _metrics(history, cfg):
    pos_err = history["target_xy"] - history["pos"][:, :2]
    force_err = history["desired_force"] - history["force"]
    tau_norm = np.linalg.norm(history["tau"], axis=1)
    return {
        "rms_xy_error_m": float(np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))),
        "rms_force_error_n": float(np.sqrt(np.mean(force_err**2))),
        "mean_tau_norm": float(np.mean(tau_norm)),
        "peak_joint_tau_nm": float(np.max(np.abs(history["tau"]))),
        "torque_cost_integral": float(np.trapz(0.5 * tau_norm**2, dx=cfg.controller.dt)),
        "worst_constraint_margin": float(np.min(history["worst_margin"])),
    }


def print_metrics_3d(name, history, cfg):
    report = {"name": name, **_metrics(history, cfg)}
    print(report)
    return report


def plot_summary_3d(history, cfg, title, filename):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = history["t"]
    pos = history["pos"]
    target = history["target_xy"]
    q = history["q"]
    dq = history["dq"]
    tau = history["tau"]

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    fig.suptitle(title)
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="end-effector path")
    ax3d.plot(target[:, 0], target[:, 1], np.zeros_like(t), "--", label="desired contact path")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(t, history["force"], label="measured contact force")
    ax.plot(t, history["desired_force"], "--", label="desired contact force")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 3)
    xy_err = np.linalg.norm(target - pos[:, :2], axis=1)
    ax.plot(t, xy_err, label="xy tracking error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m]")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 4)
    for idx in range(q.shape[1]):
        ax.plot(t, np.rad2deg(q[:, idx]), label=f"joint {idx + 1}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint angle [deg]")
    ax.grid(True)
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(2, 3, 5)
    for idx in range(dq.shape[1]):
        ax.plot(t, np.rad2deg(dq[:, idx]), label=f"joint {idx + 1}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint velocity [deg/s]")
    ax.grid(True)
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, np.linalg.norm(tau, axis=1), label="||tau||")
    ax.plot(t, history["worst_margin"], label="minimum constraint margin")
    ax.axhline(0.0, color="0.25", linewidth=1.0, label="constraint boundary")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _set_3d_axes(ax, cfg):
    ax.set_xlim(-0.55, 0.65)
    ax.set_ylim(-0.55, 0.65)
    ax.set_zlim(-0.08, 1.15)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    xx, yy = np.meshgrid(np.linspace(-0.55, 0.65, 2), np.linspace(-0.55, 0.65, 2))
    zz = np.zeros_like(xx) + cfg.surface.z_contact
    ax.plot_surface(xx, yy, zz, alpha=0.18, color="0.55", linewidth=0)
    ax.plot_surface(xx, yy, zz + cfg.surface.link_clearance, alpha=0.08, color="#8b5a2b", linewidth=0)


def plot_arm_snapshots_3d(history, cfg, arm, title, filename, frame_count=6):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(history["q"]) - 1, frame_count, dtype=int)
    cols = 3
    rows = int(np.ceil(frame_count / cols))
    fig = plt.figure(figsize=(15, 4.5 * rows), constrained_layout=True)
    fig.suptitle(title)

    for plot_idx, idx in enumerate(indices, start=1):
        ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")
        _set_3d_axes(ax, cfg)
        points = arm.link_points(history["q"][idx])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color="#3d6f99", linewidth=4, markersize=5, label="KUKA iiwa links")
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color="#b5332f", s=60, label="end effector")
        ax.scatter(history["target_xy"][idx, 0], history["target_xy"][idx, 1], cfg.surface.z_contact, marker="x", color="#e28b21", s=70, label="target point")
        ax.set_title(f"t={history['t'][idx]:.2f}s, F={history['force'][idx]:.2f}N")
        if plot_idx == 1:
            ax.legend(fontsize=7)

    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_optimization_comparison_3d(no_opt, opt, cfg, title, filename):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = no_opt["t"]
    tau_no = np.linalg.norm(no_opt["tau"], axis=1)
    tau_opt = np.linalg.norm(opt["tau"], axis=1)
    cost_no = np.cumsum(0.5 * tau_no**2) * cfg.controller.dt
    cost_opt = np.cumsum(0.5 * tau_opt**2) * cfg.controller.dt
    force_no = no_opt["desired_force"] - no_opt["force"]
    force_opt = opt["desired_force"] - opt["force"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title)
    axes[0, 0].plot(t, tau_no, label="without torque optimization")
    axes[0, 0].plot(t, tau_opt, label="with torque optimization")
    axes[0, 0].set_ylabel("||tau|| [Nm]")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t, cost_no, label="without torque optimization")
    axes[0, 1].plot(t, cost_opt, label="with torque optimization")
    axes[0, 1].set_ylabel("cumulative 0.5 ||tau||^2")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(t, force_no, label="without torque optimization")
    axes[1, 0].plot(t, force_opt, label="with torque optimization")
    axes[1, 0].axhline(0.0, color="0.25")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("force error [N]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    labels = ["mean ||tau||", "peak joint tau", "cost integral", "force RMS"]
    m_no = _metrics(no_opt, cfg)
    m_opt = _metrics(opt, cfg)
    vals_no = [m_no["mean_tau_norm"], m_no["peak_joint_tau_nm"], m_no["torque_cost_integral"], m_no["rms_force_error_n"]]
    vals_opt = [m_opt["mean_tau_norm"], m_opt["peak_joint_tau_nm"], m_opt["torque_cost_integral"], m_opt["rms_force_error_n"]]
    x = np.arange(len(labels))
    width = 0.36
    axes[1, 1].bar(x - width / 2, vals_no, width, label="without torque optimization")
    axes[1, 1].bar(x + width / 2, vals_opt, width, label="with torque optimization")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1, 1].grid(True, axis="y")
    axes[1, 1].legend()

    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
