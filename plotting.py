from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np


def plot_summary(history, cfg, title, filename):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = history["t"]
    q = history["q"]
    dq = history["dq"]
    tau = history["tau"]
    pos = history["pos"]
    force = history["force"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title)

    axes[0, 0].plot(pos[:, 0], pos[:, 1], label="end-effector path")
    axes[0, 0].plot(
        history["target_x"],
        np.zeros_like(t),
        "--",
        label="desired contact path",
    )
    axes[0, 0].axhline(
        cfg.surface.y_contact,
        color="k",
        linewidth=1.0,
        label="contact surface y=0",
    )
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, force, label="measured contact force")
    axes[0, 1].plot(t, history["desired_force"], "--", label="desired contact force")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("force [N]")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    for idx in range(q.shape[1]):
        axes[1, 0].plot(t, np.rad2deg(q[:, idx]), label=f"joint {idx + 1}")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("joint angle [deg]")
    axes[1, 0].grid(True)

    for idx in range(dq.shape[1]):
        axes[1, 1].plot(t, np.rad2deg(dq[:, idx]), label=f"joint {idx + 1}")
        limit = np.rad2deg(cfg.robot.dq_max[idx])
        limit_label = "velocity limits" if idx == 0 else None
        axes[1, 1].axhline(limit, color="0.75", linewidth=0.7, label=limit_label)
        axes[1, 1].axhline(-limit, color="0.75", linewidth=0.7)
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("joint velocity [deg/s]")
    axes[1, 1].grid(True)

    for idx in range(tau.shape[1]):
        axes[2, 0].plot(t, tau[:, idx], label=f"joint {idx + 1}")
        limit = cfg.robot.tau_max[idx]
        limit_label = "torque limits" if idx == 0 else None
        axes[2, 0].axhline(limit, color="0.75", linewidth=0.7, label=limit_label)
        axes[2, 0].axhline(-limit, color="0.75", linewidth=0.7)
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("quasistatic torque [Nm]")
    axes[2, 0].grid(True)

    axes[2, 1].plot(t, history["worst_margin"], label="minimum constraint margin")
    axes[2, 1].axhline(0.0, color="k", linewidth=1.0, label="constraint boundary")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("minimum constraint margin")
    axes[2, 1].grid(True)

    for ax in axes.ravel():
        ax.legend(fontsize=8)

    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _set_arm_axes(ax, arm, cfg):
    reach = float(np.sum(arm.link_lengths))
    ax.set_xlim(-0.15, reach + 0.12)
    ax.set_ylim(-0.32, reach * 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(
        Rectangle(
            (-0.15, cfg.surface.y_contact - 0.028),
            reach + 0.27,
            0.028,
            facecolor="0.72",
            edgecolor="0.35",
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.axhline(cfg.surface.y_contact, color="0.20", linewidth=1.4, zorder=1)
    ax.axhline(
        cfg.surface.y_contact + cfg.surface.link_clearance,
        color="#8b5a2b",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
    )
    ax.grid(True, linewidth=0.5, alpha=0.22)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _draw_arm_pose(ax, points, target_x, cfg, force=None, desired_force=None):
    segments = np.stack([points[:-1], points[1:]], axis=1)
    shadows = LineCollection(
        segments,
        linewidths=18,
        colors="0.20",
        alpha=0.14,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    links = LineCollection(
        segments,
        linewidths=12,
        colors="#3d6f99",
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    highlights = LineCollection(
        segments,
        linewidths=4,
        colors="#8fc0e3",
        alpha=0.85,
        capstyle="round",
        joinstyle="round",
        zorder=4,
    )
    ax.add_collection(shadows)
    ax.add_collection(links)
    ax.add_collection(highlights)

    for idx, point in enumerate(points[:-1]):
        radius = 0.036 if idx == 0 else 0.030
        ax.add_patch(
            Circle(point, radius, facecolor="#263746", edgecolor="white", linewidth=1.4, zorder=5)
        )
        ax.add_patch(
            Circle(point, radius * 0.48, facecolor="#d7dde2", edgecolor="0.25", linewidth=0.7, zorder=6)
        )

    tip = points[-1]
    ax.add_patch(
        Circle(tip, 0.026, facecolor="#b5332f", edgecolor="white", linewidth=1.2, zorder=6)
    )
    ax.plot(
        [tip[0], tip[0]],
        [tip[1], cfg.surface.y_contact],
        color="#b5332f",
        linewidth=1.2,
        alpha=0.55,
        zorder=2,
    )
    ax.plot(target_x, cfg.surface.y_contact, marker="x", color="#e28b21", markersize=9, mew=2.0, zorder=7)

    if force is not None and desired_force is not None:
        ratio = min(1.0, max(0.0, force / max(desired_force, 1.0e-9)))
        ax.annotate(
            "",
            xy=(tip[0], cfg.surface.y_contact + 0.012),
            xytext=(tip[0], cfg.surface.y_contact + 0.10 * ratio + 0.03),
            arrowprops=dict(arrowstyle="-|>", color="#b5332f", lw=1.8, alpha=0.8),
            zorder=8,
        )


def _arm_legend_handles():
    return [
        Line2D([0], [0], color="#3d6f99", linewidth=8, solid_capstyle="round", label="manipulator link"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#263746", markeredgecolor="white", markersize=9, label="joint"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#b5332f", markeredgecolor="white", markersize=8, label="end effector"),
        Line2D([0], [0], marker="x", color="#e28b21", markersize=8, markeredgewidth=2.0, linestyle="none", label="target point"),
        Line2D([0], [0], color="0.20", linewidth=2, label="contact surface"),
        Line2D([0], [0], color="#8b5a2b", linewidth=1.5, linestyle="--", label="link clearance"),
        Line2D([0], [0], color="#b5332f", linewidth=2, label="contact force direction"),
    ]


def plot_arm_snapshots(history, cfg, arm, title, filename, frame_count=8):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)

    q_hist = history["q"]
    t_hist = history["t"]
    target_x = history["target_x"]
    desired_force = history["desired_force"]
    force = history["force"]
    indices = np.linspace(0, len(q_hist) - 1, frame_count, dtype=int)

    cols = 4
    rows = int(np.ceil(frame_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows), constrained_layout=True)
    fig.suptitle(title)
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, indices):
        points = arm.link_points(q_hist[idx])
        _set_arm_axes(ax, arm, cfg)
        _draw_arm_pose(
            ax,
            points,
            target_x[idx],
            cfg,
            force=force[idx],
            desired_force=desired_force[idx],
        )
        ax.set_title(
            f"t={t_hist[idx]:.2f}s, F={force[idx]:.2f}/{desired_force[idx]:.2f}N"
        )

    for ax in axes[len(indices) :]:
        ax.axis("off")

    axes[0].legend(handles=_arm_legend_handles(), loc="upper right", fontsize=7)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_arm_animation(history, cfg, arm, title, filename, max_frames=160):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)

    q_hist = history["q"]
    t_hist = history["t"]
    force = history["force"]
    desired_force = history["desired_force"]
    target_x = history["target_x"]
    stride = max(1, int(np.ceil(len(q_hist) / max_frames)))
    indices = np.arange(0, len(q_hist), stride)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.set_title(title)

    def update(frame_idx):
        idx = indices[frame_idx]
        ax.clear()
        _set_arm_axes(ax, arm, cfg)
        ax.set_title(title)
        points = arm.link_points(q_hist[idx])
        _draw_arm_pose(
            ax,
            points,
            target_x[idx],
            cfg,
            force=force[idx],
            desired_force=desired_force[idx],
        )
        ax.text(
            0.02,
            0.96,
            f"t={t_hist[idx]:.2f}s\nF={force[idx]:.2f}/{desired_force[idx]:.2f} N",
            transform=ax.transAxes,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="0.85",
                alpha=0.86,
            ),
        )
        ax.legend(handles=_arm_legend_handles(), loc="upper right", fontsize=7)
        return []

    anim = FuncAnimation(fig, update, frames=len(indices), interval=35, blit=False)
    try:
        anim.save(out, writer=PillowWriter(fps=25))
    except Exception as exc:
        print(f"Animation was not saved: {exc}")
        out = None
    finally:
        plt.close(fig)
    return out


def _torque_metrics(history, cfg):
    tau = history["tau"]
    force_error = history["desired_force"] - history["force"]
    x_error = history["target_x"] - history["pos"][:, 0]
    tau_norm = np.linalg.norm(tau, axis=1)
    torque_cost = 0.5 * tau_norm**2
    dt = cfg.controller.dt
    return {
        "peak_tau_norm": float(np.max(tau_norm)),
        "mean_tau_norm": float(np.mean(tau_norm)),
        "torque_cost_integral": float(np.trapz(torque_cost, dx=dt)),
        "peak_joint_tau": float(np.max(np.abs(tau))),
        "rms_force_error": float(np.sqrt(np.mean(force_error**2))),
        "rms_x_error": float(np.sqrt(np.mean(x_error**2))),
    }


def plot_optimization_comparison(no_opt_history, opt_history, cfg, title, filename):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = no_opt_history["t"]
    tau_norm_no = np.linalg.norm(no_opt_history["tau"], axis=1)
    tau_norm_opt = np.linalg.norm(opt_history["tau"], axis=1)
    effort_no = np.cumsum(0.5 * tau_norm_no**2) * cfg.controller.dt
    effort_opt = np.cumsum(0.5 * tau_norm_opt**2) * cfg.controller.dt
    force_err_no = no_opt_history["desired_force"] - no_opt_history["force"]
    force_err_opt = opt_history["desired_force"] - opt_history["force"]

    no_metrics = _torque_metrics(no_opt_history, cfg)
    opt_metrics = _torque_metrics(opt_history, cfg)
    labels = [
        "mean torque norm",
        "peak joint torque",
        "torque cost integral",
        "force RMS error",
    ]
    no_values = [
        no_metrics["mean_tau_norm"],
        no_metrics["peak_joint_tau"],
        no_metrics["torque_cost_integral"],
        no_metrics["rms_force_error"],
    ]
    opt_values = [
        opt_metrics["mean_tau_norm"],
        opt_metrics["peak_joint_tau"],
        opt_metrics["torque_cost_integral"],
        opt_metrics["rms_force_error"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title)

    axes[0, 0].plot(t, tau_norm_no, label="without torque optimization")
    axes[0, 0].plot(t, tau_norm_opt, label="with torque optimization")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("||tau|| [Nm]")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t, effort_no, label="without torque optimization")
    axes[0, 1].plot(t, effort_opt, label="with torque optimization")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("cumulative 0.5 ||tau||^2")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(t, force_err_no, label="without torque optimization")
    axes[1, 0].plot(t, force_err_opt, label="with torque optimization")
    axes[1, 0].axhline(0.0, color="0.25", linewidth=1.0)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("force error [N]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    x = np.arange(len(labels))
    width = 0.36
    axes[1, 1].bar(x - width / 2, no_values, width, label="without torque optimization")
    axes[1, 1].bar(x + width / 2, opt_values, width, label="with torque optimization")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1, 1].grid(True, axis="y")
    axes[1, 1].legend()

    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def print_optimization_comparison(name, no_opt_history, opt_history, cfg):
    no_metrics = _torque_metrics(no_opt_history, cfg)
    opt_metrics = _torque_metrics(opt_history, cfg)
    report = {"name": name}
    for key, no_value in no_metrics.items():
        opt_value = opt_metrics[key]
        if abs(no_value) > 1.0e-12:
            change_percent = 100.0 * (no_value - opt_value) / no_value
        else:
            change_percent = 0.0
        report[key] = {
            "without": no_value,
            "with": opt_value,
            "improvement_percent": change_percent,
        }
    print(report)
    return report


def print_metrics(name, history):
    pos_error = history["target_x"] - history["pos"][:, 0]
    force_error = history["desired_force"] - history["force"]
    tail = slice(int(0.75 * len(pos_error)), None)
    metrics = {
        "name": name,
        "rms_x_error_m": float(np.sqrt(np.mean(pos_error[tail] ** 2))),
        "rms_force_error_n": float(np.sqrt(np.mean(force_error[tail] ** 2))),
        "max_abs_tau_nm": float(np.max(np.abs(history["tau"]))),
        "max_abs_dq_deg_s": float(np.max(np.abs(np.rad2deg(history["dq"])))),
        "worst_constraint_margin": float(np.min(history["worst_margin"])),
    }
    print(metrics)
    return metrics
