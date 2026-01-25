"""
Plot robot and payload trajectories from HDF5 experiment data.
"""

import sys
sys.path.append('/home/manav/my_isaaclab_project')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hdf5_logger_utils import HDF5Reader

# Data directory
DATA_DIR = Path('/home/manav/Desktop/data_collection/simulation/flat_terrain')

# =============================================================================
# Configuration: Skip initial seconds (robot drop phase)
# =============================================================================
SKIP_SECONDS = 1.5  # Skip first N seconds of data (robot drop/stabilization)
DT = 0.02           # Data collection timestep

def plot_trajectory_with_z(filepath: str, save_fig: bool = True):
    """
    Plot 2D (X-Y) trajectory on top and Z variation on bottom.

    Args:
        filepath: Path to HDF5 experiment file
        save_fig: Whether to save the figure
    """
    # Load data
    data = HDF5Reader.load(filepath, dt=0.02)

    print(f"Loaded: {data}")
    print(f"Fields: {data.list_fields()}")

    # Get position data
    robot_pos = data.get_field('robot_pos_w')
    payload_pos = data.get_field('payload_pos_w')
    time = data.get_field('sim_time').flatten()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # Top plot: 2D X-Y trajectory
    ax1.plot(robot_pos[:, 0], robot_pos[:, 1],
             'b-', linewidth=2, label='Robot', alpha=0.8)
    ax1.plot(payload_pos[:, 0], payload_pos[:, 1],
             'orange', linewidth=2, label='Payload', alpha=0.8, linestyle='--')
    ax1.plot(robot_pos[0, 0], robot_pos[0, 1],
             'go', markersize=12, label='Start', zorder=5)
    ax1.plot(robot_pos[-1, 0], robot_pos[-1, 1],
             'ro', markersize=12, label='End', zorder=5)

    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title(f'X-Y Trajectory', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Bottom plot: Z position vs time
    ax2.plot(time, robot_pos[:, 2], 'b-', linewidth=2, label='Robot Z', alpha=0.8)
    ax2.plot(time, payload_pos[:, 2], 'orange', linewidth=2, label='Payload Z', alpha=0.8, linestyle='--')

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Z Position (m)', fontsize=12)
    ax2.set_title('Z Position Over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{Path(filepath).stem}', fontsize=11, y=0.995, color='gray')
    plt.tight_layout()

    if save_fig:
        output_path = Path(filepath).stem + '_trajectory_with_z.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def compute_acceleration_metrics(save_table: bool = True, skip_seconds: float = None):
    """
    Compute summary metrics (RMS) for linear and angular acceleration across all experiments.
    Creates a comparison table and bar chart for the paper.

    Args:
        save_table: Whether to save the figure
        skip_seconds: Seconds to skip from beginning (uses global SKIP_SECONDS if None)
    """
    if skip_seconds is None:
        skip_seconds = SKIP_SECONDS
    skip_samples = int(skip_seconds / DT)

    experiments = [
        ('baseline_flat_without_payload_waypoint_20260123_182835.h5', 'Baseline', True),
        ('flat_terrain_center_payload_5kg_waypoint_20260123_184135.h5', 'Center', False),
        ('flat_terrain_payload_5kg_forward_0.15_meter_waypoint_20260123_184832.h5', 'Fwd 0.15m', False),
        ('flat_terrain_payload_5kg_forward_0.25_meter_waypoint_20260123_184710.h5', 'Fwd 0.25m', False),
        ('flat_terrain_payload_5kg_backwards_0.15_meter_waypoint_20260123_184958.h5', 'Bwd 0.15m', False),
        ('flat_terrain_payload_5kg_backwards_0.25_meter_waypoint_20260123_185320.h5', 'Bwd 0.25m', False),
    ]

    print(f"\n[Config] Skipping first {skip_seconds}s ({skip_samples} samples) of each experiment")

    # Store results
    results = []

    for filename, label, is_baseline in experiments:
        filepath = DATA_DIR / filename
        data = HDF5Reader.load(str(filepath), dt=DT)

        # Linear acceleration (trimmed)
        if is_baseline:
            lin_acc = data.get_field('robot_lin_acc_b')[skip_samples:]
        else:
            lin_acc = data.get_field('payload_lin_acc_b')[skip_samples:]

        # Compute linear RMS for each axis
        lin_rms_x = np.sqrt(np.mean(lin_acc[:, 0]**2))
        lin_rms_y = np.sqrt(np.mean(lin_acc[:, 1]**2))
        lin_rms_z = np.sqrt(np.mean(lin_acc[:, 2]**2))

        # Angular acceleration (only for payload experiments, trimmed)
        if is_baseline:
            ang_rms_x, ang_rms_y, ang_rms_z = None, None, None
        else:
            ang_acc = data.get_field('payload_ang_acc_b')[skip_samples:]
            ang_rms_x = np.sqrt(np.mean(ang_acc[:, 0]**2))
            ang_rms_y = np.sqrt(np.mean(ang_acc[:, 1]**2))
            ang_rms_z = np.sqrt(np.mean(ang_acc[:, 2]**2))

        results.append({
            'Experiment': label,
            'is_baseline': is_baseline,
            'Lin_RMS_X': lin_rms_x, 'Lin_RMS_Y': lin_rms_y, 'Lin_RMS_Z': lin_rms_z,
            'Ang_RMS_X': ang_rms_x, 'Ang_RMS_Y': ang_rms_y, 'Ang_RMS_Z': ang_rms_z,
        })

    # Print table
    # print("\n" + "="*90)
    # print("ACCELERATION METRICS - RMS Values")
    # print("="*90)
    # print(f"{'Experiment':<12} | {'Lin_X':>7} {'Lin_Y':>7} {'Lin_Z':>7} (m/s^2) | {'Ang_X':>7} {'Ang_Y':>7} {'Ang_Z':>7} (rad/s^2)")
    # print("-"*90)
    # for r in results:
    #     ang_x = f"{r['Ang_RMS_X']:>7.2f}" if r['Ang_RMS_X'] else "    N/A"
    #     ang_y = f"{r['Ang_RMS_Y']:>7.2f}" if r['Ang_RMS_Y'] else "    N/A"
    #     ang_z = f"{r['Ang_RMS_Z']:>7.2f}" if r['Ang_RMS_Z'] else "    N/A"
    #     print(f"{r['Experiment']:<12} | {r['Lin_RMS_X']:>7.2f} {r['Lin_RMS_Y']:>7.2f} {r['Lin_RMS_Z']:>7.2f}        | {ang_x} {ang_y} {ang_z}")
    # print("="*90)

    # Create bar chart - 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    labels = [r['Experiment'] for r in results]
    x = np.arange(len(labels))
    width = 0.6

    # Colors: baseline black, others blue
    colors = ['black'] + ['#1f77b4'] * (len(results) - 1)

    # Top row: Linear acceleration
    # Add row label for linear acceleration (vertical)
    fig.text(0.02, 0.75, 'Linear Acc (Baseline: Robot CoM, Others: Payload CoM)',
             fontsize=8, va='center', ha='center', style='italic', color='black', rotation=90)

    for col, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = axes[0, col]
        values = [r[f'Lin_RMS_{axis_name}'] for r in results]
        bars = ax.bar(x, values, width, color=colors, alpha=0.8)
        ax.set_ylabel('RMS (m/s^2)', fontsize=9)
        ax.set_title(f'Linear Acc {axis_name}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=6)

    # Bottom row: Angular acceleration (skip baseline)
    # Add row label for angular acceleration (vertical)
    fig.text(0.02, 0.25, 'Angular Acc (Payload CoM only)',
             fontsize=8, va='center', ha='center', style='italic', color='black', rotation=90)

    labels_ang = [r['Experiment'] for r in results if not r['is_baseline']]
    x_ang = np.arange(len(labels_ang))
    colors_ang = ['#1f77b4'] * len(labels_ang)

    for col, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = axes[1, col]
        values = [r[f'Ang_RMS_{axis_name}'] for r in results if not r['is_baseline']]
        bars = ax.bar(x_ang, values, width, color=colors_ang, alpha=0.8)
        ax.set_ylabel('RMS (rad/s^2)', fontsize=9)
        ax.set_title(f'Angular Acc {axis_name}', fontsize=10, fontweight='bold')
        ax.set_xticks(x_ang)
        ax.set_xticklabels(labels_ang, rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)  # Make room for row labels

    if save_table:
        fig.savefig('acceleration_rms_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: acceleration_rms_comparison.png")

    return results, fig


def plot_comparison_all_experiments(save_fig: bool = True, skip_seconds: float = None):
    """
    Compare all 6 experiments: Z position, payload linear acc X, payload angular acc Z.
    Designed for paper - compact layout with baseline highlighted.

    Args:
        save_fig: Whether to save the figure
        skip_seconds: Seconds to skip from beginning (uses global SKIP_SECONDS if None)
    """
    if skip_seconds is None:
        skip_seconds = SKIP_SECONDS
    skip_samples = int(skip_seconds / DT)

    # Define experiments with short labels
    experiments = [
        ('baseline_flat_without_payload_waypoint_20260123_182835.h5', 'Baseline (no payload)', True),
        ('flat_terrain_center_payload_5kg_waypoint_20260123_184135.h5', 'Center', False),
        ('flat_terrain_payload_5kg_forward_0.15_meter_waypoint_20260123_184832.h5', 'Forward 0.15m', False),
        # ('flat_terrain_payload_5kg_forward_0.25_meter_waypoint_20260123_184710.h5', 'Forward 0.25m', False),
        ('flat_terrain_payload_5kg_backwards_0.15_meter_waypoint_20260123_184958.h5', 'Backward 0.15m', False),
        # ('flat_terrain_payload_5kg_backwards_0.25_meter_waypoint_20260123_185320.h5', 'Backward 0.25m', False),
    ]

    # Colors: baseline black, others use colormap
    colors = ['black', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    print(f"\n[Config] Skipping first {skip_seconds}s ({skip_samples} samples) of each experiment")

    # Load all data
    all_data = []
    for filename, label, is_baseline in experiments:
        filepath = DATA_DIR / filename
        data = HDF5Reader.load(str(filepath), dt=DT)
        all_data.append((data, label, is_baseline))
        print(f"Loaded: {label} ({data.timesteps} timesteps, using {data.timesteps - skip_samples} after trim)")

    # Create figure - 7 rows, compact for paper
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    ax_z, ax_lx, ax_ly, ax_lz, ax_ax, ax_ay, ax_az = axes

    for i, (data, label, is_baseline) in enumerate(all_data):
        time = data.get_field('sim_time').flatten()[skip_samples:]
        color = colors[i]
        lw = 2.0 if is_baseline else 1.0
        alpha = 1.0 if is_baseline else 0.7

        # Plot 1: Z Position (trimmed)
        if is_baseline:
            z_pos = data.get_field('robot_pos_w')[skip_samples:, 2]
        else:
            z_pos = data.get_field('payload_pos_w')[skip_samples:, 2]
        ax_z.plot(time, z_pos, color=color, linewidth=lw, label=label, alpha=alpha)

        # Linear Accelerations (trimmed)
        if is_baseline:
            lin_acc = data.get_field('robot_lin_acc_b')[skip_samples:]
        else:
            lin_acc = data.get_field('payload_lin_acc_b')[skip_samples:]
        ax_lx.plot(time, lin_acc[:, 0], color=color, linewidth=lw, label=label, alpha=alpha)
        ax_ly.plot(time, lin_acc[:, 1], color=color, linewidth=lw, label=label, alpha=alpha)
        ax_lz.plot(time, lin_acc[:, 2], color=color, linewidth=lw, label=label, alpha=alpha)

        # Angular Accelerations (skip baseline - no data, trimmed)
        if not is_baseline:
            ang_acc = data.get_field('payload_ang_acc_b')[skip_samples:]
            ax_ax.plot(time, ang_acc[:, 0], color=color, linewidth=lw, label=label, alpha=alpha)
            ax_ay.plot(time, ang_acc[:, 1], color=color, linewidth=lw, label=label, alpha=alpha)
            ax_az.plot(time, ang_acc[:, 2], color=color, linewidth=lw, label=label, alpha=alpha)

    # Format axes
    ax_z.set_ylabel('Z Pos (m)', fontsize=9)
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(fontsize=7, loc='upper right', ncol=2)

    ax_lx.set_ylabel('Lin Acc X (m/s^2)', fontsize=9)
    ax_lx.grid(True, alpha=0.3)

    ax_ly.set_ylabel('Lin Acc Y (m/s^2)', fontsize=9)
    ax_ly.grid(True, alpha=0.3)

    ax_lz.set_ylabel('Lin Acc Z (m/s^2)', fontsize=9)
    ax_lz.grid(True, alpha=0.3)

    ax_ax.set_ylabel('Ang Acc X (rad/s^2)', fontsize=9)
    ax_ax.grid(True, alpha=0.3)

    ax_ay.set_ylabel('Ang Acc Y (rad/s^2)', fontsize=9)
    ax_ay.grid(True, alpha=0.3)

    ax_az.set_ylabel('Ang Acc Z (rad/s^2)', fontsize=9)
    ax_az.set_xlabel('Time (s)', fontsize=10)
    ax_az.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    if save_fig:
        output_path = 'comparison_all_experiments.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_trajectory_full(filepath: str, save_fig: bool = True):
    """
    Plot full experiment data: X-Y trajectory, Z position, linear and angular acceleration.

    Args:
        filepath: Path to HDF5 experiment file
        save_fig: Whether to save the figure
    """
    # Load data
    data = HDF5Reader.load(filepath, dt=0.02)

    print(f"Loaded: {data}")
    print(f"Fields: {data.list_fields()}")

    # Get data
    robot_pos = data.get_field('robot_pos_w')
    payload_pos = data.get_field('payload_pos_w')
    time = data.get_field('sim_time').flatten()
    # robot_lin_acc = data.get_field('robot_lin_acc_b')
    # robot_ang_acc = data.get_field('robot_ang_acc_b')
    payload_lin_acc = data.get_field('payload_lin_acc_b')
    payload_ang_acc = data.get_field('payload_ang_acc_b')

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[2, 1, 1, 1])
    ax1, ax2, ax3, ax4 = axes

    # Plot 1: X-Y Trajectory
    ax1.plot(robot_pos[:, 0], robot_pos[:, 1],
             'b-', linewidth=2, label='Robot', alpha=0.8)
    ax1.plot(payload_pos[:, 0], payload_pos[:, 1],
             'orange', linewidth=2, label='Payload', alpha=0.8, linestyle='--')
    ax1.plot(robot_pos[0, 0], robot_pos[0, 1],
             'go', markersize=12, label='Start', zorder=5)
    ax1.plot(robot_pos[-1, 0], robot_pos[-1, 1],
             'ro', markersize=12, label='End', zorder=5)
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('X-Y Trajectory', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Z Position
    ax2.plot(time, robot_pos[:, 2], 'b-', linewidth=1.5, label='Robot Z', alpha=0.8)
    ax2.plot(time, payload_pos[:, 2], 'orange', linewidth=1.5, label='Payload Z', alpha=0.8, linestyle='--')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Z Position (m)', fontsize=11)
    # ax2.set_title('Z Position Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Linear Acceleration
    # ax3.plot(time, robot_lin_acc[:, 0], 'b-', linewidth=1, label='Robot X', alpha=0.7)
    # ax3.plot(time, robot_lin_acc[:, 1], 'g-', linewidth=1, label='Robot Y', alpha=0.7)
    # ax3.plot(time, robot_lin_acc[:, 2], 'r-', linewidth=1, label='Robot Z', alpha=0.7)
    ax3.plot(time, payload_lin_acc[:, 0], 'b--', linewidth=1, label='Payload X', alpha=0.7)
    ax3.plot(time, payload_lin_acc[:, 1], 'g--', linewidth=1, label='Payload Y', alpha=0.7)
    ax3.plot(time, payload_lin_acc[:, 2], 'r--', linewidth=1, label='Payload Z', alpha=0.7)

    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Payload Linear Acc (m/s^2) body frame', fontsize=11)
    # ax3.set_title('Linear Acceleration (Body Frame)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Angular Acceleration (payload only, robot_ang_acc may not exist)
    ax4.plot(time, payload_ang_acc[:, 0], 'b-', linewidth=1, label='Payload Roll', alpha=0.8)
    ax4.plot(time, payload_ang_acc[:, 1], 'g-', linewidth=1, label='Payload Pitch', alpha=0.8)
    # ax4.plot(time, payload_ang_acc[:, 2], 'r-', linewidth=1, label='Payload Yaw', alpha=0.8)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Payload Angular Acc (rad/s^2) body frame', fontsize=11)
    # ax4.set_title('Payload Angular Acceleration (Body Frame)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'{Path(filepath).stem}', fontsize=11, y=0.995, color='gray')
    plt.tight_layout()

    if save_fig:
        output_path = Path(filepath).stem + '_full.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_trajectory_2d(filepath: str, save_fig: bool = True):
    """
    Plot 2D (X-Y) trajectories of robot and payload.

    Args:
        filepath: Path to HDF5 experiment file
        save_fig: Whether to save the figure
    """
    # Load data
    data = HDF5Reader.load(filepath, dt=0.02)

    print(f"Loaded: {data}")
    print(f"Fields: {data.list_fields()}")

    # Get position data
    robot_pos = data.get_field('robot_pos_w')
    payload_pos = data.get_field('payload_pos_w')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot robot trajectory
    ax.plot(robot_pos[:, 0], robot_pos[:, 1],
            'b-', linewidth=2, label='Robot', alpha=0.8)
    ax.plot(robot_pos[0, 0], robot_pos[0, 1],
            'go', markersize=12, label='Start', zorder=5)
    ax.plot(robot_pos[-1, 0], robot_pos[-1, 1],
            'ro', markersize=12, label='End', zorder=5)

    # Plot payload trajectory
    ax.plot(payload_pos[:, 0], payload_pos[:, 1],
            'orange', linewidth=2, label='Payload', alpha=0.8, linestyle='--')

    # Labels and formatting
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Robot & Payload Trajectory\n{Path(filepath).stem}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_fig:
        output_path = Path(filepath).stem + '_trajectory.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_joint_torques(filepath: str, skip_seconds: float = None, save_fig: bool = True):
    """
    Plot joint torques for a single experiment.
    Spot robot has 12 joints (3 per leg x 4 legs).

    Args:
        filepath: Path to HDF5 experiment file
        skip_seconds: Seconds to skip from beginning (uses global SKIP_SECONDS if None)
        save_fig: Whether to save the figure
    """
    if skip_seconds is None:
        skip_seconds = SKIP_SECONDS
    skip_samples = int(skip_seconds / DT)

    # Load data
    data = HDF5Reader.load(filepath, dt=DT)

    print(f"Loaded: {data}")
    print(f"Fields: {data.list_fields()}")

    # Get data (trimmed)
    time = data.get_field('sim_time').flatten()[skip_samples:]
    torques = data.get_field('robot_joint_torques')[skip_samples:]

    # Spot joint order from IsaacLab:
    # ['fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn']
    # Indices 0-3: Hip X (all legs), 4-7: Hip Y (all legs), 8-11: Knee (all legs)
    # Within each type: FL, FR, HL, HR

    # Create figure - 4 rows (legs) x 3 cols (joint types)
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)

    leg_names = ['Front Left', 'Front Right', 'Hind Left', 'Hind Right']
    joint_types = ['Hip X (Abd)', 'Hip Y (Flex)', 'Knee']

    for row in range(4):  # legs: FL, FR, HL, HR
        for col in range(3):  # joint types: Hip X, Hip Y, Knee
            ax = axes[row, col]
            # Data index: joint_type * 4 + leg
            data_idx = col * 4 + row

            ax.plot(time, torques[:, data_idx], linewidth=1, alpha=0.8)
            ax.set_ylabel('Torque (N·m)', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add title only for top row
            if row == 0:
                ax.set_title(joint_types[col], fontsize=10, fontweight='bold')

            # Add leg label on left
            if col == 0:
                ax.text(-0.15, 0.5, leg_names[row], transform=ax.transAxes,
                        fontsize=9, va='center', ha='right', fontweight='bold')

    # Add x-label only for bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(f'Joint Torques - {Path(filepath).stem}', fontsize=11, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)

    if save_fig:
        output_path = Path(filepath).stem + '_joint_torques.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_joint_torques_comparison(skip_seconds: float = None, save_fig: bool = True):
    """
    Compare joint torques across all 6 experiments using box plots.
    6 subplots (one per experiment), each with 12 box plots (one per joint).

    Args:
        skip_seconds: Seconds to skip from beginning (uses global SKIP_SECONDS if None)
        save_fig: Whether to save the figure
    """
    if skip_seconds is None:
        skip_seconds = SKIP_SECONDS
    skip_samples = int(skip_seconds / DT)

    # Define experiments
    experiments = [
        ('baseline_flat_without_payload_waypoint_20260123_182835.h5', 'Baseline'),
        ('flat_terrain_center_payload_5kg_waypoint_20260123_184135.h5', 'Center'),
        ('flat_terrain_payload_5kg_forward_0.15_meter_waypoint_20260123_184832.h5', 'Fwd 0.15m'),
        ('flat_terrain_payload_5kg_forward_0.25_meter_waypoint_20260123_184710.h5', 'Fwd 0.25m'),
        ('flat_terrain_payload_5kg_backwards_0.15_meter_waypoint_20260123_184958.h5', 'Bwd 0.15m'),
        ('flat_terrain_payload_5kg_backwards_0.25_meter_waypoint_20260123_185320.h5', 'Bwd 0.25m'),
    ]

    # Joint names in order (from IsaacLab)
    # ['fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn']
    joint_labels = [
        'FL_hx', 'FR_hx', 'HL_hx', 'HR_hx',
        'FL_hy', 'FR_hy', 'HL_hy', 'HR_hy',
        'FL_kn', 'FR_kn', 'HL_kn', 'HR_kn'
    ]

    print(f"\n[Config] Skipping first {skip_seconds}s ({skip_samples} samples) of each experiment")

    # Create figure - 2 rows x 3 cols for 6 experiments
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for exp_idx, (filename, label) in enumerate(experiments):
        filepath = DATA_DIR / filename
        data = HDF5Reader.load(str(filepath), dt=DT)
        torques = data.get_field('robot_joint_torques')[skip_samples:]

        print(f"Loaded: {label} ({torques.shape[0]} samples after trim)")

        ax = axes[exp_idx]

        # Prepare data for box plot (list of 12 arrays)
        box_data = [torques[:, i] for i in range(12)]

        # Create box plot
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False)

        # Color by joint type: Hip X (blue), Hip Y (green), Knee (orange)
        colors = ['#1f77b4'] * 4 + ['#2ca02c'] * 4 + ['#ff7f0e'] * 4
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Change median line color (default is orange, change to black for visibility)
        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(1.25)

        ax.set_xticklabels(joint_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Torque (N·m)', fontsize=9)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical separators between joint types
        ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=8.5, color='gray', linestyle='--', alpha=0.5)

    # Add legend for joint types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='Hip X (Abduction)'),
        Patch(facecolor='#2ca02c', alpha=0.7, label='Hip Y (Flexion)'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='Knee'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle('Joint Torque Distribution Comparison', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_fig:
        output_path = 'joint_torques_comparison_boxplot.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

    return fig


if __name__ == "__main__":
    # List available files
    print("Available experiment files:")
    for f in sorted(DATA_DIR.glob('*.h5')):
        print(f"  - {f.name}")
    print()

    # Plot the first payload experiment
    experiment_file = DATA_DIR / 'flat_terrain_center_payload_5kg_waypoint_20260123_184135.h5'

    # Just to plot robot and payload position in x,y.
    # fig_2d = plot_trajectory_2d(str(experiment_file))

    # Plots robot and payload position in x,y + another plot for only z component
    # fig_with_z = plot_trajectory_with_z(str(experiment_file))

    # Full plot: X-Y, Z, linear acc, angular acc
    # fig_full = plot_trajectory_full(str(experiment_file))

    # Comparison plot for paper
    # fig_compare = plot_comparison_all_experiments()

    # Compute acceleration metrics (RMS) with labeled data sources
    results, fig_metrics = compute_acceleration_metrics()

    # Plot joint torques for single experiment
    # fig_torques = plot_joint_torques(str(experiment_file))

    # Compare joint torques across all experiments (box plots)
    # fig_torques_compare = plot_joint_torques_comparison()

    plt.show()
