"""
Plot robot and payload trajectories from HDF5 experiment data.
"""

import sys
sys.path.append('/home/manav/my_isaaclab_project')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hdf5_logger_utils import HDF5Reader

# Data directories
DATA_DIR = Path('/home/manav/Desktop/data_collection/simulation/flat_terrain')
DATA_DIR_5KG = Path('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_robot_ang_acc')
DATA_DIR_8KG = Path('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload')
DATA_DIR_BASELINE = Path('/home/manav/Desktop/data_collection/simulation/baseline_vel_1.1')

# =============================================================================
# Configuration: Skip initial seconds (robot drop phase)
# =============================================================================
SKIP_SECONDS = 3.5  # Skip first N seconds of data (robot drop/stabilization)
DT = 0.02           # Data collection timestep


def plot_joint_torques_comparison(experiments, title="Joint Torque Comparison", output_file="joint_torques_comparison.png"):
    """
    Compare joint torques across experiments using box plots.
    12 subplots (one per joint), each with N box plots (one per experiment).

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
                    - folder_path: Path to folder containing .h5 files
                    - label: Display name for this experiment group
        title: Plot title
        output_file: Output filename for the saved plot
    """
    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s\n")

    # Column labels (leg names) and row labels (joint types)
    leg_labels = ['FL', 'FR', 'HL', 'HR']
    row_titles = ['Abductor', 'Hip', 'Knee']

    # Load all experiment data
    all_experiment_data = []
    global_max = 0

    for folder_path, label, _ in experiments:
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        print(f"Processing: {label} ({len(h5_files)} files)")

        all_torques = []
        for h5_file in h5_files:
            data = HDF5Reader.load(str(h5_file), dt=DT)
            torques = data.get_field('robot_joint_torques')

            if torques.shape[0] <= skip_samples + 10:
                print(f"  Skipping {h5_file.name} (empty)")
                continue

            all_torques.append(np.abs(torques[skip_samples:]))

        combined_torques = np.concatenate(all_torques, axis=0)
        print(f"  Total samples: {combined_torques.shape[0]}")

        all_experiment_data.append((label, combined_torques))
        global_max = max(global_max, combined_torques.max())

    # 12 subplots: 3 rows (Hip X, Hip Y, Knee) x 4 cols (FL, FR, HL, HR)
    fig, axes = plt.subplots(3, 4, figsize=(10, 5))
    fig.set_facecolor('white')

    exp_labels = [label for label, _ in all_experiment_data]

    for joint_idx in range(12):
        row = joint_idx // 4
        col = joint_idx % 4
        ax = axes[row, col]
        ax.set_facecolor('white')

        # Collect data for this joint from all experiments
        box_data = [exp_data[:, joint_idx] for _, exp_data in all_experiment_data]

        # Create box plot (professional: black outline, no fill)
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False)

        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        for whisker in bp['whiskers']:
            whisker.set_color('black')

        for cap in bp['caps']:
            cap.set_color('black')

        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(1.5)

        # Only show x-axis labels on bottom row
        if row == 2:
            ax.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=10.5)
        else:
            ax.set_xticklabels([])

        # Only show row title on first column
        if col == 0:
            ax.set_ylabel(row_titles[row], fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('')

        ax.set_title(leg_labels[col], fontsize=10, fontweight='bold')
        ax.set_ylim(0, global_max * 1.05)

    # Shared y-axis label for entire figure
    fig.text(0.03, 0.5, '|Torque| (N·m)', va='center', rotation='vertical', fontsize=14)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.04, 0, 1, 1])  # Leave space for shared ylabel

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

    return fig


def compute_jerk_mae(experiments, title="Jerk MAE Comparison", output_file="jerk_mae.png"):
    """
    Compute and compare jerk MAE across multiple experiments.

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
                    - folder_path: Path to folder containing .h5 files
                    - label: Display name for this experiment group
                    - is_baseline: True for baseline (black color), False for others (blue)
        title: Plot title
        output_file: Output filename for the saved plot

    Example:
        experiments = [
            ('/path/to/baseline', 'Baseline', True),
            ('/path/to/center', 'Center', False),
            ('/path/to/forward', 'Forward', False),
        ]
        compute_jerk_mae(experiments, title="8kg Payload", output_file="jerk_8kg.png")
    """
    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s\n")

    results = []

    for folder_path, label, is_baseline in experiments:
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        print(f"Processing: {label} ({len(h5_files)} files)")

        lin_jerk_mae = {'x': [], 'y': [], 'z': []}
        ang_jerk_mae = {'x': [], 'y': [], 'z': []}

        for h5_file in h5_files:
            data = HDF5Reader.load(str(h5_file), dt=DT)
            lin_acc = data.get_field('robot_lin_acc_b')

            # Skip empty files
            if lin_acc.shape[0] <= skip_samples + 10:
                print(f"  Skipping {h5_file.name} (empty)")
                continue

            # Compute jerk = d(acc)/dt
            lin_acc = lin_acc[skip_samples:]
            ang_acc = data.get_field('robot_ang_acc_b')[skip_samples:]

            lin_jerk = np.diff(lin_acc, axis=0) / DT
            ang_jerk = np.diff(ang_acc, axis=0) / DT

            for i, axis in enumerate(['x', 'y', 'z']):
                lin_jerk_mae[axis].append(np.mean(np.abs(lin_jerk[:, i])))
                ang_jerk_mae[axis].append(np.mean(np.abs(ang_jerk[:, i])))

        # Store results
        results.append({
            'label': label,
            'is_baseline': is_baseline,
            'lin_x': np.mean(lin_jerk_mae['x']), 'lin_x_std': np.std(lin_jerk_mae['x']),
            'lin_y': np.mean(lin_jerk_mae['y']), 'lin_y_std': np.std(lin_jerk_mae['y']),
            'lin_z': np.mean(lin_jerk_mae['z']), 'lin_z_std': np.std(lin_jerk_mae['z']),
            'ang_x': np.mean(ang_jerk_mae['x']), 'ang_x_std': np.std(ang_jerk_mae['x']),
            'ang_y': np.mean(ang_jerk_mae['y']), 'ang_y_std': np.std(ang_jerk_mae['y']),
            'ang_z': np.mean(ang_jerk_mae['z']), 'ang_z_std': np.std(ang_jerk_mae['z']),
        })

        r = results[-1]
        print(f"  Linear:  X={r['lin_x']:.1f}±{r['lin_x_std']:.1f}, Y={r['lin_y']:.1f}±{r['lin_y_std']:.1f}, Z={r['lin_z']:.1f}±{r['lin_z_std']:.1f}")
        print(f"  Angular: X={r['ang_x']:.1f}±{r['ang_x_std']:.1f}, Y={r['ang_y']:.1f}±{r['ang_y_std']:.1f}, Z={r['ang_z']:.1f}±{r['ang_z_std']:.1f}")

    # Find max values for consistent scaling
    lin_max = max(r[f'lin_{axis}'] + r[f'lin_{axis}_std'] for r in results for axis in ['x', 'y', 'z'])
    ang_max = max(r[f'ang_{axis}'] + r[f'ang_{axis}_std'] for r in results for axis in ['x', 'y', 'z'])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))
    fig.set_facecolor('white')
    labels = [r['label'] for r in results]
    x = np.arange(len(labels))

    # Top: Linear jerk, Bottom: Angular jerk
    for row, (prefix, y_max) in enumerate([('lin', lin_max), ('ang', ang_max)]):
        for col, axis in enumerate(['x', 'y', 'z']):
            ax = axes[row, col]
            ax.set_facecolor('white')
            vals = [r[f'{prefix}_{axis}'] for r in results]
            stds = [r[f'{prefix}_{axis}_std'] for r in results]

            ax.bar(x, vals, 0.4, color='white', edgecolor='black', linewidth=1, yerr=stds, capsize=3, error_kw={'linewidth': 1, 'ecolor': 'red'})
            ax.set_title(f'{"Linear" if prefix=="lin" else "Angular"} Jerk {axis.upper()}', fontweight='bold', fontsize=12)
            ax.set_xticks(x)
            ax.set_ylim(0, y_max * 1.1)

            # Only show x labels on bottom row
            if row == 1:
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            else:
                ax.set_xticklabels([])

    # Merged Y labels
    fig.text(0.03, 0.72, 'Jerk MAE (m/s^3)', va='center', rotation='vertical', fontsize=10)
    fig.text(0.03, 0.28, 'Jerk MAE (rad/s^3)', va='center', rotation='vertical', fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.98])
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Lowest jerk per axis")
    print(f"{'='*70}")
    for prefix, name in [('lin', 'Linear'), ('ang', 'Angular')]:
        print(f"{name}:")
        for axis in ['x', 'y', 'z']:
            best = min(results, key=lambda r: r[f'{prefix}_{axis}'])
            print(f"  {axis.upper()}: {best['label']} ({best[f'{prefix}_{axis}']:.1f})")

    return fig, results


def plot_acceleration_signals(experiments, title="Acceleration Comparison", output_file="acceleration_signals.png", exp_index=0):
    """
    Compare acceleration signals across experiments as time-series plots.

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
                    - folder_path: Path to folder containing .h5 files
                    - label: Display name for this experiment group
                    - is_baseline: Not used, for interface consistency
        title: Plot title
        output_file: Output filename for the saved plot
        exp_index: Which experiment file to use from each folder (0-4)
    """
    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s")
    print(f"[Config] Using experiment file index: {exp_index}\n")

    # Load data from each experiment
    all_data = []
    for folder_path, label, _ in experiments:
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        if exp_index >= len(h5_files):
            print(f"  Warning: {label} has only {len(h5_files)} files, using last one")
            h5_file = h5_files[-1]
        else:
            h5_file = h5_files[exp_index]

        print(f"Processing: {label} ({h5_file.name})")

        data = HDF5Reader.load(str(h5_file), dt=DT)
        lin_acc = data.get_field('robot_lin_acc_b')[skip_samples:]
        ang_acc = data.get_field('robot_ang_acc_b')[skip_samples:]

        # Create time array
        time = np.arange(lin_acc.shape[0]) * DT

        all_data.append({
            'label': label,
            'time': time,
            'lin_acc': lin_acc,
            'ang_acc': ang_acc
        })

        print(f"  Samples: {lin_acc.shape[0]}, Duration: {time[-1]:.1f}s")

    # Create figure - 6 rows x 1 column layout
    fig, axes = plt.subplots(6, 1, figsize=(12, 10))
    fig.set_facecolor('white')

    axis_names = ['X', 'Y', 'Z']

    # First 3 rows: Linear acceleration (X, Y, Z)
    for i in range(3):
        ax = axes[i]
        ax.set_facecolor('white')

        for exp_data in all_data:
            ax.plot(exp_data['time'], exp_data['lin_acc'][:, i], label=exp_data['label'], linewidth=0.8)

        ax.set_ylabel(f'Lin Acc {axis_names[i]}\n(m/s^2)', fontweight='bold', fontsize=9, labelpad=2)
        ax.set_xticklabels([])

    # Last 3 rows: Angular acceleration (X, Y, Z)
    for i in range(3):
        ax = axes[i + 3]
        ax.set_facecolor('white')

        for exp_data in all_data:
            ax.plot(exp_data['time'], exp_data['ang_acc'][:, i], label=exp_data['label'], linewidth=0.8)

        ax.set_ylabel(f'Ang Acc {axis_names[i]}\n(rad/s^2)', fontweight='bold', fontsize=9, labelpad=2)
        # Only show X label on bottom subplot
        if i == 2:
            ax.set_xlabel('Time (s)', fontsize=9)
        else:
            ax.set_xticklabels([])

    # Add legend to first subplot with border
    axes[0].legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black', facecolor='white')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

    return fig


def plot_xy_trajectory(experiments, title="XY Trajectory Comparison", output_file="xy_trajectory.png", exp_index=0):
    """
    Plot XY trajectory for each experiment in separate subplots.

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
                    - folder_path: Path to folder containing .h5 files
                    - label: Display name for this experiment group
                    - is_baseline: Not used, for interface consistency
        title: Plot title
        output_file: Output filename for the saved plot
        exp_index: Which experiment file to use from each folder (0-4)
    """
    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s")
    print(f"[Config] Using experiment file index: {exp_index}\n")

    # Reference waypoints from main_with_control_toggle.py
    waypoints_xy = np.array([
        [-1, 0], [0.34, 0], [1.52, 0], [2.90, 0], [4, 1.2], [5.2, 1.2],
        [6.57, 1.2], [6.57, 0], [5.57, 0], [4.07, 0], [3.16, 1.26],
        [1.87, 1.26], [0.64, 1.26], [0.64, 0.5]
    ])

    # Calculate grid size
    n_experiments = len(experiments)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 1.5 * n_rows))
    fig.set_facecolor('white')
    axes = axes.flatten() if n_experiments > 1 else [axes]

    for exp_idx, (folder_path, label, _) in enumerate(experiments):
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        if exp_index >= len(h5_files):
            print(f"  Warning: {label} has only {len(h5_files)} files, using last one")
            h5_file = h5_files[-1]
        else:
            h5_file = h5_files[exp_index]

        print(f"Processing: {label} ({h5_file.name})")

        data = HDF5Reader.load(str(h5_file), dt=DT)
        pos = data.get_field('robot_pos_w')

        x = pos[:, 0]
        y = pos[:, 1]

        # Apply offset to waypoints based on robot's starting position
        offset_x = x[0] - waypoints_xy[0, 0]
        offset_y = y[0] - waypoints_xy[0, 1]
        wp_x = waypoints_xy[:, 0] + offset_x
        wp_y = waypoints_xy[:, 1] + offset_y

        print(f"  Samples: {pos.shape[0]}")

        row = exp_idx // n_cols
        col = exp_idx % n_cols
        ax = axes[exp_idx]
        ax.set_facecolor('white')

        # Plot waypoints (dots only)
        ax.plot(wp_x, wp_y, 'b.', markersize=3.5)

        # Plot actual trajectory
        ax.plot(x, y, 'k-', linewidth=1)
        ax.plot(x[0], y[0], 'go', markersize=5)
        ax.plot(x[-1], y[-1], 'ro', markersize=5)

        # Only show X label on bottom row
        if row == n_rows - 1:
            ax.set_xlabel('X (m)', fontsize=9)

        # Only show Y label on first column
        if col == 0:
            ax.set_ylabel('Y (m)', fontsize=9)

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_aspect('equal')

    # Hide unused axes
    for i in range(n_experiments, len(axes)):
        axes[i].set_visible(False)

    # Add shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='.', color='w', markerfacecolor='b', markersize=8, label='Waypoints'),
        Line2D([0], [0], color='k', linestyle='-', linewidth=0.8, label='Actual'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=5, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=5, label='End'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=10.5,
               frameon=True, edgecolor='black', facecolor='white')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

    return fig


if __name__ == "__main__":
    # List available files
    print("Available experiment files:")
    for f in sorted(DATA_DIR.glob('*.h5')):
        print(f"  - {f.name}")
    print()

    # Plot the first payload experiment
    experiment_file = DATA_DIR / 'flat_terrain_center_payload_5kg_waypoint_20260123_184135.h5'

    # ==========================================================================
    # Define experiments: (folder_path, label, is_baseline)
    # ==========================================================================
    experiments_8kg = [
        ('/home/manav/Desktop/data_collection/simulation/baseline_vel_1.1', 'Baseline', True),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/center_payload_8kg', 'Center', False),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/forward_0.15_payload_8kg', 'Fwd 0.15m', False),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/forward_0.25_payload_8kg', 'Fwd 0.25m', False),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/backward_0.15_payload_8kg', 'Bwd 0.15m', False),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/backward_0.25_payload_8kg', 'Bwd 0.25m', False),
    ]

  
    # Compute and plot jerk MAE
    # fig_jerk, jerk_results = compute_jerk_mae(
    #     experiments_8kg,
    #     title="Jerk MAE Comparison - 8kg Payload",
    #     output_file="jerk_mae_8kg.png"
    # )

    # fig = plot_joint_torques_comparison(experiments_8kg, title="Joint Torque Comparison", output_file="joint_torques_comparison.png")

    # fig = plot_acceleration_signals(experiments_8kg, title="Acceleration Comparison", output_file="acc.png", exp_index=0)

    fig = plot_xy_trajectory(experiments_8kg, title="XY Trajectory", output_file="xy.png", exp_index=2)

    plt.show()
