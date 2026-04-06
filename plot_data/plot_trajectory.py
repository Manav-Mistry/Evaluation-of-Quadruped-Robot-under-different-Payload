"""
Plot robot and payload trajectories from HDF5 experiment data.
"""

import sys
sys.path.append('/home/manav/my_isaaclab_project')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mstats
from matplotlib.collections import LineCollection
from hdf5_logger_utils import HDF5Reader

# Data directories
DATA_DIR = Path('/home/manav/Desktop/data_collection/simulation/flat_terrain')
DATA_DIR_5KG = Path('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_robot_ang_acc')
DATA_DIR_8KG = Path('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload')
DATA_DIR_BASELINE = Path('/home/manav/Desktop/data_collection/simulation/baseline_vel_1.1')

# =============================================================================
# Configuration: Skip initial seconds (robot drop phase)
# =============================================================================
SKIP_SECONDS = 1.6  # Skip first N seconds of data (robot drop/stabilization)
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

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {output_file}")

    return fig


def compute_jerk_rms(experiments, title="Jerk RMS Comparison", output_file="jerk_rms.png"):
    """
    Compute and compare jerk RMS across multiple experiments.
        jerk[i] = (acc[i+1] - acc[i-1]) / (2 * dt)

    compute_jerk_rms(experiments, title="8kg Payload", output_file="jerk_8kg.png")
    """
    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s")
    print(f"[Method] Central difference, RMS metric")
    print(f"[Filter] Clipping: lin_acc ±20 m/s^2, ang_acc ±50 rad/s^2")
    print(f"[Filter] Low-pass: Butterworth 2nd order, 10 Hz cutoff\n")

    results = []

    for folder_path, label, is_baseline in experiments:
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        print(f"Processing: {label} ({len(h5_files)} files)")

        lin_jerk_rms = {'x': [], 'y': [], 'z': []}
        ang_jerk_rms = {'x': [], 'y': [], 'z': []}
        lin_acc_rms = {'x': [], 'y': [], 'z': []}
        ang_acc_rms = {'x': [], 'y': [], 'z': []}
        lin_acc_peak = {'x': [], 'y': [], 'z': []}
        ang_acc_peak = {'x': [], 'y': [], 'z': []}

        for h5_file in h5_files:
            data = HDF5Reader.load(str(h5_file), dt=DT)
            lin_acc = data.get_field('robot_lin_acc_b')

            # Skip empty files
            if lin_acc.shape[0] <= skip_samples + 10:
                print(f"  Skipping {h5_file.name} (empty)")
                continue

            # Skip initial samples
            lin_acc = lin_acc[skip_samples:]
            ang_acc = data.get_field('robot_ang_acc_b')[skip_samples:]

            # Clip acceleration
            lin_acc = np.clip(lin_acc, -20, 20)   # m/s^2 (~2g)
            ang_acc = np.clip(ang_acc, -50, 50)   # rad/s^2 (realistic limit for walking)

            # low-pass filter (Butterworth 2nd order, 10 Hz cutoff, SOS form)
            fs = 1 / DT  # Sampling frequency (50 Hz)
            cutoff = 10  # Hz
            nyq = 0.5 * fs
            sos = butter(2, cutoff / nyq, btype='low', output='sos')
            lin_acc = sosfiltfilt(sos, lin_acc, axis=0)
            ang_acc = sosfiltfilt(sos, ang_acc, axis=0)

            # Compute jerk using central difference: jerk[i] = (acc[i+1] - acc[i-1]) / (2*dt)
            lin_jerk = (lin_acc[2:] - lin_acc[:-2]) / (2 * DT)
            ang_jerk = (ang_acc[2:] - ang_acc[:-2]) / (2 * DT)

            for i, axis in enumerate(['x', 'y', 'z']):
                lin_jerk_rms[axis].append(np.sqrt(np.mean(lin_jerk[:, i]**2)))
                ang_jerk_rms[axis].append(np.sqrt(np.mean(ang_jerk[:, i]**2)))
                lin_acc_rms[axis].append(np.sqrt(np.mean(lin_acc[:, i]**2)))
                ang_acc_rms[axis].append(np.sqrt(np.mean(ang_acc[:, i]**2)))
                lin_acc_peak[axis].append(np.max(np.abs(lin_acc[:, i])))
                ang_acc_peak[axis].append(np.max(np.abs(ang_acc[:, i])))

        # Store results
        results.append({
            'label': label,
            'is_baseline': is_baseline,
            'lin_x': np.mean(lin_jerk_rms['x']), 'lin_x_std': np.std(lin_jerk_rms['x']),
            'lin_y': np.mean(lin_jerk_rms['y']), 'lin_y_std': np.std(lin_jerk_rms['y']),
            'lin_z': np.mean(lin_jerk_rms['z']), 'lin_z_std': np.std(lin_jerk_rms['z']),
            'ang_x': np.mean(ang_jerk_rms['x']), 'ang_x_std': np.std(ang_jerk_rms['x']),
            'ang_y': np.mean(ang_jerk_rms['y']), 'ang_y_std': np.std(ang_jerk_rms['y']),
            'ang_z': np.mean(ang_jerk_rms['z']), 'ang_z_std': np.std(ang_jerk_rms['z']),
            'acc_lin_x': np.mean(lin_acc_rms['x']), 'acc_lin_x_std': np.std(lin_acc_rms['x']),
            'acc_lin_y': np.mean(lin_acc_rms['y']), 'acc_lin_y_std': np.std(lin_acc_rms['y']),
            'acc_lin_z': np.mean(lin_acc_rms['z']), 'acc_lin_z_std': np.std(lin_acc_rms['z']),
            'acc_ang_x': np.mean(ang_acc_rms['x']), 'acc_ang_x_std': np.std(ang_acc_rms['x']),
            'acc_ang_y': np.mean(ang_acc_rms['y']), 'acc_ang_y_std': np.std(ang_acc_rms['y']),
            'acc_ang_z': np.mean(ang_acc_rms['z']), 'acc_ang_z_std': np.std(ang_acc_rms['z']),
            'peak_lin_x': np.mean(lin_acc_peak['x']), 'peak_lin_x_std': np.std(lin_acc_peak['x']),
            'peak_lin_y': np.mean(lin_acc_peak['y']), 'peak_lin_y_std': np.std(lin_acc_peak['y']),
            'peak_lin_z': np.mean(lin_acc_peak['z']), 'peak_lin_z_std': np.std(lin_acc_peak['z']),
            'peak_ang_x': np.mean(ang_acc_peak['x']), 'peak_ang_x_std': np.std(ang_acc_peak['x']),
            'peak_ang_y': np.mean(ang_acc_peak['y']), 'peak_ang_y_std': np.std(ang_acc_peak['y']),
            'peak_ang_z': np.mean(ang_acc_peak['z']), 'peak_ang_z_std': np.std(ang_acc_peak['z']),
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
    fig.text(0.03, 0.72, 'Jerk RMS (m/s^3)', va='center', rotation='vertical', fontsize=10)
    fig.text(0.03, 0.28, 'Jerk RMS (rad/s^3)', va='center', rotation='vertical', fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.98])
    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {output_file}")

    # Print full results table
    print(f"\n{'='*90}")
    print("Jerk RMS Results (mean ± std)")
    print(f"{'='*90}")
    header = f"{'Experiment':<14} | {'Lin X':>10} | {'Lin Y':>10} | {'Lin Z':>10} | {'Ang X':>10} | {'Ang Y':>10} | {'Ang Z':>10}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['label']:<14} | {r['lin_x']:>5.2f}±{r['lin_x_std']:<4.2f} | {r['lin_y']:>5.2f}±{r['lin_y_std']:<4.2f} | {r['lin_z']:>5.2f}±{r['lin_z_std']:<4.2f} | {r['ang_x']:>5.2f}±{r['ang_x_std']:<4.2f} | {r['ang_y']:>5.2f}±{r['ang_y_std']:<4.2f} | {r['ang_z']:>5.2f}±{r['ang_z_std']:<4.2f}")

    # Acceleration RMS table
    print(f"\n{'='*90}")
    print("Acceleration RMS Results (mean ± std)")
    print(f"{'='*90}")
    header = f"{'Experiment':<14} | {'Lin X':>10} | {'Lin Y':>10} | {'Lin Z':>10} | {'Ang X':>10} | {'Ang Y':>10} | {'Ang Z':>10}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['label']:<14} | {r['acc_lin_x']:>5.2f}±{r['acc_lin_x_std']:<4.2f} | {r['acc_lin_y']:>5.2f}±{r['acc_lin_y_std']:<4.2f} | {r['acc_lin_z']:>5.2f}±{r['acc_lin_z_std']:<4.2f} | {r['acc_ang_x']:>5.2f}±{r['acc_ang_x_std']:<4.2f} | {r['acc_ang_y']:>5.2f}±{r['acc_ang_y_std']:<4.2f} | {r['acc_ang_z']:>5.2f}±{r['acc_ang_z_std']:<4.2f}")

    # Peak Acceleration table
    print(f"\n{'='*90}")
    print("Peak Acceleration Results (mean ± std)")
    print(f"{'='*90}")
    header = f"{'Experiment':<14} | {'Lin X':>10} | {'Lin Y':>10} | {'Lin Z':>10} | {'Ang X':>10} | {'Ang Y':>10} | {'Ang Z':>10}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['label']:<14} | {r['peak_lin_x']:>5.2f}±{r['peak_lin_x_std']:<4.2f} | {r['peak_lin_y']:>5.2f}±{r['peak_lin_y_std']:<4.2f} | {r['peak_lin_z']:>5.2f}±{r['peak_lin_z_std']:<4.2f} | {r['peak_ang_x']:>5.2f}±{r['peak_ang_x_std']:<4.2f} | {r['peak_ang_y']:>5.2f}±{r['peak_ang_y_std']:<4.2f} | {r['peak_ang_z']:>5.2f}±{r['peak_ang_z_std']:<4.2f}")

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

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
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

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {output_file}")

    return fig


def plot_xyz_trajectory(experiments, title="XYZ Trajectory", output_file="xyz_trajectory.png", exp_index=0):
    """
    Plot X, Y, Z position vs time for each experiment.

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
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
        pos = data.get_field('robot_pos_w')[skip_samples:]
        time = np.arange(pos.shape[0]) * DT

        all_data.append({
            'label': label,
            'time': time,
            'pos': pos,
        })

        print(f"  Samples: {pos.shape[0]}, Duration: {time[-1]:.1f}s")

    # Create figure - 3 rows x 1 column (X, Y, Z)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    fig.set_facecolor('white')

    axis_names = ['X', 'Y', 'Z']
    units = 'm'

    for i in range(3):
        ax = axes[i]
        ax.set_facecolor('white')

        for exp_data in all_data:
            ax.plot(exp_data['time'], exp_data['pos'][:, i], label=exp_data['label'], linewidth=0.8)

        ax.set_ylabel(f'{axis_names[i]} ({units})', fontweight='bold', fontsize=9, labelpad=2)

        if i == 2:
            ax.set_xlabel('Time (s)', fontsize=9)
        else:
            ax.set_xticklabels([])

    # Legend on first subplot
    axes[0].legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black', facecolor='white')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {output_file}")

    return fig


def plot_gait_cycle_diagnostic(folder_path, exp_index=0, output_file="gait_cycle_diagnostic.png"):
    """
    Diagnostic plot to visualize knee torques and identify gait cycles.

    Joint order: ['fl_hx','fr_hx','hl_hx','hr_hx','fl_hy','fr_hy','hl_hy','hr_hy','fl_kn','fr_kn','hl_kn','hr_kn']
    Knee (kn) indices: 8=fl_kn, 9=fr_kn, 10=hl_kn, 11=hr_kn
    """
    from scipy.signal import find_peaks

    skip_samples = int(SKIP_SECONDS / DT)

    folder = Path(folder_path)
    h5_files = sorted(folder.glob('*.h5'))
    h5_file = h5_files[exp_index]

    print(f"\nGait Cycle Diagnostic: {h5_file.name}")

    data = HDF5Reader.load(str(h5_file), dt=DT)
    torques = data.get_field('robot_joint_torques')[skip_samples:]
    time = np.arange(torques.shape[0]) * DT

    # Knee (kn) joints
    kn_indices = [8, 9, 10, 11]
    kn_labels = ['FL Knee (kn)', 'FR Knee (kn)', 'HL Knee (kn)', 'HR Knee (kn)']

    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    fig.set_facecolor('white')

    for i, (idx, label) in enumerate(zip(kn_indices, kn_labels)):
        ax = axes[i]
        ax.set_facecolor('white')
        signal = torques[:, idx]

        ax.plot(time, signal, 'k-', linewidth=0.6)

        # Find peaks for gait cycle detection
        peaks, _ = find_peaks(signal, distance=int(0.3 / DT), prominence=20)

        if len(peaks) > 0:
            ax.plot(time[peaks], signal[peaks], 'rv', markersize=5, label=f'{len(peaks)} peaks')

            # Estimate gait period
            if len(peaks) > 1:
                periods = np.diff(time[peaks])
                ax.set_title(f'{label} | Period: {np.mean(periods):.3f}±{np.std(periods):.3f}s | Freq: {1/np.mean(periods):.1f}Hz',
                            fontsize=9, fontweight='bold')
            else:
                ax.set_title(label, fontsize=9, fontweight='bold')

            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.set_title(f'{label} | No peaks found', fontsize=9, fontweight='bold')

        ax.set_ylabel('Torque (N·m)', fontsize=8)
        if i == 3:
            ax.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle('Gait Cycle Diagnostic - Knee Torques', fontsize=12, fontweight='bold')
    plt.tight_layout()
    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"Saved: {output_file}")

    return fig


def compute_torque_symmetry_ratio(experiments, title="Torque Symmetry Ratio", output_file="torque_symmetry.png", trial_idx=1):
    """
    Compute and plot per-gait-cycle torque symmetry ratio (left/right).

    Gait cycles are segmented using FL knee torque peaks.
    For each cycle, RMS torque is computed for left and right legs,
    then the ratio left/right is calculated per joint type.

    Symmetry ratio = RMS(left) / RMS(right)
        1.0 = perfect symmetry
        >1.0 = left leg works harder
        <1.0 = right leg works harder

    Joint order: [fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn]

    Left/Right pairs per joint type:
        Abductor (hx): FL(0) vs FR(1), HL(2) vs HR(3)
        Hip (hy):      FL(4) vs FR(5), HL(6) vs HR(7)
        Knee (kn):     FL(8) vs FR(9), HL(10) vs HR(11)

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
        title: Plot title
        output_file: Output filename
        trial_idx: Which trial to use from each experiment (default: 0)
    """
    from scipy.signal import find_peaks

    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s")
    print(f"[Config] Trial index: {trial_idx}")
    print(f"[Method] Gait segmentation: FL knee peaks (distance=15, prominence=20)")
    print(f"[Metric] RMS symmetry ratio: RMS(left) / RMS(right) per gait cycle\n")

    # Define left/right joint pairs: (name, left_idx, right_idx)
    joint_pairs = [
        ('Front HX', 0, 1),   # fl_hx vs fr_hx
        ('Front HY', 4, 5),   # fl_hy vs fr_hy
        ('Front Knee', 8, 9),   # fl_kn vs fr_kn
        ('Hind HX', 2, 3),   # hl_hx vs hr_hx
        ('Hind HY', 6, 7),   # hl_hy vs hr_hy
        ('Hind Knee', 10, 11),  # hl_kn vs hr_kn
    ]

    pair_names = [name for name, _, _ in joint_pairs]
    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange']

    # --- Compute per-cycle TSR for each experiment ---
    all_exp_data = []

    for folder_path, label, is_baseline in experiments:
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        if trial_idx >= len(h5_files):
            print(f"  Warning: {label} has only {len(h5_files)} files, using last one")
            h5_file = h5_files[-1]
        else:
            h5_file = h5_files[trial_idx]

        print(f"Processing: {label} ({h5_file.name})")

        data = HDF5Reader.load(str(h5_file), dt=DT)
        torques = data.get_field('robot_joint_torques')

        if torques.shape[0] <= skip_samples + 10:
            print(f"  Skipping {label} (empty)")
            continue

        torques = torques[skip_samples:]
        time = np.arange(torques.shape[0]) * DT

        # Segment gait cycles using FL knee (index 8) peaks
        fl_knee = torques[:, 8]
        peaks, _ = find_peaks(fl_knee, distance=int(0.3 / DT), prominence=20)

        if len(peaks) < 2:
            print(f"  Skipping {label} (insufficient peaks: {len(peaks)})")
            continue

        n_cycles = len(peaks) - 1
        print(f"  Gait cycles: {n_cycles}")

        # Gait cycle indices
        cycle_indices = np.arange(1, n_cycles + 1)

        # Per-cycle TSR for each joint pair
        cycle_tsr = {}
        for name, left_idx, right_idx in joint_pairs:
            ratios = []
            for c in range(n_cycles):
                start = peaks[c]
                end = peaks[c + 1]
                cycle_torques = torques[start:end]

                # # RMS-based TSR
                # left_rms = np.sqrt(np.mean(cycle_torques[:, left_idx]**2))
                # right_rms = np.sqrt(np.mean(cycle_torques[:, right_idx]**2))
                # if right_rms > 0.01:
                #     ratios.append(left_rms / right_rms)
                # else:
                #     ratios.append(np.nan)

                # Peak-based TSR
                left_peak = np.max(np.abs(cycle_torques[:, left_idx]))
                right_peak = np.max(np.abs(cycle_torques[:, right_idx]))
                if right_peak > 0.01:
                    ratios.append(left_peak / right_peak)
                else:
                    ratios.append(np.nan)

            cycle_tsr[name] = np.array(ratios)
            print(f"  {name}: mean={np.nanmean(cycle_tsr[name]):.3f} ± {np.nanstd(cycle_tsr[name]):.3f}")

        all_exp_data.append({
            'label': label,
            'cycle_indices': cycle_indices,
            'cycle_tsr': cycle_tsr,
        })

    # --- Winsorize: clip top 10% outliers using scipy ---
    # for exp_data in all_exp_data:
    #     for name in pair_names:
    #         exp_data['cycle_tsr'][name] = mstats.winsorize(exp_data['cycle_tsr'][name], limits=[0.5, 0.1])

    # --- Plot: 2x3 grid with experiments overlaid ---
    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(10, 5))
    fig.set_facecolor('white')
    axes = axes.flatten()

    for exp_idx, exp_data in enumerate(all_exp_data):
        color = colors[exp_idx % len(colors)]
        ci = exp_data['cycle_indices']

        for i, name in enumerate(pair_names):
            ax = axes[i]
            tsr = exp_data['cycle_tsr'][name]
            ax.plot(ci, tsr, color=color, linewidth=0.6, zorder=1)
            ax.scatter(ci, tsr, s=15, color=color, zorder=2,
                       label=exp_data['label'] if i == 0 else None)

    for i, name in enumerate(pair_names):
        ax = axes[i]
        ax.set_facecolor('white')
        ax.axhline(y=1.0, color='darkgray', linestyle='--', linewidth=1)
        ax.axhline(y=2.0, color='darkred', linestyle='--', linewidth=0.8)
        ax.set_title(name, fontsize=10, fontweight='normal')
        ax.tick_params(axis='both', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)

    # fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.supxlabel('Gait Cycle', fontsize=10)
    fig.legend(loc='upper right', fontsize=9, framealpha=0.5)
    plt.tight_layout()
    # plt.ylim(0, 100)

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {pdf_file}")

    # Print results table
    print(f"\n{'='*100}")
    print("Torque Symmetry Ratio: RMS(Left) / RMS(Right)  [1.0 = symmetric]")
    print(f"{'='*100}")
    header = f"{'Experiment':<14}"
    for name in pair_names:
        short = name.replace('Front ', 'F.').replace('Hind ', 'H.')
        header += f" | {short:>12}"
    print(header)
    print('-' * len(header))
    for exp_data in all_exp_data:
        row = f"{exp_data['label']:<14}"
        for name in pair_names:
            tsr = exp_data['cycle_tsr'][name]
            row += f" | {np.nanmean(tsr):>6.3f}±{np.nanstd(tsr):<5.3f}"
        print(row)

    return fig, all_exp_data


def plot_xy_colored_by_z(experiments, title="XY Trajectory (colored by Z)", output_file="xy_z_colored.png", n_trials=3):
    """
    Plot XY trajectory colored by Z height for each experiment.

    Each subplot shows up to n_trials overlaid trajectories for one payload
    distribution. Line color encodes Z position using a coolwarm colormap.
    Thicker lines are drawn first so thinner ones remain visible on top.

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
        title: Plot title
        output_file: Output filename
        n_trials: Number of trials to overlay per experiment (default: 3)
    """
    skip_samples = int(SKIP_SECONDS / DT)
    trial_widths = [3.5, 2.0, 1.0][:n_trials]

    n_exp = len(experiments)
    n_cols = 3
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2.0 * n_rows),
                             constrained_layout=True)
    fig.set_facecolor('white')
    axes = axes.flatten() if n_exp > 1 else [axes]

    for exp_idx, (folder_path, label, _) in enumerate(experiments):
        ax = axes[exp_idx]
        ax.set_facecolor('white')
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))[:n_trials]

        # Compute per-experiment z range for consistent coloring
        exp_z_min, exp_z_max = np.inf, -np.inf
        for h5_file in h5_files:
            data = HDF5Reader.load(str(h5_file), dt=DT)
            pos = data.get_field('robot_pos_w')[skip_samples:]
            exp_z_min = min(exp_z_min, pos[:, 2].min())
            exp_z_max = max(exp_z_max, pos[:, 2].max())

        # Draw trials thick-first so thin lines end up on top
        for i, h5_file in enumerate(h5_files):
            data = HDF5Reader.load(str(h5_file), dt=DT)
            pos = data.get_field('robot_pos_w')[skip_samples:]
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            z_segments = (z[:-1] + z[1:]) / 2

            lc = LineCollection(segments, cmap='coolwarm', linewidth=trial_widths[i])
            lc.set_array(z_segments)
            lc.set_clim(exp_z_min, exp_z_max)
            ax.add_collection(lc)

            # Start/end markers
            label_start = 'Start' if (exp_idx == 0 and i == 0) else None
            label_end = 'End' if (exp_idx == 0 and i == 0) else None
            ax.plot(x[0], y[0], 'o', color='green', markersize=5, zorder=5, label=label_start)
            ax.plot(x[-1], y[-1], 's', color='red', markersize=5, zorder=5, label=label_end)

        ax.autoscale()
        ax.margins(0.05)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(label, fontsize=10, fontweight='normal')
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
        if exp_idx % n_cols != 0:
            ax.set_yticklabels([])

        # Per-subplot colorbar
        sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                   norm=plt.Normalize(vmin=exp_z_min, vmax=exp_z_max))
        cbar = fig.colorbar(sm, ax=ax, shrink=0.9, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        if exp_idx % n_cols == n_cols - 1:
            cbar.set_label('Z (m)', fontsize=8)

    # Hide unused subplots
    for i in range(n_exp, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel('X (m)', fontsize=12)
    fig.supylabel('Y (m)', fontsize=12)
    fig.legend(loc='lower right', ncol=2, fontsize=8, frameon=True)
    # fig.suptitle(title, fontsize=13, fontweight='bold')

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {output_file}")

    return fig


# =============================================================================
# Trajectory Smoothness (Curvature)
# =============================================================================
N_RESAMPLE = 400  # Number of equally-spaced arc-length points


def _generate_laplacian(n_pts):
    """Build discrete Laplacian matrix for curvature estimation."""
    L = (
        2.0 * np.diag(np.ones((n_pts,)))
        - np.diag(np.ones((n_pts - 1,)), 1)
        - np.diag(np.ones((n_pts - 1,)), -1)
    )
    L[0, 1] = -2.0
    L[-1, -2] = -2.0
    L = L / 2.0
    return L


def _resample_by_arc_length(x, y, z, n_pts):
    """Resample a 3D trajectory at equally-spaced arc-length intervals.

    Returns:
        X_resampled: (n_pts, 3) array of resampled positions
        s_total: total arc length of the trajectory
    """
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    s_total = s[-1]

    s_uniform = np.linspace(0, s_total, n_pts)

    x_resamp = np.interp(s_uniform, s, x)
    y_resamp = np.interp(s_uniform, s, y)
    z_resamp = np.interp(s_uniform, s, z)

    return np.column_stack([x_resamp, y_resamp, z_resamp]), s_total


def _compute_curvature(X_resampled, s_total):
    """Compute curvature at each point using the Laplacian.

    Returns:
        kappa: (n_pts-4,) curvature at interior points (1/m)
    """
    n_pts = X_resampled.shape[0]
    L = _generate_laplacian(n_pts)

    curvature_vectors = L @ X_resampled
    ds = s_total / (n_pts - 1)
    curvature_vectors = curvature_vectors / (ds ** 2)

    kappa = np.linalg.norm(curvature_vectors, axis=1)

    # Discard boundary points (first/last 2) — unreliable due to boundary effects
    kappa = kappa[2:-2]
    return kappa


def _compute_curvature_per_axis(X_resampled, s_total):
    """Compute curvature per axis (x, y, z) using the Laplacian.

    Unlike _compute_curvature which returns ||L @ X|| / ds^2 (scalar curvature),
    this returns the signed curvature components: (L @ x) / ds^2, (L @ y) / ds^2, (L @ z) / ds^2.

    Returns:
        kappa_x, kappa_y, kappa_z: each (n_pts-4,) signed curvature per axis (1/m)
    """
    n_pts = X_resampled.shape[0]
    L = _generate_laplacian(n_pts)
    ds = s_total / (n_pts - 1)

    kappa_x = (L @ X_resampled[:, 0]) / (ds ** 2)
    kappa_y = (L @ X_resampled[:, 1]) / (ds ** 2)
    kappa_z = (L @ X_resampled[:, 2]) / (ds ** 2)

    # Discard boundary points (first/last 2)
    return kappa_x[2:-2], kappa_y[2:-2], kappa_z[2:-2]


def compute_trajectory_smoothness(experiments, title="Trajectory Smoothness", output_file="trajectory_smoothness.png"):
    """
    Compute trajectory smoothness using RMS curvature (Laplacian-based).

    For each experiment, loads all .h5 files, computes curvature of the
    3D trajectory (robot_pos_w), and reports RMS curvature as a smoothness score.

    Lower RMS curvature = smoother trajectory.

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
        title: Plot title
        output_file: Output filename
    """
    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s")
    print(f"[Config] Arc-length resampling: {N_RESAMPLE} points")
    print(f"[Method] Laplacian-based curvature, RMS metric\n")

    exp_names = []
    all_means = []
    all_sds = []

    print(f"{'Experiment':<14} ", end='')
    print(f"{'Trials':>8} {'Mean RMS':>10} {'± SD':>10} {'Mean Arc(m)':>12}")
    print("-" * 60)

    for folder_path, label, is_baseline in experiments:
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        trial_rms = []
        trial_arc = []

        for h5_file in h5_files:
            data = HDF5Reader.load(str(h5_file), dt=DT)
            pos = data.get_field('robot_pos_w')

            if pos.shape[0] <= skip_samples + 10:
                continue

            pos = pos[skip_samples:]
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

            X_resamp, s_total = _resample_by_arc_length(x, y, z, N_RESAMPLE)
            kappa = _compute_curvature(X_resamp, s_total)

            trial_rms.append(np.sqrt(np.mean(kappa ** 2)))
            trial_arc.append(s_total)

        mean_rms = np.mean(trial_rms)
        sd_rms = np.std(trial_rms)
        mean_arc = np.mean(trial_arc)

        exp_names.append(label)
        all_means.append(mean_rms)
        all_sds.append(sd_rms)

        print(f"{label:<14} {len(trial_rms):>8} {mean_rms:>10.4f} {sd_rms:>10.4f} {mean_arc:>12.3f}")

    # --- Plot: dot plot matching real robot style ---
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    x = np.arange(len(exp_names))
    ax.errorbar(x, all_means, yerr=all_sds, fmt='ko-', markersize=4,
                capsize=3, capthick=1, ecolor='red', elinewidth=1,
                color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('RMS Curvature (1/m)')
    ax.margins(x=0.15)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {output_file}")

    return fig, exp_names, all_means, all_sds


def compute_trajectory_smoothness_per_axis(experiments, title="Trajectory Smoothness", output_file="trajectory_smoothness_per_axis.png"):
    """
    Compute trajectory smoothness per axis (X, Y, Z) using RMS curvature.

    Same Laplacian-based method as compute_trajectory_smoothness, but reports
    RMS curvature for each axis separately: (L @ x)/ds^2, (L @ y)/ds^2, (L @ z)/ds^2.

    3 subplots (one per axis), each with a dot plot across experiments.

    Args:
        experiments: List of tuples (folder_path, label, is_baseline)
        title: Plot title
        output_file: Output filename
    """
    skip_samples = int(SKIP_SECONDS / DT)

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"[Config] Skip: {SKIP_SECONDS}s ({skip_samples} samples), dt: {DT}s")
    print(f"[Config] Arc-length resampling: {N_RESAMPLE} points")
    print(f"[Method] Laplacian-based curvature per axis, RMS metric\n")

    axis_names = ['X', 'Y', 'Z']
    exp_names = []
    # {axis: {'means': [], 'sds': []}}
    axis_results = {a: {'means': [], 'sds': []} for a in axis_names}

    print(f"{'Experiment':<14} {'Trials':>8} {'RMS X':>16} {'RMS Y':>16} {'RMS Z':>16}")
    print("-" * 80)

    for folder_path, label, is_baseline in experiments:
        folder = Path(folder_path)
        h5_files = sorted(folder.glob('*.h5'))

        trial_rms = {'X': [], 'Y': [], 'Z': []}

        for h5_file in h5_files:
            data = HDF5Reader.load(str(h5_file), dt=DT)
            pos = data.get_field('robot_pos_w')

            if pos.shape[0] <= skip_samples + 10:
                continue

            pos = pos[skip_samples:]
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

            X_resamp, s_total = _resample_by_arc_length(x, y, z, N_RESAMPLE)
            kx, ky, kz = _compute_curvature_per_axis(X_resamp, s_total)

            trial_rms['X'].append(np.sqrt(np.mean(kx ** 2)))
            trial_rms['Y'].append(np.sqrt(np.mean(ky ** 2)))
            trial_rms['Z'].append(np.sqrt(np.mean(kz ** 2)))

        exp_names.append(label)
        for a in axis_names:
            axis_results[a]['means'].append(np.mean(trial_rms[a]))
            axis_results[a]['sds'].append(np.std(trial_rms[a]))

        print(f"{label:<14} {len(trial_rms['X']):>8} "
              f"{axis_results['X']['means'][-1]:>7.4f}±{axis_results['X']['sds'][-1]:<7.4f} "
              f"{axis_results['Y']['means'][-1]:>7.4f}±{axis_results['Y']['sds'][-1]:<7.4f} "
              f"{axis_results['Z']['means'][-1]:>7.4f}±{axis_results['Z']['sds'][-1]:<7.4f}")

    # --- Plot: 1x3 subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    fig.set_facecolor('white')

    x = np.arange(len(exp_names))

    for i, a in enumerate(axis_names):
        ax = axes[i]
        ax.set_facecolor('white')

        means = axis_results[a]['means']
        sds = axis_results[a]['sds']

        ax.errorbar(x, means, yerr=sds, fmt='ko-', markersize=4,
                    capsize=3, capthick=1, ecolor='red', elinewidth=1,
                    color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, fontsize=12, rotation=45, ha='right')
        ax.set_title(f'{a}-axis', fontsize=11, fontweight='bold')
        ax.margins(x=0.15)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    axes[0].set_ylabel('RMS Curvature (1/m)')

    # fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    pdf_file = Path(output_file).with_suffix('.pdf')
    fig.savefig(pdf_file, dpi=300, bbox_inches='tight', format='pdf')
    png_dir = pdf_file.parent / 'png'
    png_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / pdf_file.with_suffix('.png').name, dpi=300, bbox_inches='tight', format='png')
    print(f"\nSaved: {output_file}")

    return fig, exp_names, axis_results


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
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/forward_0.15_payload_8kg', 'Forward 0.15m', False),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/forward_0.25_payload_8kg', 'Forward 0.25m', False),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/backward_0.15_payload_8kg', 'Backward 0.15m', False),
        ('/home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/backward_0.25_payload_8kg', 'Backward 0.25m', False),
    ]

  
    # Compute and plot jerk RMS
    # fig_jerk, jerk_results = compute_jerk_rms(
    #     experiments_8kg,
    #     title="Jerk RMS Comparison - 8kg Payload (n = 5 experiments)",
    #     output_file="jerk_rms_8kg.png"
    # )

    # fig = plot_joint_torques_comparison(experiments_8kg, title="Joint Torque Comparison", output_file="joint_torques_comparison.png")

    # fig = plot_gait_cycle_diagnostic(experiments_8kg[1][0], exp_index=0)

    # fig, sym_results = compute_torque_symmetry_ratio(
    #     experiments_8kg,
    #     title="Torque Symmetry Ratio - 8kg Payload",
    #     output_file="torque_symmetry_8kg.png"
    # )

    # fig = plot_acceleration_signals(experiments_8kg, title="Acceleration Comparison", output_file="acc.png", exp_index=0)

    # fig = plot_xy_trajectory(experiments_8kg, title="XY Trajectory", output_file="xy.png", exp_index=2)

    # fig = plot_xyz_trajectory(experiments_8kg, title="XYZ Trajectory - 8kg Payload", output_file="xyz_trajectory_8kg.png", exp_index=0)
    # fig = compute_torque_symmetry_ratio(experiments_8kg, title="Torque Symmetry Ratio (per Gait Cycle)", output_file="torque_symmetry.png", trial_idx=1)

    # fig = compute_trajectory_smoothness(experiments_8kg, title="Trajectory Smoothness", output_file="trajectory_smoothness.png")
    
    fig = plot_xy_colored_by_z(experiments_8kg, title="XY Trajectory (colored by Z)", output_file="xy_z_colored.png")

    # fig, names, axis_results = compute_trajectory_smoothness_per_axis(
    #     experiments_8kg, output_file="trajectory_smoothness_per_axis.png"
    # )


    plt.show()
