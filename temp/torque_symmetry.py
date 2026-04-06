import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DATA_DIR = '/home/nerve/Desktop/data_collected'

EXPERIMENTS = {
    'baseline': [
        f'{DATA_DIR}/baseline_empty_center_crate_Feb_4/exp{i}/baseline_empty_crate_exp{i}.csv'
        for i in range(1, 4)
    ],
    'center_5.2kg': [
        f'{DATA_DIR}/flat_terrain_center_crate_8kg_Feb_4/exp{i}/flat_terrain_8kg_center_crate_exp{i}.csv'
        for i in range(1, 4)
    ],
    'front_5.2kg': [
        f'{DATA_DIR}/flat_terrain_front_crate_8kg_Feb_4/exp{i}/flat_terrain_8kg_front_crate_exp{i}.csv'
        for i in range(1, 4)
    ],
    'rear_5.2kg': [
        f'{DATA_DIR}/flat_terrain_rear_crate_8kg_Feb_4/exp{i}/flat_terrain_8kg_rear_crate_exp{i}.csv'
        for i in range(1, 4)
    ],
}

# Walking region (exclude standing portions)
WALK_START = 5.0
WALK_END = 29.0

# Peak detection parameters
REF_JOINT = 'fl.kn_load'
PEAK_DISTANCE = 3
PEAK_PROMINENCE = 20

# Symmetric joint pairs: (left, right, label)
JOINT_PAIRS = [
    ('fl.hx_load', 'fr.hx_load', 'Front Hip X'),
    ('fl.hy_load', 'fr.hy_load', 'Front Hip Y'),
    ('fl.kn_load', 'fr.kn_load', 'Front Knee'),
    ('hl.hx_load', 'hr.hx_load', 'Hind Hip X'),
    ('hl.hy_load', 'hr.hy_load', 'Hind Hip Y'),
    ('hl.kn_load', 'hr.kn_load', 'Hind Knee'),
]

# All individual joints for peak torque analysis
ALL_JOINTS = [
    ('fl.hx_load', 'FL Hip X'), ('fl.hy_load', 'FL Hip Y'), ('fl.kn_load', 'FL Knee'),
    ('fr.hx_load', 'FR Hip X'), ('fr.hy_load', 'FR Hip Y'), ('fr.kn_load', 'FR Knee'),
    ('hl.hx_load', 'HL Hip X'), ('hl.hy_load', 'HL Hip Y'), ('hl.kn_load', 'HL Knee'),
    ('hr.hx_load', 'HR Hip X'), ('hr.hy_load', 'HR Hip Y'), ('hr.kn_load', 'HR Knee'),
]


def detect_gait_cycles(time, signal):
    """Detect gait cycle boundaries using peak detection. Returns peak indices."""
    peaks, _ = find_peaks(signal, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    return peaks


def compute_trial_tsr(file_path):
    """Compute torque symmetry ratio for each joint pair, per gait cycle.

    Returns:
        cycle_times: array of mid-cycle elapsed times (one per gait cycle)
        results: dict mapping joint label -> array of per-cycle TSR values
    """
    df = pd.read_csv(file_path)
    time = df['elapsed_time'].values

    # Isolate walking region
    walk_mask = (time >= WALK_START) & (time <= WALK_END)
    walk_df = df[walk_mask].reset_index(drop=True)
    walk_time = walk_df['elapsed_time'].values
    walk_signal = walk_df[REF_JOINT].values

    # Detect gait cycle boundaries
    peaks = detect_gait_cycles(walk_df['elapsed_time'].values, walk_signal)

    # Mid-cycle times for plotting
    cycle_times = np.array([(walk_time[peaks[i]] + walk_time[peaks[i+1]]) / 2
                            for i in range(len(peaks) - 1)])

    results = {}
    for left_col, right_col, label in JOINT_PAIRS:
        left_signal = walk_df[left_col].values
        right_signal = walk_df[right_col].values

        cycle_ratios = []
        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i + 1]
            left_cycle = left_signal[start:end]
            right_cycle = right_signal[start:end]

            rms_left = np.sqrt(np.mean(left_cycle ** 2))
            rms_right = np.sqrt(np.mean(right_cycle ** 2))

            if rms_right > 0:
                cycle_ratios.append(rms_left / rms_right)
            else:
                cycle_ratios.append(np.nan)

        results[label] = np.array(cycle_ratios)

    return cycle_times, results


def compute_trial_rms_torque(file_path):
    """Compute RMS torque for each joint during walking region."""
    df = pd.read_csv(file_path)
    time = df['elapsed_time'].values

    walk_mask = (time >= WALK_START) & (time <= WALK_END)
    walk_df = df[walk_mask].reset_index(drop=True)

    results = {}
    for col, label in ALL_JOINTS:
        results[label] = np.sqrt(np.mean(walk_df[col].values ** 2))

    return results


def compute_all_rms_torque():
    """Compute RMS torque for all experiments, averaging across trials."""
    exp_names = list(EXPERIMENTS.keys())
    joint_labels = [label for _, label in ALL_JOINTS]

    print(f"\n{'Joint':<15}", end='')
    for name in exp_names:
        print(f"{'| ' + name:<22}", end='')
    print()
    print("-" * 105)

    all_means = {}
    all_sds = {}
    for label in joint_labels:
        means = []
        sds = []
        for exp_name in exp_names:
            trial_values = [compute_trial_rms_torque(f)[label] for f in EXPERIMENTS[exp_name]]
            means.append(np.mean(trial_values))
            sds.append(np.std(trial_values))
        all_means[label] = means
        all_sds[label] = sds
        print(f"{label:<15}", end='')
        for i in range(len(exp_names)):
            print(f"| {means[i]:>7.3f} ± {sds[i]:.3f}   ", end='')
        print()

    return exp_names, joint_labels, all_means, all_sds


def compute_trial_peak_torque(file_path):
    """Compute peak absolute torque for each joint during walking region."""
    df = pd.read_csv(file_path)
    time = df['elapsed_time'].values

    walk_mask = (time >= WALK_START) & (time <= WALK_END)
    walk_df = df[walk_mask].reset_index(drop=True)

    results = {}
    for col, label in ALL_JOINTS:
        results[label] = np.max(np.abs(walk_df[col].values))

    return results


def compute_all_peak_torque():
    """Compute peak torque for all experiments, averaging across trials."""
    exp_names = list(EXPERIMENTS.keys())
    joint_labels = [label for _, label in ALL_JOINTS]

    print(f"\n{'Joint':<15}", end='')
    for name in exp_names:
        print(f"{'| ' + name:<22}", end='')
    print()
    print("-" * 105)

    all_means = {}
    all_sds = {}
    for label in joint_labels:
        means = []
        sds = []
        for exp_name in exp_names:
            trial_values = [compute_trial_peak_torque(f)[label] for f in EXPERIMENTS[exp_name]]
            means.append(np.mean(trial_values))
            sds.append(np.std(trial_values))
        all_means[label] = means
        all_sds[label] = sds
        print(f"{label:<15}", end='')
        for i in range(len(exp_names)):
            print(f"| {means[i]:>7.3f} ± {sds[i]:.3f}   ", end='')
        print()

    return exp_names, joint_labels, all_means, all_sds


def plot_peak_torque():
    """Plot peak torque bar charts in 3x4 grid (rows: joint type, cols: leg)."""
    exp_names, joint_labels, means, sds = compute_all_peak_torque()

    # 3 rows (Hip X, Hip Y, Knee) x 4 cols (FL, FR, HL, HR)
    ROWS = ['Hip X', 'Hip Y', 'Knee']
    COLS = ['FL', 'FR', 'HL', 'HR']

    fig, axes = plt.subplots(3, 4, figsize=(10, 6))

    x = np.arange(len(exp_names))
    bar_width = 0.35

    for row_idx, joint_type in enumerate(ROWS):
        for col_idx, leg in enumerate(COLS):
            ax = axes[row_idx, col_idx]
            label = f'{leg} {joint_type}'

            ax.bar(x, means[label], width=bar_width, yerr=sds[label],
                   color='white', edgecolor='black', linewidth=0.8,
                   error_kw={'ecolor': 'red', 'capsize': 4, 'capthick': 1})
            ax.set_title(label, fontsize=10)
            ax.margins(x=0.05)
            ax.tick_params(axis='y', labelsize=8)

            # X labels only on bottom row
            if row_idx == 2:
                ax.set_xticks(x)
                ax.set_xticklabels(exp_names, fontsize=10, rotation=45, ha='right')
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])

    fig.suptitle('Peak Joint Torque (Nm)', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_torque_boxplot():
    """Plot box plots for absolute torque in 3x4 grid (rows: joint type, cols: leg)."""
    exp_names = list(EXPERIMENTS.keys())

    ROWS = ['Hip X', 'Hip Y', 'Knee']
    COLS = ['FL', 'FR', 'HL', 'HR']

    # Collect raw absolute torque data per joint per experiment (pooled across trials)
    torque_data = {}
    for col, label in ALL_JOINTS:
        torque_data[label] = {}
        for exp_name in exp_names:
            pooled = []
            for f in EXPERIMENTS[exp_name]:
                df = pd.read_csv(f)
                time = df['elapsed_time'].values
                walk_mask = (time >= WALK_START) & (time <= WALK_END)
                pooled.append(np.abs(df.loc[walk_mask, col].values))
            torque_data[label][exp_name] = np.concatenate(pooled)

    fig, axes = plt.subplots(3, 4, figsize=(10, 6))

    for row_idx, joint_type in enumerate(ROWS):
        for col_idx, leg in enumerate(COLS):
            ax = axes[row_idx, col_idx]
            label = f'{leg} {joint_type}'

            data = [torque_data[label][exp] for exp in exp_names]
            bp = ax.boxplot(data, widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.8),
                            medianprops=dict(color='red', linewidth=1),
                            whiskerprops=dict(color='black', linewidth=0.8),
                            capprops=dict(color='black', linewidth=0.8),
                            showfliers=False)
            ax.set_title(label, fontsize=10)
            ax.tick_params(axis='y', labelsize=8)

            if row_idx == 2:
                ax.set_xticklabels(exp_names, fontsize=8, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])

    fig.suptitle('Absolute Joint Torque Distribution (Nm)', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_torque_boxplot_signed():
    """Plot box plots for signed torque in 3x4 grid (rows: joint type, cols: leg)."""
    exp_names = list(EXPERIMENTS.keys())

    ROWS = ['Hip X', 'Hip Y', 'Knee']
    COLS = ['FL', 'FR', 'HL', 'HR']

    # Collect raw signed torque data per joint per experiment (pooled across trials)
    torque_data = {}
    for col, label in ALL_JOINTS:
        torque_data[label] = {}
        for exp_name in exp_names:
            pooled = []
            for f in EXPERIMENTS[exp_name]:
                df = pd.read_csv(f)
                time = df['elapsed_time'].values
                walk_mask = (time >= WALK_START) & (time <= WALK_END)
                pooled.append(df.loc[walk_mask, col].values)
            torque_data[label][exp_name] = np.concatenate(pooled)

    fig, axes = plt.subplots(3, 4, figsize=(10, 6))

    for row_idx, joint_type in enumerate(ROWS):
        for col_idx, leg in enumerate(COLS):
            ax = axes[row_idx, col_idx]
            label = f'{leg} {joint_type}'

            data = [torque_data[label][exp] for exp in exp_names]
            bp = ax.boxplot(data, widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.8),
                            medianprops=dict(color='red', linewidth=1),
                            whiskerprops=dict(color='black', linewidth=0.8),
                            capprops=dict(color='black', linewidth=0.8),
                            showfliers=False)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_title(label, fontsize=10)
            ax.tick_params(axis='y', labelsize=8)

            if row_idx == 2:
                ax.set_xticklabels(exp_names, fontsize=8, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])

    fig.suptitle('Joint Torque Distribution (Nm)', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_tsr_trajectory(file_path):
    """Plot per-cycle TSR as scatter points for all joint pairs in a 2x3 grid."""
    cycle_times, tsr = compute_trial_tsr(file_path)
    cycle_times = cycle_times - cycle_times[0]

    joint_labels = [label for _, _, label in JOINT_PAIRS]
    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(10, 5))
    axes = axes.flatten()

    for idx, label in enumerate(joint_labels):
        ax = axes[idx]
        ax.plot(cycle_times, tsr[label], color='black', linewidth=0.6, zorder=1)
        ax.scatter(cycle_times, tsr[label], s=20, color='black', edgecolors='black', zorder=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle('TSR Trajectory per Gait Cycle', fontsize=12)
    fig.supxlabel('Time (s)', fontsize=10)
    fig.supylabel('TSR', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_tsr_trajectory_comparison(trial_idx=0):
    """Plot TSR trajectories for all experiments overlaid, one trial per experiment.

    Args:
        trial_idx: which trial to pick from each experiment (default: 0 = first trial)
    """
    exp_names = list(EXPERIMENTS.keys())
    colors = ['black', 'blue', 'red', 'green']
    joint_labels = [label for _, _, label in JOINT_PAIRS]

    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(10, 5))
    axes = axes.flatten()

    for exp_name, color in zip(exp_names, colors):
        file_path = EXPERIMENTS[exp_name][trial_idx]
        cycle_times, tsr = compute_trial_tsr(file_path)
        cycle_times = cycle_times - cycle_times[0]

        for idx, label in enumerate(joint_labels):
            ax = axes[idx]
            ax.plot(cycle_times, tsr[label], color=color, linewidth=0.6, zorder=1)
            ax.scatter(cycle_times, tsr[label], s=15, color=color, zorder=2,
                       label=exp_name if idx == 0 else None)

    for idx, label in enumerate(joint_labels):
        ax = axes[idx]
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle('Torque Symmetry Ratio (per Gait Cycle)', fontsize=12)
    fig.supxlabel('Time (s)', fontsize=10)
    fig.supylabel('TSR', fontsize=10)
    fig.legend(loc='upper right', fontsize=9, framealpha=0.5)
    plt.tight_layout()
    plt.show()


def compute_all_tsr():
    """Compute TSR for all experiments, averaging across trials."""
    exp_names = list(EXPERIMENTS.keys())
    joint_labels = [label for _, _, label in JOINT_PAIRS]

    print(f"{'Joint Pair':<20}", end='')
    for name in exp_names:
        print(f"{'| ' + name:<22}", end='')
    print()
    print("-" * 110)

    # Gather all data
    all_means = {}
    all_sds = {}
    for label in joint_labels:
        means = []
        sds = []
        for exp_name in exp_names:
            trial_means = [np.mean(compute_trial_tsr(f)[1][label]) for f in EXPERIMENTS[exp_name]]
            means.append(np.mean(trial_means))
            sds.append(np.std(trial_means))
        all_means[label] = means
        all_sds[label] = sds
        print(f"{label:<20}", end='')
        for i in range(len(exp_names)):
            print(f"| {means[i]:>7.3f} ± {sds[i]:.3f}   ", end='')
        print()

    return exp_names, joint_labels, all_means, all_sds


def plot_tsr():
    """Plot TSR bar charts for each joint pair."""
    exp_names, joint_labels, means, sds = compute_all_tsr()

    n = len(joint_labels)
    fig, axes = plt.subplots(n, 1, figsize=(4, 1.6 * n))

    x = np.arange(len(exp_names))
    bar_width = 0.35

    for idx, (ax, label) in enumerate(zip(axes, joint_labels)):
        ax.bar(x, means[label], width=bar_width, yerr=sds[label],
               color='white', edgecolor='black', linewidth=0.8,
               error_kw={'ecolor': 'red', 'capsize': 4, 'capthick': 1})
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(x)
        ax.margins(x=0.05)
        ax.tick_params(axis='y', labelsize=8)

        if idx == n - 1:
            ax.set_xticklabels(exp_names, fontsize=10, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])

    fig.suptitle('Torque Symmetry Ratio (Left/Right)', fontsize=12)
    fig.align_ylabels(axes)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_tsr()
    # plot_peak_torque()
    # plot_torque_boxplot()
    # plot_torque_boxplot_signed()
    # plot_tsr_trajectory("/home/nerve/Desktop/data_collected/flat_terrain_center_crate_8kg_Feb_4/exp2/flat_terrain_8kg_center_crate_exp2.csv")
    # plot_tsr_trajectory_comparison(trial_idx=1)

