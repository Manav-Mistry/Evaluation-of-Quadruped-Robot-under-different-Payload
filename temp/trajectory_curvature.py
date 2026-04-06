import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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

WALK_START = 5.0
WALK_END = 29.0

# Number of equally-spaced points for resampling
N_RESAMPLE = 400


def generate_laplacian(n_pts):
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


def resample_by_arc_length(x, y, z, n_pts):
    """Resample a 3D trajectory at equally-spaced arc-length intervals using linear interpolation.

    Returns:
        X_resampled: (n_pts, 3) array of resampled positions
        s_total: total arc length of the trajectory
    """
    # Compute cumulative arc length
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    s_total = s[-1]

    # Equally-spaced arc-length targets
    s_uniform = np.linspace(0, s_total, n_pts)

    # Linear interpolation at uniform arc-length positions
    x_resamp = np.interp(s_uniform, s, x)
    y_resamp = np.interp(s_uniform, s, y)
    z_resamp = np.interp(s_uniform, s, z)

    return np.column_stack([x_resamp, y_resamp, z_resamp]), s_total


def compute_curvature(X_resampled, s_total):
    """Compute curvature at each point using the Laplacian.

    Args:
        X_resampled: (n_pts, 3) positions resampled at equal arc-length intervals
        s_total: total arc length

    Returns:
        kappa: (n_pts,) curvature at each point (1/m)
    """
    n_pts = X_resampled.shape[0]
    L = generate_laplacian(n_pts)

    # L @ X gives discrete second derivative (unitless, in "per sample² " units)
    # To get actual curvature (1/m), scale by (n_pts - 1)² / s_total²
    # because ds_sample = s_total / (n_pts - 1)
    curvature_vectors = L @ X_resampled
    ds = s_total / (n_pts - 1)
    curvature_vectors = curvature_vectors / (ds ** 2)

    kappa = np.linalg.norm(curvature_vectors, axis=1)

    # Discard boundary points — first/last rows of L compute first difference
    # (tangent), not curvature. Second-from-boundary points are also unreliable
    # due to the robot decelerating/stopping at walk region edges.
    kappa = kappa[2:-2]
    return kappa


def compute_trial_curvature(file_path, source='odom'):
    """Compute curvature for a single trial.

    Args:
        file_path: path to CSV
        source: 'odom' or 'vision'

    Returns:
        kappa: curvature array
        s_total: total arc length
        mean_kappa: mean curvature (1/m)
        max_kappa: max curvature (1/m)
        sum_kappa_sq: sum of squared curvatures (smoothness energy)
    """
    df = pd.read_csv(file_path)
    time = df['elapsed_time'].values

    walk_mask = (time >= WALK_START) & (time <= WALK_END)
    walk_df = df[walk_mask].reset_index(drop=True)

    prefix = 'odom' if source == 'odom' else 'vision'
    x = walk_df[f'{prefix}_x'].values
    y = walk_df[f'{prefix}_y'].values
    z = walk_df[f'{prefix}_z'].values

    X_resamp, s_total = resample_by_arc_length(x, y, z, N_RESAMPLE)
    kappa = compute_curvature(X_resamp, s_total)

    return {
        'kappa': kappa,
        's_total': s_total,
        'mean_kappa': np.mean(kappa),
        'std_kappa': np.std(kappa),
        'rms_kappa': np.sqrt(np.mean(kappa ** 2)),
        'max_kappa': np.max(kappa),
    }


def analyze_single_file(file_path, source='odom'):
    """Analyze and plot curvature for a single file."""
    result = compute_trial_curvature(file_path, source)
    kappa = result['kappa']
    s_total = result['s_total']
    ds = s_total / (N_RESAMPLE - 1)
    # kappa has boundary points removed, so arc-length axis starts at ds
    s_axis = np.linspace(ds, s_total - ds, len(kappa))

    print(f"Source: {source}")
    print(f"Total arc length: {s_total:.3f} m")
    print(f"Mean curvature:   {result['mean_kappa']:.4f} ± {result['std_kappa']:.4f} 1/m")
    print(f"RMS curvature:    {result['rms_kappa']:.4f} 1/m")
    print(f"Max curvature:    {result['max_kappa']:.4f} 1/m")

    # Load raw data for trajectory plot
    df = pd.read_csv(file_path)
    time = df['elapsed_time'].values
    walk_mask = (time >= WALK_START) & (time <= WALK_END)
    walk_df = df[walk_mask]
    prefix = 'odom' if source == 'odom' else 'vision'

    x = walk_df[f'{prefix}_x'].values
    y = walk_df[f'{prefix}_y'].values
    z = walk_df[f'{prefix}_z'].values

    walk_time = walk_df['elapsed_time'].values

    fig, axes = plt.subplots(5, 1, figsize=(10, 12))

    # XY trajectory
    ax = axes[0]
    ax.plot(x, y, 'k-', linewidth=0.8)
    ax.plot(x[0], y[0], 'go', markersize=8, label='start')
    ax.plot(x[-1], y[-1], 'rs', markersize=8, label='end')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{source.upper()} Trajectory (XY)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    # X vs time
    ax = axes[1]
    ax.plot(walk_time, x, 'k-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X (m)')
    ax.set_title(f'{source.upper()} X vs Time')

    # Y vs time
    ax = axes[2]
    ax.plot(walk_time, y, 'k-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{source.upper()} Y vs Time')

    # Z vs time
    ax = axes[3]
    ax.plot(walk_time, z, 'k-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'{source.upper()} Z vs Time')

    # Curvature along arc length
    ax = axes[4]
    ax.plot(s_axis, kappa, 'k-', linewidth=0.8)
    ax.set_xlabel('Arc length (m)')
    ax.set_ylabel('Curvature (1/m)')
    ax.set_title(f'Curvature along path  |  mean={result["mean_kappa"]:.4f}  max={result["max_kappa"]:.4f}')

    plt.tight_layout()
    plt.show()


def plot_xy_colored_by_z(file_path, source='odom'):
    """Plot XY trajectory with line colored by Z position (dark=high, light=low)."""
    df = pd.read_csv(file_path)
    time = df['elapsed_time'].values
    walk_mask = (time >= WALK_START) & (time <= WALK_END)
    walk_df = df[walk_mask]
    prefix = 'odom' if source == 'odom' else 'vision'

    x = walk_df[f'{prefix}_x'].values
    y = walk_df[f'{prefix}_y'].values
    z = walk_df[f'{prefix}_z'].values

    # Build line segments for LineCollection
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color each segment by average z of its two endpoints
    z_segments = (z[:-1] + z[1:]) / 2

    lc = LineCollection(segments, cmap='coolwarm', linewidth=1.5)
    lc.set_array(z_segments)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{source.upper()} Trajectory — colored by Z height')
    ax.plot(x[0], y[0], 'go', markersize=8, label='start')
    ax.plot(x[-1], y[-1], 'rs', markersize=8, label='end')
    ax.legend(fontsize=8)
    cbar = fig.colorbar(lc, ax=ax, shrink=0.8)
    cbar.set_label('Z (m)')
    plt.tight_layout()
    plt.show()


def plot_xy_colored_by_z_overlay(exp_name, source='odom'):
    """Overlay all 3 trial trajectories for one load distribution, each colored by Z."""
    trial_files = EXPERIMENTS[exp_name]
    trial_labels = ['Trial 1', 'Trial 2', 'Trial 3']

    fig, ax = plt.subplots(figsize=(8, 7))

    # Collect global z range across all trials for consistent coloring
    all_z = []
    for f in trial_files:
        df = pd.read_csv(f)
        time = df['elapsed_time'].values
        walk_mask = (time >= WALK_START) & (time <= WALK_END)
        prefix = 'odom' if source == 'odom' else 'vision'
        all_z.append(df.loc[walk_mask, f'{prefix}_z'].values)
    z_min = min(z.min() for z in all_z)
    z_max = max(z.max() for z in all_z)

    # Draw thick lines first, thin lines on top so nothing is hidden
    trial_widths = [3.5, 2.0, 1.0]
    draw_order = [0, 1, 2]  # thick first, thin last (on top)

    for i in draw_order:
        f = trial_files[i]
        df = pd.read_csv(f)
        time = df['elapsed_time'].values
        walk_mask = (time >= WALK_START) & (time <= WALK_END)
        walk_df = df[walk_mask]
        prefix = 'odom' if source == 'odom' else 'vision'

        x = walk_df[f'{prefix}_x'].values
        y = walk_df[f'{prefix}_y'].values
        z = walk_df[f'{prefix}_z'].values

        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        z_segments = (z[:-1] + z[1:]) / 2

        lc = LineCollection(segments, cmap='coolwarm', linewidth=trial_widths[i])
        lc.set_array(z_segments)
        lc.set_clim(z_min, z_max)
        ax.add_collection(lc)

        # Add start/end markers for each trial
        ax.plot(x[0], y[0], 'o', color='green', markersize=6)
        ax.plot(x[-1], y[-1], 's', color='red', markersize=6)

    # Single shared colorbar
    cbar = fig.colorbar(lc, ax=ax, shrink=0.8)
    cbar.set_label('Z (m)')

    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{exp_name} — {source.upper()} Trajectories (3 trials, colored by Z)')
    plt.tight_layout()
    plt.show()


def plot_all_experiments_xy_z(source='odom'):
    """2x2 grid: each subplot shows 3 trial trajectories for one distribution, colored by Z."""
    exp_names = list(EXPERIMENTS.keys())
    prefix = 'odom' if source == 'odom' else 'vision'
    trial_widths = [3.5, 2.0, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    for idx, exp_name in enumerate(exp_names):
        ax = axes[idx]
        trial_files = EXPERIMENTS[exp_name]

        # Per-experiment z range for maximum color contrast
        exp_z_min, exp_z_max = np.inf, -np.inf
        for f in trial_files:
            df = pd.read_csv(f)
            time = df['elapsed_time'].values
            walk_mask = (time >= WALK_START) & (time <= WALK_END)
            z = df.loc[walk_mask, f'{prefix}_z'].values
            exp_z_min = min(exp_z_min, z.min())
            exp_z_max = max(exp_z_max, z.max())

        for i in [0, 1, 2]:  # thick first, thin last
            df = pd.read_csv(trial_files[i])
            time = df['elapsed_time'].values
            walk_mask = (time >= WALK_START) & (time <= WALK_END)
            walk_df = df[walk_mask]

            x = walk_df[f'{prefix}_x'].values
            y = walk_df[f'{prefix}_y'].values
            z = walk_df[f'{prefix}_z'].values

            # Swap axes: Y horizontal, X vertical
            points = np.column_stack([y, x]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            z_segments = (z[:-1] + z[1:]) / 2

            lc = LineCollection(segments, cmap='coolwarm', linewidth=trial_widths[i])
            lc.set_array(z_segments)
            lc.set_clim(exp_z_min, exp_z_max)
            ax.add_collection(lc)

        # Start/end markers — only label once (first experiment) for shared legend
        for j, f in enumerate(trial_files):
            df = pd.read_csv(f)
            time = df['elapsed_time'].values
            walk_mask = (time >= WALK_START) & (time <= WALK_END)
            walk_df = df[walk_mask]
            label_start = 'Start' if (idx == 0 and j == 0) else None
            label_end = 'End' if (idx == 0 and j == 0) else None
            ax.plot(walk_df[f'{prefix}_y'].iloc[0], walk_df[f'{prefix}_x'].iloc[0],
                    'o', color='green', markersize=5, zorder=5, label=label_start)
            ax.plot(walk_df[f'{prefix}_y'].iloc[-1], walk_df[f'{prefix}_x'].iloc[-1],
                    's', color='red', markersize=5, zorder=5, label=label_end)

        # ax.autoscale()
        # ax.set_aspect('equal')
        ax.set_title(exp_name, fontsize=11)
        # ax.set_xlabel('Y (m)', fontsize=9)
        # ax.set_ylabel('X (m)', fontsize=9)
        fig.supylabel('x (m)', fontsize=12)
        fig.supxlabel('y (m)', fontsize=12)
        ax.tick_params(labelsize=8)

        # Per-subplot colorbar
        sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                   norm=plt.Normalize(vmin=exp_z_min, vmax=exp_z_max))
        cbar = fig.colorbar(sm, ax=ax, shrink=0.9, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        if idx % 2 == 1:  # only label right-column colorbars
            cbar.set_label('Z (m)', fontsize=8)


    fig.legend(loc='upper center', ncol=2, fontsize=8, frameon=True)
    # fig.suptitle(f'{source.upper()} Trajectories — colored by Z height', fontsize=13)
    # fig.subplots_adjust(bottom=0.08)
    # plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.show()


def compare_experiments(source='vision'):
    """Compare RMS curvature across all experiments.

    For each experiment (distribution), compute RMS curvature for each of the 3 trials,
    then report mean ± SD of those 3 RMS values.
    """
    exp_names = list(EXPERIMENTS.keys())

    print(f"\nTrajectory Curvature Comparison ({source.upper()})")
    print(f"{'Experiment':<20} {'Trial 1':>10} {'Trial 2':>10} {'Trial 3':>10} {'Mean ± SD':>20}")
    print("-" * 75)

    all_means = []
    all_sds = []

    for exp_name in exp_names:
        trial_rms = []
        for f in EXPERIMENTS[exp_name]:
            result = compute_trial_curvature(f, source)
            trial_rms.append(result['rms_kappa'])

        mean_rms = np.mean(trial_rms)
        sd_rms = np.std(trial_rms)
        all_means.append(mean_rms)
        all_sds.append(sd_rms)

        print(f"{exp_name:<20} {trial_rms[0]:>10.4f} {trial_rms[1]:>10.4f} {trial_rms[2]:>10.4f} "
              f"{mean_rms:>8.4f} ± {sd_rms:.4f}")

    # Dot plot
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(exp_names))
    ax.errorbar(x, all_means, yerr=all_sds, fmt='ko-', markersize=4,
                capsize=3, capthick=1, ecolor='red', elinewidth=1,
                color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('RMS Curvature (1/m)')
    # ax.set_title(f'Trajectory Smoothness — {source.upper()}')
    ax.margins(x=0.15)
    plt.tight_layout()
    plt.show()

    return exp_names, all_means, all_sds


if __name__ == '__main__':
    # plot_xy_colored_by_z_overlay('baseline', source="vision")
    # plot_all_experiments_xy_z(source='vision')
    compare_experiments(source="vision")
