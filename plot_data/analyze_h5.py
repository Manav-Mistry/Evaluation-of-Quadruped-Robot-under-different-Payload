"""
Analyze HDF5 experiment data files.

Usage:
    python analyze_h5.py <path_to_h5_file>
    python analyze_h5.py <path_to_folder>  # analyzes all .h5 files in folder
"""

import h5py
import numpy as np
from pathlib import Path
import sys


def print_header(title):
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")


def analyze_file(file_path):
    """Analyze a single HDF5 file."""
    print_header(f"Analyzing: {Path(file_path).name}")

    with h5py.File(file_path, 'r') as f:
        # Print structure
        print("\n[Structure]")
        datasets = {}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    'shape': obj.shape,
                    'dtype': obj.dtype
                }
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  {name}/ (group)")

        f.visititems(visitor)

        # Analyze timing
        print("\n[Timing Analysis]")
        time_key = None
        for key in datasets:
            if 'time' in key.lower():
                time_key = key
                break

        if time_key:
            time_data = f[time_key][:]
            if time_data.ndim > 1:
                time_data = time_data[:, 0]

            # Find valid data (non-zero)
            nonzero_idx = np.where(time_data > 0)[0]
            if len(nonzero_idx) > 0:
                valid_time = time_data[nonzero_idx[0]:nonzero_idx[-1]+1]
                diffs = np.diff(valid_time)

                print(f"  Time field: {time_key}")
                print(f"  Total samples: {len(time_data)}")
                print(f"  Valid samples: {len(valid_time)} (non-zero)")
                print(f"  Duration: {valid_time[-1] - valid_time[0]:.2f} seconds")
                print(f"  dt (mean): {np.mean(diffs):.6f} s")
                print(f"  dt (range): [{np.min(diffs):.6f}, {np.max(diffs):.6f}] s")
                print(f"  Sampling rate: {1/np.mean(diffs):.1f} Hz")
            else:
                print(f"  Warning: No valid time data found")
        else:
            print("  No time field found")

        # Analyze each data field
        print("\n[Data Statistics]")
        for name, info in datasets.items():
            if 'time' in name.lower():
                continue

            data = f[name][:]

            # Find valid data range using time if available
            if time_key:
                time_data = f[time_key][:]
                if time_data.ndim > 1:
                    time_data = time_data[:, 0]
                nonzero_idx = np.where(time_data > 0)[0]
                if len(nonzero_idx) > 0:
                    data = data[nonzero_idx[0]:nonzero_idx[-1]+1]

            # Use absolute values for acceleration fields
            is_acc = 'acc' in name.lower()
            if is_acc:
                data = np.abs(data)

            print(f"\n  {name}:" + (" (absolute values)" if is_acc else ""))
            print(f"    Shape: {data.shape}")

            if data.ndim == 1:
                print(f"    Mean:   {np.mean(data):.4f}")
                print(f"    Median: {np.median(data):.4f}")
                print(f"    Std:    {np.std(data):.4f}")
            elif data.ndim == 2:
                axis_labels = ['X', 'Y', 'Z'] if data.shape[1] == 3 else [str(i) for i in range(data.shape[1])]
                for i in range(min(data.shape[1], 12)):  # Limit to 12 columns
                    label = axis_labels[i] if i < len(axis_labels) else str(i)
                    col = data[:, i]
                    print(f"    [{label}] Mean: {np.mean(col):8.4f}, Median: {np.median(col):8.4f}, Std: {np.std(col):8.4f}")


def analyze_folder(folder_path):
    """Analyze all HDF5 files in a folder."""
    folder = Path(folder_path)
    h5_files = sorted(folder.glob('*.h5'))

    if not h5_files:
        print(f"No .h5 files found in {folder_path}")
        return

    print_header(f"Folder Analysis: {folder.name}")
    print(f"Found {len(h5_files)} files")

    # Summary statistics across all files
    all_durations = []
    all_samples = []

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            # Find time field
            time_key = None
            for key in f.keys():
                def find_time(name, obj):
                    nonlocal time_key
                    if isinstance(obj, h5py.Dataset) and 'time' in name.lower():
                        time_key = name
                f.visititems(find_time)

            if time_key:
                time_data = f[time_key][:]
                if time_data.ndim > 1:
                    time_data = time_data[:, 0]

                nonzero_idx = np.where(time_data > 0)[0]
                if len(nonzero_idx) > 0:
                    valid_time = time_data[nonzero_idx[0]:nonzero_idx[-1]+1]
                    duration = valid_time[-1] - valid_time[0]
                    all_durations.append(duration)
                    all_samples.append(len(valid_time))
                    print(f"  {h5_file.name}: {len(valid_time)} samples, {duration:.2f}s")

    if all_durations:
        print(f"\n[Summary]")
        print(f"  Total files: {len(all_durations)}")
        print(f"  Duration: {np.mean(all_durations):.2f} +/- {np.std(all_durations):.2f} s")
        print(f"  Samples: {np.mean(all_samples):.0f} +/- {np.std(all_samples):.0f}")


def compare_experiments(folders):
    """Compare statistics across multiple experiment folders."""
    print_header("Experiment Comparison")

    for folder_path in folders:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"  {folder.name}: NOT FOUND")
            continue

        h5_files = sorted(folder.glob('*.h5'))
        if not h5_files:
            print(f"  {folder.name}: No .h5 files")
            continue

        durations = []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                # Find time field
                time_key = None
                def find_time(name, obj):
                    nonlocal time_key
                    if isinstance(obj, h5py.Dataset) and 'time' in name.lower():
                        time_key = name
                f.visititems(find_time)

                if time_key:
                    time_data = f[time_key][:]
                    if time_data.ndim > 1:
                        time_data = time_data[:, 0]
                    nonzero_idx = np.where(time_data > 0)[0]
                    if len(nonzero_idx) > 0:
                        valid_time = time_data[nonzero_idx[0]:nonzero_idx[-1]+1]
                        durations.append(valid_time[-1] - valid_time[0])

        if durations:
            print(f"  {folder.name}: {len(h5_files)} files, {np.mean(durations):.2f} +/- {np.std(durations):.2f} s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample paths:")
        print("  /home/manav/Desktop/data_collection/simulation/flat_terrain_with_8kg_payload/center_payload_8kg/")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file() and path.suffix == '.h5':
        analyze_file(path)
    elif path.is_dir():
        # Check if it contains .h5 files directly or subfolders
        h5_files = list(path.glob('*.h5'))
        subfolders = [p for p in path.iterdir() if p.is_dir()]

        if h5_files:
            analyze_folder(path)
            if len(sys.argv) > 2 and sys.argv[2] == '--detail':
                for h5_file in h5_files:
                    analyze_file(h5_file)
        elif subfolders:
            # Compare across subfolders
            compare_experiments(subfolders)
        else:
            print(f"No .h5 files or subfolders found in {path}")
    else:
        print(f"Invalid path: {path}")
