#!/usr/bin/env python3
"""
Simple script to read and inspect HDF5 logged data.
"""

import h5py
import numpy as np
import sys

def inspect_hdf5_file(filename):
    """
    Inspect the structure and contents of an HDF5 file.
    
    Args:
        filename (str): Path to HDF5 file
    """
    print("="*70)
    print(f"Inspecting HDF5 File: {filename}")
    print("="*70)
    
    with h5py.File(filename, 'r') as f:
        
        # ========== Print File Metadata ==========
        print("\n📋 FILE METADATA:")
        print("-"*70)
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # ========== Print File Structure ==========
        print("\n📁 FILE STRUCTURE:")
        print("-"*70)
        
        def print_structure(name, obj):
            """Recursively print HDF5 structure."""
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}📂 {name}/")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 {name}")
                print(f"{indent}   Shape: {obj.shape}")
                print(f"{indent}   Type: {obj.dtype}")
                # Print dataset attributes
                if obj.attrs:
                    for attr_key, attr_val in obj.attrs.items():
                        print(f"{indent}   {attr_key}: {attr_val}")
        
        f.visititems(print_structure)
        
        # ========== Print Data Summary for env_0 ==========
        print("\n📊 DATA SUMMARY (Environment 0):")
        print("-"*70)
        
        if 'env_0/kinematics/robot_position' in f:
            robot_pos = f['env_0/kinematics/robot_position'][:]
            actual_timesteps = f.attrs.get('actual_timesteps', len(robot_pos))
            
            # Get only valid data (non-zero timesteps)
            valid_data = robot_pos[:actual_timesteps]
            
            print(f"\n🤖 Robot Position:")
            print(f"  Total timesteps logged: {actual_timesteps}")
            print(f"  Data shape: {robot_pos.shape}")
            print(f"  Valid data shape: {valid_data.shape}")
            
            print(f"\n  First 5 positions:")
            for i in range(min(5, len(valid_data))):
                print(f"    [{i}] x={valid_data[i,0]:.3f}, y={valid_data[i,1]:.3f}, z={valid_data[i,2]:.3f}")
            
            print(f"\n  Last 5 positions:")
            for i in range(max(0, actual_timesteps-5), actual_timesteps):
                print(f"    [{i}] x={valid_data[i,0]:.3f}, y={valid_data[i,1]:.3f}, z={valid_data[i,2]:.3f}")
            
            print(f"\n  Statistics:")
            print(f"    X: min={valid_data[:,0].min():.3f}, max={valid_data[:,0].max():.3f}, mean={valid_data[:,0].mean():.3f}")
            print(f"    Y: min={valid_data[:,1].min():.3f}, max={valid_data[:,1].max():.3f}, mean={valid_data[:,1].mean():.3f}")
            print(f"    Z: min={valid_data[:,2].min():.3f}, max={valid_data[:,2].max():.3f}, mean={valid_data[:,2].mean():.3f}")
            
            # Calculate distance traveled
            distances = np.sqrt(np.sum(np.diff(valid_data[:, :2], axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            print(f"\n  Distance traveled: {total_distance:.3f} meters")
        
        else:
            print("  ⚠️ No robot position data found")
    
    print("\n" + "="*70)


def load_robot_trajectory(filename, env_id=0):
    """
    Load robot trajectory from HDF5 file.
    
    Args:
        filename (str): Path to HDF5 file
        env_id (int): Environment ID to load
    
    Returns:
        dict: Dictionary with trajectory data
    """
    with h5py.File(filename, 'r') as f:
        actual_timesteps = f.attrs.get('actual_timesteps', None)
        
        # Load robot position
        robot_pos = f[f'env_{env_id}/kinematics/robot_position'][:]
        
        # Trim to actual logged timesteps
        if actual_timesteps is not None:
            robot_pos = robot_pos[:actual_timesteps]
        
        trajectory = {
            'positions': robot_pos,
            'timesteps': actual_timesteps,
            'x': robot_pos[:, 0],
            'y': robot_pos[:, 1],
            'z': robot_pos[:, 2],
        }
        
        return trajectory


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_logged_data.py <path_to_hdf5_file>")
        print("\nExample:")
        print("  python read_logged_data.py /home/manav/IsaacLab/logged_data/spot_demo_keyboard_20250131_143022.h5")
        sys.exit(1)
    
    filename = sys.argv[1]
    inspect_hdf5_file(filename)