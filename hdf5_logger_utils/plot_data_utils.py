"""
Plotting Utilities for Payload IMU Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional

from . import ExperimentData, HDF5Reader


# Default plotting style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = plt.cm.tab10.colors


def plot_imu_acceleration(
    data: ExperimentData,
    accel_field: str = 'payload_lin_acc_b',
    title: Optional[str] = None
) -> Figure:
    """
    Plot IMU linear acceleration (x, y, z components).
    
    Args:
        data: Experiment data
        accel_field: Field name for linear acceleration
        title: Plot title

    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time = data.get_time_array()
    accel = data.get_field(accel_field)
    
    ax.plot(time, accel[:, 0], label='X', linewidth=2, color=COLORS[0])
    ax.plot(time, accel[:, 1], label='Y', linewidth=2, color=COLORS[1])
    ax.plot(time, accel[:, 2], label='Z', linewidth=2, color=COLORS[2])
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Acceleration', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Payload Linear Acceleration', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_imu_angular(
    data: ExperimentData,
    gyro_field: str = 'payload_ang_acc_b',
    title: Optional[str] = None
) -> Figure:
    """
    Plot IMU angular acceleration/velocity (roll, pitch, yaw).
    
    Args:
        data: Experiment data
        gyro_field: Field name for angular data
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time = data.get_time_array()
    gyro = data.get_field(gyro_field)
    
    ax.plot(time, gyro[:, 0], label='Roll', linewidth=2, color=COLORS[0])
    ax.plot(time, gyro[:, 1], label='Pitch', linewidth=2, color=COLORS[1])
    ax.plot(time, gyro[:, 2], label='Yaw', linewidth=2, color=COLORS[2])
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angular Acceleration', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Payload Angular Acceleration', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_imu_combined(
    data: ExperimentData,
    accel_field: str = 'payload_lin_acc_b',
    gyro_field: str = 'payload_ang_acc_b',
    title: Optional[str] = None
) -> Figure:
    """
    Plot both linear and angular IMU data in one figure (2 subplots).
    
    Args:
        data: Experiment data
        accel_field: Field name for linear acceleration
        gyro_field: Field name for angular acceleration
        title: Figure title
        
    Returns:
        Matplotlib figure with 2 subplots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    time = data.get_time_array()
    
    # Linear acceleration
    accel = data.get_field(accel_field)
    ax1.plot(time, accel[:, 0], label='X', linewidth=2, color=COLORS[0])
    ax1.plot(time, accel[:, 1], label='Y', linewidth=2, color=COLORS[1])
    ax1.plot(time, accel[:, 2], label='Z', linewidth=2, color=COLORS[2])
    ax1.set_ylabel('Acceleration', fontsize=12)
    ax1.set_title('Linear Acceleration', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Angular acceleration
    gyro = data.get_field(gyro_field)
    ax2.plot(time, gyro[:, 0], label='Roll', linewidth=2, color=COLORS[0])
    ax2.plot(time, gyro[:, 1], label='Pitch', linewidth=2, color=COLORS[1])
    ax2.plot(time, gyro[:, 2], label='Yaw', linewidth=2, color=COLORS[2])
    ax2.set_ylabel('Angular Acceleration', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_title('Angular Acceleration', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    else:
        fig.suptitle('Payload IMU Data', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    return fig


def plot_payload_position_3d(
    data: ExperimentData,
    position_field: str = 'payload_pos_w',
    title: Optional[str] = None
) -> Figure:
    """
    Plot 3D trajectory of payload position.
    
    Args:
        data: Experiment data
        position_field: Field name for position
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pos = data.get_field(position_field)
    time = data.get_time_array()
    
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # Color trajectory by time
    scatter = ax.scatter(x, y, z, c=time, cmap='viridis', s=20, alpha=0.6)
    
    # Start and end markers
    ax.plot([x[0]], [y[0]], [z[0]], 'go', markersize=12, label='Start', zorder=5)
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'ro', markersize=12, label='End', zorder=5)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Time (s)', fontsize=12)
    
    ax.legend()
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Payload 3D Position', fontsize=14, fontweight='bold')
    
    return fig


def save_figure(fig: Figure, filename: str, dpi: int = 300):
    """Save figure to file."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to: {filename}")


def show_all():
    """Show all open figures."""
    plt.show()

