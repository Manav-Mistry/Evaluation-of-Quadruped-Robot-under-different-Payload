"""
Visualization utilities for HDF5 logged data.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List


def plot_2d_trajectory(
    filename: str,
    env_id: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    show_plot: bool = True,
    color_by_time: bool = True,
    title: Optional[str] = None,
    dpi: int = 150
) -> None:
    """
    Plot 2D (x-y) trajectory of the robot from HDF5 logged data.
    
    Args:
        filename (str): Path to HDF5 file
        env_id (int): Environment ID to plot (default: 0)
        save_path (str, optional): Path to save the figure. If None, auto-generates name
        figsize (tuple): Figure size in inches (width, height)
        show_plot (bool): Whether to display the plot
        color_by_time (bool): If True, colors trajectory by timestep progression
        title (str, optional): Custom title for the plot
        dpi (int): Resolution for saved figure
    
    Returns:
        None
    
    Example:
        >>> plot_2d_trajectory('data.h5', env_id=0, save_path='trajectory.png')
        >>> plot_2d_trajectory('data.h5', color_by_time=False, show_plot=True)
    """
    
    # ========== Load Data ==========
    try:
        with h5py.File(filename, 'r') as f:
            # Get actual timesteps logged
            actual_timesteps = f.attrs.get('actual_timesteps', None)
            
            # Load robot positions
            robot_pos = f[f'env_{env_id}/kinematics/robot_position'][:]
            
            # Trim to actual logged data
            if actual_timesteps is not None:
                robot_pos = robot_pos[:actual_timesteps]
            
            # Extract metadata
            control_mode = f.attrs.get('control_mode', 'unknown')
            date = f.attrs.get('date', 'unknown')
            
    except KeyError as e:
        raise ValueError(f"Data not found in file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading HDF5 file: {e}")
    
    # Extract x, y coordinates
    x = robot_pos[:, 0]
    y = robot_pos[:, 1]
    
    # ========== Create Figure ==========
    fig, ax = plt.subplots(figsize=figsize)
    
    # ========== Plot Trajectory ==========
    if color_by_time:
        # Create color gradient based on timestep
        timesteps = np.arange(len(x))
        scatter = ax.scatter(
            x, y, 
            c=timesteps, 
            cmap='viridis', 
            s=20, 
            alpha=0.6,
            edgecolors='none',
            label='Trajectory'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Timestep', fontsize=11, labelpad=10)
        cbar.ax.tick_params(labelsize=10)
    else:
        # Single color trajectory
        ax.plot(x, y, 'k-', linewidth=1.5, alpha=0.7, label='Trajectory')
    
    # ========== Mark Start and End Points ==========
    ax.plot(
        x[0], y[0], 
        'go', 
        markersize=12, 
        label='Start', 
        zorder=5,
        markeredgewidth=2,
        markeredgecolor='darkgreen'
    )
    
    ax.plot(
        x[-1], y[-1], 
        'ro', 
        markersize=12, 
        label='End', 
        zorder=5,
        markeredgewidth=2,
        markeredgecolor='darkred'
    )
    
    # ========== Add Direction Arrows (Every N points) ==========
    # Show direction with arrows at regular intervals
    arrow_interval = max(len(x) // 10, 1)  # ~10 arrows along path
    
    for i in range(0, len(x) - 1, arrow_interval):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        
        # Only draw arrow if movement is significant
        if np.sqrt(dx**2 + dy**2) > 0.01:
            ax.arrow(
                x[i], y[i], dx, dy,
                head_width=0.2,
                head_length=0.08,
                fc='black',
                ec='black',
                alpha=1,
                length_includes_head=True,
                zorder=3
            )
    
    # ========== Calculate Statistics ==========
    # Total distance traveled
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_distance = np.sum(distances)
    
    # Straight-line displacement
    displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    
    # Path efficiency (how straight the path is)
    path_efficiency = (displacement / total_distance * 100) if total_distance > 0 else 0
    
    # ========== Add Statistics Text Box ==========
    stats_text = (
        f"Timesteps: {len(x)}\n"
        f"Distance: {total_distance:.2f} m\n"
        f"Displacement: {displacement:.2f} m\n"
        f"Efficiency: {path_efficiency:.1f}%"
    )
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10,
        family='monospace'
    )
    
    # ========== Formatting ==========
    ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    
    # Set title
    if title is None:
        filename_short = Path(filename).stem
        title = f'Robot 2D Trajectory - {control_mode.upper()} Mode\n{filename_short}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Equal aspect ratio for accurate representation
    ax.axis('equal')
    
    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add margins
    x_margin = (x.max() - x.min()) * 0.1
    y_margin = (y.max() - y.min()) * 0.1
    ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
    
    plt.tight_layout()
    
    # ========== Save Figure ==========
    if save_path is None:
        # Auto-generate save path
        save_path = str(Path(filename).with_suffix('')) + '_trajectory_2d.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved 2D trajectory plot: {save_path}")
    
    # ========== Display ==========
    if show_plot:
        plt.show()
    else:
        plt.close()

