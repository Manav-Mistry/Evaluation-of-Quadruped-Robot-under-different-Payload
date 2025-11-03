from hdf5_logger_utils import plot_2d_trajectory


# ========== Example Usage (for testing) ==========
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python utils.py <path_to_hdf5_file>")
        print("\nExample:")
        print("  python utils.py /home/manav/IsaacLab/logged_data/spot_demo_keyboard_20251031_180612.h5")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Test the function
    plot_2d_trajectory(
        filename=filename,
        env_id=0,
        show_plot=True,
        color_by_time=False,
        save_path="./position_plots"
    )