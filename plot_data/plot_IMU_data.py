from hdf5_logger_utils import HDF5Reader, plot_imu_combined, save_figure, plot_payload_position_3d, plot_imu_acceleration
from pathlib import Path
import re

try:
    file_path = "/home/manav/my_isaaclab_project/logged_data/complete_demo_keyboard_20251230_124054.h5"
    
    # Extract date and time from filename (format: YYYYMMDD_HHMMSS)
    filename = Path(file_path).stem  # Gets filename without extension
    datetime_match = re.search(r'(\d{8}_\d{6})', filename)
    datetime_suffix = f"_{datetime_match.group(1)}" if datetime_match else ""
    
    data = HDF5Reader.load(file_path, dt=0.2)

    print(f"Loaded: {data}")
    print(f"Fields: {data.list_fields()}\n")
    
    print()
    fig = plot_imu_acceleration(data)
    save_figure(fig, f"imu_data_test{datetime_suffix}")
    
    if data.has_field('payload_pos_w'):
        fig = plot_payload_position_3d(data)
        save_figure(fig, f"payload_pos_data_{datetime_suffix}")
    
    print("\nDisplaying plots...")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run the logger first to generate data.")