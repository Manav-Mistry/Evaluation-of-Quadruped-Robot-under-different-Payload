
from hdf5_logger_utils import HDF5Reader, plot_imu_combined, save_figure, plot_payload_position_3d
from pathlib import Path

try:

    data = HDF5Reader.load("/home/manav/my_isaaclab_project/logged_data/imu_test_keyboard_20251124_185344.h5", dt=0.2)

    print(f"Loaded: {data}")
    print(f"Fields: {data.list_fields()}\n")
    
    fig = plot_imu_combined(data)
    save_figure(fig, "imu_data_test")
    
    if data.has_field('payload_pos_w'):
        plot_payload_position_3d(data)
    
    print("\nDisplaying plots...")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run the logger first to generate data.")