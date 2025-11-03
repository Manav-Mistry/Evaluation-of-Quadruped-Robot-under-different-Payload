import h5py
import numpy as np
from datetime import datetime
import os
from pathlib import Path


class HDF5Logger:
    """
    Handles HDF5 data logging for robot experiments.
    
    Features:
    - Buffered writing for performance
    - Automatic file naming with timestamps
    - Metadata storage
    - Pre-allocated datasets for efficiency
    - Graceful handling of early termination
    """
    
    def __init__(self, config, control_mode, num_envs=1, max_timesteps=100000):
        """
        Initialize the HDF5 logger.
        
        Args:
            config (dict): Configuration dictionary with keys:
                - 'enabled': bool, toggle logging
                - 'buffer_size': int, number of timesteps to buffer before writing
                - 'save_dir': str, directory to save HDF5 files
                - 'experiment_prefix': str, prefix for experiment files
            control_mode (str): 'keyboard' or 'waypoint'
            num_envs (int): Number of parallel environments
            max_timesteps (int): Maximum expected timesteps (for pre-allocation)
        """
        self.config = config
        self.control_mode = control_mode
        self.num_envs = num_envs
        self.max_timesteps = max_timesteps
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            print("[HDF5Logger] Logging disabled by configuration")
            return
        
        # Buffer configuration
        self.buffer_size = config.get('buffer_size', 100)
        self.current_buffer_idx = 0
        
        # Track actual number of logged timesteps
        self.logged_timesteps = 0
        
        # Create save directory if it doesn't exist
        self.save_dir = Path(config.get('save_dir', './logged_data'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_prefix = config.get('experiment_prefix', 'spot_experiment')
        self.filename = self.save_dir / f"{experiment_prefix}_{control_mode}_{timestamp}.h5"
        
        # Initialize HDF5 file
        self._initialize_file()
        
        # Initialize buffers
        self._initialize_buffers()
        
        print(f"[HDF5Logger] Initialized")
        print(f"  File: {self.filename}")
        print(f"  Buffer size: {self.buffer_size} timesteps")
        print(f"  Max timesteps: {self.max_timesteps}")
        print(f"  Control mode: {self.control_mode}")
    
    def _initialize_file(self):
        """Create HDF5 file and set up structure."""
        self.file = h5py.File(self.filename, 'w')
        
        # Store experiment-level metadata
        self.file.attrs['experiment_name'] = f"{self.config.get('experiment_prefix', 'spot')}_{self.control_mode}"
        self.file.attrs['date'] = datetime.now().isoformat()
        self.file.attrs['control_mode'] = self.control_mode
        self.file.attrs['num_envs'] = self.num_envs
        self.file.attrs['max_timesteps'] = self.max_timesteps
        self.file.attrs['buffer_size'] = self.buffer_size
        
        # Create groups for each environment
        for env_id in range(self.num_envs):
            env_group = self.file.create_group(f'env_{env_id}')
            
            # Create kinematics subgroup
            kinematics_group = env_group.create_group('kinematics')
            
            # Pre-allocate dataset for robot position
            # Shape: [max_timesteps, 3] for (x, y, z)
            kinematics_group.create_dataset(
                'robot_position',
                shape=(self.max_timesteps, 3),
                dtype=np.float32,
                chunks=(self.buffer_size, 3),  # Chunk along time dimension
                compression='gzip',
                compression_opts=4  # Compression level (1-9, 4 is good balance)
            )
            
            # Add metadata to dataset
            kinematics_group['robot_position'].attrs['units'] = 'meters'
            kinematics_group['robot_position'].attrs['frame'] = 'world'
            kinematics_group['robot_position'].attrs['description'] = 'Robot base position in world frame (x, y, z)'
            kinematics_group['robot_position'].attrs['coordinates'] = 'x, y, z'
        
        print(f"[HDF5Logger] File structure created: {self.filename}")
    
    def _initialize_buffers(self):
        """Initialize in-memory buffers for each environment."""
        self.buffers = {}
        
        for env_id in range(self.num_envs):
            self.buffers[env_id] = {
                'robot_position': np.zeros((self.buffer_size, 3), dtype=np.float32)
            }
    
    def log_timestep(self, timestep, robot_pos, env_id=0):
        """
        Log data for a single timestep.
        
        Args:
            timestep (int): Current timestep number
            robot_pos (np.ndarray): Robot position [x, y, z] as numpy array
            env_id (int): Environment ID (default 0 for single environment)
        """
        if not self.enabled:
            return
        
        # Validate env_id
        if env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} exceeds num_envs {self.num_envs}")
        
        # Convert to numpy if it's a tensor
        if not isinstance(robot_pos, np.ndarray):
            robot_pos = np.array(robot_pos)
        
        # Store in buffer. the second idx named 'robot_position' is a dataset name defined in _initialize_file
        self.buffers[env_id]['robot_position'][self.current_buffer_idx] = robot_pos
        
        # Increment buffer index
        self.current_buffer_idx += 1
        
        # Flush if buffer is full
        if self.current_buffer_idx >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered data to HDF5 file."""
        if not self.enabled:
            return
        
        if self.current_buffer_idx == 0:
            return  # Nothing to flush
        
        # Write data for each environment
        for env_id in range(self.num_envs):
            # Calculate write range
            start_idx = self.logged_timesteps
            end_idx = self.logged_timesteps + self.current_buffer_idx
            
            # Check if we're exceeding pre-allocated space
            if end_idx > self.max_timesteps:
                print(f"[HDF5Logger] WARNING: Exceeded max_timesteps ({self.max_timesteps})")
                print(f"[HDF5Logger] Truncating data. Consider increasing max_timesteps.")
                end_idx = self.max_timesteps
                actual_write_size = end_idx - start_idx
            else:
                actual_write_size = self.current_buffer_idx
            
            # Write robot position
            self.file[f'env_{env_id}/kinematics/robot_position'][start_idx:end_idx] = \
                self.buffers[env_id]['robot_position'][:actual_write_size]
        
        # Update logged timesteps counter
        self.logged_timesteps += self.current_buffer_idx
        
        # Reset buffer index
        self.current_buffer_idx = 0
        
        # Flush to disk (ensure data is written)
        self.file.flush()
        
        print(f"[HDF5Logger] Flushed {actual_write_size} timesteps to disk (Total: {self.logged_timesteps})")
    
    def close(self):
        """Close the HDF5 file and save final metadata."""
        if not self.enabled:
            return
        
        print(f"[HDF5Logger] Closing file...")
        
        # Flush any remaining buffered data
        self.flush()
        
        # Store actual number of timesteps logged
        self.file.attrs['actual_timesteps'] = self.logged_timesteps
        self.file.attrs['end_time'] = datetime.now().isoformat()
        
        # Close file
        self.file.close()
        
        # Calculate file size
        file_size_mb = os.path.getsize(self.filename) / (1024 * 1024)
        
        print(f"[HDF5Logger] File closed successfully")
        print(f"  Total timesteps logged: {self.logged_timesteps}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Location: {self.filename}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures file is closed."""
        self.close()
        return False


# ========================================
# Usage Example (for reference)
# ========================================
# if __name__ == "__main__":
#     """
#     Example usage and test of HDF5Logger.
#     Run this file directly to test the logger.
#     """
    
#     # Configuration
#     test_config = {
#         'enabled': True,
#         'buffer_size': 10,  # Small buffer for testing
#         'save_dir': './test_logs',
#         'experiment_prefix': 'test_spot'
#     }
    
#     # Create logger
#     logger = HDF5Logger(
#         config=test_config,
#         control_mode='test',
#         num_envs=1,
#         max_timesteps=100
#     )
    
#     print("\n[TEST] Simulating robot movement...")
    
#     # Simulate 50 timesteps of robot movement
#     for t in range(50):
#         # Simulate robot moving in a circle
#         x = np.cos(t * 0.1) * 2.0
#         y = np.sin(t * 0.1) * 2.0
#         z = 0.5 + np.sin(t * 0.05) * 0.1
        
#         robot_pos = np.array([x, y, z], dtype=np.float32)
        
#         # Log the data
#         logger.log_timestep(timestep=t, robot_pos=robot_pos, env_id=0)
        
#         if t % 10 == 0:
#             print(f"  Logged timestep {t}: pos=[{x:.2f}, {y:.2f}, {z:.2f}]")
    
#     # Close logger
#     logger.close()
    
#     print("\n[TEST] Reading back logged data...")
    
#     # Verify data was written correctly
#     with h5py.File(logger.filename, 'r') as f:
#         print(f"\nFile contents:")
#         print(f"  Groups: {list(f.keys())}")
        
#         print(f"\nMetadata:")
#         for key, value in f.attrs.items():
#             print(f"  {key}: {value}")
        
#         print(f"\nEnvironment 0 data:")
#         print(f"  Groups: {list(f['env_0'].keys())}")
#         print(f"  Kinematics datasets: {list(f['env_0/kinematics'].keys())}")
        
#         # Read robot positions
#         robot_positions = f['env_0/kinematics/robot_position'][:]
#         actual_timesteps = f.attrs['actual_timesteps']
        
#         print(f"\nRobot position data:")
#         print(f"  Shape: {robot_positions.shape}")
#         print(f"  Actual timesteps logged: {actual_timesteps}")
#         print(f"  First 5 positions:")
#         print(robot_positions[:5])
#         print(f"  Last 5 logged positions:")
#         print(robot_positions[actual_timesteps-5:actual_timesteps])
    
#     print("\n[TEST] Test completed successfully!")