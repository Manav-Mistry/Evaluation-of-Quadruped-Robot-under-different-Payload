"""
Schema-Driven HDF5 Logger

"""

import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DataField:
    """
    Defines a single data field to be logged.
    
    Attributes:
        shape: Tuple defining the shape of data per timestep (e.g., (3,) for xyz)
        dtype: NumPy dtype for the data
        units: String describing units (for metadata)
        description: Human-readable description
        frame: Reference frame if applicable (e.g., 'world', 'body')
    """
    shape: tuple
    dtype: type = np.float32
    units: str = ""
    description: str = ""
    frame: str = ""


class LoggingSchema:
    """
    Defines what data fields to log.
    
    Use this to configure the logger without modifying its code.
    """
    
    def __init__(self):
        self.fields: Dict[str, DataField] = {}
    
    def add_field(self, name: str, shape: tuple, dtype=np.float32,
                  units: str = "", description: str = "", frame: str = "") -> 'LoggingSchema':
        """Add a data field to the schema. Returns self for chaining."""
        self.fields[name] = DataField(
            shape=shape,
            dtype=dtype,
            units=units,
            description=description,
            frame=frame
        )
        return self
    
    def remove_field(self, name: str) -> 'LoggingSchema':
        """Remove a field from the schema. Returns self for chaining."""
        if name in self.fields:
            del self.fields[name]
        return self
    
    @classmethod
    def payload_experiment(cls) -> 'LoggingSchema':
        """Preset: Payload experiment with IMU data."""
        schema = cls()
        schema.add_field('payload_pos_w', (3,), description='Payload IMU position_w', frame='sensor')
        schema.add_field('payload_quat_w', (4,),  description='Payload IMU quaternion_w', frame='sensor')
        schema.add_field('payload_lin_acc_b', (3,),  description='Payload IMU linear acceleration base frame', frame='sensor')
        schema.add_field('payload_ang_acc_b', (3,), description='Payload IMU angular acceleration base frame', frame='sensor')
        return schema


class HDF5Logger:

    def __init__(
        self,
        config: Dict[str, Any],
        schema: LoggingSchema,
        control_mode: str = 'unknown',
        num_envs: int = 1,
        max_timesteps: int = 100000
    ):
        """
        Initialize the HDF5 logger.
        
        Args:
            max_timesteps: Maximum expected timesteps (for pre-allocation)
        """
        self.config = config
        self.schema = schema
        self.control_mode = control_mode
        self.num_envs = num_envs
        self.max_timesteps = max_timesteps
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            print("[HDF5Logger] Logging disabled by configuration")
            return
        
        if not schema.fields:
            print("[HDF5Logger] WARNING: Empty schema - nothing will be logged")
            self.enabled = False
            return
        
        self.buffer_size = config.get('buffer_size', 100)
        self.current_buffer_idx = 0
        self.logged_timesteps = 0
        
        self.save_dir = Path(config.get('save_dir', './logged_data'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_prefix = config.get('experiment_prefix', 'spot_experiment')
        self.filename = self.save_dir / f"{experiment_prefix}_{control_mode}_{timestamp}.h5"
        
        self._initialize_file()
        self._initialize_buffers()
        
        print(f"[HDF5Logger] Initialized")
        print(f"  File: {self.filename}")
        print(f"  Schema fields: {list(self.schema.fields.keys())}")
        print(f"  Buffer size: {self.buffer_size}")
        print(f"  Control mode: {self.control_mode}")
    
    def _initialize_file(self):
        """Create HDF5 file structure based on schema."""
        self.file = h5py.File(self.filename, 'w')
        
        # Experiment-level metadata
        self.file.attrs['experiment_name'] = f"{self.config.get('experiment_prefix', 'spot')}_{self.control_mode}"
        self.file.attrs['date'] = datetime.now().isoformat()
        self.file.attrs['control_mode'] = self.control_mode
        self.file.attrs['num_envs'] = self.num_envs
        self.file.attrs['max_timesteps'] = self.max_timesteps
        self.file.attrs['buffer_size'] = self.buffer_size
        self.file.attrs['schema_fields'] = list(self.schema.fields.keys())
        
        # Create structure for each environment
        for env_id in range(self.num_envs):
            env_group = self.file.create_group(f'env_{env_id}')
            data_group = env_group.create_group('data')
            
            # Create datasets based on schema
            for field_name, field_def in self.schema.fields.items():
                dataset_shape = (self.max_timesteps,) + field_def.shape
                chunk_shape = (self.buffer_size,) + field_def.shape
                
                ds = data_group.create_dataset(
                    field_name,
                    shape=dataset_shape,
                    dtype=field_def.dtype,
                    chunks=chunk_shape,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Attach metadata
                if field_def.units:
                    ds.attrs['units'] = field_def.units
                if field_def.description:
                    ds.attrs['description'] = field_def.description
                if field_def.frame:
                    ds.attrs['frame'] = field_def.frame
                ds.attrs['shape_per_timestep'] = field_def.shape
    
    def _initialize_buffers(self):
        """Initialize in-memory buffers based on schema."""
        self.buffers = {}
        
        for env_id in range(self.num_envs):
            self.buffers[env_id] = {}
            for field_name, field_def in self.schema.fields.items():
                buffer_shape = (self.buffer_size,) + field_def.shape
                self.buffers[env_id][field_name] = np.zeros(buffer_shape, dtype=field_def.dtype)
    
    def log(self, data: Dict[str, Any], env_id: int = 0):
        """
        Log data for a single timestep.
        
        Args:
            data: Dictionary mapping field names to values.
                  Values can be numpy arrays or torch tensors.
                  Missing fields will be logged as zeros with a warning.
            env_id: Environment ID (default 0)
        """
        if not self.enabled:
            return
        
        if env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} exceeds num_envs {self.num_envs}")
        
        # Log each field
        for field_name in self.schema.fields:
            if field_name in data:
                value = data[field_name]
                
                # Convert torch tensor to numpy if needed
                if hasattr(value, 'cpu'):
                    value = value.cpu().numpy()
                elif not isinstance(value, np.ndarray):
                    value = np.array(value)
                
                self.buffers[env_id][field_name][self.current_buffer_idx] = value
            else:
                # Field missing - log zeros (could also raise warning)
                pass  # Buffer already initialized to zeros
        
        self.current_buffer_idx += 1
        
        if self.current_buffer_idx >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered data to HDF5 file."""
        if not self.enabled or self.current_buffer_idx == 0:
            return
        
        for env_id in range(self.num_envs):
            start_idx = self.logged_timesteps
            end_idx = self.logged_timesteps + self.current_buffer_idx
            
            if end_idx > self.max_timesteps:
                print(f"[HDF5Logger] WARNING: Exceeded max_timesteps ({self.max_timesteps})")
                end_idx = self.max_timesteps
                actual_write_size = end_idx - start_idx
            else:
                actual_write_size = self.current_buffer_idx
            
            # Write all fields from schema
            for field_name in self.schema.fields:
                self.file[f'env_{env_id}/data/{field_name}'][start_idx:end_idx] = \
                    self.buffers[env_id][field_name][:actual_write_size]
        
        self.logged_timesteps += self.current_buffer_idx
        self.current_buffer_idx = 0
        self.file.flush()
        
        print(f"[HDF5Logger] Flushed to disk (Total: {self.logged_timesteps} timesteps)")
    
    def close(self):
        """Close the HDF5 file and save final metadata."""
        if not self.enabled:
            return
        
        self.flush()
        
        self.file.attrs['actual_timesteps'] = self.logged_timesteps
        self.file.attrs['end_time'] = datetime.now().isoformat()
        self.file.close()
        
        import os
        file_size_mb = os.path.getsize(self.filename) / (1024 * 1024)
        
        print(f"[HDF5Logger] Closed")
        print(f"  Timesteps: {self.logged_timesteps}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  File: {self.filename}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Utility function to inspect logged data
# =============================================================================

# def inspect_hdf5(filepath: str):
#     """Utility to inspect contents of a logged HDF5 file."""
#     with h5py.File(filepath, 'r') as f:
#         print(f"\n{'='*60}")
#         print(f"HDF5 File: {filepath}")
#         print(f"{'='*60}")
        
#         print("\nMetadata:")
#         for key, value in f.attrs.items():
#             print(f"  {key}: {value}")
        
#         print("\nStructure:")
#         def print_structure(name, obj):
#             indent = "  " * name.count('/')
#             if isinstance(obj, h5py.Dataset):
#                 print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
#             else:
#                 print(f"{indent}{name}/")
        
#         f.visititems(print_structure)
        
#         # Show actual data range
#         actual_ts = f.attrs.get('actual_timesteps', 0)
#         print(f"\nActual timesteps logged: {actual_ts}")
        
#         if actual_ts > 0 and 'env_0/data' in f:
#             print("\nFirst 3 timesteps of each field:")
#             for field_name in f['env_0/data'].keys():
#                 data = f[f'env_0/data/{field_name}'][:min(3, actual_ts)]
#                 print(f"  {field_name}: {data}")