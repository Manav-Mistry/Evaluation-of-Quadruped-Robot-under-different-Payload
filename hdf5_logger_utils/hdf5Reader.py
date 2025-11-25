"""
HDF5 Reader Utilities
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ExperimentData:
    """
    Container for loaded experiment data.
    
    Attributes:
        filepath: Path to the HDF5 file
        metadata: Experiment-level metadata (dict)
        fields: Dictionary mapping field names to numpy arrays
        timesteps: Actual number of timesteps logged
        control_mode: Control mode used ('keyboard', 'waypoint', etc.)
        dt: Time between control steps (seconds)
        env_id: Environment ID (for multi-env experiments)
    """
    filepath: Path
    metadata: Dict = field(default_factory=dict)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    timesteps: int = 0
    control_mode: str = "unknown"
    dt: float = 0.02  # Default 50 Hz
    env_id: int = 0
    
    def get_time_array(self) -> np.ndarray:
        """Get time array for plotting (0, dt, 2*dt, ...)."""
        return np.arange(self.timesteps) * self.dt
    
    def get_field(self, field_name: str) -> np.ndarray:
        """Get data for a specific field."""
        if field_name not in self.fields:
            raise KeyError(f"Field '{field_name}' not found. Available: {list(self.fields.keys())}")
        return self.fields[field_name]
    
    def has_field(self, field_name: str) -> bool:
        """Check if a field exists in the data."""
        return field_name in self.fields
    
    def list_fields(self) -> List[str]:
        """List all available fields."""
        return list(self.fields.keys())
    
    def get_field_shape(self, field_name: str) -> Tuple:
        """Get shape of a field (excluding time dimension)."""
        return self.fields[field_name].shape[1:]
    
    def __repr__(self) -> str:
        return (f"ExperimentData(file={self.filepath.name}, "
                f"timesteps={self.timesteps}, "
                f"fields={len(self.fields)}, "
                f"mode={self.control_mode})")


class HDF5Reader:
  
    @staticmethod
    def load(filepath: str, env_id: int = 0, dt = 0.02) -> ExperimentData:
        """
        Load experiment data from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            env_id: Environment ID to load (default 0)
            dt: data collection rate. In simulation I am doing count%n so it changes accordingly
            
        Returns:
            ExperimentData object with loaded data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")
        
        data = ExperimentData(filepath=filepath, env_id=env_id, dt=dt)
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            data.metadata = dict(f.attrs)
            data.timesteps = f.attrs.get('actual_timesteps', 0)
            data.control_mode = f.attrs.get('control_mode', 'unknown')
            
            # Calculate dt from decimation if available
            if 'decimation' in f.attrs and 'sim_dt' in f.attrs:
                data.dt = f.attrs['decimation'] * f.attrs['sim_dt']
            
            # Load all fields from the specified environment
            env_group = f[f'env_{env_id}/data']
            
            for field_name in env_group.keys():
                # Load only the actual logged timesteps (not pre-allocated space)
                data.fields[field_name] = env_group[field_name][:data.timesteps]
        
        return data
    
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict:
        """
        Get quick summary of file contents without loading all data.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'r') as f:
            info = {
                'filepath': str(filepath),
                'filename': filepath.name,
                'size_mb': filepath.stat().st_size / (1024 * 1024),
                'metadata': dict(f.attrs),
                'num_envs': f.attrs.get('num_envs', 1),
                'timesteps': f.attrs.get('actual_timesteps', 0),
                'control_mode': f.attrs.get('control_mode', 'unknown'),
                'fields': list(f['env_0/data'].keys()) if 'env_0/data' in f else [],
            }
        
        return info
    



class DataFilter:
    """
    Utilities for filtering and processing loaded data.
    """
    
    # @staticmethod
    # def get_acceleration(data: ExperimentData, acc_field: str) -> np.ndarray:
    #     acc = data.get_field(acc_field)
    #     return acc
    pass

