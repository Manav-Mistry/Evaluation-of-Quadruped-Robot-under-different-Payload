# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

import torch
from pxr import Gf, Sdf
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from isaaclab.utils.math import quat_apply

class ThirdPersonCamera:
    """Manages a third-person camera that follows a robot."""
    
    def __init__(self, camera_path="/World/Camera", viewport_name="Viewport"):
        """
        Initialize third-person camera.
        
        Args:
            camera_path: USD path for the camera prim
            viewport_name: Name of the viewport window
        """
        self.camera_path = camera_path
        self.perspective_path = "/OmniverseKit_Persp"
        self.viewport = get_viewport_from_window_name(viewport_name)
        self._camera_local_transform = None
        
        self._create_camera()
    
    def _create_camera(self):
        """Creates the camera prim in the USD stage."""
        stage = get_current_stage()
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        
        # Set center of interest
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", 
                Sdf.ValueTypeNames.Vector3d, 
                True, 
                Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        
        # Start with perspective view
        self.viewport.set_active_camera(self.perspective_path)
    
    def set_local_transform(self, local_transform):
        """
        Set the camera's local transform relative to the robot.
        
        Args:
            local_transform: torch.Tensor of [x, y, z] offset in robot frame
        """
        self._camera_local_transform = local_transform
    
    def activate(self):
        """Switch viewport to use this camera."""
        self.viewport.set_active_camera(self.camera_path)
    
    def update(self, robot_pos, robot_quat, target_offset_z=0.6):
        """
        Update camera position to follow the robot.
        
        Args:
            robot_pos: Robot base position tensor [x, y, z]
            robot_quat: Robot base quaternion [w, x, y, z]
            target_offset_z: Vertical offset for camera target point
        """
        if self._camera_local_transform is None:
            return
        
        # Calculate camera position in world frame
        camera_pos = quat_apply(robot_quat, self._camera_local_transform) + robot_pos
        
        # Set camera state
        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(
            camera_pos[0].item(), 
            camera_pos[1].item(), 
            camera_pos[2].item()
        )
        target = Gf.Vec3d(
            robot_pos[0].item(), 
            robot_pos[1].item(), 
            robot_pos[2].item() + target_offset_z
        )
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)