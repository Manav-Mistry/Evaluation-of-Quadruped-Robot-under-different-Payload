import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import VisualizationMarkersCfg
from collections import deque
from enum import Enum


class FollowerState(Enum):
    FOLLOWING = 1
    FINAL_APPROACH = 2
    COMPLETE = 3


class ProgressBasedWaypointFollower:
    def __init__(self, waypoints, kp=2.0, kd=0.3, 
                 threshold_normal=0.35, threshold_final=0.15,
                 velocity_threshold_final=0.1, lookahead_distance=0.5,
                 stuck_timeout=10.0):
        """
        Progress-based waypoint follower with smooth motion through waypoints.
        
        Args:
            waypoints: list of [x, y, yaw] in world frame
            kp: Proportional gain for position control
            kd: Derivative gain for position control
            threshold_normal: Distance threshold to advance to next waypoint (meters)
            threshold_final: Tighter threshold for final waypoint (meters)
            velocity_threshold_final: Max velocity to consider "stopped" at final waypoint (m/s)
            lookahead_distance: Distance to start blending toward next waypoint (meters)
            stuck_timeout: Max time at one waypoint before skipping (seconds)
        """
        self.waypoints = np.array(waypoints, dtype=np.float32)
        self.kp = kp
        self.kd = kd
        self.threshold_normal = threshold_normal
        self.threshold_final = threshold_final
        self.velocity_threshold_final = velocity_threshold_final
        self.lookahead_distance = lookahead_distance
        self.stuck_timeout = stuck_timeout
        
        # State tracking
        self.current_waypoint_index = 0
        self.state = FollowerState.FOLLOWING
        self.errors = deque(maxlen=2)
        
        # Stuck detection
        self.time_at_current_waypoint = 0.0
        self.last_error_magnitude = float('inf')
        
        # Separate marker objects for waypoints and current target
        self.waypoint_markers = None
        self.target_marker = None

    def setup_markers(self):
        """Setup visualization markers for drawing the path"""
        # Marker for waypoints (cyan spheres)
        waypoint_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/WaypointPath",
            markers={
                "waypoint_sphere": sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
            },
        )
        self.waypoint_markers = VisualizationMarkers(waypoint_marker_cfg)
        
        # Marker for current target (red sphere)
        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CurrentTarget",
            markers={
                "target_sphere": sim_utils.SphereCfg(
                    radius=0.15,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.target_marker = VisualizationMarkers(target_marker_cfg)
        
    def draw_path(self):
        """Draw the waypoint trajectory as spheres (call once or periodically)"""
        if self.waypoint_markers is None:
            print("Markers not initialized. Call setup_markers() first.")
            return
            
        # Create positions from waypoints
        positions = np.array([[wp[0], wp[1], 0.15] for wp in self.waypoints], dtype=np.float32)
        
        # All waypoints use index 0 (all same marker type)
        marker_indices = np.zeros(len(self.waypoints), dtype=np.int32)
        
        # Visualize the waypoints
        self.waypoint_markers.visualize(translations=positions, marker_indices=marker_indices)

    def draw_current_target(self, target_pos):
        """Draw the current target position (call every frame)"""
        if self.target_marker is None:
            return
        
        # Current target - single sphere at index 0
        position = np.array([[target_pos[0], target_pos[1], 0.15]], dtype=np.float32)
        marker_index = np.array([0], dtype=np.int32)
        
        self.target_marker.visualize(translations=position, marker_indices=marker_index)

    def get_current_target(self, current_pos):
        """
        Get the current target position with optional look-ahead blending.
        Returns blended target for smooth motion through waypoints.
        """
        if self.state == FollowerState.COMPLETE:
            return self.waypoints[-1]
        
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # If at last waypoint or far from current waypoint, no blending
        if self.current_waypoint_index >= len(self.waypoints) - 1:
            return current_waypoint
        
        # Calculate distance to current waypoint
        distance = np.linalg.norm(current_pos[:2] - current_waypoint[:2])
        
        # Start blending when within lookahead distance
        if distance < self.lookahead_distance:
            next_waypoint = self.waypoints[self.current_waypoint_index + 1]
            
            # Blend factor: 0 at lookahead_distance, 1 at threshold
            blend = np.clip((self.lookahead_distance - distance) / 
                          (self.lookahead_distance - self.threshold_normal), 0, 1)
            
            # Smooth blending using cosine interpolation for smoother curves
            blend = (1 - np.cos(blend * np.pi)) / 2
            
            blended = (1 - blend) * current_waypoint + blend * next_waypoint
            return blended
        
        return current_waypoint

    def check_advancement(self, current_pos, current_velocity, dt):
        """
        Check if we should advance to the next waypoint.
        Updates current_waypoint_index and state.
        """
        if self.state == FollowerState.COMPLETE:
            return
        
        self.time_at_current_waypoint += dt
        
        current_waypoint = self.waypoints[self.current_waypoint_index]
        distance = np.linalg.norm(current_pos[:2] - current_waypoint[:2])
        
        # Check for stuck condition
        if self.time_at_current_waypoint > self.stuck_timeout:
            if distance > self.last_error_magnitude * 0.95:  # Not making progress
                print(f"WARNING: Stuck at waypoint {self.current_waypoint_index}. Skipping.")
                self._advance_waypoint()
                return
        
        self.last_error_magnitude = distance
        
        # Normal waypoint advancement
        if self.state == FollowerState.FOLLOWING:
            if distance < self.threshold_normal:
                self._advance_waypoint()
                
        # Final waypoint requires tighter criteria
        elif self.state == FollowerState.FINAL_APPROACH:
            velocity_magnitude = np.linalg.norm(current_velocity[:2])
            if distance < self.threshold_final and velocity_magnitude < self.velocity_threshold_final:
                self.state = FollowerState.COMPLETE
                print("Reached final waypoint!")

    def _advance_waypoint(self):
        """Helper to advance to next waypoint"""
        self.current_waypoint_index += 1
        self.time_at_current_waypoint = 0.0
        self.last_error_magnitude = float('inf')
        
        if self.current_waypoint_index >= len(self.waypoints) - 1:
            self.current_waypoint_index = len(self.waypoints) - 1
            self.state = FollowerState.FINAL_APPROACH
            print(f"Approaching final waypoint...")
        else:
            print(f"Advanced to waypoint {self.current_waypoint_index}")

    def compute_desired_heading(self, current_pos, target_pos, current_yaw):
        """
        Compute desired heading from motion direction.
        For final waypoint, use specified yaw.
        """
        if self.state == FollowerState.FINAL_APPROACH or self.state == FollowerState.COMPLETE:
            # Use specified yaw at final waypoint
            return self.waypoints[-1][2]
        
        # Compute heading from direction to target
        direction = target_pos[:2] - current_pos[:2]
        distance = np.linalg.norm(direction)
        
        if distance < 0.01:  # Too close, maintain current heading
            return current_yaw
        
        desired_yaw = np.arctan2(direction[1], direction[0])
        return desired_yaw

    def get_command_PD(self, dt, current_pos, current_yaw, current_velocity):
        """
        Compute velocity command using PD control for smooth waypoint following.
        
        Args:
            dt: Time step (seconds)
            current_pos: Current position [x, y, z]
            current_yaw: Current yaw angle (radians)
            current_velocity: Current velocity [vx, vy, vz] in world frame
            
        Returns:
            error: Tuple of (error_x, error_y, yaw_error)
            command: np.array [vx_base, vy_base, yaw_rate] in base frame
        """
        # Check advancement
        self.check_advancement(current_pos, current_velocity, dt)
        
        if self.state == FollowerState.COMPLETE:
            # Stop commanding, we're done
            return (0, 0, 0), np.zeros(3, dtype=np.float32)
        
        # Get current target (with potential blending)
        target = self.get_current_target(current_pos)
        
        # Visualize current target (separate marker object)
        self.draw_current_target(target)
        
        # Position error in world frame
        error_x_world = target[0] - current_pos[0]
        error_y_world = target[1] - current_pos[1]
        
        # Compute desired heading
        desired_yaw = self.compute_desired_heading(current_pos, target, current_yaw)
        yaw_err = np.arctan2(np.sin(desired_yaw - current_yaw), 
                            np.cos(desired_yaw - current_yaw))
        
        error = (error_x_world, error_y_world, yaw_err)
        self.errors.append(error)
        
        # PD Control
        kp = self.kp
        kd = self.kd
        
        if len(self.errors) == 2:
            # Derivative term (change in error, no division by dt for stability)
            e_rate_x = self.errors[-1][0] - self.errors[-2][0]
            e_rate_y = self.errors[-1][1] - self.errors[-2][1]
            
            dx_world = kp * error_x_world + kd * e_rate_x
            dy_world = kp * error_y_world + kd * e_rate_y
        else:
            # First step: pure P control
            dx_world = kp * error_x_world
            dy_world = kp * error_y_world
        
        # Transform velocities from world frame to base frame
        dx_base = dx_world * np.cos(current_yaw) + dy_world * np.sin(current_yaw)
        dy_base = -dx_world * np.sin(current_yaw) + dy_world * np.cos(current_yaw)
        
        # Yaw rate control (proportional)
        yaw_gain = 1.0
        yaw_rate = yaw_gain * yaw_err
        yaw_rate = np.clip(yaw_rate, -1.0, 1.0)


        
        # Optional: velocity limiting for safety
        max_linear_vel = 2.0
        velocity_mag = np.sqrt(dx_base**2 + dy_base**2)
        if velocity_mag > max_linear_vel:
            scale = max_linear_vel / velocity_mag
            dx_base *= scale
            dy_base *= scale

        if yaw_err >= np.pi/7: # 25 degree
            return error, np.array([0.0, 0.0, yaw_rate], dtype=np.float32)
        
        return error, np.array([dx_base, dy_base, yaw_rate], dtype=np.float32)

    def is_complete(self):
        """Check if trajectory is complete"""
        return self.state == FollowerState.COMPLETE

    def reset(self):
        """Reset the follower to start from beginning"""
        self.current_waypoint_index = 0
        self.state = FollowerState.FOLLOWING
        self.errors.clear()
        self.time_at_current_waypoint = 0.0
        self.last_error_magnitude = float('inf')
        print("Waypoint follower reset to start.")