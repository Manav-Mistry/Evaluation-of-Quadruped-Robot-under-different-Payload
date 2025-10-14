import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import VisualizationMarkersCfg
from collections import deque


class WaypointTrajectoryFollower:
    def __init__(self, waypoints, segment_time=2.0, kp=3, kd=1):
        """
        waypoints: list of [x, y, yaw] in world frame
        segment_time: duration (s) to move between consecutive waypoints
        """
        self.waypoints = np.array(waypoints, dtype=np.float32)
        self.segment_time = segment_time
        self.kp = kp
        self.kd = kd
        self.errors = deque(maxlen=2)

        # total trajectory duration
        self.total_time = segment_time * (len(waypoints) - 1)

        self.markers = None

    
    def setup_markers(self):
        """Setup visualization markers for drawing the path"""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/WaypointPath",
            markers={
                "waypoint_sphere": sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),  # Cyan
                ),
            },
        )
        self.markers = VisualizationMarkers(marker_cfg)
        
    def draw_path(self):
        """Draw the waypoint trajectory as spheres"""
        if self.markers is None:
            print("Markers not initialized. Call setup_markers() first.")
            return
            
        # Create positions from waypoints (x, y, z=0.15 to keep above ground)
        positions = np.array([[wp[0], wp[1], 0.15] for wp in self.waypoints], dtype=np.float32)
        
        # Create marker indices (all use the same marker type - index 0)
        marker_indices = np.zeros(len(self.waypoints), dtype=np.int32)
        
        # Visualize the waypoints
        self.markers.visualize(translations=positions, marker_indices=marker_indices)



    def get_reference(self, t):
        """
        Return desired [x, y, yaw] at time t using linear interpolation.
        """
        if hasattr(t, "item"):
            t = t.item()
        

        if t >= self.total_time:
            return self.waypoints[-1]

        # which segment are we in?
        seg_idx = int(t // self.segment_time)
        tau = (t % self.segment_time) / self.segment_time  # normalized [0,1]

        p0 = self.waypoints[seg_idx]
        p1 = self.waypoints[seg_idx + 1]

        # simple linear interpolation
        interp = (1 - tau) * p0 + tau * p1
        return interp
    

    def get_command(self, t, dt, current_yaw):
        """
        Compute velocity command [vx, vy, yaw_rate] at time t
        from finite differences of reference trajectory.
        """
        
        if t >= self.total_time:
            return (None, None), np.zeros(3, dtype=np.float32)

        ref_now = self.get_reference(t)
        ref_next = self.get_reference(t + dt)

        error_x = (ref_next[0] - ref_now[0])
        error_y = (ref_next[1] - ref_now[1])
        
        error = (error_x, error_y)


        dx_world = (ref_next[0] - ref_now[0]) / dt
        dy_world = (ref_next[1] - ref_now[1]) / dt
        
        # Transform from world frame to base frame
        dx_base = dx_world * np.cos(current_yaw) + dy_world * np.sin(current_yaw)
        dy_base = -dx_world * np.sin(current_yaw) + dy_world * np.cos(current_yaw)

        # handle yaw properly (wrap-around)
        yaw_now, yaw_next = ref_now[2], ref_next[2]
        yaw_err = np.arctan2(np.sin(yaw_next - yaw_now), np.cos(yaw_next - yaw_now))
        yaw_rate = yaw_err / dt

        return error, np.array([dx_base, dy_base, yaw_rate], dtype=np.float32)
    

    def get_command_with_feedback(self, t, dt, current_pos, current_yaw):
        # Get CURRENT waypoint goal (not next timestep reference)
        ref_now = self.get_reference(t)
        
        # Error to the GOAL (not to next interpolation point)
        error_x_world = ref_now[0] - current_pos[0]
        error_y_world = ref_now[1] - current_pos[1]
        
        # Use proportional control
        kp = self.kp  # Tune this
        dx_world = kp * error_x_world
        dy_world = kp * error_y_world
        
        # Clip to training limits
        dx_world = np.clip(dx_world, -2.0, 3.0)
        dy_world = np.clip(dy_world, -1.5, 1.5)
        
        # Transform to base frame
        dx_base = dx_world * np.cos(current_yaw) + dy_world * np.sin(current_yaw)
        dy_base = -dx_world * np.sin(current_yaw) + dy_world * np.cos(current_yaw)
        
        # Yaw control (same as before)
        yaw_err = np.arctan2(np.sin(ref_now[2] - current_yaw), 
                            np.cos(ref_now[2] - current_yaw))
        yaw_rate = 2.0 * yaw_err  # Proportional yaw control
        yaw_rate = np.clip(yaw_rate, -2.0, 2.0)
        
        error = (error_x_world, error_y_world, yaw_err)
        
        return error, np.array([dx_base, dy_base, yaw_rate], dtype=np.float32)
    

    def get_command_with_feedback_PD(self, t, dt, current_pos, current_yaw):
        # Get CURRENT waypoint goal (not next timestep reference)
        ref_now = self.get_reference(t)
        
        # Error to the GOAL (not to next interpolation point)
        error_x_world = ref_now[0] - current_pos[0]
        error_y_world = ref_now[1] - current_pos[1]
        # Yaw control (same as before)
        yaw_err = np.arctan2(np.sin(ref_now[2] - current_yaw), 
                            np.cos(ref_now[2] - current_yaw))
        # yaw_rate = np.clip(yaw_rate, -2.0, 2.0)
        yaw_rate = 2.0 * yaw_err
        
        error = (error_x_world, error_y_world, yaw_err)
        self.errors.append(error)

        # Use proportional derivative control
        kp = self.kp  # Tune this   
        kd = self.kd
        
        if len(self.errors) == 2:
            e_rate_x = (self.errors[1][0] - self.errors[0][0]) / 1
            e_rate_y = (self.errors[1][1] - self.errors[0][1]) / 1
            e_rate_yaw = (self.errors[1][2] - self.errors[0][2]) / 1

            dx_world = kp * error_x_world + kd * e_rate_x
            dy_world = kp * error_y_world + kd * e_rate_y
            # yaw_rate = 2.0 * yaw_err + kd * e_rate_yaw  # Proportional derivative yaw control
        else:
            dx_world = kp * error_x_world
            dy_world = kp * error_y_world

        
        
        # Clip to training limits
        # dx_world = np.clip(dx_world, -2.0, 3.0)
        # dy_world = np.clip(dy_world, -1.5, 1.5)
        
        # Transform to base frame
        dx_base = dx_world * np.cos(current_yaw) + dy_world * np.sin(current_yaw)
        dy_base = -dx_world * np.sin(current_yaw) + dy_world * np.cos(current_yaw)
        
        
        return error, np.array([dx_base, dy_base, yaw_rate], dtype=np.float32)
    
