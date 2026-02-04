import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import VisualizationMarkersCfg
from collections import deque

class WaypointTrajectoryFollower:
    def __init__(self, waypoints, segment_time=2.0, kp=1.0, ki=0.0, kd=0.1):
        """
        waypoints: list of [x, y, yaw] in world frame
        segment_time: duration (s) to move between consecutive waypoints
        kp: proportional gain
        ki: integral gain
        kd: derivative gain
        """
        self.waypoints = np.array(waypoints, dtype=np.float32)
        self.segment_time = segment_time
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.errors = deque(maxlen=2)

        # total trajectory duration
        self.total_time = segment_time * (len(waypoints) - 1)

        self.markers = None
        # new
        self.waypoint_idx = 0

        # PID state variables
        self.prev_error = np.zeros(3)  # [error_x, error_y, yaw_err]
        self.integral = np.zeros(3)    # accumulated integral error

    
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
    

    def _find_closest_point_on_segment(self, p, a, b):
        """
        Find the closest point on line segment AB to point P.

        Returns:
            closest_point: [x, y] closest point on segment
            t: parameter in [0, 1] indicating position along segment (0=A, 1=B)
        """
        ap = p - a
        ab = b - a
        ab_squared = np.dot(ab, ab)

        if ab_squared < 1e-8:  # A and B are essentially the same point
            return a.copy(), 0.0

        t = np.dot(ap, ab) / ab_squared
        t = np.clip(t, 0.0, 1.0)

        closest = a + t * ab
        return closest, t

    def _find_look_ahead_point(self, current_pos, look_ahead_dist):
        """
        Find the look-ahead point on the path at a specified distance ahead.

        Args:
            current_pos: [x, y] current robot position
            look_ahead_dist: distance to look ahead on the path
        """
        pos = np.array(current_pos[:2])

        # Check if we've completed all waypoints
        if self.waypoint_idx >= len(self.waypoints) - 1:
            return self.waypoints[-1].copy(), True

        # Only search from current segment forward (not the entire path!)
        # This prevents "skipping" when the path loops back on itself
        min_dist = float('inf')
        closest_segment_idx = self.waypoint_idx
        closest_t = 0.0

        # Search only current segment and next few (limit lookahead to avoid skipping)
        search_limit = min(self.waypoint_idx + 3, len(self.waypoints) - 1)

        for i in range(self.waypoint_idx, search_limit):
            a = self.waypoints[i][:2]
            b = self.waypoints[i + 1][:2]
            closest, t = self._find_closest_point_on_segment(pos, a, b)
            dist = np.linalg.norm(pos - closest)

            if dist < min_dist:
                min_dist = dist
                closest_segment_idx = i
                closest_t = t

        # Update waypoint_idx if we've moved to a new segment
        # Only advance forward, never backward
        if closest_segment_idx > self.waypoint_idx:
            self.waypoint_idx = closest_segment_idx

        # Also check if we've passed the current waypoint (for segment advancement)
        # This handles the case where we're past the end of current segment
        if closest_t > 0.95 and self.waypoint_idx < len(self.waypoints) - 2:
            # Check distance to next waypoint
            next_wp = self.waypoints[self.waypoint_idx + 1][:2]
            if np.linalg.norm(pos - next_wp) < look_ahead_dist:
                self.waypoint_idx += 1
                closest_segment_idx = self.waypoint_idx
                closest_t = 0.0

        # Now find the look-ahead point starting from our closest point
        # and moving forward along the path by look_ahead_dist
        remaining_dist = look_ahead_dist
        current_seg = closest_segment_idx

        # Distance remaining in current segment from closest point
        a = self.waypoints[current_seg][:2]
        b = self.waypoints[current_seg + 1][:2]
        segment_remaining = (1.0 - closest_t) * np.linalg.norm(b - a)

        while remaining_dist > segment_remaining:
            remaining_dist -= segment_remaining
            current_seg += 1

            # Check if we've reached the end of the path
            if current_seg >= len(self.waypoints) - 1:
                # Return the final waypoint
                return self.waypoints[-1].copy(), True

            a = self.waypoints[current_seg][:2]
            b = self.waypoints[current_seg + 1][:2]
            segment_remaining = np.linalg.norm(b - a)

        # Interpolate within the final segment
        a = self.waypoints[current_seg][:2]
        b = self.waypoints[current_seg + 1][:2]
        seg_length = np.linalg.norm(b - a)

        if seg_length < 1e-8:
            look_ahead_xy = a
            t_final = 0.0
        else:
            # How far along this segment?
            if current_seg == closest_segment_idx:
                # Still in same segment, account for starting position
                dist_from_a = closest_t * seg_length + remaining_dist
            else:
                # New segment, start from beginning
                dist_from_a = remaining_dist

            t_final = dist_from_a / seg_length
            t_final = np.clip(t_final, 0.0, 1.0)
            look_ahead_xy = a + t_final * (b - a)

        # Interpolate yaw between waypoints
        yaw_a = self.waypoints[current_seg][2]
        yaw_b = self.waypoints[current_seg + 1][2]
        # Handle yaw wrap-around for interpolation
        yaw_diff = np.arctan2(np.sin(yaw_b - yaw_a), np.cos(yaw_b - yaw_a))
        look_ahead_yaw = yaw_a + t_final * yaw_diff

        look_ahead_point = np.array([look_ahead_xy[0], look_ahead_xy[1], look_ahead_yaw], dtype=np.float32)

        return look_ahead_point, False

    def get_command_pure_pursuit(self, current_pos, current_yaw, look_ahead_dist=0.5,
                                   target_speed=1.0, yaw_threshold=0.785, kp_yaw=2.0):
        """
        Pure pursuit controller with in-place rotation for sharp turns.

        Instead of targeting discrete waypoints, this controller targets a point
        that is always 'look_ahead_dist' ahead on the path, creating smooth
        continuous motion without velocity jumps at waypoints.

        When yaw error exceeds yaw_threshold, the robot stops linear motion
        and rotates in place until aligned.

        Args:
            current_pos: [x, y, z] current robot position
            current_yaw: current robot yaw angle in radians
            look_ahead_dist: distance to look ahead on the path (meters)
            target_speed: desired linear speed (m/s)
            yaw_threshold: if |yaw_error| > this, stop and rotate in place (default 45° = 0.785 rad)
            kp_yaw: proportional gain for yaw control
        """
        # Find the look-ahead point on the path
        look_ahead_point, path_complete = self._find_look_ahead_point(current_pos, look_ahead_dist)

        if path_complete:
            # Near end of path - switch to position control for final approach
            final_target = self.waypoints[-1]
            error_x = final_target[0] - current_pos[0]
            error_y = final_target[1] - current_pos[1]
            dist_to_goal = np.sqrt(error_x**2 + error_y**2)

            if dist_to_goal < 0.1:  # Close enough to final goal
                return (0.0, 0.0, 0.0), np.zeros(3, dtype=np.float32)

            # Yaw control
            target_yaw = final_target[2]
            yaw_err = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
            yaw_rate = kp_yaw * yaw_err
            yaw_rate = np.clip(yaw_rate, -1.5, 1.5)

            # In-place rotation at final approach if needed
            # if abs(yaw_err) > yaw_threshold:
            #     error = (error_x, error_y, yaw_err)
            #     command = np.array([0.0, 0.0, yaw_rate], dtype=np.float32)
            #     return error, command

            # Slow down as we approach final goal
            speed = min(target_speed, self.kp * dist_to_goal)

            # Direction to goal
            direction = np.array([error_x, error_y]) / dist_to_goal
            vx_world = speed * direction[0]
            vy_world = speed * direction[1]

            # Transform to base frame
            vx_base = vx_world * np.cos(current_yaw) + vy_world * np.sin(current_yaw)
            vy_base = -vx_world * np.sin(current_yaw) + vy_world * np.cos(current_yaw)

            # Clip to robot velocity limits
            vx_base = np.clip(vx_base, -1.0, 1.6)
            vy_base = np.clip(vy_base, -1.0, 1.6)

            error = (error_x, error_y, yaw_err)
            command = np.array([vx_base, vy_base, yaw_rate], dtype=np.float32)
            return error, command

        # Compute error to look-ahead point
        target_x, target_y, target_yaw = look_ahead_point
        error_x = target_x - current_pos[0]
        error_y = target_y - current_pos[1]
        dist_to_look_ahead = np.sqrt(error_x**2 + error_y**2)

        # Compute yaw error
        yaw_err = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
        yaw_rate = kp_yaw * yaw_err
        yaw_rate = np.clip(yaw_rate, -1.5, 1.5)

        # IN-PLACE ROTATION: If yaw error is large, stop and rotate
        if abs(yaw_err) > yaw_threshold:
            error = (error_x, error_y, yaw_err)
            command = np.array([0.0, 0.0, yaw_rate], dtype=np.float32)
            return error, command

        # Normal operation: compute linear velocity
        if dist_to_look_ahead > 1e-6:
            direction = np.array([error_x, error_y]) / dist_to_look_ahead
            vx_world = target_speed * direction[0]
            vy_world = target_speed * direction[1]
        else:
            vx_world = 0.0
            vy_world = 0.0

        # Transform velocity from world frame to base frame
        vx_base = vx_world * np.cos(current_yaw) + vy_world * np.sin(current_yaw)
        vy_base = -vx_world * np.sin(current_yaw) + vy_world * np.cos(current_yaw)

        # Clip to robot velocity limits
        vx_base = np.clip(vx_base, -1.0, 1.6)
        vy_base = np.clip(vy_base, -1.0, 1.6)

        error = (error_x, error_y, yaw_err)
        command = np.array([vx_base, vy_base, yaw_rate], dtype=np.float32)

        return error, command

    def is_trajectory_complete(self, current_pos=None, threshold=0.15):
        """
        Check if all waypoints have been visited.

        Args:
            current_pos: [x, y] current robot position (optional, for distance check)
            threshold: distance threshold to final waypoint (default 0.15m)

        Returns:
            True if trajectory is complete
        """
        # waypoint_idx is the segment index (0 to n-2 for n waypoints)
        # When waypoint_idx >= n-1, we're on or past the last segment
        if self.waypoint_idx < len(self.waypoints) - 1:
            return False

        # If no position provided, just check segment index
        if current_pos is None:
            return True

        # Check if robot is close enough to final waypoint
        final_wp = self.waypoints[-1][:2]
        pos = np.array(current_pos[:2])
        dist = np.linalg.norm(pos - final_wp)
        return dist < threshold

    def reset_waypoint_index(self):
        """Reset waypoint index and PID state to start trajectory from beginning."""
        self.waypoint_idx = 0
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)

    def get_current_waypoint_idx(self):
        """Get current waypoint index."""
        return self.waypoint_idx


    
# OLD Code that uses expected waypoint logic according to time to find position error 
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
    
