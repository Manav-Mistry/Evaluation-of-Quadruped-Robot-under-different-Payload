
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

# """Keyboard control interface for robot teleoperation."""

# import torch
# import carb
# import omni


# class KeyboardController:
#     """Handles keyboard input for manual robot control."""
    
#     def __init__(self, num_envs=1, device="cuda"):
#         """
#         Initialize keyboard controller.
        
#         Args:
#             num_envs: Number of environments
#             device: Device for tensor operations
#         """
#         self.num_envs = num_envs
#         self.device = device
#         self.commands = torch.zeros(num_envs, 3, device=device)
#         self._selected_id = 0
        
#         # Setup keyboard interface
#         self._setup_keyboard()
        
#     def _setup_keyboard(self):
#         """Setup keyboard input interface and register key mappings."""
#         self._input = carb.input.acquire_input_interface()
#         self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
#         self._sub_keyboard = self._input.subscribe_to_keyboard_events(
#             self._keyboard, 
#             self._on_keyboard_event
#         )
        
#         # Define control mappings [vx, vy, yaw_rate]
#         self._key_to_control = {
#             "UP": torch.tensor([2.0, 0.0, 0.0], device=self.device),      # Forward
#             "DOWN": torch.tensor([-2.0, 0.0, 0.0], device=self.device),   # Backward
#             "LEFT": torch.tensor([0.0, 2.0, 0.0], device=self.device),    # Strafe left
#             "RIGHT": torch.tensor([0.0, -2.0, 0.0], device=self.device),  # Strafe right
#             "N": torch.tensor([0.0, 0.0, 2.0], device=self.device),       # Turn left
#             "M": torch.tensor([0.0, 0.0, -2.0], device=self.device),      # Turn right
#             "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device)    # Stop
#         }
    
#     def _on_keyboard_event(self, event):
#         """
#         Handle keyboard events and update commands.
        
#         Args:
#             event: Keyboard event from carb.input
#         """
#         if event.type == carb.input.KeyboardEventType.KEY_PRESS:
#             # Apply command based on key pressed
#             if event.input.name in self._key_to_control:
#                 self.commands[self._selected_id] = self._key_to_control[event.input.name]
#             # Additional key: SPACE for emergency stop
#             elif event.input.name == "SPACE":
#                 self.commands[self._selected_id] = self._key_to_control["ZEROS"]
                
#         elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
#             # Stop on key release
#             if event.input.name in ["UP", "DOWN", "LEFT", "RIGHT", "N", "M"]:
#                 self.commands[self._selected_id] = self._key_to_control["ZEROS"]
    
#     def get_command(self):
#         """
#         Get current command for selected robot.
        
#         Returns:
#             Tensor of shape (3,) with [vx, vy, yaw_rate]
#         """
#         return self.commands[self._selected_id]
    
#     def get_all_commands(self):
#         """
#         Get commands for all robots.
        
#         Returns:
#             Tensor of shape (num_envs, 3)
#         """
#         return self.commands
    
#     def set_selected_id(self, robot_id):
#         """
#         Set which robot to control.
        
#         Args:
#             robot_id: Index of robot to control
#         """
#         if 0 <= robot_id < self.num_envs:
#             self._selected_id = robot_id
    
#     def reset_commands(self):
#         """Reset all commands to zero."""
#         self.commands.zero_()
    
#     def cleanup(self):
#         """Cleanup keyboard subscription."""
#         if hasattr(self, '_sub_keyboard'):
#             self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub_keyboard)
    
#     def print_controls(self):
#         """Print available keyboard controls."""
#         print("\n" + "="*60)
#         print("KEYBOARD CONTROLS")
#         print("="*60)
#         print("  UP ARROW    : Move forward")
#         print("  DOWN ARROW  : Move backward")
#         print("  LEFT ARROW  : Strafe left")
#         print("  RIGHT ARROW : Strafe right")
#         print("  N           : Turn left")
#         print("  M           : Turn right")
#         print("  SPACE       : Emergency stop")
#         print("="*60 + "\n")


# class CameraController:
#     """Handles camera view switching for keyboard control demo."""
    
#     def __init__(self, camera_path, perspective_path, viewport):
#         """
#         Initialize camera controller.
        
#         Args:
#             camera_path: USD path to third-person camera
#             perspective_path: USD path to perspective camera
#             viewport: Viewport object
#         """
#         self.camera_path = camera_path
#         self.perspective_path = perspective_path
#         self.viewport = viewport
#         self._current_camera = perspective_path
        
#         # Setup keyboard interface for camera switching
#         self._setup_keyboard()
    
#     def _setup_keyboard(self):
#         """Setup keyboard interface for camera control."""
#         self._input = carb.input.acquire_input_interface()
#         self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
#         self._sub_keyboard = self._input.subscribe_to_keyboard_events(
#             self._keyboard,
#             self._on_keyboard_event
#         )
    
#     def _on_keyboard_event(self, event):
#         """Handle camera switching events."""
#         if event.type == carb.input.KeyboardEventType.KEY_PRESS:
#             # C key switches between cameras
#             if event.input.name == "C":
#                 self.toggle_camera()
#             # ESC exits third-person view
#             elif event.input.name == "ESCAPE":
#                 self.set_perspective_view()
    
#     def toggle_camera(self):
#         """Toggle between third-person and perspective cameras."""
#         if self._current_camera == self.camera_path:
#             self.set_perspective_view()
#         else:
#             self.set_third_person_view()
    
#     def set_third_person_view(self):
#         """Switch to third-person camera."""
#         self.viewport.set_active_camera(self.camera_path)
#         self._current_camera = self.camera_path
    
#     def set_perspective_view(self):
#         """Switch to perspective camera."""
#         self.viewport.set_active_camera(self.perspective_path)
#         self._current_camera = self.perspective_path
    
#     def cleanup(self):
#         """Cleanup keyboard subscription."""
#         if hasattr(self, '_sub_keyboard'):
#             self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub_keyboard)
    
#     def print_controls(self):
#         """Print camera control instructions."""
#         print("\n" + "="*60)
#         print("CAMERA CONTROLS")
#         print("="*60)
#         print("  C      : Toggle between third-person and perspective view")
#         print("  ESCAPE : Return to perspective view")
#         print("="*60 + "\n")

"""Keyboard control interface for robot teleoperation."""

import torch
import carb
import omni


class KeyboardController:
    """Handles keyboard input for manual robot control AND camera switching."""
    
    def __init__(self, num_envs=1, device="cuda", camera_path=None, perspective_path=None, viewport=None):
        """
        Initialize keyboard controller.
        
        Args:
            num_envs: Number of environments
            device: Device for tensor operations
            camera_path: USD path to third-person camera (optional)
            perspective_path: USD path to perspective camera (optional)
            viewport: Viewport object (optional)
        """
        self.num_envs = num_envs
        self.device = device
        self.commands = torch.zeros(num_envs, 3, device=device)
        self._selected_id = 0
        
        # Camera properties
        self.camera_path = camera_path
        self.perspective_path = perspective_path
        self.viewport = viewport
        self._current_camera = perspective_path if perspective_path else None
        
        # Setup keyboard interface
        self._setup_keyboard()
        
    def _setup_keyboard(self):
        """Setup keyboard input interface and register key mappings."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, 
            self._on_keyboard_event
        )
        
        # Define control mappings [vx, vy, yaw_rate]
        self._key_to_control = {
            "UP": torch.tensor([2.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([-2.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([0.0, 2.0, 0.0], device=self.device),
            "RIGHT": torch.tensor([0.0, -2.0, 0.0], device=self.device),
            "N": torch.tensor([0.0, 0.0, 2.0], device=self.device),
            "M": torch.tensor([0.0, 0.0, -2.0], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device)
        }
    
    def _on_keyboard_event(self, event):
        """Handle keyboard events for both robot control and camera."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Robot control keys
            if event.input.name in self._key_to_control:
                self.commands[self._selected_id] = self._key_to_control[event.input.name]
            # Emergency stop
            elif event.input.name == "SPACE":
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]
            # Camera toggle
            elif event.input.name == "C":
                self.toggle_camera()
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Stop on key release
            if event.input.name in ["UP", "DOWN", "LEFT", "RIGHT", "N", "M"]:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]
    
    def toggle_camera(self):
        """Toggle between third-person and perspective cameras."""
        if self.viewport is None or self.camera_path is None:
            return
            
        if self.viewport.get_active_camera() == self.camera_path:
            self.viewport.set_active_camera(self.perspective_path)
            self._current_camera = self.perspective_path
            print("Switched to PERSPECTIVE view")
        else:
            self.viewport.set_active_camera(self.camera_path)
            self._current_camera = self.camera_path
            print("Switched to THIRD-PERSON view")
    
    def get_command(self):
        """Get current command for selected robot."""
        return self.commands[self._selected_id]
    
    def get_all_commands(self):
        """Get commands for all robots."""
        return self.commands
    
    def reset_commands(self):
        """Reset all commands to zero."""
        self.commands.zero_()
    
    def cleanup(self):
        """Cleanup keyboard subscription."""
        if hasattr(self, '_sub_keyboard'):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub_keyboard)
    
    def print_controls(self):
        """Print available keyboard controls."""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS")
        print("="*60)
        print("  UP ARROW    : Move forward")
        print("  DOWN ARROW  : Move backward")
        print("  LEFT ARROW  : Strafe left")
        print("  RIGHT ARROW : Strafe right")
        print("  N           : Turn left")
        print("  M           : Turn right")
        print("  SPACE       : Emergency stop")
        print("  C           : Toggle camera view")
        print("="*60 + "\n")