"""
Waypoint Library - Collection of different waypoint trajectories for robot navigation
"""

import numpy as np


def create_basic_square():
    """
    Basic square path (8 waypoints).
    Robot moves in a 2x2 meter square with rotations at corners.
    """
    return [
        [0, 0, 0],
        [2, 0, 0],
        [2, 0, np.pi/2],
        [2, 2, np.pi/2],
        [2, 2, np.pi],
        [0, 2, np.pi],
        [0, 2, 3*np.pi/2],
        [0, 0, 3*np.pi/2]
    ]


def create_square_with_midpoints():
    """
    Square path with midpoints on each side (16 waypoints).
    More controlled motion with waypoints at middle of each side.
    """
    return [
        # Start at origin
        [0, 0, 0],
        
        # First side: move right (0,0) → (2,0)
        [1, 0, 0],              # Midpoint
        [2, 0, 0],              # Corner
        [2, 0, np.pi/2],        # Turn to face up
        
        # Second side: move up (2,0) → (2,2)
        [2, 1, np.pi/2],        # Midpoint
        [2, 2, np.pi/2],        # Corner
        [2, 2, np.pi],          # Turn to face left
        
        # Third side: move left (2,2) → (0,2)
        [1, 2, np.pi],          # Midpoint
        [0, 2, np.pi],          # Corner
        [0, 2, 3*np.pi/2],      # Turn to face down
        
        # Fourth side: move down (0,2) → (0,0)
        [0, 1, 3*np.pi/2],      # Midpoint
        [0, 0, 3*np.pi/2],      # Back to origin
        [0, 0, 0],              # Turn to face original direction
        
        # Extra waypoints for more detail
        [0.5, 0, 0],            # Quarter point on first side
        [1.5, 0, 0],            # Three-quarter point on first side
        [2, 0, 0]               # Back to first corner
    ]


def create_square_with_intermediate_rotations():
    """
    Square with extra rotation waypoints at each corner (16 waypoints).
    Gradual rotation at corners for smoother turning.
    """
    return [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [2, 0, np.pi/4],        # Intermediate rotation (45°)
        [2, 0, np.pi/2],        # Full rotation (90°)
        [2, 1, np.pi/2],
        [2, 2, np.pi/2],
        [2, 2, 3*np.pi/4],      # Intermediate rotation (135°)
        [2, 2, np.pi],          # Full rotation (180°)
        [1, 2, np.pi],
        [0, 2, np.pi],
        [0, 2, 5*np.pi/4],      # Intermediate rotation (225°)
        [0, 2, 3*np.pi/2],      # Full rotation (270°)
        [0, 1, 3*np.pi/2],
        [0, 0, 3*np.pi/2],
        [0, 0, 0]               # Back to start orientation
    ]


def create_fine_grid_square():
    """
    Square with fine position resolution (16 waypoints).
    More waypoints along each side for tighter control.
    """
    return [
        [0, 0, 0],
        [0.5, 0, 0],
        [1, 0, 0],
        [1.5, 0, 0],
        [2, 0, 0],
        [2, 0, np.pi/2],
        [2, 0.5, np.pi/2],
        [2, 1, np.pi/2],
        [2, 1.5, np.pi/2],
        [2, 2, np.pi/2],
        [2, 2, np.pi],
        [1.5, 2, np.pi],
        [1, 2, np.pi],
        [0.5, 2, np.pi],
        [0, 2, np.pi],
        [0, 2, 3*np.pi/2]
    ]


def create_figure_eight():
    """
    Figure-8 pattern (16 waypoints).
    Robot traces a figure-8 shape with upper and lower loops.
    """
    return [
        # Upper loop
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, np.pi/2],
        [2, 1, np.pi/2],
        [2, 2, np.pi],
        [1, 2, np.pi],
        [0, 2, 3*np.pi/2],
        [0, 1, 3*np.pi/2],
        # Back to center, then lower loop
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, -np.pi/2],       # Go down instead
        [2, -1, -np.pi/2],
        [2, -2, np.pi],
        [1, -2, np.pi],
        [0, -2, np.pi/2],
        [0, 0, np.pi/2]
    ]


def create_l_shape():
    """
    L-shaped path (6 waypoints).
    Simple L trajectory for testing.
    """
    return [
        [0, 0, 0],
        [2, 0, 0],
        [2, 0, np.pi/2],
        [2, 2, np.pi/2],
        [2, 2, np.pi],
        [2, 2, 0]               # Final orientation facing right
    ]


def create_zigzag():
    """
    Zigzag pattern (12 waypoints).
    Robot moves in a zigzag pattern with sharp turns.
    """
    return [
        [0, 0, 0],
        [1, 0, np.pi/4],        # 45° up-right
        [2, 1, np.pi/4],
        [2, 1, -np.pi/4],       # 45° down-right
        [3, 0, -np.pi/4],
        [3, 0, np.pi/4],        # 45° up-right
        [4, 1, np.pi/4],
        [4, 1, -np.pi/4],       # 45° down-right
        [5, 0, -np.pi/4],
        [5, 0, 0],              # Face forward
        [6, 0, 0],
        [6, 0, np.pi/2]         # Final turn
    ]


def create_circle_approximation(radius=2.0, num_points=16):
    """
    Circular path approximated by waypoints.
    
    Args:
        radius: Circle radius in meters
        num_points: Number of waypoints around the circle
    
    Returns:
        List of waypoints forming a circle
    """
    waypoints = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # Heading is tangent to circle (perpendicular to radius)
        heading = angle + np.pi/2
        waypoints.append([x, y, heading])
    return waypoints


def create_spiral(num_loops=2, points_per_loop=8, max_radius=2.0):
    """
    Spiral pattern (expanding or contracting).
    
    Args:
        num_loops: Number of complete loops
        points_per_loop: Waypoints per loop
        max_radius: Maximum radius of spiral
    
    Returns:
        List of waypoints forming a spiral
    """
    waypoints = []
    total_points = num_loops * points_per_loop
    
    for i in range(total_points):
        angle = 2 * np.pi * i / points_per_loop
        # Radius increases linearly
        radius = max_radius * (i / total_points)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        heading = angle + np.pi/2
        waypoints.append([x, y, heading])
    
    return waypoints


def create_slalom(num_gates=4, gate_spacing=2.0, gate_width=1.0):
    """
    Slalom course (weaving between gates).
    
    Args:
        num_gates: Number of gates to weave through
        gate_spacing: Distance between gates
        gate_width: Width to move left/right
    
    Returns:
        List of waypoints for slalom course
    """
    waypoints = [[0, 0, 0]]  # Start
    
    for i in range(num_gates):
        x = gate_spacing * (i + 1)
        # Alternate left and right
        y = gate_width if i % 2 == 0 else -gate_width
        heading = np.pi/6 if i % 2 == 0 else -np.pi/6  # Slight angle
        waypoints.append([x, y, heading])
    
    # Final straight section
    waypoints.append([gate_spacing * (num_gates + 1), 0, 0])
    
    return waypoints


def create_parking_maneuver():
    """
    Parallel parking maneuver (10 waypoints).
    Simulates a parallel parking trajectory.
    """
    return [
        [0, 0, 0],              # Start position
        [2, 0, 0],              # Drive forward
        [2, 0, -np.pi/4],       # Start backing, turn right
        [1.5, -0.5, -np.pi/4],  # Back and right
        [1, -0.5, -np.pi/2],    # Continue backing
        [0.5, -0.5, -np.pi/2],  # More backing
        [0.5, -0.5, np.pi],     # Turn to face backward
        [0.3, -0.5, np.pi],     # Final position
        [0.3, -0.5, 3*np.pi/2], # Turn to face left
        [0.3, -0.5, 0]          # Face forward (parked)
    ]


# Example usage
if __name__ == "__main__":
    # Test all waypoint generators
    print("Available waypoint patterns:")
    print("1. Basic Square (8 points)")
    print("2. Square with Midpoints (16 points)")
    print("3. Square with Intermediate Rotations (16 points)")
    print("4. Fine Grid Square (16 points)")
    print("5. Figure-8 (16 points)")
    print("6. L-Shape (6 points)")
    print("7. Zigzag (12 points)")
    print("8. Circle Approximation (configurable)")
    print("9. Spiral (configurable)")
    print("10. Slalom (configurable)")
    print("11. Parking Maneuver (10 points)")
    
    # Example: Create and print basic square
    waypoints = create_basic_square()
    print(f"\nBasic Square waypoints ({len(waypoints)} points):")
    for i, wp in enumerate(waypoints):
        print(f"  {i}: [{wp[0]:.2f}, {wp[1]:.2f}, {np.degrees(wp[2]):.1f}°]")