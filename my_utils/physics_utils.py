# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

from pxr import Gf, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage

def attach_payload_to_robot(robot_body_path, payload_path, local_offset=(0.0, 0.0, 0.14343)):
    """
    Attach a payload object to the robot using a fixed joint.
    
    Args:
        robot_body_path: USD path to the robot's body link
        payload_path: USD path to the payload object
        local_offset: Local position offset as (x, y, z) tuple
    """
    stage = get_current_stage()
    
    # Create fixed joint
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/FixedJoint")
    
    # Set body relationships
    fixed_joint.CreateBody0Rel().SetTargets([robot_body_path])
    fixed_joint.CreateBody1Rel().SetTargets([payload_path])
    
    # Set local transform
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_offset))