# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess = false

from pxr import Gf, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage

def attach_payload_to_robot(robot_body_path, payload_path, env_idx, local_offset=(0.0, 0.0, 0.14343)):
    """
    Attach a payload object to the robot using a fixed joint.
    
    Args:
        robot_body_path: USD path to the robot's body link
        payload_path: USD path to the payload object
        local_offset: Local position offset as (x, y, z) tuple
    """
    stage = get_current_stage()
    
    # Create fixed joint /World/envs/env_0/Robot/body or /World/FixedJoint 
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"/World/envs/env_{env_idx}/FixedJoint")
    
    # Set body relationships
    fixed_joint.CreateBody0Rel().SetTargets([robot_body_path])
    fixed_joint.CreateBody1Rel().SetTargets([payload_path])
    
    # Set local transform
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_offset))

    # Making it EXTREMELY rigid (because I want less relative acceleration effect due to robot movement)
    fixed_joint.CreateBreakForceAttr().Set(1e10)
    fixed_joint.CreateBreakTorqueAttr().Set(1e10)


def update_payload_position(new_offset=(0.1, 0.1, 0.3)):
    """
    Update the payload position by modifying the fixed joint's local transform.
    
    Args:
        new_offset: New local position offset as (x, y, z) tuple
    """
    stage = get_current_stage()
    fixed_joint_prim = stage.GetPrimAtPath("/World/FixedJoint")
    
    if fixed_joint_prim.IsValid():
        fixed_joint = UsdPhysics.FixedJoint(fixed_joint_prim)
        fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*new_offset))
        print(f"Updated payload position to: {new_offset}")
    else:
        print("Fixed joint not found!")