import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

def target_pos_rel(
    env: ManagerBasedRLEnv, 
    target_cfg: SceneEntityCfg, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns the position of the target relative to the robot's root frame."""
    target_pos = env.scene[target_cfg.name].data.root_pos_w
    robot_pos = env.scene[robot_cfg.name].data.root_pos_w
    robot_quat = env.scene[robot_cfg.name].data.root_quat_w
    
    # Position difference
    diff = target_pos - robot_pos
    # Transform to robot frame
    return math_utils.quat_apply_inverse(robot_quat, diff)

def target_quat_rel(
    env: ManagerBasedRLEnv, 
    target_cfg: SceneEntityCfg, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns the orientation of the target relative to the robot's root frame."""
    target_quat = env.scene[target_cfg.name].data.root_quat_w
    robot_quat = env.scene[robot_cfg.name].data.root_quat_w
    
    # Relative orientation: q_rel = q_robot^-1 * q_target
    return math_utils.quat_mul(math_utils.quat_conjugate(robot_quat), target_quat)

def ee_pos_rel(
    env: ManagerBasedRLEnv, 
    ee_cfg: SceneEntityCfg, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns the position of the end-effector relative to the robot's root (pelvis) frame."""
    # Resolve body ID if needed
    if ee_cfg.body_ids is None or len(ee_cfg.body_ids) == 0:
        body_id, _ = env.scene[ee_cfg.name].find_bodies(ee_cfg.body_names)
        body_idx = body_id[0]
    else:
        body_idx = ee_cfg.body_ids[0]

    ee_pos = env.scene[ee_cfg.name].data.body_pos_w[:, body_idx]
    robot_pos = env.scene[robot_cfg.name].data.root_pos_w
    robot_quat = env.scene[robot_cfg.name].data.root_quat_w
    
    # Position difference
    diff = ee_pos - robot_pos
    # Transform to robot frame
    return math_utils.quat_apply_inverse(robot_quat, diff)

def ee_quat_rel(
    env: ManagerBasedRLEnv, 
    ee_cfg: SceneEntityCfg, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns the orientation of the end-effector relative to the robot's root (pelvis) frame."""
    # Resolve body ID if needed
    if ee_cfg.body_ids is None or len(ee_cfg.body_ids) == 0:
        body_id, _ = env.scene[ee_cfg.name].find_bodies(ee_cfg.body_names)
        body_idx = body_id[0]
    else:
        body_idx = ee_cfg.body_ids[0]

    ee_quat = env.scene[ee_cfg.name].data.body_quat_w[:, body_idx]
    robot_quat = env.scene[robot_cfg.name].data.root_quat_w
    
    # Relative orientation: q_rel = q_robot^-1 * q_ee
    return math_utils.quat_mul(math_utils.quat_conjugate(robot_quat), ee_quat)

def arm_contacts(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 1.0
) -> torch.Tensor:
    """Returns a boolean contact tensor (1.0 for contact, 0.0 otherwise) for the specified bodies."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Resolve body ids if needed
    if sensor_cfg.body_ids is None or len(sensor_cfg.body_ids) == 0:
        body_ids, _ = contact_sensor.find_bodies(sensor_cfg.body_names)
    else:
        body_ids = sensor_cfg.body_ids
        
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Shape of net_contact_forces: (num_envs, history_length, num_total_bodies, 3)
    # Check if contact force magnitude for each tracked body is above the threshold
    contact_forces_norm = torch.norm(net_contact_forces[:, :, body_ids], dim=-1) # (num_envs, history_length, num_tracked_bodies)
    max_contact_forces = torch.max(contact_forces_norm, dim=1)[0] # (num_envs, num_tracked_bodies)
    
    is_contact = (max_contact_forces > threshold).float() # (num_envs, num_tracked_bodies)
    return is_contact
