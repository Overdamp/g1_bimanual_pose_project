import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def distance_to_target_tanh(
    env: ManagerBasedRLEnv, 
    std: float, 
    target_cfg: SceneEntityCfg, 
    ee_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Reward function that uses a tanh kernel to encourage the end-effector to reach the target.
    """
    # Get target root position
    target_pos = env.scene[target_cfg.name].data.root_pos_w
    
    # Resolve body ID if needed
    if ee_cfg.body_ids is None or len(ee_cfg.body_ids) == 0:
        # resolve body names
        body_id, _ = env.scene[ee_cfg.name].find_bodies(ee_cfg.body_names)
        body_idx = body_id[0]
    else:
        body_idx = ee_cfg.body_ids[0]

    # Get robot end-effector position
    ee_pos = env.scene[ee_cfg.name].data.body_pos_w[:, body_idx]
    
    # Calculate Euclidean distance
    distance = torch.norm(target_pos - ee_pos, dim=-1)
    
    # Calculate tanh reward
    reward = 1.0 - torch.tanh(distance / std)
    
    return reward

def orientation_to_target_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    target_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Reward function that uses a tanh kernel on orientation error to align the end-effector with the target frame.
    """
    # Get target root orientation (quaternion)
    target_quat = env.scene[target_cfg.name].data.root_quat_w
    
    # Resolve body ID if needed
    if ee_cfg.body_ids is None or len(ee_cfg.body_ids) == 0:
        body_id, _ = env.scene[ee_cfg.name].find_bodies(ee_cfg.body_names)
        body_idx = body_id[0]
    else:
        body_idx = ee_cfg.body_ids[0]

    # Get robot end-effector orientation (quaternion)
    ee_quat = env.scene[ee_cfg.name].data.body_quat_w[:, body_idx]
    
    # Calculate quaternion similarity: dot product squared
    # dot_prod is in [-1.0, 1.0]. A value of 1.0 or -1.0 means perfect alignment.
    dot_prod = torch.sum(target_quat * ee_quat, dim=-1)
    quat_diff_sq = 1.0 - torch.square(dot_prod)
    
    # Calculate tanh reward
    reward = 1.0 - torch.tanh(quat_diff_sq / std)
    
    return reward
