import argparse
import sys
import os
import glob
import time

from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Play trained PPO policy for G1 Bimanual Coordinate Pose Reaching.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate during playback.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model checkpoint (e.g. model_1500.pt). If None, loads the latest checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment and runner.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch simulation application (Viewer mode by default)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import dependencies after simulation starts
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import importlib.metadata as metadata
    import torch
    from env_cfg import G1BimanualReacherEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import configclass
    
    # RSL-RL VecEnv wrapper, configs, and runner
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
        handle_deprecated_rsl_rl_cfg,
    )
    from rsl_rl.runners import OnPolicyRunner

    # 1. Create environment config
    env_cfg = G1BimanualReacherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # Disable domain randomization/offsets for evaluation if desired,
    # but we keep standard setups to see how it performs in different poses.
    print("[INFO] Creating G1 Bimanual environment for playback...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 2. Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # ปรับระดับความยากการสุ่มเป้าหมายสำหรับเปิดแสดงผล (Play)
    # ตั้งค่า 10000 จะได้ระดับความยากปานกลาง (เป้าหมายเกิดในระยะเอื้อมกลางๆ ไม่ใกล้เกินไปและไม่ไกลเกินไป)
    env.unwrapped.common_step_counter = 10000
    env.common_step_counter = 10000

    # 3. Define PPO configuration matching train.py
    @configclass
    class G1BimanualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        experiment_name = "g1_bimanual_pose_reach"
        run_name = "bimanual_pose_ppo"
        seed = args_cli.seed
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

    # Convert/handle deprecations
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = G1BimanualPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # 4. Find checkpoint path
    checkpoint_path = args_cli.checkpoint
    if checkpoint_path is None:
        checkpoint_dir = os.path.join("logs", agent_cfg.experiment_name, agent_cfg.run_name)
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
        if checkpoint_files:
            # Sort model files by the iteration number (model_<iteration>.pt)
            def get_iteration(file_path):
                filename = os.path.basename(file_path)
                try:
                    return int(filename.split("_")[-1].split(".")[0])
                except ValueError:
                    return 0
            checkpoint_files.sort(key=get_iteration)
            checkpoint_path = checkpoint_files[-1]
            print(f"[INFO] Automatically found latest checkpoint: {checkpoint_path}")
        else:
            print(f"[WARNING] No checkpoints found in {checkpoint_dir}. Running with random policy.")
            checkpoint_path = None
    else:
        print(f"[INFO] Using specified checkpoint: {checkpoint_path}")

    # 5. Setup Runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        runner.load(checkpoint_path)
        print("[INFO] Model loaded successfully.")
    
    # Obtain the policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 6. Playback simulation loop
    dt = env.unwrapped.step_dt
    print("[INFO] Starting playback loop. Press Ctrl+C to stop.")
    
    obs = env.get_observations()
    
    try:
        while simulation_app.is_running():
            start_time = time.time()
            
            with torch.inference_mode():
                # Inference action from policy
                actions = policy(obs)
                # Step environment
                obs, _, _, _ = env.step(actions)
                
            # Slow down simulation to match real time
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[INFO] Playback stopped by user.")

    # Close environment
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
