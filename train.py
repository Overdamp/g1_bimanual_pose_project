import argparse
import sys
import os

from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Train PPO for G1 Bimanual Coordinate Pose Reaching.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
    parser.add_argument("--max_iterations", type=int, default=1500, help="RL Policy training iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment and runner.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch simulation application
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
    
    # 2. Initialize environment
    print("[INFO] Creating G1 Bimanual environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 3. Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # 4. Define PPO configuration using RslRl classes for compatibility
    @configclass
    class G1BimanualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 24
        max_iterations = args_cli.max_iterations
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

    # 5. Setup Runner
    log_dir = os.path.join("logs", agent_cfg.experiment_name, agent_cfg.run_name)
    print(f"[INFO] Setting up OnPolicyRunner. Logs will be saved to: {log_dir}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    # 6. Start Training
    print(f"[INFO] Starting PPO training for {agent_cfg.max_iterations} iterations...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("[INFO] Training finished.")
    # Close environment
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
