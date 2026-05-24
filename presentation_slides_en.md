# Presentation Slides: G1 Bimanual Reaching & Pose Control

---

## 🛝 Slide 1: Title
**Title:** G1 Bimanual Pose Control using Deep Reinforcement Learning
**Subtitle:** Training the Unitree G1 Humanoid Robot for Simultaneous Dual-Arm Reaching via IsaacLab
**Presenter:** [Your Name]

---

## 🛝 Slide 2: Project Objective
**Main Goal:** 
Develop and train an Artificial Intelligence (AI) policy to control both arms of the Unitree G1 robot, guiding its palms to dynamically generated 3D coordinates and orientations with high precision and smooth motion.

**Key Challenges:**
- Controlling 14 independent joints simultaneously (7 joints per arm).
- Avoiding self-collisions and environmental collisions (e.g., the table).
- Ensuring natural, human-like motion without jitter or stiffness (Smoothness).

---

## 🛝 Slide 3: Technology Stack
- **Simulator:** **NVIDIA Isaac Sim & IsaacLab**
  - Utilized for Massively Parallel Simulation using GPU-accelerated physics.
- **Algorithm:** **PPO (Proximal Policy Optimization)**
  - An advanced Deep Reinforcement Learning algorithm via the `rsl_rl` library.
- **Robot Model:** **Unitree G1 Humanoid**
  - Focused specifically on the upper body (arms and wrists) for the required tasks.

---

## 🛝 Slide 4: Environment & MDP Setup
The Markov Decision Process (MDP) design consists of:

- **Observation Space (What the AI sees - 87 Dimensions):**
  - Joint States: Positions and Velocities.
  - Target Coordinates: Distance (Position) and Rotation (Quaternion) relative to the robot's base.
  - Current Palm Coordinates: Relative to the targets.
  - Contact Forces: Collision detection data.
- **Action Space (AI Outputs - 14 Dimensions):**
  - Outputs Position Commands directly to the PD Controller of each joint.

---

## 🛝 Slide 5: Reward Engineering
The core logic that guides the robot to learn the correct behaviors:

**1. Task Rewards (Success Metrics):**
- `distance_reward`: Rewards the robot as its palms move closer to the targets.
- `distance_fine_reward`: High-precision reward when the hands are in extremely close proximity to the targets.
- `orientation_reward`: Rewards aligning the palm's rotation with the target's XYZ coordinate frame.

**2. Regularization Penalties (Smoothness & Safety):**
- `action_rate` & `action_l2`: Penalizes abrupt command changes to reduce jerky movements.
- `joint_vel` & `joint_acc`: Limits joint speed and acceleration for physical safety.
- `undesired_contacts`: Heavy penalties for self-collisions or hitting the table.

---

## 🛝 Slide 6: Curriculum Learning
To ensure rapid learning and prevent early-stage failures, we implemented a progressive difficulty scaling technique:

- **Phase 1 (Beginner):** Targets spawn overlapping or very close to the palms (max 5 cm). The AI learns that surviving near the target yields rewards.
- **Phase 2 (Scaling):** Target spawn distance automatically and gradually expands as training steps increase.
- **Phase 3 (Maximum Difficulty):** Targets spawn randomly across the table space (Medium Workspace up to 18-30 cm away), forcing the robot to learn full-arm reaching behaviors.

---

## 🛝 Slide 7: Training Process & Overcoming Issues
- **Parallel Environments:** Simulating 64 robots simultaneously on a single GPU (reducing training time from days to minutes).
- **Overcoming the "Lazy Agent" Problem:**
  - *Issue:* Initially, the robot refused to move its arms to avoid accumulating penalty deductions (becoming a "Lazy Agent").
  - *Solution:* Fine-tuned the penalty weights (e.g., heavily reducing `action_l2` and `action_rate`) and doubled the `distance` rewards. This achieved a perfect balance between reaching the targets and maintaining smooth motion.

---

## 🛝 Slide 8: Results & Conclusion
- **Zero-Shot Transfer:** The trained robot can reach dynamically moving, unseen targets in real-time during Play mode.
- **Stable Tracking:** Successfully maintains palm elevation and rotation precisely aligned with the target frames without jitters.
- **Conclusion:** IsaacLab combined with PPO is highly effective for training complex Bimanual Control systems and provides a strong foundation for future Sim-to-Real hardware deployment.

---
## 🛝 Slide 9: Q&A
**Q & A**
Thank you for listening!
