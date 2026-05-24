import torch
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils

def reset_target_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    target_side: str, # "left" หรือ "right"
    max_steps: int = 150000, # จำนวนสเต็ปที่แอดวานซ์จากง่ายสุดไปยากสุด
):
    """
    ฟังก์ชันสุ่มพิกัดเป้าหมายแบบไต่ระดับการเรียนรู้ (Curriculum Reset)
    - เริ่มแรกช่วงต้นฝึกฝน เป้าหมายจะเกิดตำแหน่งและทิศทางตรงกับมือเริ่มต้นหุ่นยนต์ (Euler error ~ 0)
    - เมื่อเวลา/สเต็ปการเทรนรวมเพิ่มขึ้น จะสุ่มระยะห่างและมุมบิดเบี้ยวค่อยๆ ยากขึ้นเป็นลำดับ
    - ควบคุมขอบเขตให้อยู่ในขอบข่ายโต๊ะเสมอ
    """
    asset_name = f"{target_side}_target"
    asset = env.scene[asset_name]
    num_resets = len(env_ids)
    
    # คำนวณอัตราความยากง่ายของ Curriculum (0.0 = ง่ายสุด, 1.0 = ยากสุด/กว้างสุด)
    step_counter = getattr(env, "common_step_counter", 0)
    alpha = min(1.0, float(step_counter) / max_steps)
    
    # กำหนดพิกัดฝ่ามือเริ่มต้นโดยอิงจากตำแหน่งจริงของหุ่นยนต์ (Physical Palm Position)
    # ซ้าย: X=-0.17, Y=0.22, Z=0.80, ขวา: X=0.16, Y=0.24, Z=0.85 เมตร
    if target_side == "left":
        default_pos = torch.tensor([-0.17, 0.22, 0.80], device=asset.device).repeat(num_resets, 1)
    else:
        default_pos = torch.tensor([0.16, 0.24, 0.85], device=asset.device).repeat(num_resets, 1)
        
    # กำหนดทิศทางเริ่มต้นหมุนรอบแกน Z 90 องศา [W, X, Y, Z] = [0.7071, 0, 0, 0.7071]
    # เพื่อให้ทิศทางตรงกับตัวหุ่นยนต์ G1 ที่สปอนแบบหมุน 90 องศารอบแกน Z พอดี
    default_quat = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=asset.device).repeat(num_resets, 1)
    
    # คำนวณตำแหน่งระดับโลก (World Position) ของปลายมือโดยการรวมตำแหน่งจุดเกิดของสภาพแวดล้อม
    hand_pos_w = default_pos + env.scene.env_origins[env_ids]
    hand_quat_w = default_quat
    
    # ระยะสุ่มจะเพิ่มขึ้นตามความก้าวหน้าของการเทรน (เริ่มต้น 5 ซม. ไปจนถึง 18 ซม. ในระดับยากสุด เพื่อให้เอื้อมถึงระยะกลาง)
    pos_limit = 0.05 + alpha * (0.18 - 0.05)
    ranges_pos = torch.tensor([[-pos_limit, pos_limit]] * 3, device=asset.device)
    pos_offset = math_utils.sample_uniform(ranges_pos[:, 0], ranges_pos[:, 1], (num_resets, 3), device=asset.device)
    
    # คำนวณตำแหน่งเป้าหมายเบื้องต้น
    target_pos = hand_pos_w + pos_offset
    
    # ควบคุมขอบเขตเป้าหมายให้อยู่ในระยะเอื้อมกลางๆ เหนือโต๊ะ (Medium Workspace)
    pos_rel = target_pos - env.scene.env_origins[env_ids]
    if target_side == "left":
        pos_rel[:, 0] = torch.clamp(pos_rel[:, 0], min=-0.35, max=-0.05)
    else:
        pos_rel[:, 0] = torch.clamp(pos_rel[:, 0], min=0.05, max=0.35)
        
    pos_rel[:, 1] = torch.clamp(pos_rel[:, 1], min=0.18, max=0.38) # ระยะลึก Y (จากหน้าอกไปจนถึงกลางโต๊ะ)
    pos_rel[:, 2] = torch.clamp(pos_rel[:, 2], min=0.70, max=0.95) # ความสูง Z (ตั้งแต่ระดับโต๊ะถึงระดับหน้าอก)
    
    # แปลงกลับเป็นตำแหน่งระดับโลกหลังทำการควบคุม (Clamping)
    target_pos = pos_rel + env.scene.env_origins[env_ids]
    
    # --- 2. คำนวณและสุ่มทิศทางมุมหมุนเบี่ยงเบน (Orientation Offset) ---
    # จำกัดมุมสุ่มเบี่ยงเบนไม่ให้บิดหมุนมากเกินไป (เริ่มต้น 0.05 เรเดียน (~3 องศา) ถึงสูงสุดเพียง 0.2618 เรเดียน (~15 องศา))
    rot_limit = 0.05 + alpha * (0.2618 - 0.05)
    ranges_rot = torch.tensor([[-rot_limit, rot_limit]] * 3, device=asset.device)
    rot_samples = math_utils.sample_uniform(ranges_rot[:, 0], ranges_rot[:, 1], (num_resets, 3), device=asset.device)
    
    # สร้างการแปลงทิศทางเบี่ยงเบนแบบ Quaternion
    quat_delta = math_utils.quat_from_euler_xyz(rot_samples[:, 0], rot_samples[:, 1], rot_samples[:, 2])
    # ประยุกต์ทิศทางเบี่ยงเบนเข้ากับมุมข้อมือเริ่มต้น
    target_quat = math_utils.quat_mul(hand_quat_w, quat_delta)
    
    # บันทึกพิกัดและทิศทางเป้าหมายกลับคืนสู่ PhysX Stage
    asset.write_root_pose_to_sim(torch.cat([target_pos, target_quat], dim=-1), env_ids=env_ids)
