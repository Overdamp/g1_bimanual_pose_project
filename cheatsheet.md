# 📋 G1 Bimanual Reaching Command-Line Cheatsheet

หน้านี้รวบรวมคำสั่ง (Commands) ที่จำเป็นสำหรับการพัฒนา ฝึกฝน ทดสอบ และติดตามผลการเรียนรู้ของหุ่นยนต์ Unitree G1 Bimanual Reaching

---

## ⚙️ 1. การเตรียมสภาพแวดล้อม (Environment Setup)

ก่อนเริ่มต้นรันคำสั่งใดๆ ให้เปิดใช้งาน Conda Environment ที่ติดตั้ง Isaac Lab ไว้:

```bash
# เปิดใช้งาน Conda Env ของ Isaac Lab 2026
source /home/luke/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab2026
```

---

## 🏋️ 2. การฝึกฝนโมเดล (Training Policy)

ไฟล์หลักสำหรับการเทรนคือ `train.py`

### 2.1 ทดสอบความถูกต้องเบื้องหลังอย่างรวดเร็ว (Headless Dry-run)
สำหรับเช็คบั๊ก โหลดฉาก และทดสอบ Syntax (ไม่ต้องเปิดหน้าจอ GUI):
```bash
CUDA_VISIBLE_DEVICES=0 python g1_bimanual_pose_project/train.py --num_envs 2 --max_iterations 2 --headless
```

### 2.2 ฝึกฝนแบบเต็มรูปแบบโหมดไม่มีหน้าจอ (Headless Full Training)
(แนะนำสำหรับการฝึกฝนจริงเพื่อความรวดเร็วสูงสุด):
```bash
CUDA_VISIBLE_DEVICES=0 python g1_bimanual_pose_project/train.py --num_envs 64 --max_iterations 1500 --headless
```

### 2.3 ฝึกฝนแบบสังเกตหน้าจอ (GUI Full Training)
(เปิดฉากจำลอง 3D เพื่อดูความเคลื่อนไหว):
```bash
CUDA_VISIBLE_DEVICES=0 python g1_bimanual_pose_project/train.py --num_envs 64 --max_iterations 1500
```

### 2.4 ระบุค่า Seed ที่เฉพาะเจาะจง
```bash
CUDA_VISIBLE_DEVICES=0 python g1_bimanual_pose_project/train.py --num_envs 64 --max_iterations 1500 --seed 42 --headless
```

---

## 🎮 3. การแสดงผลและประเมินนโยบาย (Policy Playback / Evaluation)

ไฟล์หลักสำหรับการเล่นนโยบายคือ `play.py` (รันในโหมดมี GUI โดยค่าเริ่มต้นเพื่อประเมินทางสายตา)

### 3.1 ค้นหาและโหลดโมเดลล่าสุดอัตโนมัติ
```bash
CUDA_VISIBLE_DEVICES=0 python g1_bimanual_pose_project/play.py --num_envs 16
```

### 3.2 เจาะจงไฟล์โมเดล Checkpoint ที่ต้องการรัน
```bash
CUDA_VISIBLE_DEVICES=0 python g1_bimanual_pose_project/play.py --num_envs 16 --checkpoint logs/g1_bimanual_pose_reach/bimanual_pose_ppo/model_1500.pt
```

### 3.3 กำหนด Seed เฉพาะเพื่อทดสอบการตอบสนองรูปแบบเดียวกัน
```bash
CUDA_VISIBLE_DEVICES=0 python g1_bimanual_pose_project/play.py --num_envs 16 --seed 42
```

---

## 📊 4. การวิเคราะห์ผลผ่าน TensorBoard (Monitoring Logs)

เปิดหน้าต่างวิเคราะห์ข้อมูลกราฟคะแนนความคืบหน้า (Reward, Actor/Critic Loss)

### 4.1 รัน TensorBoard Server
```bash
tensorboard --logdir logs/g1_bimanual_pose_reach/
```

> [!TIP]
> เมื่อรันคำสั่งเสร็จสิ้น สามารถเปิดบราวเซอร์เพื่อดูผลลัพธ์ผ่าน URL: **`http://localhost:6006/`**

### 4.2 ระบุพอร์ตบราวเซอร์กรณีพอร์ตชนกัน
```bash
tensorboard --logdir logs/g1_bimanual_pose_reach/ --port 6007
```

---

## 🧹 5. การจัดการกับ Logs (Log Management)

### 5.1 ล้าง Logs เก่าเพื่อเริ่มเก็บสถิติการรันใหม่
```bash
# ลบโฟลเดอร์ Log การเทรนทั้งหมด (ระมัดระวัง! คำสั่งนี้จะลบไฟล์โมเดลที่บันทึกไว้ด้วย)
rm -rf logs/g1_bimanual_pose_reach/
```
