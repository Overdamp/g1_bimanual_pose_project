# 🔄 ลำดับการรันโปรแกรมของระบบ (System Program Execution Flow)

คู่มือนี้อธิบายลำดับการรันคำสั่งและการทำงานของระบบควบคุมหุ่นยนต์ Unitree G1 Bimanual Reaching ตั้งแต่จุดเริ่มต้นการประมวลผลจนถึงสเต็ปการทำซ้ำ (Step Loop) ของแต่ละสภาพแวดล้อมครับ

---

## 🗺️ แผนผังลำดับภาพรวม (Execution Flow Diagram)

```mermaid
graph TD
    A[1. Entry Point: train.py / play.py] --> B[2. AppLauncher: เปิดระบบจำลอง Isaac Sim Headless/GUI]
    B --> C[3. Load env_cfg.py: โหลดโครงสร้างโมเดลและฟังก์ชันนำทาง]
    C --> D[4. Init ManagerBasedRLEnv: สร้าง Environment และโหลด Assets ลงในฉาก]
    D --> E[5. Event Manager: เรียก reset_scene เพื่อวางจุดสปอนเริ่มต้นและจัดท่าหุ่นยนต์]
    E --> F[6. Wrap env for RSL-RL & Load Policy]
    F --> G[7. Main Step Loop: ลูปการตัดสินใจและการเทรน]
    
    subgraph Main Step Loop (ลูป 30Hz)
        G1[7.1 ดึงค่าสังเกต Observations จาก mdp/observations.py] --> G2[7.2 โมเดลทำนาย Action: Joint Delta]
        G2 --> G3[7.3 Action Manager ส่งคำสั่งขยับข้อต่อหุ่นยนต์]
        G3 --> G4[7.4 ฟิสิกส์ประมวลผลการเคลื่อนที่ Step Sim]
        G4 --> G5[7.5 คำนวณคะแนน Rewards จาก mdp/rewards.py]
        G5 --> G6[7.6 ตรวจสอบ Termination: ครบ 5 วินาทีหรือไม่?]
        G6 -- หมดเวลา / ชนรุนแรง --> G7[7.7 เรียกสุ่มค่าใหม่ Event Manager: mdp/events.py]
        G6 -- ยังไม่หมดเวลา --> G1
        G7 --> G1
    end
```

---

## 🔍 คำอธิบายการทำงานแต่ละขั้นตอนอย่างละเอียด (Step-by-Step Details)

### ขั้นตอนที่ 1: จุดเริ่มต้น (Entry Point)
* **ไฟล์ที่เรียกใช้**: [train.py](file:///home/luke/Isaac_directory/IsaacLab2026/g1_bimanual_pose_project/train.py) (สำหรับเทรน) หรือ [play.py](file:///home/luke/Isaac_directory/IsaacLab2026/g1_bimanual_pose_project/play.py) (สำหรับแสดงผล)
* **สิ่งที่เกิดขึ้น**:
  1. ระบบจะรันคำสั่งจอง GPU และตั้งค่าพารามิเตอร์เริ่มต้น (เช่น จำนวน Envs, พอร์ต Tensorboard, และโหมด Headless)
  2. เรียกใช้งาน `AppLauncher` เพื่อทำการรันระบบจำลองฟิสิกส์เบื้องหลังของ **Isaac Sim (Omniverse)**

### ขั้นตอนที่ 2: โหลดข้อมูลการตั้งค่าฉาก (Load Configuration)
* **ไฟล์ที่เรียกใช้**: [env_cfg.py](file:///home/luke/Isaac_directory/IsaacLab2026/g1_bimanual_pose_project/env_cfg.py)
* **สิ่งที่เกิดขึ้น**:
  ระบบอ่านค่า Config Class เพื่อเตรียมสร้างแผนที่จำลอง:
  * **Scene Layout**: ระบุว่าจะใช้หุ่นยนต์ `G1`, โต๊ะ `PackingTable` (ที่ปิดตะกร้าผ่าน `spawn_clean_table`), และพิกัดเป้าหมายซ้าย/ขวา (`left_target`, `right_target`) ขนาด 10 ซม.
  * **Observation Terms**: เชื่อมโยงข้อมูลที่ส่งให้ AI เรียนรู้ เข้ากับฟังก์ชันใน [mdp/observations.py](file:///home/luke/Isaac_directory/IsaacLab2026/g1_bimanual_pose_project/mdp/observations.py) (พิกัดมือ, มุมข้อต่อ, พิกัดเป้าหมายเทียบกับตัวหุ่น)
  * **Reward Terms**: เชื่อมโยงน้ำหนักคะแนนเข้ากับฟังก์ชันใน [mdp/rewards.py](file:///home/luke/Isaac_directory/IsaacLab2026/g1_bimanual_pose_project/mdp/rewards.py) (ระยะห่างแกน, ทิศทางข้อมือ, บทลงโทษการชน)
  * **Event Terms**: เชื่อมโยงกฎตอนเกิดรีเซ็ตเข้ากับฟังก์ชันใน [mdp/events.py](file:///home/luke/Isaac_directory/IsaacLab2026/g1_bimanual_pose_project/mdp/events.py) (การสุ่มเป้าหมายแบบ Curriculum)

### ขั้นตอนที่ 3: สร้างและจัดฉากสิ่งแวดล้อม (Environment Instantiation)
* **ไฟล์ที่เรียกใช้**: `ManagerBasedRLEnv(cfg=env_cfg)` (ภายใน Isaac Lab Core)
* **สิ่งที่เกิดขึ้น**:
  1. ตัวจำลองฉากจะสปอนวัตถุทั้งหมดลงในแต่ละ Sub-environment (เช่น 64 ช่อง Envs ขนานกัน)
  2. เรียกฟังก์ชันเริ่มต้นในกลุ่ม **Startup & Reset Events** ใน [mdp/events.py](file:///home/luke/Isaac_directory/IsaacLab2026/g1_bimanual_pose_project/mdp/events.py):
     * จัดท่าทางของหุ่นยนต์ให้อยู่ในท่ายืนเซ็ตอัพ
     * สุ่มพิกัดและแกนหมุนของเป้าหมาย `left_target` และ `right_target` บนผิวหน้าโต๊ะครั้งแรกสุด

### ขั้นตอนที่ 4: เชื่อมต่ออัลกอริทึม PPO และลูปหลัก (RL Wrapper & Main Step Loop)
* **ไฟล์ที่เรียกใช้**: `RslRlVecEnvWrapper` และ `OnPolicyRunner` (RSL-RL Library)
* **สิ่งที่เกิดขึ้น**:
  1. ตัว Wrapper จะทำหน้าที่แปลงรูปแบบข้อมูลภาพรวมให้เข้ากับอัลกอริทึม PPO
  2. โหลดน้ำหนักโครงข่ายประสาท (เช่น `model_1499.pt` ในสคริปต์ play.py) หรือเริ่มสุ่มค่าน้ำหนักเพื่อฝึกฝนใหม่
  3. **เริ่มลูปขั้นตอนจำลองหลัก (Main Step Loop)** ซึ่งจะวนลูปซ้ำด้วยความถี่ควบคุม 30Hz:
     * **อ่านค่า (Read Observations)**: เรียกฟังก์ชันใน `mdp/observations.py` ดึงข้อมูลข้อต่อและระยะเป้าหมายส่งให้โมเดล AI
     * **ทำนายคำสั่ง (Predict Action)**: โมเดลแปลง Observation ออกมาเป็น Joint Position Deltas (Action) ขนาด 14 มิติตามแกนแขน
     * **ขยับจริง (Step Physics)**: คอนโทรลเลอร์ขับมอเตอร์ขยับแขนหุ่นยนต์ตามคำสั่ง และสเต็ปเวลาจำลองฟิสิกส์ขยับตาม
     * **ให้คะแนน (Compute Rewards)**: เรียกฟังก์ชันใน `mdp/rewards.py` เพื่อวัดว่าการเข้าใกล้เป้าหมายดีขึ้นไหม และการขยับสั่นสั่นน้อยลงตามเกณฑ์ที่เราตั้งไว้หรือไม่
     * **ตรวจสอบจุดสิ้นสุดตอน (Check Termination)**: หากครบรอบ 5 วินาทีจำลอง (150 steps) หรือเกิดการหลุดฉาก/ชนรุนแรง จะสั่งสัญญาณ Reset เป็น True
     * **สุ่มเป้าหมายใหม่ (Execute Reset Event)**: เมื่อ Envs ใดเกิดการ Reset จะเรียกฟังก์ชันใน `mdp/events.py` เพื่อสุ่มสปอนพิกัดเป้าหมายใหม่และจัดท่าเริ่มข้อมือหุ่นยนต์ เพื่อเริ่มวนลูปใหม่อีกครั้งอย่างต่อเนื่อง
