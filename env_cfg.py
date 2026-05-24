import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.utils import clone
from pxr import UsdPhysics, Usd

# นำเข้าฟังก์ชัน MDP แบบกำหนดเองที่เราเขียนขึ้นมาในโฟลเดอร์ mdp
import mdp as custom_mdp

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

@clone
def spawn_target_frame(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """ฟังก์ชันสปอน์เพื่อบังคับให้ Visual USD Frame มีคุณสมบัติ RigidBody เพื่อให้ Isaac Lab รู้จักในฐานะ RigidObject"""
    # โหลดไฟล์ USD ปกติ
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # บังคับลงทะเบียน RigidBodyAPI และอัปเดต properties เช่น kinematic_enabled
    if cfg.rigid_props is not None:
        from isaaclab.sim.schemas import schemas
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    return prim


@clone
def spawn_clean_table(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """ฟังก์ชันโหลดโต๊ะ PackingTable และปิดการใช้งาน (Deactivate) กล่องและตะกร้าด้านบนทั้งหมด"""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    
    # รายการ Prim paths ย่อยของตะกร้าและกล่องที่จะเอาออก
    paths_to_deactivate = [
        f"{prim_path}/container_h20",
        f"{prim_path}/SM_CratePacking_Table_A1/Crate_02_GRP",
        f"{prim_path}/SM_CratePacking_Table_A1/Crate_03_GRP",
        f"{prim_path}/SM_CratePacking_Table_A1/SM_Crate_A07_Yellow_05",
        f"{prim_path}/SM_CratePacking_Table_A1/SM_Crate_A07_Yellow_06",
        f"{prim_path}/SM_CratePacking_Table_A1/SM_Crate_A07_Yellow_07",
        f"{prim_path}/SM_CratePacking_Table_A1/SM_Crate_A08_Blue_03",
        f"{prim_path}/SM_CratePacking_Table_A1/SM_Crate_A08_Blue_04",
    ]
    
    for path in paths_to_deactivate:
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            p.SetActive(False)
            
    return prim


@configclass
class G1BimanualSceneCfg(InteractiveSceneCfg):
    """การกำหนดองค์ประกอบจำลองในฉาก (G1 Articulation, Coordinate Frames, Ground, Lights, Table)"""
    
    # 1. หุ่นยนต์ Unitree G1 (29 DOF)
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 2. พิกัดเป้าหมายแกนอ้างอิง XYZ สำหรับมือซ้าย (กำหนดขนาดเล็กลงเป็น 0.10 และใช้ kinematic_enabled=True)
    left_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/left_target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.7071068, 0.0, 0.0, 0.7071068]),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_target_frame,
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        )
    )
    
    # 3. พิกัดเป้าหมายแกนอ้างอิง XYZ สำหรับมือขวา (กำหนดขนาดเล็กลงเป็น 0.10 และใช้ kinematic_enabled=True)
    right_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/right_target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[0.7071068, 0.0, 0.0, 0.7071068]),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_target_frame,
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        )
    )
    
    # 4. เซ็นเซอร์ตรวจจับการชน (Contact Sensor) เพื่อตรวจเช็คการชนของแขนและลำตัว
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False
    )
    
    # 5. โต๊ะ PackingTable ตั้งไว้ด้านหน้าหุ่นยนต์ (หุ่นยนต์หันหน้าไปทางแกน +Y)
    # วางไว้ที่ตำแหน่ง Y=0.55 และปรับ Z=-0.3 พร้อมเปิดใช้ CollisionPropertiesCfg เพื่อป้องกันการชน
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_clean_table,
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    
    # 6. Coordinate Frame Marker ที่ฝ่ามือซ้าย (left_hand_palm_link)
    left_ee_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_hand_palm_link/ee_marker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.10, 0.10, 0.10),
        ),
    )
    
    # 7. Coordinate Frame Marker ที่ฝ่ามือขวา (right_hand_palm_link)
    right_ee_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_hand_palm_link/ee_marker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.10, 0.10, 0.10),
        ),
    )
    
    # 8. พื้นผิวจำลอง
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=GroundPlaneCfg(),
    )
    
    # 9. แสงสว่างในฉาก
    lights = AssetBaseCfg(
        prim_path="/World/defaultDomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2000.0),
    )


@configclass
class ActionsCfg:
    """การควบคุมข้อต่อของหุ่นยนต์ (จำกัดความเร็วลงมาที่ 50% เพื่อความปลอดภัย)"""
    
    # ควบคุมมุมองศาข้อต่อแขน (ไหล่, ข้อศอก, ข้อมือ) ทั้งหมด 14 ข้อต่อ
    # ใช้ RelativeJointPositionActionCfg (Delta Control) เพื่อให้สะสมการเคลื่อนไหวขยับได้เต็มพิกัดอย่างสมูท
    arms_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=0.05,
        use_zero_offset=True,
    )


@configclass
class ObservationsCfg:
    """ข้อมูลที่ AI นำไปใช้ในการประมวลผล (Observations)"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        # องศาข้อต่อปัจจุบันของแขนหุ่นยนต์แบบสัมพัทธ์
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ])}
        )
        # ความเร็วข้อต่อแขนปัจจุบันแบบสัมพัทธ์
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ])}
        )
        
        # ตำแหน่งและทิศทางเป้าหมายซ้าย เทียบกับตัวหุ่นยนต์ (Pelvis)
        left_target_pos = ObsTerm(
            func=custom_mdp.target_pos_rel, 
            params={"target_cfg": SceneEntityCfg("left_target"), "robot_cfg": SceneEntityCfg("robot")}
        )
        left_target_quat = ObsTerm(
            func=custom_mdp.target_quat_rel,
            params={"target_cfg": SceneEntityCfg("left_target"), "robot_cfg": SceneEntityCfg("robot")}
        )
        
        # ตำแหน่งและทิศทางเป้าหมายขวา เทียบกับตัวหุ่นยนต์ (Pelvis)
        right_target_pos = ObsTerm(
            func=custom_mdp.target_pos_rel, 
            params={"target_cfg": SceneEntityCfg("right_target"), "robot_cfg": SceneEntityCfg("robot")}
        )
        right_target_quat = ObsTerm(
            func=custom_mdp.target_quat_rel,
            params={"target_cfg": SceneEntityCfg("right_target"), "robot_cfg": SceneEntityCfg("robot")}
        )
        
        # ตำแหน่งและทิศทางฝ่ามือซ้าย เทียบกับตัวหุ่นยนต์ (Pelvis)
        left_ee_pos = ObsTerm(
            func=custom_mdp.ee_pos_rel,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link"]), "robot_cfg": SceneEntityCfg("robot")}
        )
        left_ee_quat = ObsTerm(
            func=custom_mdp.ee_quat_rel,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link"]), "robot_cfg": SceneEntityCfg("robot")}
        )
        
        # ตำแหน่งและทิศทางฝ่ามือขวา เทียบกับตัวหุ่นยนต์ (Pelvis)
        right_ee_pos = ObsTerm(
            func=custom_mdp.ee_pos_rel,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]), "robot_cfg": SceneEntityCfg("robot")}
        )
        right_ee_quat = ObsTerm(
            func=custom_mdp.ee_quat_rel,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]), "robot_cfg": SceneEntityCfg("robot")}
        )
        
        # การกระทำครั้งก่อนหน้า (Last Action) เพื่อความสมูทในการสั่งการ
        actions = ObsTerm(func=mdp.last_action)
        
        # ข้อมูลการชน/สัมผัสทั้งหมดของหุ่นยนต์กับตัวเองและสิ่งแวดล้อม (Self-Collision & Table collision detection)
        # โฟกัสเฉพาะร่างกายส่วนบนและแขน เพื่อหลีกเลี่ยงการชนตัวและโต๊ะ
        arm_contacts = ObsTerm(
            func=custom_mdp.arm_contacts,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        ".*_shoulder_.*",
                        ".*_elbow_.*",
                        ".*_wrist_.*",
                        "waist_.*",
                        "torso_.*",
                    ],
                ),
                "threshold": 1.0,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """กฎการสุ่มตำแหน่งวัตถุและท่าทางเริ่มต้นหุ่นยนต์ในแต่ละรอบ (Domain Randomization & Resets)"""
    
    # 1. รีเซ็ตทุกข้อต่อของหุ่นยนต์ให้อยู่ในท่ายืนมาตรฐานเริ่มต้นก่อนเสมอ
    reset_all_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # 2. สุ่มเบี่ยงเบนเล็กน้อยเฉพาะองศาข้อต่อแขน เพื่อสร้างความหลากหลายในการเริ่มเล่น
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ]
            ),
        },
    )
    
    # 3. สุ่มตำแหน่งและมุมของเป้าหมายฝั่งซ้าย (แบบไต่ระดับความยาก Curriculum Learning)
    # เริ่มแรกช่วงต้นฝึกฝน เป้าหมายจะสปอนใกล้มือเริ่มต้นหุ่นยนต์ เมื่อเทรนขึ้นสูงขึ้นจะสุ่มกว้างออกไปจนทั่วหน้าโต๊ะ
    reset_left_target = EventTerm(
        func=custom_mdp.reset_target_curriculum,
        mode="reset",
        params={
            "target_side": "left",
            "max_steps": 20000, # ไต่ระดับความยากแบบสมูทขึ้นจนถึงประมาณ iteration ที่ 830 (20000 / 24 steps)
        },
    )
    
    # 4. สุ่มตำแหน่งและมุมของเป้าหมายฝั่งขวา (แบบไต่ระดับความยาก Curriculum Learning)
    reset_right_target = EventTerm(
        func=custom_mdp.reset_target_curriculum,
        mode="reset",
        params={
            "target_side": "right",
            "max_steps": 20000,
        },
    )


@configclass
class RewardsCfg:
    """ฟังก์ชันการให้คะแนนและลงโทษเพื่อนำทาง AI (Reward / Penalty Terms)"""
    
    # -- ส่วนที่ 1: รางวัลตำแหน่งฝ่ามือเข้าใกล้เป้าหมาย
    left_distance = RewTerm(
        func=custom_mdp.distance_to_target_tanh,
        weight=2.5,
        params={
            "std": 0.15,
            "target_cfg": SceneEntityCfg("left_target"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link"]),
        },
    )
    
    right_distance = RewTerm(
        func=custom_mdp.distance_to_target_tanh,
        weight=2.5,
        params={
            "std": 0.15,
            "target_cfg": SceneEntityCfg("right_target"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
        },
    )

    # -- คะแนนส่วนขยายความแม่นยำสูงระยะชิด (Fine reaching reward)
    left_distance_fine = RewTerm(
        func=custom_mdp.distance_to_target_tanh,
        weight=2.5,
        params={
            "std": 0.05,
            "target_cfg": SceneEntityCfg("left_target"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link"]),
        },
    )
    
    right_distance_fine = RewTerm(
        func=custom_mdp.distance_to_target_tanh,
        weight=2.5,
        params={
            "std": 0.05,
            "target_cfg": SceneEntityCfg("right_target"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
        },
    )
    
    # -- ส่วนที่ 2: รางวัลแนวแกนหมุนตรงกัน (Orientation matching reward)
    # นำทางให้ปลายข้อหันตาม XYZ frame เป้าหมาย
    left_orientation = RewTerm(
        func=custom_mdp.orientation_to_target_tanh,
        weight=1.5,
        params={
            "std": 0.2,
            "target_cfg": SceneEntityCfg("left_target"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["left_hand_palm_link"]),
        },
    )
    
    right_orientation = RewTerm(
        func=custom_mdp.orientation_to_target_tanh,
        weight=1.5,
        params={
            "std": 0.2,
            "target_cfg": SceneEntityCfg("right_target"),
            "ee_cfg": SceneEntityCfg("robot", body_names=["right_hand_palm_link"]),
        },
    )
    
    # -- ส่วนที่ 3: ลงโทษชนตัวเองหรือชนโต๊ะอย่างรุนแรง (Self-collision and table collision penalty) 
    # โฟกัสเฉพาะส่วนแขน เอว และลำตัวที่ห้ามชนตัวเองหรือสิ่งของอื่นเด็ดขาด
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_.*",
                    "waist_.*",
                    "torso_.*",
                ]
            ),
            "threshold": 1.0,
        },
    )
    
    # -- ส่วนเสริมความสวยงาม: ลงโทษการเคลื่อนไหวที่ไม่ราบเรียบ เพื่อลดอาการกระตุกหรือสะบัด
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.10
    )
    
    # ลงโทษขนาดของการขยับ (Action L2 Penalty) เพื่อบังคับให้ Actions เข้าใกล้ 0 เมื่อเอื้อมถึงเป้าหมาย
    # เนื่องจากค่า Action คือ Delta ข้อต่อ ดังนั้น 0 หมายถึงหยุดนิ่งขยับแขนอยู่กับที่
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.01
    )
    
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"])}
    )

    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"])}
    )


@configclass
class TerminationsCfg:
    """เงื่อนไขในการเริ่มรอบใหม่ (Done Conditions)"""
    
    # 1. หมดเวลารอบจำลอง (Timeout) เช่น ขยับเกิน 5 วินาที
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1BimanualReacherEnvCfg(ManagerBasedRLEnvCfg):
    """การประมวลรวมการตั้งค่าทั้งหมดของสภาพแวดล้อม G1 Bimanual Reacher"""
    
    # สภาพแวดล้อมในฉากจำลอง (64 Environments)
    scene: G1BimanualSceneCfg = G1BimanualSceneCfg(num_envs=64, env_spacing=3.0)
    
    # ตัวจัดการย่อย (Managers)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        """ฟังก์ชันตั้งค่าเพิ่มเติมหลังสร้าง Class"""
        self.decimation = 2  # ทำงานทุก 2 สเต็ปฟิสิกส์
        self.episode_length_s = 5.0  # รอบละ 5 วินาที
        self.step_dt = 1.0 / 60.0  # ความถี่ในการตัดสินใจของ AI (60 Hz)
        
        # ปรับความถี่ในการ Render ให้เข้ากับ Decimation
        self.sim.render_interval = self.decimation
        
        # 1. บังคับล็อกฐานล่างของหุ่นยนต์ให้อยู่กับที่ (Pelvis Fixed base) เพื่อโฟกัสแค่ร่างกายส่วนบน
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        
        # 2. เปิดระบบตรวจสอบการชน (Contact Sensors) และเปิดใช้งาน self-collisions
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        
        # 3. ตั้งค่ามุมกล้องสำหรับหน้าต่างจำลอง (Viewport Viewer)
        self.viewer.eye = [1.5, 0.0, 1.3]
        self.viewer.lookat = [0.0, 0.0, 0.95]
