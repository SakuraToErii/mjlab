"""Unitree G1 velocity environment configurations.

Homework TODOs in this file: **3**, **4**, **5**  (of 10 total)
Function: unitree_g1_flat_height_env_cfg()
Index: docs/HOMEWORK_TODO.md · grep: 【作业 TODO
"""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import (
  UniformBaseHeightCommandCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # --- mjwarp(GPU)仿真缓冲区与求解器参数 ---
  # njmax: 每个 env 的约束(constraint)缓冲区上限。求解器把每个接触展开成
  #   若干约束(法向 + 摩擦锥)，超过此数会丢弃约束并可能报溢出。
  cfg.sim.njmax = 300
  # ccd_iterations: 凸-凸碰撞检测(GJK/CCD)迭代次数，用于 mesh-mesh 碰撞；
  #   越大越准越慢，50 已足够。
  cfg.sim.mujoco.ccd_iterations = 50
  # contact_sensor_maxmatch: 一次 forward 中每个接触传感器最多匹配的接触数，
  #   决定 ContactSensor 的 found/force 读取上限。
  cfg.sim.contact_sensor_maxmatch = 64
  # nconmax: 每个 env 的接触(contact)缓冲区上限。broadphase 生成的候选接触
  #   存这里，超容量会丢弃并报 "broadphase overflow - increase nconmax ..."。
  #   None 走启发式估值，蹲姿任务接触多时容易估少，故显式设 64(告警最低要 54，
  #   留余量)。若再报 njmax 溢出，把上面的 njmax 提到 512~1000。
  cfg.sim.nconmax = 64

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  velocity_cmd = cfg.commands["velocity"]
  assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
  velocity_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    velocity_cmd = cfg.commands["velocity"]
    assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
    velocity_cmd.ranges.lin_vel_x = (-2, 3.0)
    velocity_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  return unitree_g1_base_env_cfg(play=play)


def unitree_g1_flat_height_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity + height command configuration."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # >>> HOMEWORK_TODO_3_START
  # ==============================================================================
  # 【作业 TODO 3/10】注册 base_height 命令
  # 位置: config/g1/env_cfgs.py · unitree_g1_flat_height_env_cfg
  # 提示: 高度指令是 height 任务的输入；entity_name 应对应 scene 中的 robot。
  # 概念: UniformBaseHeightCommandCfg、resampling_time_range、ranges.height
  # 索引: docs/HOMEWORK_TODO.md · 完整说明见 docs/HW3_蹲姿行走策略.md
  # ==============================================================================
  # 注册高度指令，供策略学习不同蹲姿目标。
  cfg.commands["base_height"] = UniformBaseHeightCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    debug_vis=True,
    ranges=UniformBaseHeightCommandCfg.Ranges(
      height=(0.45, 0.80)
    )
  )

  # --- 实现提示 ---
  # - 向 cfg.commands 注册 "base_height"
  # - 参考 velocity_env_cfg.py 中 commands["velocity"] 的写法
  # - 需配置 entity_name、resampling_time_range、ranges.height（见 HW3 §2.2 / §6）
  # <<< HOMEWORK_TODO_3_END

  # >>> HOMEWORK_TODO_4_START
  # ==============================================================================
  # 【作业 TODO 4/10】接入 height_command 观测（蹲姿/高度任务核心观测）
  # 位置: config/g1/env_cfgs.py · unitree_g1_flat_height_env_cfg
  # 提示: 策略必须能观测到 base_height 指令；command_name 与 commands 键一致。
  # 概念: ObservationTermCfg、generated_commands、actor/critic 观测拼接
  # 索引: docs/HOMEWORK_TODO.md · 完整说明见 docs/HW3_蹲姿行走策略.md
  # ==============================================================================
  height_command_obs = ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "base_height"},  # TODO 4: 替换为正确的 command 名称
  )
  # actor 和 critic 都需要观测当前目标高度。
  cfg.observations["actor"].terms["height_command"] = height_command_obs
  cfg.observations["critic"].terms["height_command"] = height_command_obs

  # --- 实现提示 ---
  # - params["command_name"] 须与 TODO 3 注册的 commands 键一致
  # - 将 height_command 观测项加入 cfg.observations["actor"] 与 ["critic"] 的 terms
  # <<< HOMEWORK_TODO_4_END

  # >>> HOMEWORK_TODO_5_START
  # ==============================================================================
  # 【作业 TODO 5/10】注册 track_base_height 奖励项（蹲姿/高度任务核心奖励）
  # 位置: config/g1/env_cfgs.py · unitree_g1_flat_height_env_cfg
  # 提示: 将 track_base_height 接入奖励管理器；weight/std 已预填，无需修改。
  # 概念: RewardTermCfg、command_name="base_height"
  # 索引: docs/HOMEWORK_TODO.md · 完整说明见 docs/HW3_蹲姿行走策略.md
  # ==============================================================================
  # 高度奖励鼓励骨盆高度跟随 base_height 指令。
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=mdp.track_base_height,
    weight=2.0,
    params={
      "command_name": "base_height",
      "std": 0.05,
    },
  )
  # --- 实现提示 ---
  # - 向 cfg.rewards 注册 "track_base_height"
  # - func 指向 mdp.track_base_height；params 含 command_name 与 std（见 HW3 §6 TODO 5）
  # <<< HOMEWORK_TODO_5_END

  if play:

    height_cmd = cfg.commands["base_height"]
    assert isinstance(height_cmd, UniformBaseHeightCommandCfg)
    height_cmd.ranges.height = (0.45, 0.80)

  return cfg



def unitree_g1_flat_height_env_cfg2(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Height velocity env with reward tweaks for crouch gait.

  Differences vs ``unitree_g1_flat_height_env_cfg``:
  - Angular velocity reward tracks yaw only (no roll/pitch rate in the exp term).
  - Pose std loosens only when base_height command is below nominal stand height.
  """
  cfg = unitree_g1_flat_height_env_cfg(play=play)

  # cfg2-only: replace angular term with yaw-only tracker (base cfg keeps xy+z).
  # Roll/pitch rates remain handled by body_ang_vel / upright on this env.
  ang = cfg.rewards["track_angular_velocity"]
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_yaw_velocity,
    weight=ang.weight,
    params={
      "command_name": ang.params["command_name"],
      "std": ang.params["std"],
    },
  )

  # Keep base speed std tables; only crouch tables are looser on lower body.
  # Blend: height_cmd >= height_nominal → base std; height_cmd <= height_crouch → crouch std.
  pose = cfg.rewards["pose"]
  crouch_standing = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.12,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  crouch_walking = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.55,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.12,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  crouch_running = {
    r".*hip_pitch.*": 0.65,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.7,
    r".*ankle_pitch.*": 0.4,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.25,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"] = RewardTermCfg(
    func=mdp.variable_posture_height,
    weight=pose.weight,
    params={
      **pose.params,
      "height_command_name": "base_height",
      # Matches height command upper end / near standing root height.
      "height_nominal": 0.80,
      "height_crouch": 0.45,
      "std_crouch_standing": crouch_standing,
      "std_crouch_walking": crouch_walking,
      "std_crouch_running": crouch_running,
    },
  )

  return cfg
