# MyoSuite Walk 阶段性实施记录

本文记录 `myoLegWalk-v0` 迁移到 UniLab 的当前阶段成果。它是阶段性工程文档：
说明已经改了什么、为什么这样改、验证到什么程度，以及后续还需要继续收敛的问题。

方案与 MDP 拆解见 {doc}`6-myosuite_walk_mdp`。

## 阶段目标

当前阶段采用 **faithful-first, contract-aligned** 路线：

- 先复用 MyoSuite 原始 MyoLeg XML 物理模型。
- 先保持 `myoLegWalk-v0` 的动作、观测、奖励、终止和 reset 语义。
- 所有 MuJoCo 特有能力通过 backend contract 暴露，env 热路径不解析 XML、不读取
  backend 私有字段。
- 初版只支持 MuJoCo backend，任务入口为 `task=myoleg_walk_flat/mujoco`。

当前阶段不是训练出最终 policy，而是交付一个可构建、可 step、可最小 PPO smoke、
并有数值对齐测试保护的 MVP。

## 已完成改动

### 任务与配置

- 新增 `MyoLegWalkFlat` env config 与 env 注册。
- 新增 PPO owner YAML：`conf/ppo/task/myoleg_walk_flat/mujoco.yaml`。
- `training.task_name = MyoLegWalkFlat`，`training.sim_backend = mujoco`。
- action dim 直接来自 MuJoCo actuator count，当前真实 MyoLeg XML 下为 `80`。
- PPO observation set 对齐 RSL-RL wrapper，使用 `actor: [actor]` 与
  `critic: [critic]`。

### 原始资产迁入

MyoLeg 资产以 MyoSuite / `myo_sim` 的原始 `leg/myolegs.xml` 为基线迁入：

- `src/unilab/assets/robots/myoleg/leg/myolegs.xml`
- `src/unilab/assets/robots/myoleg/leg/assets/*`
- `src/unilab/assets/robots/myoleg/torso/*`
- `src/unilab/assets/robots/myoleg/scene/*`
- `src/unilab/assets/robots/myoleg/myo_sim/*`
- upstream `LICENSE`

保留了原 body / joint / tendon / muscle / actuator / keyframe 语义。只调整 XML
include / asset 路径，使模型在 UniLab 仓库内自包含加载。

### Backend contract

`SimBackend` 增加了 MyoLeg 需要的通用能力：

- actuator names
- actuator length / velocity / force
- indexed keyframe `qpos` / `qvel`
- full `qpos` / `qvel`
- COM position
- COM velocity xy

MuJoCo backend 实现这些接口。actuator sensors、body tracking sensors、subtree COM
sensor 通过 backend 冷路径临时 XML 注入，不修改原始资产。

### Env MDP 实现

`MyoLegWalkFlatEnv` 当前实现了：

- MyoSuite muscle action normalization。
- `reset_type="init"`：使用 keyframe 2。
- `reset_type="random"`：在 keyframe 2 / 3 间随机，`qpos` 加 `0.02` 高斯噪声，
  保持 root height 与 root quaternion。
- observation 组件按 MyoSuite 顺序拼接，并保留 `_obs_slices` 便于测试。
- reward 五项：
  `vel_reward`、`done`、`cyclic_hip`、`ref_rot`、`joint_angle_rew`。
- termination：
  COM height `< min_height`，以及 MyoSuite root quaternion 方向判据。
- time-limit truncation 继续走 UniLab `NpEnv` contract。

### 坐标系对齐

本阶段专门复核了 XML、MyoSuite 源码和 UniLab 实现的坐标系方向。

关键结论：

- `target_y_vel=1.2` 是 MuJoCo world `+Y` target，不是 body-local forward。
- XML keyframe 2/3 的 root `qvel[1] = -1.5`，但 MyoSuite COM velocity 公式使用
  `-mj_data.cvel` 后取 `[3:5]`，因此 reward 看到的 `com_vel[1]` 为正。
- `feet_rel_positions` 是 `xpos[talus] - xpos[pelvis]`，表达轴仍是 world axis，
  不是 pelvis/body-local。
- `torso_angle` 是 torso body world quaternion，四元数顺序为 MuJoCo `wxyz`。
- termination 的旋转项使用 root quaternion `qpos[3:7]` 作用 `[1,0,0]` 后的
  world `x` 分量。
- `ref_rot` 的默认参考姿态不是 reset keyframe 2，而是 MyoSuite setup 阶段的
  `key_qpos[0][3:7]`。UniLab 已单独保存 `_ref_qpos`，reset 初始状态和
  reference rotation 不混用。

## MyoSuite 包安装结论

当前不建议把 `myosuite` 直接安装进 UniLab 主 uv 环境做端到端 diff。

实测 `uv pip install myosuite` 会拉入官方 `mujoco==3.6.0`，覆盖 UniLab 当前依赖的
`mujoco-uni==3.8.0` runtime，导致 MuJoCo backend `BatchEnvPool` materialization
失败。

当前策略：

- 资产层复用 MyoSuite 原始 XML。
- MDP 公式用仓库内 MuJoCo 参考计算测试锁定。
- 后续如需 MyoSuite package 级数值 diff，应使用隔离环境或独立进程，不能污染
  UniLab 主环境。

## 已验证内容

### Targeted regression

命令：

```bash
uv run pytest tests/envs/locomotion/myoleg/test_myoleg_walk_contract.py \
  tests/config/test_config_system.py tests/base/test_sim_backend_smoke.py \
  tests/training/test_training_helpers.py tests/utils/test_xml_utils.py \
  -q -k 'myoleg or actuator_sensors or model_variants_preserve_actuator_sensors or indexed_keyframe or fixed_base_dof_views or backend_adapter or geom_overrides or post_step_forward_sensor'
```

当前结果：

```text
43 passed, 153 deselected
```

覆盖范围：

- env 注册与 MuJoCo-only 约束。
- actuator sensor contract。
- indexed keyframe contract。
- full `qvel` contract。
- temporary XML `geom_overrides` contract。
- MyoLeg-specific `post_step_forward_sensor` contract。
- model-variant path preserves actuator sensor injection。
- 真实 MyoLeg XML 构建、reset、step。
- action dim 等于 `80`。
- actor / critic observation dim 等于 `403`，包含 MyoSuite wrapper 自动追加的
  `act` 80 维。
- observation slices 与 MyoSuite MDP 公式重算值一致。
- COM position / velocity 与 MyoSuite 公式一致。
- reward 分项与总 reward 公式一致。
- world `+Y` velocity direction 与 `target_y_vel` 对齐。
- `ref_rot` 使用 keyframe 0 reference。
- `reset_type="random"` 保持 root height / quaternion。
- subset reset / autoreset contract：`reset(env_indices)` 只返回被 reset env 的 obs，
  避免 partial done autoreset 时全量 obs 回填导致 shape mismatch。

### Ruff

命令：

```bash
uv run ruff check src/unilab/envs/locomotion/myoleg/walk.py \
  tests/envs/locomotion/myoleg/test_myoleg_walk_contract.py
uv run ruff format --check src/unilab/envs/locomotion/myoleg/walk.py \
  tests/envs/locomotion/myoleg/test_myoleg_walk_contract.py
```

结果均通过。

### 最小 PPO smoke

命令：

```bash
uv run python scripts/train_rsl_rl.py task=myoleg_walk_flat/mujoco \
  algo.num_envs=2 algo.num_steps_per_env=2 algo.max_iterations=1 \
  algo.save_interval=1 algo.empirical_normalization=false \
  training.no_play=true training.logger=tensorboard \
  training.nan_guard.enabled=false training.log_root=/tmp/unilab_myoleg_smoke \
  algo.policy.actor_hidden_dims='[32,16]' \
  algo.policy.critic_hidden_dims='[32,16]' \
  algo.algorithm.num_learning_epochs=1 algo.algorithm.num_mini_batches=1
```

结果：成功完成 `Learning iteration 0/1`。

观测 / 动作维度：

- actor input dim：`403`
- critic input dim：`403`
- action dim：`80`

### Faithful PPO smoke

在 faithful timing（`sim_dt=0.001`、`sim_substeps=10`、
`post_step_forward_sensor=true`）下执行了更长一点的 smoke：

```bash
uv run python scripts/train_rsl_rl.py task=myoleg_walk_flat/mujoco \
  algo.num_envs=16 algo.num_steps_per_env=16 algo.max_iterations=20 \
  algo.save_interval=20 algo.empirical_normalization=false \
  training.no_play=true training.logger=tensorboard \
  training.nan_guard.enabled=true training.log_root=/tmp/unilab_myoleg_act_faithful_smoke \
  algo.policy.actor_hidden_dims='[64,32]' \
  algo.policy.critic_hidden_dims='[64,32]' \
  algo.algorithm.num_learning_epochs=1 algo.algorithm.num_mini_batches=1
```

结果：完成 `Learning iteration 19/20`，总步数 `5120`，未触发 NaN guard。训练过程中
mean reward 约从 `30.54` 上升到 `58.00`，mean episode length 约为 `33.75`。

### Stability PPO smoke

随后执行了一次 50-iteration stability smoke，重点验证 403 维 observation、faithful
timing、NaN guard、autoreset 与 episode 统计在更长 rollout 下是否稳定：

```bash
uv run python scripts/train_rsl_rl.py task=myoleg_walk_flat/mujoco \
  algo.num_envs=32 algo.num_steps_per_env=32 algo.max_iterations=50 \
  algo.save_interval=50 algo.empirical_normalization=false \
  training.no_play=true training.logger=tensorboard \
  training.nan_guard.enabled=true training.log_root=/tmp/unilab_myoleg_stability_smoke \
  algo.policy.actor_hidden_dims='[64,32]' \
  algo.policy.critic_hidden_dims='[64,32]' \
  algo.algorithm.num_learning_epochs=1 algo.algorithm.num_mini_batches=2
```

结果：完成 `Learning iteration 49/50`，总步数 `51200`，未触发 NaN guard。actor /
critic 输入维度为 `403`，action dim 为 `80`。mean reward 约从 `36.85`
上升到 `61.18`，mean episode length 从约 `30.87` 稳定到约 `34.92`。

## 当前风险与限制

- 这是 MuJoCo-only MVP，Motrix 尚未实现 MyoLeg muscle contract。
- MyoSuite package 级 diff 已在隔离环境 `/tmp/unilab-myoleg-myodiff` 中执行，避免污染
  UniLab 主 uv 环境。结论是 reset 后核心 MDP 字段基本一致；为对齐 step dynamics，
  MyoLeg owner config 采用 MyoSuite faithful timing：`sim_dt=0.001`、
  `ctrl_dt=0.01`、`sim_substeps=10`，并开启 `post_step_forward_sensor=true`。
  这会牺牲 rollout 性能，但只作用于 `myoleg_walk_flat/mujoco` owner YAML。
- `post_step_forward_sensor=true` 不是 MuJoCo 任务的全局默认。G1 是 UniLab-native
  任务，当前 obs / reward contract 接受 backend step 返回的 sensor cache；强制开启
  会改变既有训练分布并增加一次 sensor forward 开销。MyoLeg 开启该选项是因为迁移目标
  是复刻 MyoSuite `step()->forward()` 后的 sensor-backed observation / reward 语义。
- 原 XML 的 terrain hfield 已保留，但 UniLab 在 MuJoCo backend 初始化冷路径复现
  MyoSuite flat walk 的 terrain hiding：临时 XML 将 `terrain` 下移到 `z=-10`
  并保持透明，避免与 floor 同时参与接触。原始资产不改，env hot path 也不解析 XML。
- 当前 reward / obs 对齐到 MyoSuite 公式和 faithful timing，但还没有做长训练稳定性评估。
- 原始 MyoSuite XML 资产较多，后续可以继续裁剪未引用资产，但必须保证 XML 编译和
  license 保留。

## 下一步建议

1. 做更长的 PPO smoke，观察 reward 曲线、NaN guard、episode length。
2. 梳理资产依赖，删除确认未引用的冗余 scene / mesh 文件。
3. 若要支持 Motrix，先扩展 Motrix backend muscle actuator contract，不要在 env
   层分支读取 backend 私有能力。
