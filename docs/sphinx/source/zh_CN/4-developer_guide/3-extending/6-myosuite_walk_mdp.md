# MyoSuite Walk 任务 MDP 设计方案

这页整理 `MyoSuite` 的 `myoLegWalk-v0` 在 UniLab 中的迁移方案。目标不是马上定死
所有实现细节，而是先把任务语义说清楚，避免在“像 walk”但“已不是原任务”的中间态里
反复返工。

相关背景可先看：

- {doc}`5-task_mdp_and_frames`

## 结论先行

推荐先落一条 **faithful-first, contract-aligned** 路线：

- 任务名先定为 `myoleg_walk_flat`。
- 初版只支持 `mujoco` backend。
- 初版优先复现 `myoLegWalk-v0` 的 mixed-frame observation、奖励、终止和 reset
  语义。
- 不建议一开始就把它硬改造成现有 locomotion 模板那种“纯 local actor + PD joint
  target”任务。

原因很简单：`myoLegWalk-v0` 的核心不是“有个 walk reward”，而是
“肌肉驱动 + mixed-frame observation + 节律先验 + world-fixed velocity target”这一
整套组合。

## 原任务语义速记

`myoLegWalk-v0` 注册参数的关键事实：

- `max_episode_steps = 1000`
- `min_height = 0.8`
- `max_rot = 0.8`
- `hip_period = 100`
- `reset_type = "init"`
- `target_x_vel = 0.0`
- `target_y_vel = 1.2`
- `target_rot = None`

对应源码位置：

- `walk_v0.py` 中的 `WalkEnvV0`
- `myobase/__init__.py` 中的 `myoLegWalk-v0` 注册

### 任务目标

原任务目标不是“沿当前身体前方走”，而是更接近：

- 在世界系下保持参考朝向；
- 将质心速度跟踪到 `x=0.0, y=1.2 m/s`；
- 同时维持一定的步态节律和姿态约束。

这意味着它本质上是一个 **world-fixed velocity tracking** 任务，而不是纯 body-local
  前进任务。

## 推荐的 MDP 拆解

## 状态 state

从真实 MDP 看，至少包含这些物理量：

- 根部 free joint 的完整位姿与速度
- 全部关节位置与速度
- 全部肌肉 actuator internal state
- 刚体质心、身体姿态、接触引起的隐含动力学状态
- 任务时钟 `phase_var`

如果未来引入 `fatigue` / `sarcopenia` 变体，还要把对应肌肉内部状态视为状态的一部分。

对 UniLab 而言，这层通常不直接暴露给 actor，但在设计 reward、termination、reset 时
要以它为准。

## 动作 action

原任务动作不是关节 PD target，而是 **肌肉控制输入**。

建议初版动作定义：

- `action[t]`: 每个 muscle actuator 的归一化控制输入
- 维度：`nu == actuator count`
- 初版语义：尽量保持与 MyoSuite 一致的 muscle control path

这里有一个非常重要的实现提醒：

- `MyoSuite` 在 `normalize_act=True` 时，会对 muscle actuator 走一条显式的非线性
  投影路径，而不是简单线性映射。
- 这条路径属于原任务控制语义的一部分，初版 faithful migration 不建议“顺手修正”为
  UniLab 里更好看的动作定义。

换句话说，**如果初版目标是复现原任务，不要把 muscle action 偷偷改成 joint target
或线性 `[0,1]` excitation。**

## 观测 observation

### 设计原则

推荐把 `myoLegWalk-v0` 初版 actor observation 视为 **mixed-frame observation**，
而不是强行归类成纯 local 或纯 world。

建议为后续实现先整理一张字段表：

| 字段 | 原始语义 | frame 标签 | 是否保留到 actor |
|------|----------|------------|------------------|
| `qpos_without_xy` | 去掉根部 world `x/y` 的 `qpos` | `J + W-P + W-O` 混合 | 是 |
| `qvel` | 全部广义速度乘 `dt` | `J + W-V` 混合 | 是 |
| `com_vel` | 质心线速度的 `x/y` 分量 | `W-V` | 是 |
| `torso_angle` | torso 四元数 | `W-O` | 是 |
| `feet_heights` | 双脚 world `z` | `W-P` | 是 |
| `height` | 质心高度 | `W-P` | 是 |
| `feet_rel_positions` | 双脚减 pelvis 的相对位移，但轴仍在 world | `W-P` | 是 |
| `phase_var` | 步态时钟 | `T` | 是 |
| `muscle_length` | 肌肉长度 | `J` | 是 |
| `muscle_velocity` | 肌肉速度 | `J` | 是 |
| `muscle_force` | 肌肉力 | `J` | 是 |

### Actor 方案

初版推荐：

- actor 先保留与原任务等价的 mixed-frame observation；
- 不额外删除 `torso_angle`、`com_vel` 这类世界系信息；
- 不把 `feet_rel_positions` 误当成 body-local 量。

这样做的理由是：原任务 reward 本身就依赖 world-fixed 速度目标和参考姿态，如果
observation 先被本地化重写，任务就已经不是原任务了。

### Critic 方案

初版最稳妥的选择是：

- `critic_obs = actor_obs`

这样更容易先对齐任务语义，再决定是否要加 privileged critic。

第二阶段如果要增强训练稳定性，可以额外给 critic 补：

- 根部 world `x/y` 位置
- 更多 body world pose / velocity
- 接触或外力统计

但这些应视为训练技巧，而不是初版任务定义本身。

## 奖励 reward

原任务默认 dense reward 是加权和：

| 项 | 含义 | 默认权重 | 说明 |
|----|------|----------|------|
| `vel_reward` | world `x/y` 速度跟踪 | `5.0` | 核心任务项 |
| `done` | 失败惩罚 | `-100` | 由 termination 触发 |
| `cyclic_hip` | 髋关节周期参考 | `-10` | 本质是节律误差惩罚 |
| `ref_rot` | 根部姿态接近参考旋转 | `10.0` | 维持参考朝向 |
| `joint_angle_rew` | 特定髋关节角接近 0 | `5.0` | 抑制特定姿态偏差 |

### 推荐保留的 reward 定义

初版建议一项不删地保留五项默认 reward，并沿用原公式。

特别注意两点：

- `cyclic_hip` 返回的是误差范数，本身不是 reward bonus，而是靠负权重形成惩罚。
- `done` 不是独立 termination API 外的另一个逻辑，而是 termination 信号在 reward
  里的显式成本。

### reward 语义解读

把它翻成任务语言，大概就是：

- 我要你以世界系目标速度前进；
- 我要你别倒；
- 我要你保持接近参考朝向；
- 我要你带一点预设步态周期；
- 我要你抑制某些髋关节偏转。

所以这不是“纯速度跟踪 walk”，而是一个带**姿态与节律先验**的肌肉步行任务。

## 终止 termination 与截断 truncation

### termination

初版建议忠实保留以下失败条件：

- `com_height < 0.8`
- 根部旋转条件超过阈值 `max_rot = 0.8`

这里第二项不是普通的“roll/pitch 大小”，而是通过 root quaternion 旋转后的方向向量来
判断姿态是否偏离参考方向。实现时要保留它的原始判据，不要擅自改成欧拉角阈值。

### truncation

建议单独表达：

- `time_limit = 1000 env steps`

不要把 time limit 和 failure termination 混成同一个 `done` 语义。

## reset 分布

原任务 `reset_type` 支持三种语义：

- `init`: 使用 keyframe 2
- `random`: 在 keyframe 2 / 3 之间随机，并对 `qpos` 注入小高斯噪声，但保持 root
  height 与 root rotation 不变
- 其他：退回 keyframe 0

对 UniLab 初版，建议：

- 默认只实现并暴露 `init`
- 若要支持数据增强式 reset，再补 `random`

这样能减少一开始的变量数量，更容易验证 reward/termination 是否已对齐。

## 推荐的迁移路线

## 路线 A：faithful migration

这是推荐的第一阶段方案。

### 任务定义

- 任务名：`myoleg_walk_flat`
- backend：仅 `mujoco`
- 控制：muscle actuator control
- actor obs：保留 mixed-frame
- critic obs：先与 actor 相同
- reward：保留默认五项及权重
- termination：保留原阈值
- truncation：`1000 steps`
- reset：先支持 `init`

### 适用场景

- 想验证 UniLab 是否能承载 MyoSuite 这类肌肉任务
- 想做语义对齐测试
- 想把“仿真差异”和“任务重写差异”拆开

### 优点

- 语义最清晰
- 最容易与原任务做逐项对照
- 更容易定位偏差来自 backend、动作语义还是 reward

### 代价

- actor observation 不够“UniLab locomotion 风格”
- 需要补齐 muscle 相关 backend contract / env 适配能力

## 路线 B：UniLab-native redesign

这是第二阶段才建议考虑的路线。

### 可以改什么

- actor observation 改成 local / yaw-aligned
- 把 world pose 更多地下放到 critic
- 将 `phase_var` 改成显式 gait command 或去掉节律先验
- 重新审视 reward，把“预设步态”改为更开放的 locomotion 目标

### 什么时候做

- faithful migration 已稳定
- 已经确认需要与现有 UniLab locomotion 任务族风格统一
- 目标是长期维护的新任务，而不是复现 benchmark

## 不建议的中间态

以下做法最容易制造误解：

- 名字叫 `myosuite walk`，但动作已经换成 joint PD
- 奖励还在跟踪 world 速度，actor 却只给纯 local 信息
- 把 `feet_rel_positions` 误记成 body-local 特征
- 为了“更标准”把 root quaternion 判据改成另一套 termination
- 一边说要 faithful，一边删掉 `phase_var` 或 `ref_rot`

这类方案表面看起来更“工程化”，但很难再回答“你复现的到底是不是原任务”。

## 对 UniLab 契约的直接要求

如果要真的开始实现，至少需要先明确这些契约点：

- backend / env 层是否能稳定提供 muscle length / velocity / force
- 动作语义是否允许 muscle-specific control path
- `obs_groups_spec` 如何表达 actor / critic 观测分组
- reward 是否通过 owner YAML 注入
- `--sim mujoco` 是否作为唯一初版入口

如果这些点没先定清楚，不建议直接开写 env 代码。

## 一页版实施建议

如果后续进入实现，我建议按这个顺序推进：

1. 先定义 `myoleg_walk_flat` 的 owner YAML 与任务身份。
2. 明确 muscle 观测与动作所需的 `SimBackend` 抽象接口。
3. 实现 faithful 版 actor observation。
4. 逐项对齐 reward、termination、reset。
5. 为 mixed-frame observation 和 termination 判据补最小测试。
6. 只有在 faithful 版本稳定后，再讨论 UniLab-native redesign。

一句话总结：

**MyoSuite walk 的正确打开方式，不是先把它改成 UniLab 现有 locomotion 模板，而是先把它当成一个“肌肉驱动、mixed-frame、world-fixed velocity tracking”的原始任务，完整迁进来，再决定哪些地方值得重构。**

## 最终实施方案

初版实现采用 **faithful-first, contract-aligned** 路线。目标是在 UniLab 中先复现
`myoLegWalk-v0` 的原始 MDP，而不是直接重写成 UniLab-native locomotion 任务。

### 任务身份

- 任务名：`myoleg_walk_flat`
- 初版 backend：仅 `mujoco`
- 初版训练入口：先接 `ppo` owner YAML，后续再按需要补 `sac` / `appo`
- 任务归属：建议放在 locomotion 下的 muscle 子族，例如
  `src/unilab/envs/locomotion/myoleg/`

### MDP 约束

- action 保持 MyoSuite muscle actuator control，维度等于 muscle actuator 数量 `nu`。
- 如果原任务 `normalize_act=True` 走非线性 action 投影路径，初版应复现该路径。
- actor observation 保持 mixed-frame，不强行 local 化。
- critic observation 初版等于 actor observation。
- reward 保留原五项及默认权重：`vel_reward=5.0`、`done=-100`、
  `cyclic_hip=-10`、`ref_rot=10.0`、`joint_angle_rew=5.0`。
- termination 保留 `com_height < 0.8` 与原 root quaternion 方向判据
  `max_rot=0.8`。
- truncation 单独表达为 `1000 env steps`。
- reset 支持 `reset_type="init"` 与 `reset_type="random"`。`init` 使用 keyframe 2；
  `random` 在 keyframe 2 / 3 间随机，并对 `qpos` 加 0.02 高斯噪声，但保持 root
  height 和 root quaternion 不变。

### 坐标系记录要求

每个 MDP 字段都必须同时记录“数值来源”和“表达坐标系”。对 MyoSuite walk，尤其要
避免把 world 轴上的相对量误认为 body-local 量。

| 字段 | 坐标 / 语义系 | 初版迁移要求 |
|------|---------------|--------------|
| `qpos_without_xy` | joint + MuJoCo world pose 混合 | 去掉 world `x/y`，保留 root `z` 和 root quaternion |
| `qvel * dt` | joint + MuJoCo world velocity 混合 | 不转 body frame |
| `com_vel` | MuJoCo world velocity | 使用 MyoSuite 的 `-mj_data.cvel` 后取 `[3:5]`，对齐 world `x/y` target |
| `torso_angle` | MuJoCo world orientation | 使用 `mj_data.xquat[torso]` / world frame sensor，保留四元数顺序 `wxyz` |
| `feet_heights` | MuJoCo world `z` | 不 local 化 |
| `feet_rel_positions` | world-axis relative displacement | 只做 `foot_pos_w - pelvis_pos_w`，不是 body-local |
| `phase_var` | task clock | actor 可见 |
| `muscle_length` / `muscle_velocity` / `muscle_force` | actuator intrinsic | 通过 backend contract 暴露 |

坐标系审计后的关键结论：

- 原始 `myoLegWalk-v0` 是 world-fixed velocity tracking。`target_y_vel=1.2`
  对应 MuJoCo world `+Y`，不是机器人 body-local forward，也不是 world `+X`。
- 原始 reset keyframe 的 `qvel[1]` 为负，但 MyoSuite 的 COM 速度公式是
  `mass * (-mj_data.cvel)` 后取 `[3:5]`，因此初始 `com_vel[1]` 为正，并与
  `target_y_vel` 同向。
- `feet_rel_positions` 只是位置差 `xpos[talus] - xpos[pelvis]`，表达轴仍是 world
  axis；不能再乘 root inverse quaternion 转成 pelvis/body-local。
- `torso_angle` 是 torso body 的 world quaternion，不是 root quaternion，也不是 Euler
  angle。
- termination 的旋转判据使用 root quaternion `qpos[3:7]` 作用在 `[1,0,0]` 后的
  world `x` 分量：`abs((quat2mat(root_quat) @ [1,0,0])[0]) > max_rot`。
- `reset_type="init"` 的物理初始状态来自 keyframe 2；但默认 `target_rot=None`
  时，MyoSuite 的 `ref_rot` 参考来自 setup 阶段的 `init_qpos = key_qpos[0]`。
  UniLab 迁移必须保留这个区别，不能把 ref rotation reference 改成 reset keyframe 2。

### UniLab step 生命周期适配

UniLab env 对外只有一个主 rollout 生命周期：`NpEnv.step(actions)`。

```text
apply_action(actions, state)
backend.step(ctrl, sim_substeps)
update_state(state)
_compute_truncated(state)
autoreset done envs
```

因此，MyoSuite 原任务里分散在 `forward` / observation / reward / done 的逻辑，
需要收敛到 UniLab 的 `apply_action -> backend.step -> update_state` 节奏里。
MuJoCo `BatchEnvPool.forward(...)` 只用于初始化或冷路径 sensor cache 刷新；正常
rollout 不新增 env-level `forward` lifecycle。

### Backend contract 要求

实现前必须先把共享 env 需要的 muscle 能力加入 `SimBackend` 抽象接口。MuJoCo backend
实现这些接口，Motrix 初版保持显式 `NotImplementedError`。

初版至少需要明确：

- muscle length
- muscle velocity
- muscle force
- muscle actuator id / control range 映射
- full `qpos` / `qvel`
- indexed keyframe `qpos` / `qvel`
- world COM position / velocity

env 层不得直接读取 MuJoCo 私有字段，也不得在热路径解析 XML / asset / model metadata。

### MyoSuite 包安装结论

当前 UniLab 环境不能把 MyoSuite 作为同一个 uv 环境里的普通依赖直接用于端到端 diff。

实测 `uv pip install myosuite` 会额外安装官方 `mujoco==3.6.0`。这会覆盖
UniLab 依赖的 `mujoco-uni==3.8.0` runtime，并导致 MuJoCo backend 在
`BatchEnvPool` materialization 时失败，例如：

```text
mujoco.FatalError: mj_copyModel: dest and src models have different buffer size
```

卸载 `myosuite` / `mujoco` 后，还需要强制重装 `mujoco-uni==3.8.0` 才能恢复
UniLab backend runtime。

因此当前对齐策略是：

- 资产层复用 MyoSuite 原始 XML / mesh / keyframe。
- MDP 公式用仓库内 MuJoCo 参考计算测试锁定。
- 若后续要做 MyoSuite package 级端到端数值 diff，应放到隔离环境或独立进程中，
  不能污染 UniLab 主 uv 环境。

### 原始 XML 迁入与必要修改

MyoLeg 初版资产以 MyoSuite / `myo_sim` 的原始 `leg/myolegs.xml` 为基线迁入：

- 入口 XML：`src/unilab/assets/robots/myoleg/leg/myolegs.xml`
- 原始依赖：`leg/assets/*`、`torso/assets/myotorso_rigid_*`、
  `scene/myosuite_scene_noPedestal.xml`
- 外部资源：只带入该模型实际引用的 scene 贴图 / msh 与 leg、torso mesh
- 许可证：保留 upstream Apache-2.0 `LICENSE`

迁移时遵循“MDP 语义不动，只修本地可加载性”的原则：

- 保留原 body / joint / actuator / tendon / muscle / keyframe 内容，尤其是
  keyframe 2 / 3、`pelvis`、`torso`、`talus_l`、`talus_r` 与 80 个 muscle
  actuator。
- 不把 `<keyframe>` 移到 robot fragment。这里 `myolegs.xml` 本身就是
  task-level 入口 XML，keyframe 属于任务初始状态，符合 UniLab asset 约束。
- 只调整 XML 里的 asset 路径，使其在 UniLab 仓库内可通过绝对 `model_file`
  稳定编译。原因是 upstream XML 假设包外存在 `../myo_sim/...` 布局；UniLab
  需要资产自包含在 `src/unilab/assets/robots/myoleg/` 下。
- MuJoCo actuator/body sensors 不写入原始资产，而是在 backend 初始化冷路径注入到
  临时 XML。原因是这些 sensor 是 UniLab backend contract 的观测缓存机制，不属于
  MyoSuite 原始 MDP。
- `terrain` hfield 保留在原 XML 中；MyoSuite 原任务在 setup 中把它下移隐藏。UniLab
  不改原始资产，而是在 MuJoCo backend 初始化冷路径把 `terrain` 物化到
  `pos="0 0 -10"`、`rgba_alpha=0` 的临时 XML 中。这样 flat walk 只与 floor
  接触，不会出现 floor 与 terrain 双接触，同时避免在 env hot path 解析 XML 或修改
  geom metadata。

这样做的好处是：训练任务用真实 MyoLeg muscle 模型和原始 keyframe 起步，同时所有
UniLab 所需的工程适配都集中在配置、asset 路径和 backend 冷路径中，不污染 env step
热路径。

### 实施顺序

1. 建立 `myoleg_walk_flat` env config、注册与 `mujoco.yaml`。
2. 梳理 MyoSuite 原始 `myoLegWalk-v0` 的 actuator、sensor/body/site、keyframe、
   reward 字段映射。
3. 给 `SimBackend` 增加 muscle 相关抽象接口。
4. 实现 MuJoCo backend muscle contract，Motrix 保持显式不支持。
5. 实现 faithful action normalization / control path。
6. 实现 mixed-frame actor observation。
7. 实现五项 reward、两项 termination 与 time-limit truncation。
8. 实现 `init` / `random` reset，并验证 keyframe 2 / 3 初始状态与 root
   height / quaternion 保持逻辑。
9. 接入小规模 PPO smoke。

### 最小验收测试

- `obs_groups_spec` 稳定，`NpEnvState.obs` 是 dict。
- action dim 等于 MuJoCo muscle actuator 数量。
- muscle length / velocity / force 通过 `SimBackend` 获取。
- `qpos_without_xy` 和 `qvel * dt` 分别来自 backend full `qpos` / `qvel`。
- `feet_rel_positions` 保持 world-axis relative displacement。
- `vel_reward` 使用 MuJoCo world `x/y` velocity target。
- termination 使用原 quaternion 方向判据。
- time limit 进入 `truncated`，失败进入 `terminated`。
- `reset_type="init"` 使用正确 keyframe；`reset_type="random"` 保持 root height /
  quaternion 并扰动其他 `qpos`。
- 单环境 step smoke 与小规模 vectorized smoke 通过。
- 真实 MyoLeg XML 下，observation 各 slice 与 MyoSuite MDP 公式重算值一致。
- 真实 MyoLeg XML 下，五项 reward 分项与总 reward 公式一致。
- PPO owner YAML 使用 RSL-RL wrapper 暴露的 `actor` / `critic` observation set。
- faithful timing 使用 MyoSuite 的 `model.opt.timestep=0.001` 与
  `frame_skip=10` 等价设置，即 UniLab `sim_dt=0.001`、`ctrl_dt=0.01`、
  `sim_substeps=10`。
- step 后开启 `post_step_forward_sensor=true`，使 MuJoCo sensor-backed obs 与
  MyoSuite `forward()` 后语义一致。

### 隔离 MyoSuite 数值 diff 结论

MyoSuite package 级 diff 必须在隔离环境中执行，不能安装到 UniLab 主 uv 环境。当前
隔离 diff 使用 `myosuite==2.12.2` 与官方 `mujoco==3.6.0`，只把参考 JSON 写到
`/tmp/unilab-myoleg-myodiff`。

关键结论：

- reset 后核心 MDP 字段与 reward 基本一致：reset dense reward 为
  MyoSuite `12.025755215085287`，UniLab `12.025754928588867`。
- MyoSuite 底层 `model.opt.timestep=0.001`，`frame_skip=10`，control dt 为
  `0.01`。若 UniLab 使用 `sim_dt=0.005`、`sim_substeps=2`，zero-action 一步后的
  reward 会偏离：MyoSuite `12.063525532041009`，UniLab `11.94792652130127`。
- 将 UniLab owner config 改为 `sim_dt=0.001`、`sim_substeps=10` 后，一步后的
  return reward 对齐到 `12.063526153564453`。
- 进一步开启 `post_step_forward_sensor=true` 后，step 后核心 obs components
  也回到 `1e-8` 量级误差。
- MyoSuite wrapper 输出 obs 为 `403` 维，因为 `BaseV0` 会自动追加 `act` 80 维；
  UniLab 已通过 backend activation contract 纳入 `act`，actor / critic obs 也对齐为
  `403` 维。`act` reset diff 为 `0`，step 后 diff 约为 `4.5e-9`。

当前最小测试版本已通过：

```bash
uv run pytest tests/envs/locomotion/myoleg/test_myoleg_walk_contract.py \
  tests/config/test_config_system.py tests/base/test_sim_backend_smoke.py \
  tests/training/test_training_helpers.py tests/utils/test_xml_utils.py \
  -q -k 'myoleg or actuator_sensors or model_variants_preserve_actuator_sensors or indexed_keyframe or fixed_base_dof_views or backend_adapter or geom_overrides or post_step_forward_sensor'
```

结果：`43 passed, 153 deselected`。

最小 PPO smoke 也已通过：

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

结果：成功完成 `Learning iteration 0/1`，actor / critic 输入维度为 `403`，
action dim 为 `80`。

faithful profile 下的 20-iteration PPO smoke 也已通过：

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

结果：在 `act` 纳入 observation、actor / critic 输入维度对齐到 `403` 后，完成
`Learning iteration 19/20`，未触发 NaN guard。mean reward 约从 `30.54`
上升到 `58.00`，mean episode length 约为 `33.75`。首次运行暴露出的
`reset(env_indices)` 返回全量 obs 的 autoreset contract bug 已修复为只返回
requested env subset，并用 partial-done autoreset 测试覆盖。
