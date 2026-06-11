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
