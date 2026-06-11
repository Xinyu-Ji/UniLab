# 新任务 MDP 与坐标系速记

这页给后续讨论新任务时快速对齐用。重点不是推导完整 RL 理论，而是避免在
`state / observation / reward / termination` 之间混淆 local 坐标系与 world
坐标系，导致任务语义在迁移时悄悄变化。

## 为什么要先讲坐标系

新任务最常见的误解不是“奖励系数写错”，而是：

- 以为某个速度量是局部坐标系，实际上是世界系。
- 以为某个“相对位置”已经是 body-local，实际上只是世界系下做了减法。
- 以为 observation、reward、termination 用的是同一 frame，实际上是混合的。

这些误解会直接影响：

- actor 是否真的可观；
- reward 是否与 observation 对齐；
- 任务到底是在复现原任务，还是悄悄变成了另一个任务。

## 先分清 3 层

### 1. 物理状态 state

仿真器内部真实状态。通常以 joint coordinates、world pose、world velocity、
actuator internal state 为主。

这层决定真实 MDP，但不要求策略直接观测到。

### 2. 观测 observation

策略看到的输入。可以是：

- world frame
- body-local frame
- yaw-aligned frame
- 任务自定义 anchor frame
- 混合 frame

观测的 frame 设计会改变学习难度、对称性和泛化性。

### 3. 奖励与终止 reward / termination

这是任务判据，不必与 observation 使用同一个 frame。

但如果 reward 用世界系，而 observation 只给纯 local 信息，就要确认策略仍然能
恢复完成任务所需的方向信息；否则任务会变成隐式 POMDP。

## 建议统一的 frame taxonomy

后续讨论每个字段时，建议都打上一个 frame 标签。

| 标签 | 含义 | 典型量 |
|------|------|--------|
| `J` | joint / actuator intrinsic | `dof_pos`, `dof_vel`, `muscle_length` |
| `W-P` | world position | `body_pos_w`, `com_height`, `feet_heights` |
| `W-O` | world orientation | `base_quat_w`, `torso_quat_w` |
| `W-V` | world velocity | `base_lin_vel_w`, `body_lin_vel_w` |
| `B` | body-local / sensor-local | `gyro`, `local_linvel`, `upvector` |
| `A` | anchor / yaw-aligned task frame | `motion_anchor_pos_b`, anchor-frame body pose |
| `T` | task clock / hidden task context | `phase_var`, sampled command, motion frame id |

一个字段即使是“相对量”，也不一定属于 `B` 或 `A`。例如：

- `foot_pos_w - pelvis_pos_w` 只是 world 轴上的相对位移，仍然不是 body-local。

## UniLab 现有契约里的坐标系口径

### Backend contract

UniLab backend 已经显式区分了世界系和 baselink 系：

- `get_base_lin_vel()` 返回世界系线速度。
- `get_base_ang_vel()` 返回世界系角速度。
- `gyro` 传感器返回局部坐标系角速度。
- `get_body_pos_w/get_body_quat_w/get_body_lin_vel_w` 返回世界系 body 状态。
- `get_body_pos_b/get_body_quat_b/get_body_lin_vel_b` 返回 baselink 系 body 状态。

参考：

- [src/unilab/base/backend/base.py](../../../../../src/unilab/base/backend/base.py)

### Locomotion 任务的常见做法

UniLab 现有 locomotion actor observation 通常偏向局部坐标系：

- `local_linvel`
- `gyro`
- `upvector`
- 相对默认姿态的关节角
- 关节速度
- 动作历史

这类设计通常弱化绝对朝向依赖，增强对称性和泛化性。

参考：

- [src/unilab/envs/locomotion/g1/joystick.py](../../../../../src/unilab/envs/locomotion/g1/joystick.py)

### Motion tracking 任务的常见做法

Motion tracking 经常引入任务自定义 anchor frame：

- 参考动作在 world frame 存储；
- 训练时把一部分观测或误差写成 anchor-local / yaw-aligned 量；
- critic 再补充 world 或 anchor-frame 下的 privileged 信息。

参考：

- [src/unilab/envs/motion_tracking/g1/tracking.py](../../../../../src/unilab/envs/motion_tracking/g1/tracking.py)

## 坐标系如何影响 MDP 设计

### 1. 同一个底层 MDP，可以有不同 observation design

真实物理状态不变，但以下设计会显著改变训练行为：

- actor 用 world pose
- actor 用 body-local pose
- actor 用 yaw-aligned velocity
- critic 是否看 privileged world body state

所以迁移任务时要先区分：

- “我要复现原任务语义”
- “我要做更适合 UniLab 风格的重构”

这两者都合理，但不能混在一起做。

### 2. reward frame 必须与可观测性一起看

如果 reward 在 `W-V`，例如跟踪世界系 `x/y` 速度，那么 observation 至少要包含足够
信息让策略知道自己相对世界朝向的关系。

如果 reward 改成 `B` 或 yaw-aligned frame，那么就可以弱化绝对朝向依赖。

### 3. “relative” 不等于 “local”

最容易误判的量是“某位置减去 base 位置”。

只有同时做了旋转变换，把向量表达在 body 或 anchor 的轴上，它才是 local frame 量。

### 4. task clock 也是 MDP 的一部分

像 gait `phase_var`、重采样 command、motion frame index 这些不是几何量，但它们是
任务上下文。如果 reward 依赖它们，而 actor observation 不包含它们，任务语义也会变化。

## 新任务设计时的最小检查表

讨论一个新任务前，建议先列出下面这张表。

| 字段 | 作用 | frame 标签 | actor 是否可见 | critic 是否可见 | reward 是否使用 |
|------|------|------------|----------------|-----------------|----------------|
| 目标变量 | 任务 goal | `W-* / B / A / T` | 是/否 | 是/否 | 是/否 |
| base pose | 稳定性/方向 | `W-O` 或 `B` | 是/否 | 是/否 | 是/否 |
| base velocity | 运动反馈 | `W-V` 或 `B` | 是/否 | 是/否 | 是/否 |
| limb pose | 运动学 | `J / W-P / B / A` | 是/否 | 是/否 | 是/否 |
| actuator state | 控制内部态 | `J` | 是/否 | 是/否 | 是/否 |
| task clock | 节律/参考进度 | `T` | 是/否 | 是/否 | 是/否 |

只要有一行说不清，先不要写代码。

## 迁移任务时建议先回答的 4 个问题

### 1. 任务目标是 world-fixed 还是 body/yaw-fixed

例如：

- “沿世界 `+y` 方向跑”是 world-fixed。
- “沿机器人当前朝向前进”更接近 body/yaw-fixed。

### 2. actor observation 是忠实复现，还是 UniLab 风格重构

两种都可以，但要明确：

- faithful migration：允许 mixed-frame observation；
- UniLab-native redesign：倾向 local actor + privileged critic。

### 3. critic 是否允许额外 world / anchor privileged 信息

如果允许，就把它视为训练技巧，而不是任务定义本身。

### 4. termination 是物理失败，还是任务失败

例如：

- base height 太低：物理失败；
- 偏离参考轨迹过大：任务失败；
- time limit：truncation，不应与 termination 混写。

## MyoSuite walk 迁移前的特别提醒

MyoSuite `myoLegWalk-v0` 的默认 observation 不是纯 local，也不是纯 world，而是混合的。

典型例子：

- `qpos_without_xy`：去掉 world `x/y`，但仍保留 root `z` 和根四元数。
- `qvel`：free joint 部分本身就带混合语义。
- `com_vel`：更接近 world velocity。
- `torso_angle`：world orientation。
- `feet_heights`：world `z`。
- `feet_rel_positions`：world-relative displacement，不是 body-local。
- `phase_var`：task clock。
- `muscle_length/velocity/force`：actuator intrinsic state。

因此，迁移 MyoSuite walk 时至少有两条明确路线：

### 路线 A：faithful migration

- 保留 mixed-frame actor observation。
- 保留原任务的 world-fixed velocity goal。
- 优先追求 reward/obs/reset/termination 语义对齐。

### 路线 B：UniLab-native redesign

- 把 actor observation 改成 local / yaw-aligned。
- world 信息只给 critic 或显式目标编码。
- 适合做长期维护的新任务族，但不再是原任务原样复现。

不要做半忠实半重构的中间态：这最容易让任务语义失真。

## 推荐讨论顺序

后续浅议一个新任务时，建议按下面顺序讨论：

1. 目标变量是什么，属于哪个 frame。
2. actor 看到哪些量，这些量分别属于哪个 frame。
3. critic 是否补充 privileged 信息。
4. reward 每一项依赖哪个 frame 的量。
5. termination 和 truncation 各自是什么。
6. reset 分布是否改变了任务上下文。
7. 这是 faithful migration，还是 UniLab-native redesign。

只要这个顺序走通，后面的 env 实现、owner YAML 和测试就会清晰很多。
