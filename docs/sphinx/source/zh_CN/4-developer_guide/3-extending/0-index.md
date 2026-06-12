# 扩展

扩展指南说明在哪里添加任务、后端、算法与地形，
同时不跨越所有权边界搬移行为。

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 新任务
:link: 1-new_task
:link-type: doc
添加 env config、注册、owner YAML 与测试。
:::

:::{grid-item-card} 新后端
:link: 2-new_backend
:link-type: doc
添加 `SimBackend` 实现并声明能力。
:::

:::{grid-item-card} 新算法
:link: 3-new_algorithm
:link-type: doc
添加配置、runner 代码与脚本层组装。
:::

:::{grid-item-card} 新地形
:link: 4-new_terrain
:link-type: doc
在冷路径上扩展地形生成。
:::

:::{grid-item-card} MDP 与坐标系
:link: 5-task_mdp_and_frames
:link-type: doc
先对齐 frame 语义，再讨论 observation、reward 与 termination。
:::

:::{grid-item-card} MyoSuite Walk
:link: 6-myosuite_walk_mdp
:link-type: doc
把 `myoLegWalk-v0` 拆成可迁移的动作、观测、奖励与终止方案。
:::

:::{grid-item-card} MyoSuite Walk 实施记录
:link: 7-myosuite_walk_phase_report
:link-type: doc
记录 MyoLeg MVP 当前改动、验证结果、坐标系结论和剩余风险。
:::

::::

```{toctree}
:hidden:

1-new_task
2-new_backend
3-new_algorithm
4-new_terrain
5-task_mdp_and_frames
6-myosuite_walk_mdp
7-myosuite_walk_phase_report
```
