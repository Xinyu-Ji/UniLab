from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype

__all__ = [
    "MyoLegWalkFlatCfg",
    "MyoLegWalkFlatEnv",
    "MyoLegWalkRewardConfig",
]


@dataclass
class MyoLegWalkRewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "vel_reward": 5.0,
            "done": -100.0,
            "cyclic_hip": -10.0,
            "ref_rot": 10.0,
            "joint_angle_rew": 5.0,
        }
    )


@registry.envcfg("MyoLegWalkFlat")
@dataclass
class MyoLegWalkFlatCfg(EnvCfg):
    """Config anchor for a faithful MyoSuite ``myoLegWalk-v0`` migration."""

    scene: SceneCfg = field(default_factory=lambda: SceneCfg(model_file=""))
    max_episode_seconds: float = 10.0
    sim_dt: float = 0.001
    ctrl_dt: float = 0.01
    min_height: float = 0.8
    max_rot: float = 0.8
    hip_period: int = 100
    reset_type: str = "init"
    target_x_vel: float = 0.0
    target_y_vel: float = 1.2
    target_rot: float | None = None
    normalize_act: bool = True
    init_keyframe_index: int | None = 2
    init_keyframe_name: str = "init"
    torso_body_name: str = "torso"
    pelvis_body_name: str = "pelvis"
    foot_body_names: list[str] = field(default_factory=list)
    cyclic_hip_joint_names: list[str] = field(
        default_factory=lambda: ["hip_flexion_l", "hip_flexion_r"]
    )
    joint_angle_rew_joint_names: list[str] = field(
        default_factory=lambda: [
            "hip_adduction_l",
            "hip_adduction_r",
            "hip_rotation_l",
            "hip_rotation_r",
        ]
    )
    reward_config: MyoLegWalkRewardConfig = field(default_factory=MyoLegWalkRewardConfig)

    def validate(self) -> None:
        super().validate()
        if not self.scene or not self.scene.model_file:
            raise ValueError("MyoLegWalkFlat requires env.scene.model_file")
        if self.reset_type not in {"init", "random"}:
            raise ValueError("MyoLegWalkFlat supports reset_type='init' or 'random'")


@registry.env("MyoLegWalkFlat", sim_backend="mujoco")
class MyoLegWalkFlatEnv(NpEnv):
    """Minimal faithful-shape MyoSuite walk env.

    The implementation runs the UniLab lifecycle and preserves the MDP slots
    needed by ``myoLegWalk-v0``. Exact MyoSuite body/joint names are configured
    by owner YAML once the real asset mapping lands.
    """

    _OBS_COMPONENT_ORDER = (
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "phase_var",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
        "act",
    )

    _cfg: MyoLegWalkFlatCfg

    def __init__(
        self,
        cfg: MyoLegWalkFlatCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        if backend_type != "mujoco":
            raise ValueError("MyoLegWalkFlat initial migration only supports mujoco backend")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.torso_body_name,
            add_body_sensors=True,
            enable_actuator_sensors=True,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
            geom_overrides={
                "terrain": {"pos": (0.0, 0.0, -10.0), "rgba_alpha": 0.0, "required": False}
            },
        )
        super().__init__(cfg, backend, num_envs)
        self._backend.materialize()
        self._actuator_names = backend.get_actuator_names()
        self._num_actuators = backend.num_actuators
        self._ref_qpos = self._resolve_reference_qpos()
        self._init_qpos, self._init_qvel = self._resolve_init_state()
        self._random_reset_states = self._resolve_random_reset_states()
        self._torso_body_id = backend.get_body_id(cfg.torso_body_name)
        self._pelvis_body_id = backend.get_body_id(cfg.pelvis_body_name)
        self._foot_body_ids = (
            backend.get_body_ids(cfg.foot_body_names)
            if cfg.foot_body_names
            else np.empty((0,), dtype=np.int32)
        )
        self._cyclic_hip_qpos_indices = (
            backend.get_joint_dof_pos_indices(cfg.cyclic_hip_joint_names)
            if cfg.cyclic_hip_joint_names
            else np.empty((0,), dtype=np.int32)
        )
        self._joint_angle_rew_qpos_indices = (
            backend.get_joint_dof_pos_indices(cfg.joint_angle_rew_joint_names)
            if cfg.joint_angle_rew_joint_names
            else np.empty((0,), dtype=np.int32)
        )
        self._myosuite_steps = np.zeros((num_envs,), dtype=np.uint32)
        self._action_space = gym.spaces.Box(
            -np.ones((backend.num_actuators,), dtype=np.float32),
            np.ones((backend.num_actuators,), dtype=np.float32),
            dtype=np.float32,
        )
        self._obs_slices = self._build_obs_slices(self._compute_obs_components())
        self._obs_dim = sum(slice_.stop - slice_.start for slice_ in self._obs_slices.values())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._obs_dim, "critic": self._obs_dim}

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        if env_ids.size == 0:
            return self._slice_obs_dict(self._obs_dict(), env_ids), {}
        qpos, qvel = self._sample_reset_state(env_ids.size)
        self._backend.set_state(env_ids, qpos, qvel)
        self._myosuite_steps[env_ids] = 0
        obs = self._slice_obs_dict(self._obs_dict(), env_ids)
        return obs, {"phase_var": self._phase_var()[env_ids].copy()}

    def _slice_obs_dict(
        self, obs: dict[str, np.ndarray], env_ids: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {key: value[env_ids] for key, value in obs.items()}

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        ctrl = self._normalize_action(actions)
        state.info["current_actions"] = ctrl
        return ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        obs = self._obs_dict()
        terminated = self._compute_terminated()
        reward = self._compute_reward(terminated)
        state.info["phase_var"] = self._phase_var().copy()
        self._myosuite_steps += 1
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _phase_var(self) -> np.ndarray:
        phase = (self._myosuite_steps.astype(get_global_dtype()) / float(self._cfg.hip_period)) % 1
        return phase[:, None]

    def _resolve_reference_qpos(self) -> np.ndarray:
        try:
            return np.asarray(self._backend.get_keyframe_qpos_by_index(0), dtype=np.float64)
        except ValueError:
            return np.asarray(self._backend.get_default_qpos(), dtype=np.float64)

    def _resolve_init_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cfg.init_keyframe_index is not None:
            try:
                return (
                    np.asarray(
                        self._backend.get_keyframe_qpos_by_index(self._cfg.init_keyframe_index),
                        dtype=np.float64,
                    ),
                    np.asarray(
                        self._backend.get_keyframe_qvel_by_index(self._cfg.init_keyframe_index),
                        dtype=np.float64,
                    ),
                )
            except ValueError:
                pass
        try:
            return (
                np.asarray(
                    self._backend.get_keyframe_qpos(self._cfg.init_keyframe_name), dtype=np.float64
                ),
                np.asarray(self._backend.get_init_qvel(), dtype=np.float64),
            )
        except ValueError:
            return (
                np.asarray(self._backend.get_default_qpos(), dtype=np.float64),
                np.asarray(self._backend.get_init_qvel(), dtype=np.float64),
            )

    def _resolve_random_reset_states(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        states: list[tuple[np.ndarray, np.ndarray]] = []
        for keyframe_index in (2, 3):
            try:
                states.append(
                    (
                        np.asarray(
                            self._backend.get_keyframe_qpos_by_index(keyframe_index),
                            dtype=np.float64,
                        ),
                        np.asarray(
                            self._backend.get_keyframe_qvel_by_index(keyframe_index),
                            dtype=np.float64,
                        ),
                    )
                )
            except ValueError:
                continue
        return tuple(states)

    def _sample_reset_state(self, num_resets: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cfg.reset_type == "random" and self._random_reset_states:
            choices = np.random.randint(0, len(self._random_reset_states), size=(num_resets,))
            qpos = np.stack([self._random_reset_states[int(choice)][0] for choice in choices])
            qvel = np.stack([self._random_reset_states[int(choice)][1] for choice in choices])
            rot_state = qpos[:, 3:7].copy() if qpos.shape[1] >= 7 else None
            height = qpos[:, 2:3].copy() if qpos.shape[1] >= 3 else None
            qpos = qpos + np.random.normal(0.0, 0.02, size=qpos.shape)
            if rot_state is not None:
                qpos[:, 3:7] = rot_state
            if height is not None:
                qpos[:, 2:3] = height
            return qpos, qvel
        qpos = np.broadcast_to(self._init_qpos, (num_resets, self._init_qpos.shape[0])).copy()
        qvel = np.broadcast_to(self._init_qvel, (num_resets, self._init_qvel.shape[0])).copy()
        return qpos, qvel

    def _normalize_action(self, actions: np.ndarray) -> np.ndarray:
        action = np.asarray(actions, dtype=get_global_dtype())
        if action.shape != (self._num_envs, self._num_actuators):
            raise ValueError(
                f"actions must have shape {(self._num_envs, self._num_actuators)}, got {action.shape}"
            )
        clipped = np.clip(action, -1.0, 1.0)
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=clipped.dtype)
        if not self._cfg.normalize_act:
            return clipped
        low = ctrl_range[:, 0]
        high = ctrl_range[:, 1]
        ctrl = low + (clipped + 1.0) * 0.5 * (high - low)
        # MyoSuite explicitly projects muscle actuators through this sigmoid
        # before handing control to its robot wrapper.
        muscle_mask = (low >= 0.0) & (high <= 1.0)
        ctrl[:, muscle_mask] = 1.0 / (1.0 + np.exp(-5.0 * (clipped[:, muscle_mask] - 0.5)))
        return ctrl

    def _obs_dict(self) -> dict[str, np.ndarray]:
        obs = self._compute_obs()
        return {"obs": obs, "critic": obs.copy()}

    def _build_obs_slices(self, components: dict[str, np.ndarray]) -> dict[str, slice]:
        obs_slices: dict[str, slice] = {}
        cursor = 0
        for name in self._OBS_COMPONENT_ORDER:
            width = int(components[name].shape[1])
            obs_slices[name] = slice(cursor, cursor + width)
            cursor += width
        return obs_slices

    def _compute_obs_components(self) -> dict[str, np.ndarray]:
        dtype = get_global_dtype()
        qpos = np.asarray(self._backend.get_qpos(), dtype=dtype)
        qvel = np.asarray(self._backend.get_qvel(), dtype=dtype)
        qpos_without_xy = qpos[:, 2:] if qpos.shape[1] >= 2 else qpos
        com_vel = np.asarray(self._backend.get_com_velocity_xy(), dtype=dtype)
        torso_quat = self._backend.get_body_quat_w(
            np.asarray([self._torso_body_id], dtype=np.int32)
        )[:, 0, :]
        com_pos = np.asarray(self._backend.get_com_position(), dtype=dtype)
        pelvis_pos = self._backend.get_body_pos_w(
            np.asarray([self._pelvis_body_id], dtype=np.int32)
        )[:, 0, :]
        foot_pos = (
            self._backend.get_body_pos_w(self._foot_body_ids)
            if self._foot_body_ids.size
            else np.empty((self._num_envs, 0, 3), dtype=dtype)
        )
        feet_heights = (
            foot_pos[:, :, 2] if foot_pos.size else np.empty((self._num_envs, 0), dtype=dtype)
        )
        feet_rel = (foot_pos - pelvis_pos[:, None, :]).reshape(self._num_envs, -1)
        return {
            "qpos_without_xy": qpos_without_xy,
            "qvel": qvel * self._cfg.ctrl_dt,
            "com_vel": com_vel,
            "torso_angle": torso_quat,
            "feet_heights": feet_heights,
            "height": com_pos[:, 2:3],
            "feet_rel_positions": feet_rel,
            "phase_var": self._phase_var(),
            "muscle_length": np.asarray(self._backend.get_actuator_lengths(), dtype=dtype),
            "muscle_velocity": np.clip(
                np.asarray(self._backend.get_actuator_velocities(), dtype=dtype), -100.0, 100.0
            ),
            "muscle_force": np.clip(
                np.asarray(self._backend.get_actuator_forces(), dtype=dtype) / 1000.0,
                -100.0,
                100.0,
            ),
            "act": np.asarray(self._backend.get_actuator_activations(), dtype=dtype),
        }

    def _compute_obs(self) -> np.ndarray:
        dtype = get_global_dtype()
        components = self._compute_obs_components()
        return np.concatenate(
            [components[name] for name in self._OBS_COMPONENT_ORDER],
            axis=1,
            dtype=dtype,
        )

    def _compute_terminated(self) -> np.ndarray:
        com_pos = self._backend.get_com_position()
        height_done = com_pos[:, 2] < float(self._cfg.min_height)
        qpos = self._backend.get_qpos()
        rot_done = np.zeros((self._num_envs,), dtype=bool)
        if qpos.shape[1] >= 7:
            root_quat = qpos[:, 3:7]
            y, z = root_quat[:, 2], root_quat[:, 3]
            rotated_x_component = 1.0 - 2.0 * (y * y + z * z)
            rot_done = np.abs(rotated_x_component) > float(self._cfg.max_rot)
        return np.asarray(height_done | rot_done, dtype=bool)

    def _compute_reward(self, terminated: np.ndarray) -> np.ndarray:
        scales = self._cfg.reward_config.scales
        com_vel = self._backend.get_com_velocity_xy()
        vel_reward = np.exp(-np.square(self._cfg.target_y_vel - com_vel[:, 1])) + np.exp(
            -np.square(self._cfg.target_x_vel - com_vel[:, 0])
        )
        cyclic_hip = self._compute_cyclic_hip_penalty()
        ref_rot = self._compute_ref_rotation_reward()
        joint_angle_rew = self._compute_joint_angle_reward()
        reward = (
            float(scales.get("vel_reward", 0.0)) * vel_reward
            + float(scales.get("done", 0.0)) * terminated.astype(get_global_dtype())
            + float(scales.get("cyclic_hip", 0.0)) * cyclic_hip
            + float(scales.get("ref_rot", 0.0)) * ref_rot
            + float(scales.get("joint_angle_rew", 0.0)) * joint_angle_rew
        )
        return np.asarray(reward, dtype=get_global_dtype())

    def _compute_cyclic_hip_penalty(self) -> np.ndarray:
        if self._cyclic_hip_qpos_indices.size != 2:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        phase = self._phase_var()[:, 0]
        desired = np.stack(
            [
                0.8 * np.cos(phase * 2.0 * np.pi + np.pi),
                0.8 * np.cos(phase * 2.0 * np.pi),
            ],
            axis=1,
        )
        angles = self._backend.get_dof_pos()[:, self._cyclic_hip_qpos_indices]
        return np.linalg.norm(desired - angles, axis=1)

    def _compute_ref_rotation_reward(self) -> np.ndarray:
        qpos = self._backend.get_qpos()
        if qpos.shape[1] < 7:
            return np.ones((self._num_envs,), dtype=get_global_dtype())
        root_quat = qpos[:, 3:7]
        target = (
            np.asarray(self._cfg.target_rot, dtype=root_quat.dtype)
            if self._cfg.target_rot is not None
            else self._ref_qpos[3:7]
        )
        return np.exp(-np.linalg.norm(5.0 * (root_quat - target[None, :]), axis=1))

    def _compute_joint_angle_reward(self) -> np.ndarray:
        if not self._joint_angle_rew_qpos_indices.size:
            return np.ones((self._num_envs,), dtype=get_global_dtype())
        joint_angles = self._backend.get_dof_pos()[:, self._joint_angle_rew_qpos_indices]
        mag = np.mean(np.abs(joint_angles), axis=1)
        return np.exp(-5.0 * mag)
