from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[4]


def _minimal_mujoco_model(path: Path) -> str:
    path.write_text(
        """
<mujoco model="myoleg_minimal">
  <worldbody>
    <body name="torso" pos="0 0 1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="torso_geom" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_flexor" joint="slide" ctrlrange="-1 1"/>
  </actuator>
  <keyframe>
    <key name="init" qpos="0"/>
  </keyframe>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return str(path)


def _indexed_keyframe_model(path: Path) -> str:
    path.write_text(
        """
<mujoco model="myoleg_keyframes">
  <worldbody>
    <body name="torso" pos="0 0 1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="torso_geom" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_flexor" joint="slide" ctrlrange="-1 1"/>
  </actuator>
  <keyframe>
    <key name="unused0" qpos="0.1" qvel="0"/>
    <key name="unused1" qpos="0.2" qvel="0"/>
    <key name="walk_init" qpos="0.3" qvel="0"/>
  </keyframe>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return str(path)


def _freejoint_random_keyframe_model(path: Path) -> str:
    path.write_text(
        """
<mujoco model="myoleg_random_keyframes">
  <worldbody>
    <body name="torso" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="torso_geom" type="sphere" size="0.05" mass="1"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="hip" type="hinge" axis="1 0 0"/>
        <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_flexor" joint="hip" ctrlrange="-1 1"/>
  </actuator>
  <keyframe>
    <key name="unused0" qpos="0 0 1 1 0 0 0 0.1" qvel="0 0 0 0 0 0 0"/>
    <key name="unused1" qpos="0 0 1 1 0 0 0 0.2" qvel="0 0 0 0 0 0 0"/>
    <key name="walk_init_l" qpos="0 0 1 0.70710678 0 0.70710678 0 0.3" qvel="0 -1.5 0 0 0 0 0.4"/>
    <key name="walk_init_r" qpos="0 0 1 0.70710678 0 0.70710678 0 0.7" qvel="0 -1.5 0 0 0 0 -0.4"/>
  </keyframe>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return str(path)


def _reference_rotation_keyframe_model(path: Path) -> str:
    path.write_text(
        """
<mujoco model="myoleg_ref_rotation">
  <worldbody>
    <body name="torso" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="torso_geom" type="sphere" size="0.05" mass="1"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="hip" type="hinge" axis="1 0 0"/>
        <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_flexor" joint="hip" ctrlrange="-1 1"/>
  </actuator>
  <keyframe>
    <key name="reference" qpos="0 0 1 1 0 0 0 0.1" qvel="0 0 0 0 0 0 0"/>
    <key name="unused1" qpos="0 0 1 1 0 0 0 0.2" qvel="0 0 0 0 0 0 0"/>
    <key name="walk_init" qpos="0 0 1 0.70710678 0 0.70710678 0 0.3" qvel="0 -1.5 0 0 0 0 0.4"/>
  </keyframe>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return str(path)


def _muscle_like_ctrl_model(path: Path) -> str:
    path.write_text(
        """
<mujoco model="myoleg_muscle_like">
  <worldbody>
    <body name="torso" pos="0 0 1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="torso_geom" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_flexor" joint="slide" ctrlrange="0 1"/>
  </actuator>
  <keyframe>
    <key name="init" qpos="0"/>
  </keyframe>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return str(path)


def _minimal_env_override(model_file: str, **overrides):
    env_override = {
        "scene": {"model_file": model_file},
        "pelvis_body_name": "torso",
        "cyclic_hip_joint_names": [],
        "joint_angle_rew_joint_names": [],
    }
    env_override.update(overrides)
    return env_override


def test_myoleg_walk_flat_registers_mujoco_only():
    from unilab.base import registry

    registry.ensure_registries()

    registered = registry.list_registered_envs()
    assert registered["MyoLegWalkFlat"]["available_backends"] == ["mujoco"]


def test_myoleg_walk_flat_constructs_backend_with_actuator_contract(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _minimal_mujoco_model(tmp_path / "myoleg_minimal.xml")

    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file),
    )
    try:
        assert env.action_space.shape == (1,)
        assert env._backend.get_actuator_names() == ("hip_flexor",)
        assert env._backend.get_actuator_lengths().shape == (1, 1)
    finally:
        env.close()


def test_myoleg_walk_flat_init_state_and_step_smoke(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _minimal_mujoco_model(tmp_path / "myoleg_minimal.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file),
    )
    try:
        state = env.init_state()
        assert set(state.obs) == {"obs", "critic"}
        assert state.obs["obs"].shape == state.obs["critic"].shape
        assert state.obs["obs"].shape[0] == 1

        next_state = env.step(np.zeros((1, 1), dtype=np.float32))
        assert next_state.obs["obs"].shape == state.obs["obs"].shape
        assert next_state.reward.shape == (1,)
        assert np.isfinite(next_state.reward).all()
        assert not next_state.terminated[0]
        assert next_state.reward[0] == pytest.approx(5.0 * (1.0 + np.exp(-(1.2**2))) + 15.0)
        assert "phase_var" in next_state.info
    finally:
        env.close()


def test_myoleg_walk_flat_prefers_mysuite_keyframe_index_two(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _indexed_keyframe_model(tmp_path / "myoleg_keyframes.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file),
    )
    try:
        env.init_state()
        np.testing.assert_allclose(env._backend.get_qpos()[:, 0], [0.3])
    finally:
        env.close()


def test_myoleg_walk_flat_muscle_like_action_uses_myosuite_sigmoid(tmp_path: Path):
    from unilab.base import registry
    from unilab.base.np_env import NpEnvState

    registry.ensure_registries()
    model_file = _muscle_like_ctrl_model(tmp_path / "myoleg_muscle_like.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file),
    )
    try:
        state = NpEnvState(
            obs=env._obs_dict(),
            reward=np.zeros((1,), dtype=np.float32),
            terminated=np.zeros((1,), dtype=bool),
            truncated=np.zeros((1,), dtype=bool),
            info={},
        )
        ctrl = env.apply_action(np.array([[0.5]], dtype=np.float32), state)
        np.testing.assert_allclose(ctrl, [[0.5]], atol=1e-6)
        ctrl = env.apply_action(np.array([[0.0]], dtype=np.float32), state)
        np.testing.assert_allclose(ctrl, [[1.0 / (1.0 + np.exp(2.5))]], atol=1e-6)
    finally:
        env.close()


def test_myoleg_walk_flat_min_height_termination(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _minimal_mujoco_model(tmp_path / "myoleg_minimal.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file, min_height=1.5),
    )
    try:
        state = env.init_state()
        state = env.update_state(state)
        assert state.terminated[0]
        assert state.reward[0] < 0.0
    finally:
        env.close()


def test_myoleg_walk_flat_time_limit_uses_truncated(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _minimal_mujoco_model(tmp_path / "myoleg_minimal.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(
            model_file,
            max_episode_seconds=0.01,
            ctrl_dt=0.01,
        ),
    )
    try:
        state = env.init_state()
        state.info["steps"][0] = 1
        truncated = env._compute_truncated(state)
        assert truncated[0]
    finally:
        env.close()


def test_myoleg_walk_flat_requires_scene_model_file():
    from unilab.base import registry

    registry.ensure_registries()

    with pytest.raises(ValueError, match="scene.model_file"):
        registry.make("MyoLegWalkFlat", sim_backend="mujoco")


def test_myoleg_walk_flat_rejects_unknown_reset_type():
    from unilab.base.scene import SceneCfg
    from unilab.envs.locomotion.myoleg.walk import MyoLegWalkFlatCfg

    cfg = MyoLegWalkFlatCfg(scene=SceneCfg(model_file="placeholder.xml"), reset_type="unsupported")

    with pytest.raises(ValueError, match="reset_type='init' or 'random'"):
        cfg.validate()


def test_myoleg_walk_flat_random_reset_preserves_root_height_and_rotation(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _freejoint_random_keyframe_model(tmp_path / "myoleg_random_keyframes.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file, reset_type="random"),
        num_envs=4,
    )
    try:
        np.random.seed(7)
        env.reset(np.arange(4, dtype=np.int32))
        qpos = env._backend.get_qpos()
        key_qpos = env._backend.get_keyframe_qpos_by_index(2)

        np.testing.assert_allclose(qpos[:, 2], key_qpos[2])
        np.testing.assert_allclose(qpos[:, 3:7], np.broadcast_to(key_qpos[3:7], (4, 4)))
        assert not np.allclose(qpos[:, 0], 0.0)
        assert not np.all(np.isin(np.round(qpos[:, 7], 6), [0.3, 0.7]))
    finally:
        env.close()


def test_myoleg_walk_flat_subset_reset_returns_only_requested_env_obs(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _freejoint_random_keyframe_model(tmp_path / "myoleg_random_keyframes.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file),
        num_envs=4,
    )
    try:
        obs, info = env.reset(np.asarray([1, 3], dtype=np.int32))

        assert obs["obs"].shape == (2, env.obs_groups_spec["obs"])
        assert obs["critic"].shape == (2, env.obs_groups_spec["critic"])
        assert info["phase_var"].shape == (2, 1)
    finally:
        env.close()


def test_myoleg_walk_flat_autoreset_handles_partial_done_envs(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _freejoint_random_keyframe_model(tmp_path / "myoleg_random_keyframes.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file, max_episode_seconds=0.02),
        num_envs=4,
    )
    try:
        state = env.init_state()
        state.info["steps"][:] = 0
        state.info["steps"][2] = 2

        next_state = env.step(np.zeros((4, env.action_space.shape[0]), dtype=np.float32))

        assert next_state.obs["obs"].shape == (4, env.obs_groups_spec["obs"])
        assert next_state.info["steps"][2] == 0
        assert np.isfinite(next_state.reward).all()
    finally:
        env.close()


def test_myoleg_walk_flat_ref_rotation_uses_myosuite_setup_keyframe_zero(tmp_path: Path):
    from unilab.base import registry

    registry.ensure_registries()
    model_file = _reference_rotation_keyframe_model(tmp_path / "myoleg_ref_rotation.xml")
    env = registry.make(
        "MyoLegWalkFlat",
        sim_backend="mujoco",
        env_cfg_override=_minimal_env_override(model_file),
    )
    try:
        env.init_state()
        key0 = env._backend.get_keyframe_qpos_by_index(0)
        key2 = env._backend.get_keyframe_qpos_by_index(2)

        np.testing.assert_allclose(env._backend.get_qpos()[0, 3:7], key2[3:7])
        np.testing.assert_allclose(env._ref_qpos[3:7], key0[3:7])

        expected = np.exp(-np.linalg.norm(5.0 * (key2[3:7] - key0[3:7])))
        np.testing.assert_allclose(env._compute_ref_rotation_reward(), [expected], atol=1e-6)
    finally:
        env.close()


def test_myoleg_walk_flat_owner_yaml_builds_minimal_step_env():
    from unilab.base import registry
    from unilab.training import BackendAdapter

    registry.ensure_registries()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf" / "ppo")):
        cfg = compose(config_name="config", overrides=["task=myoleg_walk_flat/mujoco"])

    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    try:
        state = env.init_state()
        assert state.obs["obs"].shape[1] == env.obs_groups_spec["obs"]
        assert env.obs_groups_spec == {"obs": 403, "critic": 403}
        assert env.action_space.shape == (env._backend.num_actuators,)
        assert env._backend.num_actuators == 80
        assert env._backend.get_actuator_activations().shape == (1, 80)

        state = env.step(np.zeros((1, env._backend.num_actuators), dtype=np.float32))
        assert state.obs["obs"].shape == (1, env.obs_groups_spec["obs"])
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_myoleg_walk_flat_hides_original_terrain_hfield_on_cold_path():
    import mujoco

    from unilab.base import registry
    from unilab.training import BackendAdapter

    registry.ensure_registries()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf" / "ppo")):
        cfg = compose(config_name="config", overrides=["task=myoleg_walk_flat/mujoco"])

    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    try:
        state = env.init_state()
        model = env._backend.model
        terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        assert terrain_id >= 0
        np.testing.assert_allclose(model.geom_pos[terrain_id], (0.0, 0.0, -10.0), atol=1e-8)
        assert model.geom_rgba[terrain_id, 3] == pytest.approx(0.0)

        for _ in range(60):
            state = env.step(np.zeros((1, env._backend.num_actuators), dtype=np.float32))
        data = env._backend._scratch_data
        terrain_contacts = [
            idx
            for idx in range(data.ncon)
            if data.contact[idx].geom1 == terrain_id or data.contact[idx].geom2 == terrain_id
        ]
        assert terrain_contacts == []
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_myoleg_walk_flat_com_terms_match_myosuite_formulas():
    import mujoco

    from unilab.base import registry
    from unilab.training import BackendAdapter

    registry.ensure_registries()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf" / "ppo")):
        cfg = compose(config_name="config", overrides=["task=myoleg_walk_flat/mujoco"])

    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    try:
        env.init_state()
        model = env._backend.model
        data = mujoco.MjData(model)
        data.qpos[:] = env._backend.get_qpos()[0]
        data.qvel[:] = env._backend._physics_state[
            0, env._backend._idx_qvel : env._backend._idx_qvel + env._backend.nv
        ]
        mujoco.mj_forward(model, data)

        mass = np.expand_dims(model.body_mass, -1)
        expected_com = np.sum(mass * data.xipos, axis=0) / np.sum(mass)
        expected_vel_xy = (np.sum(mass * (-data.cvel), axis=0) / np.sum(mass))[3:5]

        np.testing.assert_allclose(env._backend.get_com_position()[0], expected_com, atol=1e-6)
        np.testing.assert_allclose(
            env._backend.get_com_velocity_xy()[0], expected_vel_xy, atol=1e-6
        )
    finally:
        env.close()


def test_myoleg_walk_flat_world_y_velocity_direction_matches_myosuite():
    from unilab.base import registry
    from unilab.training import BackendAdapter

    registry.ensure_registries()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf" / "ppo")):
        cfg = compose(config_name="config", overrides=["task=myoleg_walk_flat/mujoco"])

    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    try:
        env.init_state()
        qvel = env._backend.get_qvel()[0]
        com_vel = env._backend.get_com_velocity_xy()[0]

        assert qvel[1] < 0.0
        assert com_vel[1] > 0.0
        assert abs(env.cfg.target_y_vel - com_vel[1]) < abs(env.cfg.target_y_vel + com_vel[1])
        assert abs(com_vel[0]) < 1e-2
    finally:
        env.close()


def test_myoleg_walk_flat_observation_slices_match_myosuite_mdp_formulas():
    from unilab.base import registry
    from unilab.training import BackendAdapter

    registry.ensure_registries()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf" / "ppo")):
        cfg = compose(config_name="config", overrides=["task=myoleg_walk_flat/mujoco"])

    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    try:
        state = env.init_state()
        obs = state.obs["obs"]
        components = env._compute_obs_components()

        assert tuple(env._obs_slices) == env._OBS_COMPONENT_ORDER
        assert obs.shape[1] == sum(value.shape[1] for value in components.values())
        for name in env._OBS_COMPONENT_ORDER:
            np.testing.assert_allclose(obs[:, env._obs_slices[name]], components[name], atol=1e-6)

        qpos = env._backend.get_qpos()
        qvel = env._backend.get_qvel()
        pelvis_pos = env._backend.get_body_pos_w(np.asarray([env._pelvis_body_id], dtype=np.int32))[
            :, 0, :
        ]
        foot_pos = env._backend.get_body_pos_w(env._foot_body_ids)

        expected = {
            "qpos_without_xy": qpos[:, 2:],
            "qvel": qvel * env.cfg.ctrl_dt,
            "com_vel": env._backend.get_com_velocity_xy(),
            "torso_angle": env._backend.get_body_quat_w(
                np.asarray([env._torso_body_id], dtype=np.int32)
            )[:, 0, :],
            "feet_heights": foot_pos[:, :, 2],
            "height": env._backend.get_com_position()[:, 2:3],
            "feet_rel_positions": (foot_pos - pelvis_pos[:, None, :]).reshape(env.num_envs, -1),
            "phase_var": np.zeros((env.num_envs, 1)),
            "muscle_length": env._backend.get_actuator_lengths(),
            "muscle_velocity": np.clip(env._backend.get_actuator_velocities(), -100.0, 100.0),
            "muscle_force": np.clip(env._backend.get_actuator_forces() / 1000.0, -100.0, 100.0),
            "act": env._backend.get_actuator_activations(),
        }
        for name, value in expected.items():
            np.testing.assert_allclose(components[name], value, atol=1e-6)
    finally:
        env.close()


def test_myoleg_walk_flat_reward_matches_myosuite_mdp_terms():
    from unilab.base import registry
    from unilab.training import BackendAdapter

    registry.ensure_registries()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf" / "ppo")):
        cfg = compose(config_name="config", overrides=["task=myoleg_walk_flat/mujoco"])

    env_cfg_override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        cfg.training.task_name,
        sim_backend=cfg.training.sim_backend,
        env_cfg_override=env_cfg_override,
    )
    try:
        env.init_state()
        terminated = env._compute_terminated()
        com_vel = env._backend.get_com_velocity_xy()
        vel_reward = np.exp(-np.square(env.cfg.target_y_vel - com_vel[:, 1])) + np.exp(
            -np.square(env.cfg.target_x_vel - com_vel[:, 0])
        )
        phase = env._phase_var()[:, 0]
        desired_hip = np.stack(
            [
                0.8 * np.cos(phase * 2.0 * np.pi + np.pi),
                0.8 * np.cos(phase * 2.0 * np.pi),
            ],
            axis=1,
        )
        cyclic_angles = env._backend.get_dof_pos()[:, env._cyclic_hip_qpos_indices]
        cyclic_hip = np.linalg.norm(desired_hip - cyclic_angles, axis=1)
        root_quat = env._backend.get_qpos()[:, 3:7]
        ref_rot = np.exp(-np.linalg.norm(5.0 * (root_quat - env._ref_qpos[None, 3:7]), axis=1))
        joint_angles = env._backend.get_dof_pos()[:, env._joint_angle_rew_qpos_indices]
        joint_angle_rew = np.exp(-5.0 * np.mean(np.abs(joint_angles), axis=1))

        scales = env.cfg.reward_config.scales
        expected = (
            scales["vel_reward"] * vel_reward
            + scales["done"] * terminated.astype(np.float64)
            + scales["cyclic_hip"] * cyclic_hip
            + scales["ref_rot"] * ref_rot
            + scales["joint_angle_rew"] * joint_angle_rew
        )

        np.testing.assert_allclose(env._compute_cyclic_hip_penalty(), cyclic_hip, atol=1e-6)
        np.testing.assert_allclose(env._compute_ref_rotation_reward(), ref_rot, atol=1e-6)
        np.testing.assert_allclose(env._compute_joint_angle_reward(), joint_angle_rew, atol=1e-6)
        np.testing.assert_allclose(env._compute_reward(terminated), expected, atol=1e-6)
    finally:
        env.close()
