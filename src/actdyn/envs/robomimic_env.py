from __future__ import annotations

import sys
from typing import Any

import numpy as np

from actdyn.data.robomimic_lowdim import get_env_metadata


def build_obs_vector(obs_dict: dict[str, Any], obs_keys: list[str], obs_normalizer: Any | None = None) -> np.ndarray:
    parts = []
    for key in obs_keys:
        if key not in obs_dict:
            raise KeyError(f"Observation key '{key}' missing from environment observation.")
        parts.append(np.asarray(obs_dict[key], dtype=np.float32).reshape(-1))
    obs = np.concatenate(parts, axis=0).astype(np.float32)
    if obs_normalizer is not None and getattr(obs_normalizer, "enabled", False):
        obs = obs_normalizer.normalize(obs)
    return obs


def load_env_from_dataset(dataset_path: str, use_image_obs: bool = False):
    """
    Lazily construct an environment through RoboMimic if the package is installed.
    """
    try:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
    except ImportError as exc:
        raise RuntimeError(
            "RoboMimic rollout support is not installed. Install the optional rollout "
            "dependencies and the simulator stack before running rollout evaluation."
        ) from exc

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # Some robomimic / robosuite combinations require ObsUtils to be initialized
    # before env.reset(); otherwise OBS_KEYS_TO_MODALITIES can remain None.
    if getattr(ObsUtils, "OBS_KEYS_TO_MODALITIES", None) is None:
        env_kwargs = env_meta.get("env_kwargs", {}) if isinstance(env_meta, dict) else {}
        obs_specs = env_kwargs.get("observation_modalities")
        if not obs_specs:
            shape_meta = FileUtils.get_shape_metadata_from_dataset(
                dataset_path=dataset_path,
                all_obs_keys=[],
                verbose=False,
            )
            all_obs_keys = list(shape_meta.get("all_obs_keys", []))
            obs_specs = {"obs": {"low_dim": all_obs_keys}, "goal": {"low_dim": []}}
        # Support API differences across robomimic versions.
        try:
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_specs)
        except TypeError:
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)

    # Some RoboMimic datasets carry env kwargs that differ across robosuite versions
    # (e.g. "lite_physics"). Retry after dropping unknown kwargs to improve portability.
    while True:
        try:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=False,
                use_image_obs=use_image_obs,
            )
            break
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "mujoco_py":
                raise RuntimeError(
                    "RoboMimic is importing mujoco_py (legacy MuJoCo), which is not installed. "
                    "Install this repo's rollout stack (robomimic v0.4 from git + robosuite + DeepMind mujoco). "
                    "On Python 3.14 use a separate 3.13 venv (mujoco has no cp314 wheels): "
                    "`uv venv .venv-rollout --python 3.13` then "
                    "`UV_PROJECT_ENVIRONMENT=.venv-rollout uv sync --extra rollout` "
                    "(do not use `uv sync --python .venv-rollout/bin/python` — that repoints the default "
                    "`.venv`, not `.venv-rollout`). Or: `source .venv-rollout/bin/activate` and "
                    "`uv sync --extra rollout --active`. Then run eval with `.venv-rollout/bin/python`. "
                    "Alternatively use `actdyn.eval --offline-only`."
                ) from exc
            if missing in {"mujoco", "robosuite"}:
                py = f"{sys.version_info.major}.{sys.version_info.minor}"
                raise RuntimeError(
                    f"Missing simulation package `{missing}` (Python {py}). "
                    "Run `uv sync --extra rollout`. Rollouts need Python 3.10–3.13 for prebuilt mujoco wheels; "
                    "use `uv venv --python 3.13` if sync fails on 3.14. Or use `--offline-only`."
                ) from exc
            raise
        except TypeError as exc:
            message = str(exc)
            marker = "unexpected keyword argument '"
            if marker not in message:
                raise
            start = message.find(marker) + len(marker)
            end = message.find("'", start)
            if end == -1:
                raise
            bad_kwarg = message[start:end]
            env_kwargs = env_meta.get("env_kwargs", {})
            if bad_kwarg not in env_kwargs:
                raise
            env_kwargs.pop(bad_kwarg, None)
    return env, env_meta


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out


def step_env(env, action: np.ndarray):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info
    raise RuntimeError("Unsupported environment step return format.")


def extract_success(env, info: dict[str, Any] | None, reward: float) -> bool:
    info = info or {}
    if "success" in info:
        val = info["success"]
        if isinstance(val, dict):
            return bool(any(val.values()))
        return bool(val)
    if hasattr(env, "is_success"):
        try:
            val = env.is_success()
            if isinstance(val, dict):
                return bool(any(val.values()))
            return bool(val)
        except Exception:
            pass
    return bool(reward > 0.0)


def maybe_get_env_meta_preview(dataset_path: str) -> dict[str, Any] | None:
    return get_env_metadata(dataset_path)
