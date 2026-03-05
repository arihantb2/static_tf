"""YAML config loader for StaticTfTree.

Supports both a single config file and a base + changeset pair.
The deep merge here is the Python-side equivalent of loading a
pre-merged file — it is the authoritative merge implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation

from static_tf.tree import StaticTfTree


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on scalar conflicts.

    Lists are replaced wholesale — correct for translation/quaternion arrays
    where partial override makes no sense.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(
    base_path: str | Path,
    changeset_path: str | Path | None = None,
) -> dict:
    """Load and optionally merge a base config with a deployment changeset.

    Args:
        base_path:      Path to the base YAML config (all sensors, nominal values).
        changeset_path: Optional path to a deployment-specific changeset YAML.
                        Only keys present in the changeset override the base.

    Returns:
        Merged config dict. Plain Python types (typ='safe').
    """
    yaml = YAML(typ="safe")

    with open(base_path) as f:
        base: dict = yaml.load(f)

    if changeset_path is None:
        return base

    with open(changeset_path) as f:
        changeset: dict = yaml.load(f)

    return _deep_merge(base, changeset)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _make_T(translation: list[float], quaternion: list[float]) -> np.ndarray:
    """Build 4x4 SE(3) from translation [x,y,z] and quaternion [w,x,y,z]."""
    t = np.array(translation, dtype=float)
    q_wxyz = np.array(quaternion, dtype=float)

    if abs(np.linalg.norm(q_wxyz) - 1.0) > 1e-4:
        raise ValueError(
            f"Quaternion is not unit norm: {q_wxyz} (norm={np.linalg.norm(q_wxyz):.6f})"
        )

    # scipy.spatial.transform.Rotation expects [x, y, z, w]
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


def load_tree(
    base_path: str | Path,
    changeset_path: str | Path | None = None,
) -> StaticTfTree:
    """Load a StaticTfTree from YAML config (with optional changeset merge).

    Args:
        base_path:      Path to base YAML config.
        changeset_path: Optional deployment changeset. See load_config().

    Returns:
        StaticTfTree ready for lookup() queries.

    Raises:
        ValueError: On malformed config or duplicate frame registration.
        FileNotFoundError: If either path does not exist.
    """
    cfg = load_config(base_path, changeset_path)

    root = cfg.get("metadata", {}).get("reference_frame", "body")
    tree = StaticTfTree(root=root)

    sensors: dict = cfg.get("sensors", {})
    if not sensors:
        raise ValueError(f"Config has no 'sensors' key: {base_path}")

    for name, s in sensors.items():
        parent = s.get("parent_frame")
        if parent is None:
            raise ValueError(f"Sensor '{name}' missing 'parent_frame'")

        T_node = s.get("T_body_sensor")
        if T_node is None:
            raise ValueError(f"Sensor '{name}' missing 'T_body_sensor'")

        T = _make_T(T_node["translation"], T_node["quaternion"])
        tree.register_transform(parent=parent, child=name, T=T)

    return tree
