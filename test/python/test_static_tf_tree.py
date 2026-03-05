"""
Python tests for static_tf.

Mirror the C++ test cases 1:1 so that a failure here flags a Python/C++ divergence.
Run standalone: pytest test/python/test_static_tf_tree.py -v
"""

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from static_tf.tree import StaticTfTree
from static_tf.loader import load_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_T(translation: list[float], q_wxyz: list[float]) -> np.ndarray:
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


@pytest.fixture
def seeker_tree() -> StaticTfTree:
    """Star topology matching the Seeker sensor layout."""
    tree = StaticTfTree(root="body")
    tree.register_transform("body", "cam_left",
        make_T([-0.0375, 0.0, 0.11], [1, 0, 0, 0]))
    tree.register_transform("body", "cam_right",
        make_T([ 0.0375, 0.0, 0.11], [1, 0, 0, 0]))
    tree.register_transform("body", "nucleus1000",
        make_T([0.134, 0.0, 0.085], [1, 0, 0, 0]))
    return tree


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    cfg = textwrap.dedent("""\
        metadata:
          reference_frame: body
        sensors:
          cam_left:
            frame_id: cam_left
            parent_frame: body
            T_body_sensor:
              translation: [-0.0375, 0.0, 0.110]
              quaternion:  [1.0, 0.0, 0.0, 0.0]
          cam_right:
            frame_id: cam_right
            parent_frame: body
            T_body_sensor:
              translation: [0.0375, 0.0, 0.110]
              quaternion:  [1.0, 0.0, 0.0, 0.0]
          nucleus1000:
            frame_id: nucleus1000
            parent_frame: body
            T_body_sensor:
              translation: [0.134, 0.0, 0.085]
              quaternion:  [1.0, 0.0, 0.0, 0.0]
    """)
    p = tmp_path / "config.yaml"
    p.write_text(cfg)
    return p


# ---------------------------------------------------------------------------
# StaticTfTree unit tests
# ---------------------------------------------------------------------------

def test_identity_same_frame(seeker_tree):
    np.testing.assert_array_almost_equal(
        seeker_tree.lookup("body", "body"), np.eye(4))
    np.testing.assert_array_almost_equal(
        seeker_tree.lookup("cam_left", "cam_left"), np.eye(4))


def test_lookup_body_to_sensor(seeker_tree):
    T = seeker_tree.lookup("body", "cam_left")
    np.testing.assert_array_almost_equal(T[:3, 3], [-0.0375, 0.0, 0.11])
    np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))


def test_lookup_sensor_to_body_is_inverse(seeker_tree):
    T_body_cam = seeker_tree.lookup("body", "cam_left")
    T_cam_body = seeker_tree.lookup("cam_left", "body")
    np.testing.assert_array_almost_equal(T_body_cam @ T_cam_body, np.eye(4), decimal=10)


def test_stereo_baseline_chain_through_root(seeker_tree):
    T_body_left  = seeker_tree.lookup("body", "cam_left")
    T_body_right = seeker_tree.lookup("body", "cam_right")
    T_left_right_expected = np.linalg.inv(T_body_left) @ T_body_right
    T_left_right = seeker_tree.lookup("cam_left", "cam_right")
    np.testing.assert_array_almost_equal(T_left_right, T_left_right_expected, decimal=10)


def test_stereo_baseline_magnitude(seeker_tree):
    T = seeker_tree.lookup("cam_left", "cam_right")
    baseline = np.linalg.norm(T[:3, 3])
    np.testing.assert_almost_equal(baseline, 0.075, decimal=6)


def test_rotated_frame(seeker_tree):
    # 90-degree yaw: sensor +X maps to body +Y
    angle = math.pi / 2.0
    q_wxyz = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]  # rot about Z
    T_body_sensor = make_T([0.1, 0.0, 0.0], q_wxyz)
    seeker_tree.register_transform("body", "rotated_sensor", T_body_sensor)

    T = seeker_tree.lookup("body", "rotated_sensor")
    p_sensor = np.array([1.0, 0.0, 0.0, 1.0])
    p_body = T @ p_sensor
    np.testing.assert_almost_equal(p_body[0], 0.1, decimal=9)   # translation in X
    np.testing.assert_almost_equal(p_body[1], 1.0, decimal=9)   # sensor X → body Y
    np.testing.assert_almost_equal(p_body[2], 0.0, decimal=9)


def test_unknown_frame_raises(seeker_tree):
    with pytest.raises(ValueError, match="Unknown frame"):
        seeker_tree.lookup("body", "nonexistent")
    with pytest.raises(ValueError, match="Unknown frame"):
        seeker_tree.lookup("nonexistent", "body")


def test_duplicate_registration_raises(seeker_tree):
    with pytest.raises(ValueError, match="already registered"):
        seeker_tree.register_transform("body", "cam_left", np.eye(4))


def test_root_as_child_raises(seeker_tree):
    with pytest.raises(ValueError, match="root frame"):
        seeker_tree.register_transform("body", "body", np.eye(4))


def test_has_frame(seeker_tree):
    assert seeker_tree.has_frame("body")
    assert seeker_tree.has_frame("cam_left")
    assert not seeker_tree.has_frame("nonexistent")


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

def test_load_from_yaml(config_file):
    tree = load_tree(config_file)
    assert tree.has_frame("cam_left")
    assert tree.has_frame("cam_right")
    assert tree.has_frame("nucleus1000")

    T = tree.lookup("cam_left", "cam_right")
    np.testing.assert_almost_equal(np.linalg.norm(T[:3, 3]), 0.075, decimal=6)


def test_load_with_changeset(tmp_path: Path):
    base = textwrap.dedent("""\
        metadata:
          reference_frame: body
        sensors:
          nucleus1000:
            parent_frame: body
            T_body_sensor:
              translation: [0.120, 0.0, 0.085]
              quaternion:  [1.0, 0.0, 0.0, 0.0]
          cam_left:
            parent_frame: body
            T_body_sensor:
              translation: [-0.0375, 0.0, 0.110]
              quaternion:  [1.0, 0.0, 0.0, 0.0]
    """)
    changeset = textwrap.dedent("""\
        sensors:
          nucleus1000:
            T_body_sensor:
              translation: [0.134, 0.0, 0.085]
              quaternion:  [1.0, 0.0, 0.0, 0.0]
    """)
    base_path = tmp_path / "base.yaml"
    cs_path   = tmp_path / "changeset.yaml"
    base_path.write_text(base)
    cs_path.write_text(changeset)

    tree = load_tree(base_path, cs_path)
    # nucleus should use the changeset value 0.134, not base 0.120
    T = tree.lookup("body", "nucleus1000")
    np.testing.assert_almost_equal(T[0, 3], 0.134, decimal=9)


def test_load_non_unit_quaternion_raises(tmp_path: Path):
    cfg = textwrap.dedent("""\
        metadata:
          reference_frame: body
        sensors:
          cam_left:
            parent_frame: body
            T_body_sensor:
              translation: [0.0, 0.0, 0.0]
              quaternion:  [2.0, 0.0, 0.0, 0.0]
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(cfg)
    with pytest.raises(ValueError, match="not unit norm"):
        load_tree(p)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_tree("/tmp/does_not_exist_static_tf.yaml")
