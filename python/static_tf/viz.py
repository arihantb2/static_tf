#!/usr/bin/env python3
"""
static_tf_viz.py — Load a static_tf YAML config, print the transform tree,
and optionally show an interactive 3D visualization of all sensor frames.

Usage:
    # Print tree only
    python3 static_tf_viz.py config/seeker_base.yaml

    # Print tree + 3D plot
    python3 static_tf_viz.py config/seeker_base.yaml --viz

    # With deployment changeset
    python3 static_tf_viz.py config/seeker_base.yaml --changeset config/deploy.yaml --viz

    # Adjust axis scale (metres)
    python3 static_tf_viz.py config/seeker_base.yaml --viz --axis-scale 0.05

    # Save 2-D projection PDFs (XY top-down and XZ profile) to a directory
    python3 static_tf_viz.py config/seeker_base.yaml --output-dir figures/

    # Save projections and show 3D view
    python3 static_tf_viz.py config/seeker_base.yaml --viz --output-dir figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Minimal self-contained tree (no ruamel dependency for standalone script)
# ---------------------------------------------------------------------------

class StaticTfTree:
    def __init__(self, root: str = "body") -> None:
        self.root = root
        self._edges: dict[str, tuple[str, np.ndarray]] = {}

    def register(self, parent: str, child: str, T: np.ndarray) -> None:
        if child in self._edges:
            raise ValueError(f"Duplicate frame: {child}")
        self._edges[child] = (parent, T.copy())

    def lookup(self, target: str, source: str) -> np.ndarray:
        if target == source:
            return np.eye(4)
        return np.linalg.inv(self._chain_to_root(target)) @ self._chain_to_root(source)

    def frames(self) -> list[str]:
        return list(self._edges.keys())

    def parent_of(self, frame: str) -> Optional[str]:
        return self._edges[frame][0] if frame in self._edges else None

    def children_of(self, frame: str) -> list[str]:
        return [c for c, (p, _) in self._edges.items() if p == frame]

    def _chain_to_root(self, start: str) -> np.ndarray:
        T = np.eye(4)
        current = start
        visited = set()
        while current != self.root:
            if current not in self._edges:
                raise ValueError(f"Unknown frame: '{current}'")
            if current in visited:
                raise ValueError(f"Cycle at: {current}")
            visited.add(current)
            parent, T_parent_current = self._edges[current]
            T = T_parent_current @ T
            current = parent
        return T


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _make_T(translation: list, quaternion: list) -> np.ndarray:
    t = np.asarray(translation, dtype=float)
    q = np.asarray(quaternion, dtype=float)  # [w, x, y, z]
    if abs(np.linalg.norm(q) - 1.0) > 1e-4:
        raise ValueError(f"Non-unit quaternion: {q} (norm={np.linalg.norm(q):.6f})")
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_config(base_path: str | Path,
                changeset_path: str | Path | None = None) -> dict:
    def deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if changeset_path:
        with open(changeset_path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f))
    return cfg


def build_tree(cfg: dict) -> StaticTfTree:
    root = cfg.get("metadata", {}).get("reference_frame", "body")
    tree = StaticTfTree(root=root)
    sensors = cfg.get("sensors", {})

    # Two-pass registration so non-body parents resolve correctly
    registered = {root}
    remaining = dict(sensors)
    max_passes = len(sensors) + 1

    for _ in range(max_passes):
        if not remaining:
            break
        for name, s in list(remaining.items()):
            parent = s.get("parent_frame", root)
            if parent in registered:
                T = _make_T(
                    s["T_body_sensor"]["translation"],
                    s["T_body_sensor"]["quaternion"],
                )
                tree.register(parent, name, T)
                registered.add(name)
                del remaining[name]

    if remaining:
        raise ValueError(f"Could not register frames (unknown parents?): {list(remaining)}")

    return tree


# ---------------------------------------------------------------------------
# Tree printing
# ---------------------------------------------------------------------------

def print_tree(tree: StaticTfTree, cfg: dict) -> None:
    sensors = cfg.get("sensors", {})

    def fmt_transform(frame: str) -> str:
        if frame == tree.root:
            return ""
        parent, T = tree._edges[frame]
        t = T[:3, 3]
        q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()  # scipy: [x,y,z,w]
        q = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]   # back to [w,x,y,z]
        return (f"  t=[{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m  "
                f"q=[{q[0]:.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f}]")

    def walk(frame: str, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        tf_str    = fmt_transform(frame)
        print(f"{prefix}{connector if prefix else ''}{frame}{tf_str}")

        children   = tree.children_of(frame)
        # For root (prefix=""), children start with 4-space indent.
        # For deeper nodes, extend the existing prefix.
        extension  = "    " if is_last else "│   "
        new_prefix = (prefix + extension) if prefix else "    "

        for i, child in enumerate(children):
            walk(child, new_prefix, is_last=(i == len(children) - 1))

    meta = cfg.get("metadata", {})
    print()
    print(f"  Vehicle  : {meta.get('vehicle', 'unknown')}")
    print(f"  Root     : {tree.root}")
    print(f"  Frames   : {len(tree.frames())} sensors + root")
    print(f"  Calibrated: {meta.get('last_calibrated', 'unknown')}")
    print()
    walk(tree.root)
    print()


# ---------------------------------------------------------------------------
# 3D visualization
# ---------------------------------------------------------------------------

# Colour palette per frame — cycles if more frames than colours
_FRAME_COLOURS = [
    "#e74c3c",  # red
    "#2ecc71",  # green
    "#3498db",  # blue
    "#f39c12",  # orange
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#e67e22",  # dark orange
    "#34495e",  # dark grey
]

# Axis colours: x=red, y=green, z=blue (RGB = XYZ)
_AXIS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
_AXIS_LABELS = ["x", "y", "z"]


def _draw_frame(ax, T: np.ndarray, label: str, scale: float,
                colour: str, label_offset: np.ndarray) -> None:
    """Draw a coordinate frame as three RGB axis arrows at transform T."""
    origin_d = T[:3, 3]
    R = T[:3, :3]

    for i, (col, lbl) in enumerate(zip(_AXIS_COLORS, _AXIS_LABELS)):
        # Axis direction in NED, then mapped to display
        axis_display = R[:, i]
        ax.quiver(*origin_d, *axis_display * scale,
                  color=col, linewidth=1.5, arrow_length_ratio=0.2,
                  normalize=False)

    # Frame label
    label_pos = origin_d + label_offset
    ax.text(*label_pos, label, fontsize=8, color=colour,
            fontweight="bold", ha="center", va="center")


def _draw_parent_arrow(ax, T_parent: np.ndarray, T_child: np.ndarray,
                       colour: str) -> None:
    """Draw a solid arrow from parent origin to child origin."""
    p_start = T_parent[:3, 3]
    p_end   = T_child[:3, 3]
    delta   = p_end - p_start
    if np.linalg.norm(delta) < 1e-9:
        return
    ax.quiver(*p_start, *delta,
              color=colour, linewidth=0.8, linestyle="solid",
              arrow_length_ratio=0.05, alpha=0.6, normalize=False)


def visualize(tree: StaticTfTree, axis_scale: float = 0.05) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    all_frames = [tree.root] + tree.frames()
    colour_map = {f: _FRAME_COLOURS[i % len(_FRAME_COLOURS)]
                  for i, f in enumerate(all_frames)}

    # Collect all origins in display coords for axis scaling
    origins = []
    T_world: dict[str, np.ndarray] = {tree.root: np.eye(4)}
    for frame in tree.frames():
        T_world[frame] = tree.lookup(tree.root, frame)
        origins.append(T_world[frame][:3, 3])

    origins   = np.array(origins) if origins else np.zeros((1, 3))
    centre    = origins.mean(axis=0)
    max_range = max(np.linalg.norm(origins - centre, axis=1).max(), axis_scale * 3)

    # Label offset in display coords (above frame label in display z = world up)
    label_offset = np.array([0, 0, -1 * axis_scale * 1.8])

    # Draw root frame
    _draw_frame(ax, np.eye(4), tree.root, axis_scale,
                colour_map[tree.root], -1 * label_offset)

    # Draw sensor frames + parent arrows
    for frame in tree.frames():
        T_f      = T_world[frame]
        parent   = tree.parent_of(frame)
        T_parent = T_world[parent] if parent else np.eye(4)
        colour   = colour_map[frame]

        _draw_frame(ax, T_f, frame, axis_scale, colour, label_offset)
        _draw_parent_arrow(ax, T_parent, T_f, colour)

    # Equal-ish aspect via manual limits
    pad = max_range * 1.3
    ax.set_xlim(centre[0] - pad, centre[0] + pad)
    ax.set_ylim(centre[1] - pad, centre[1] + pad)
    ax.set_zlim(centre[2] - pad, centre[2] + pad)

    ax.set_xlabel("x  (forward)")
    ax.set_ylabel("y  (starboard)")
    ax.set_zlabel("z  (down)")

    # Legend: axis colour key + note
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=c, linewidth=2, label=f"{l}-axis")
        for c, l in zip(_AXIS_COLORS, _AXIS_LABELS)
    ]
    legend_elements += [
        Line2D([0], [0], color="grey", linewidth=1,
               linestyle="dashed", label="parent → child")
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_title(
        "static TF tree  (drag to rotate)\n"
        "Display: NED x-fwd, y-stbd, z-down",
        fontsize=9, pad=12)

    # Invert the y and z axes to match the NED convention
    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.set_box_aspect([1,1,1]) # Equal scaling for all axes

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2-D projection plots (paper-quality, saved to file)
# ---------------------------------------------------------------------------

def save_projections(tree: StaticTfTree, cfg: dict, output_dir: str | Path,
                     axis_scale: float = 0.04) -> None:
    """Save XY (top-down) and XZ (profile) projections side-by-side as a PNG.

    x (forward) is placed on the shared vertical axis so longitudinal positions
    align directly between the two views.
    Layout:  left = y(starboard) × x(forward),  right = z(down) × x(forward).
    """
    if "matplotlib" not in sys.modules:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frames = [tree.root] + tree.frames()
    colour_map = {f: _FRAME_COLOURS[i % len(_FRAME_COLOURS)]
                  for i, f in enumerate(all_frames)}

    T_world: dict[str, np.ndarray] = {tree.root: np.eye(4)}
    for frame in tree.frames():
        T_world[frame] = tree.lookup(tree.root, frame)

    # Leader-line length: 25 % of scene radius, but at least 3× axis_scale
    all_origins = np.array([T_world[f][:3, 3] for f in all_frames])
    scene_radius = max(
        np.linalg.norm(all_origins - all_origins.mean(axis=0), axis=1).max(),
        axis_scale,
    )
    leader_len = max(scene_radius * 0.30, axis_scale * 3.0)

    # Evenly-spaced label angles so no two leaders ever point the same way
    label_angles = np.linspace(np.pi / 6, np.pi / 6 + 2 * np.pi,
                               len(all_frames), endpoint=False)

    _RC = {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.dpi": 300,
    }

    def _draw_projection(ax, h_idx: int, v_idx: int) -> None:
        """Draw one projection onto *ax*.

        h_idx — world-axis index mapped to matplotlib's horizontal axis.
        v_idx — world-axis index mapped to matplotlib's vertical axis (= x=0).
        The two displayed world-frame axes (h_idx and v_idx) are drawn as
        coloured arrows at each frame origin.
        """
        # ---- per-frame: axis arrows, marker, leader + label ----
        for i, frame in enumerate(all_frames):
            T  = T_world[frame]
            R  = T[:3, :3]
            oh = T[:3, 3][h_idx]
            ov = T[:3, 3][v_idx]
            colour = colour_map[frame]

            # Project the two relevant world axes onto this plane
            for world_ax in (h_idx, v_idx):
                d = R[:, world_ax] * axis_scale
                ax.annotate(
                    "",
                    xy=(oh + d[h_idx], ov + d[v_idx]),
                    xytext=(oh, ov),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=_AXIS_COLORS[world_ax],
                        lw=1.5,
                        mutation_scale=10,
                    ),
                    zorder=3,
                )

            # Frame origin dot
            ax.scatter(oh, ov, color=colour, s=30, zorder=5, linewidths=0)

            # Leader line to label at a unique evenly-distributed angle
            ang  = label_angles[i]
            lh   = oh + np.cos(ang) * leader_len
            lv   = ov + np.sin(ang) * leader_len
            ax.plot([oh, lh], [ov, lv], color=colour, lw=0.6,
                    alpha=0.7, zorder=4)
            ax.text(
                lh, lv, frame,
                fontsize=7.5,
                color=colour,
                fontweight="bold",
                ha="center", va="center",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=colour,
                    linewidth=0.6,
                    alpha=0.92,
                ),
            )

    with plt.rc_context(_RC):
        # sharey=True → shared vertical axis = x (forward)
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2,
            figsize=(10.0, 5.5),
            sharey=True,
        )

        # ---- Left: top-down  horizontal=y(starboard)  vertical=x(forward) ----
        _draw_projection(ax_left, h_idx=1, v_idx=0)
        ax_left.set_xlabel("y  (starboard, m)")
        ax_left.set_ylabel("x  (forward, m)")
        ax_left.set_title("Top-down  (XY)", pad=8)
        ax_left.set_aspect("equal", adjustable="datalim")

        # ---- Right: profile  horizontal=z(down)  vertical=x(forward, shared) ----
        _draw_projection(ax_right, h_idx=2, v_idx=0)
        ax_right.set_xlabel("z  (down, m)")
        ax_right.set_title("Profile  (XZ)", pad=8)
        ax_right.set_aspect("equal", adjustable="datalim")

        # Legend: placed inside the right subplot to reclaim whitespace
        legend_elems = [
            Line2D([0], [0], color=_AXIS_COLORS[0], lw=1.5, label="x-axis"),
            Line2D([0], [0], color=_AXIS_COLORS[1], lw=1.5, label="y-axis"),
            Line2D([0], [0], color=_AXIS_COLORS[2], lw=1.5, label="z-axis"),
        ]
        ax_right.legend(
            handles=legend_elems,
            loc="best",
            framealpha=0.9,
            edgecolor="#cccccc",
        )

        platform = cfg.get("metadata", {}).get("name", "Sensor TF Tree")
        fig.suptitle(f"{platform} - Sensor TF Tree", fontsize=11, fontweight="bold", y=1.01)
        fig.tight_layout()

        out_path = output_dir / "tf_tree_projections.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display the static_tf sensor frame tree for the Seeker AUV."
    )
    parser.add_argument("config",
        help="Path to base YAML config (e.g. config/seeker_base.yaml)")
    parser.add_argument("--changeset", "-c", default=None,
        help="Optional deployment changeset YAML (merged on top of base)")
    parser.add_argument("--viz", "-v", action="store_true",
        help="Show interactive 3D visualisation")
    parser.add_argument("--axis-scale", "-s", type=float, default=0.04,
        help="Length of drawn axis arrows in metres (default: 0.04)")
    parser.add_argument("--output-dir", "-o", default=None, metavar="DIR",
        help="Save XY and XZ 2-D projection PNGs to DIR (implies save, "
             "not --viz)")
    args = parser.parse_args()

    # Load
    try:
        cfg  = load_config(args.config, args.changeset)
        tree = build_tree(cfg)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print tree
    print_tree(tree, cfg)

    # Visualize
    if args.viz:
        visualize(tree, axis_scale=args.axis_scale)

    # Save 2-D projections
    if args.output_dir:
        save_projections(tree, cfg, args.output_dir, axis_scale=args.axis_scale)


if __name__ == "__main__":
    main()
