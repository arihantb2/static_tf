"""Static transform tree — Python implementation.

Mirrors the C++ StaticTfTree API exactly so evaluation scripts and
the SLAM system operate on the same logical interface.

Convention:
    T_parent_child maps a point in child frame into parent frame:
        p_parent = T_parent_child @ p_child

    lookup(target, source) returns T_target_source (4x4 ndarray):
        p_target = T_target_source @ p_source
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class StaticTfTree:
    """Static transform tree for sensor frame management."""

    def __init__(self, root: str = "body") -> None:
        self.root = root
        # Maps child -> (parent, T_parent_child as 4x4 ndarray)
        self._edges: dict[str, tuple[str, np.ndarray]] = {}

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register_transform(
        self,
        parent: str,
        child: str,
        T: np.ndarray,
    ) -> None:
        """Register T_parent_child (4x4) for a child frame.

        Args:
            parent: Parent frame id.
            child:  Child frame id. Must be unique.
            T:      4x4 SE(3) transform. T @ p_child = p_parent.
        """
        if child in self._edges:
            raise ValueError(f"Frame already registered: {child}")
        if child == self.root:
            raise ValueError(f"Cannot register root frame as child: {child}")
        if T.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4, got {T.shape}")
        self._edges[child] = (parent, T.copy())

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def lookup(self, target: str, source: str) -> np.ndarray:
        """Returns T_target_source (4x4 ndarray).

        T_target_source = T_root_target^{-1} @ T_root_source
        """
        if target == source:
            return np.eye(4)
        T_root_source = self._chain_to_root(source)
        T_root_target = self._chain_to_root(target)
        return np.linalg.inv(T_root_target) @ T_root_source

    def lookup_rotation(self, target: str, source: str) -> np.ndarray:
        """Returns the 3x3 rotation component of T_target_source."""
        return self.lookup(target, source)[:3, :3]

    def lookup_translation(self, target: str, source: str) -> np.ndarray:
        """Returns the 3-vector translation component of T_target_source."""
        return self.lookup(target, source)[:3, 3]

    def has_frame(self, frame: str) -> bool:
        return frame == self.root or frame in self._edges

    def frames(self) -> list[str]:
        """Returns all registered child frame ids (does not include root)."""
        return list(self._edges.keys())

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _chain_to_root(self, start: str) -> np.ndarray:
        """Returns T_root_start by composing transforms upward.

        T_root_start satisfies: p_root = T_root_start @ p_start
        Derivation matches the C++ implementation exactly.
        """
        if not self.has_frame(start):
            raise ValueError(f"Unknown frame: '{start}'")

        T = np.eye(4)
        current = start
        max_depth = len(self._edges) + 1
        depth = 0

        while current != self.root:
            if current not in self._edges:
                raise ValueError(
                    f"Frame '{current}' is not reachable from root '{self.root}'. "
                    "Check parent_frame in config."
                )
            parent, T_parent_current = self._edges[current]
            # T_root_current = T_root_parent @ T_parent_current
            T = T_parent_current @ T
            current = parent

            depth += 1
            if depth > max_depth:
                raise ValueError(f"Cycle detected in transform tree at: {current}")

        return T
