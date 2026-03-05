# Implementation: static_tf

This document describes the internal design, data structures, algorithms, and key implementation decisions for the `static_tf` library.

---

## 1. Repository Structure

```
static_tf/
├── include/static_tf/
│   ├── static_tf_tree.hpp       # Core C++ transform tree — header-only
│   └── sensor_frame_loader.hpp  # C++ loader interface
├── src/
│   └── sensor_frame_loader.cpp  # C++ loader implementation (yaml-cpp)
├── python/static_tf/
│   ├── __init__.py
│   ├── tree.py                  # Python transform tree (NumPy)
│   ├── loader.py                # Python YAML loader + changeset merge
│   └── viz.py                   # 3D visualisation (matplotlib)
├── test/
│   ├── cpp/test_static_tf_tree.cpp
│   └── python/test_static_tf_tree.py
├── config/
│   └── seeker.yaml
├── CMakeLists.txt
└── package.xml
```

---

## 2. Core Data Structure: Transform Tree

### 2.1 Representation

The tree is stored as an **edge map** rather than a node list. Each registered child frame holds:

1. Its **parent frame id** (a string).
2. The **rigid body transform** `T_parent_child` from child to parent (a 4×4 homogeneous matrix / `Eigen::Isometry3d`).

```
edges_: { child_id → (parent_id, T_parent_child) }
```

The root frame is implicit — it has no entry in `edges_` and acts as the traversal termination condition.

**Why edge map, not adjacency list?**
Lookups traverse from frame to root, so the natural access pattern is child → parent. An edge map gives O(1) lookup per hop with minimal memory overhead.

### 2.2 Frame Identifier

Both implementations use plain strings (`std::string` in C++, `str` in Python) as frame identifiers. No integer IDs or enumerations are used — string IDs match the YAML config directly and avoid an additional mapping layer.

### 2.3 Transform Representation

| Layer | C++ | Python |
|-------|-----|--------|
| Per-edge storage | `Eigen::Isometry3d` (4×4 SE(3)) | `np.ndarray` shape `(4, 4)` |
| Lookup result | `Eigen::Isometry3d` | `np.ndarray` shape `(4, 4)` |
| Math backend | Eigen3 | NumPy |

`Isometry3d` is preferred over `Affine3d` in C++ because it constrains the bottom row to `[0, 0, 0, 1]` and uses a more numerically stable internal representation (separate `linear` and `translation` storage).

---

## 3. Transform Lookup Algorithm

### 3.1 `chain_to_root(frame)`

The fundamental primitive. Returns `T_root_frame` — the composed transform from `frame` up to the root:

```
p_root = T_root_frame * p_frame
```

**Algorithm:**

```
T ← Identity
current ← frame

while current ≠ root:
    (parent, T_parent_current) ← edges_[current]
    T ← T_parent_current * T          # prepend at each hop
    current ← parent
    depth++
    if depth > |edges_| + 1: raise CycleError

return T
```

The prepend order follows from the chain rule of homogeneous transforms:

```
T_root_frame = T_root_A * T_A_B * T_B_frame
```

Each iteration extends the chain one level toward the root, so the new `T_parent_current` is prepended (left-multiplied).

### 3.2 `lookup(target, source)`

```
T_target_source = T_root_target⁻¹ * T_root_source
```

Both frames are chained to the root independently, then combined. This automatically handles:

- **Forward lookups** (e.g., `lookup("body", "cam_fwd")`) — `T_root_target` is identity, result is `T_root_source`.
- **Inverse lookups** (e.g., `lookup("cam_fwd", "body")`) — `T_root_source` is identity, result is `T_root_target⁻¹`.
- **Cross-branch lookups** (e.g., `lookup("cam_left", "cam_right")`) — chains for both branches are computed and combined through root.

**C++ inverse:** `Isometry3d::inverse()` exploits the SE(3) structure — `R⁻¹ = Rᵀ` and `t⁻¹ = -Rᵀt` — and is numerically exact within floating-point precision.

**Python inverse:** `np.linalg.inv()` is used for simplicity. For SE(3) matrices this is equivalent to the analytic inverse; the general LU decomposition is slightly more work than necessary but avoids a separate SE(3) inversion helper.

### 3.3 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `register_transform` | O(1) | O(1) per frame |
| `chain_to_root(f)` | O(depth(f)) | O(1) |
| `lookup(t, s)` | O(depth(t) + depth(s)) | O(1) |

For the Seeker star topology (all sensors are direct children of `body`), every lookup is O(1) hops per frame — effectively O(1) total.

---

## 4. Registration and Invariant Enforcement

`register_transform` enforces three invariants before inserting an edge:

| Invariant | Check | Error |
|-----------|-------|-------|
| No duplicate frames | `child` already in `edges_` | C++: `std::invalid_argument`; Python: `ValueError` |
| Root is not a child | `child == root_` | Same |
| Valid transform shape | `T.shape != (4, 4)` | Python only (Eigen is typed) |

Cycle detection is deferred to query time via the depth counter in `chain_to_root`. This avoids an O(n) reachability check on every registration and is acceptable because cycles can only be introduced by incorrectly specifying parent frames, not by normal usage.

---

## 5. YAML Loader

### 5.1 C++ Loader (`sensor_frame_loader.cpp`)

The C++ loader is a single function `load_from_yaml(path)`:

1. **Parse file** — `YAML::LoadFile()` with `BadFile` and `ParserException` caught and rethrown as `std::runtime_error`.
2. **Extract root** — `metadata.reference_frame`, defaulting to `"body"`.
3. **Iterate sensors** — for each entry in `sensors`, extract `parent_frame`, `translation` (3 doubles), and `quaternion` (4 doubles in `[w, x, y, z]` order).
4. **Validate quaternion norm** — `|q| - 1.0 > 1e-4` throws `std::runtime_error`.
5. **Register** — calls `tree.register_transform(parent, name, translation_vec, quaternion)`.

The C++ loader reads a single flat file only. Changeset merging must be performed externally (e.g., with the Python loader) before passing the file to C++.

### 5.2 Python Loader (`loader.py`)

The Python loader is split into two stages:

**Stage 1 — `load_config(base_path, changeset_path=None)`**

Loads YAML using `ruamel.yaml` with `typ="safe"` (plain Python dicts/lists). If a changeset is provided, `_deep_merge(base, changeset)` is applied.

**Stage 2 — `load_tree(base_path, changeset_path=None)`**

Calls `load_config`, reads `metadata.reference_frame`, instantiates `StaticTfTree`, and calls `register_transform` for each sensor after constructing the 4×4 matrix via `_make_T`.

### 5.3 Changeset Deep Merge

```python
def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val   # scalar or list — override wholesale
    return result
```

**Key decision — lists are replaced wholesale.** A partial override of a 4-element quaternion or 3-element translation does not make physical sense. Replacing the entire list prevents silent partial-update bugs.

---

## 6. Quaternion Handling

Both loaders expect quaternions in **Hamilton convention, scalar-first**: `[w, x, y, z]`.

| Library | Expected order | Conversion required |
|---------|----------------|---------------------|
| `Eigen::Quaterniond` constructor | `(w, x, y, z)` | None — direct construction |
| `scipy.spatial.transform.Rotation.from_quat` | `[x, y, z, w]` | Reorder: `[q[1], q[2], q[3], q[0]]` |

The reordering in `_make_T` is explicit and documented at the call site:

```python
# scipy.spatial.transform.Rotation expects [x, y, z, w]
R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
```

Unit-norm validation tolerance is `1e-4` in both implementations, matching typical floating-point precision of YAML-serialised values.

---

## 7. Visualisation Tool (`viz.py`)

The visualisation script is intentionally **self-contained** — it re-implements a minimal version of `StaticTfTree` internally rather than importing `static_tf.tree`. This makes it runnable in environments where only `matplotlib` and `ruamel.yaml` are available (no SciPy needed for display-only use).

### Rendering

Each frame is rendered as an RGB triad:

- **Red arrow** — local x-axis
- **Green arrow** — local y-axis
- **Blue arrow** — local z-axis

Arrow length is controlled by `--axis-scale` (default 0.05 m).

Parent-to-child relationships are shown as dashed grey arrows from the parent frame origin to the child frame origin.

Frame labels are placed at the tip of the z-axis arrow with a random colour per frame for readability.

The 3D axes are scaled equal (`ax.set_aspect("equal")` via manual range computation) to prevent distortion from unequal axis ranges.

---

## 8. C++ / Python Parity Strategy

The Python `StaticTfTree` is a deliberate mirror of the C++ class:

- Same method names, same argument order, same semantics.
- Same error conditions (with language-appropriate exception types).
- `_chain_to_root` derivation is documented to match C++ step-for-step.

The test suite enforces this explicitly: `test/python/test_static_tf_tree.py` mirrors `test/cpp/test_static_tf_tree.cpp` test-for-test. A failure in Python that passes in C++ (or vice versa) immediately surfaces divergence.

---

## 9. Build System

### 9.1 Library target

```cmake
add_library(static_tf SHARED src/sensor_frame_loader.cpp)
```

Only `sensor_frame_loader.cpp` is compiled. `static_tf_tree.hpp` is header-only — it uses only Eigen, which is itself header-only, so there is nothing to compile. This keeps the build fast and avoids ODR issues.

### 9.2 Dependency exposure

```cmake
target_link_libraries(static_tf
  PUBLIC  Eigen3::Eigen    # consumers get Eigen headers transitively
  PRIVATE yaml-cpp         # yaml-cpp is an impl detail; not exposed in headers
)
```

`yaml-cpp` is kept private because `sensor_frame_loader.hpp` does not include any yaml-cpp headers. Consumers of `static_tf` only need Eigen, which is forwarded via `PUBLIC`.

### 9.3 Downstream usage

After `find_package(static_tf)`, consumers link with:

```cmake
target_link_libraries(my_target static_tf::static_tf)
```

The exported CMake config (`static_tfTargets.cmake`) provides the alias target and forwards the Eigen3 dependency automatically.

### 9.4 Python package

```cmake
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR python/${PROJECT_NAME}
)
```

Installs `python/static_tf/` as the `static_tf` Python package into the colcon install space, importable as `from static_tf.tree import StaticTfTree` after `source install/setup.bash`.

---

## 10. Testing

### 10.1 Test Coverage

| Test case | C++ | Python |
|-----------|-----|--------|
| Identity (same frame) | ✓ | ✓ |
| Forward lookup (body → sensor) | ✓ | ✓ |
| Inverse consistency (`T * T⁻¹ = I`) | ✓ | ✓ |
| Cross-branch lookup (sensor → sensor) | ✓ | ✓ |
| Stereo baseline magnitude | ✓ | ✓ |
| Rotated frame (90° yaw) | ✓ | ✓ |
| Unknown frame raises | ✓ | ✓ |
| Duplicate registration raises | ✓ | ✓ |
| Root as child raises | ✓ | ✓ |
| `has_frame` | ✓ | ✓ |
| Load from YAML | ✓ | ✓ |
| Load missing file raises | ✓ | ✓ |
| Non-unit quaternion raises | ✓ | ✓ |
| Changeset merge | — | ✓ |

### 10.2 Numerical tolerances

- Transform element comparisons: `decimal=9` (1×10⁻⁹) for composition results.
- Baseline magnitude: `decimal=6` (1×10⁻⁶ m) — looser due to cumulative floating-point error.
- Quaternion norm validation: `1e-4` — matches YAML serialisation precision.

---

## 11. Known Limitations

| Limitation | Detail |
|------------|--------|
| Static transforms only | No time-varying or dynamic frame support |
| Single root | One tree per `StaticTfTree` instance; no multi-root forest |
| No C++ changeset merge | Base + override merging is Python-only; C++ reads one file |
| Star topology assumed in config | Seeker config is all-to-body; deep trees are supported by the algorithm but untested at depth |
| No serialisation | No method to dump a tree back to YAML |
| No thread safety on write | `register_transform` is not thread-safe; all registration must complete before concurrent reads begin |
