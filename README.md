# static_tf

A lightweight static transform tree library for managing sensor frame extrinsics in autonomous systems.

Designed for SLAM, navigation, and perception pipelines where all sensor-to-body transforms are fixed at initialisation. Provides identical C++ and Python APIs, YAML-based configuration, and an optional 3D visualisation tool.

---

## Features

- **Zero runtime overhead** — transforms are registered once at startup; lookups traverse the tree in O(depth) with no allocations
- **Dual language support** — C++ (Eigen3) and Python (NumPy) implementations with matching APIs
- **YAML configuration** — all sensor extrinsics live in a single config file; no code changes required to update calibration values
- **Changeset merging** — Python loader deep-merges a base config with a deployment-specific override YAML
- **Tree invariant enforcement** — duplicate frames, cycles, and disconnected subtrees are detected at registration time
- **3D visualisation** — interactive matplotlib plot of the frame tree with RGB axis triads

---

## Requirements

### C++

| Dependency | Version | Notes |
|------------|---------|-------|
| CMake | ≥ 3.16 | |
| C++ compiler | C++17 | |
| Eigen3 | ≥ 3.3 | Re-exported to consumers |
| yaml-cpp | any | Private implementation dependency |
| Google Test | any | Tests only |

### Python

| Dependency | Notes |
|------------|-------|
| numpy | Array math |
| scipy | Quaternion → rotation matrix |
| ruamel.yaml | YAML loading with changeset merge |
| matplotlib | Visualisation (optional) |
| pytest | Tests only |

---

## Building (ROS 2 / colcon)

```bash
cd <ros2_ws>
colcon build --packages-select static_tf
source install/setup.bash
```

### CMake standalone

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local
```

---

## Quick Start

### C++

```cpp
#include "static_tf/static_tf_tree.hpp"
#include "static_tf/sensor_frame_loader.hpp"

// Load from YAML
static_tf::StaticTfTree tree = static_tf::load_from_yaml("config/seeker.yaml");

// Lookup T_body_cam_fwd: maps a point from cam_fwd frame into body frame
Eigen::Isometry3d T = tree.lookup("body", "cam_fwd");

// Convenience accessors
Eigen::Matrix3d R = tree.lookup_rotation("body", "cam_fwd");
Eigen::Vector3d t = tree.lookup_translation("body", "cam_fwd");

// Programmatic registration
Eigen::Vector3d translation(0.1, 0.0, 0.05);
Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
tree.register_transform("body", "my_sensor", translation, q);
```

### Python

```python
from static_tf.loader import load_tree

# Load from YAML
tree = load_tree("config/seeker.yaml")

# Lookup T_body_cam_fwd (4x4 ndarray)
T = tree.lookup("body", "cam_fwd")

# Convenience accessors
R = tree.lookup_rotation("body", "cam_fwd")   # 3x3
t = tree.lookup_translation("body", "cam_fwd") # shape (3,)

# With a deployment-specific changeset
tree = load_tree("config/seeker.yaml", changeset_path="config/vehicle-002.yaml")
```

### Visualisation

```bash
python -m static_tf.viz config/seeker.yaml          # ASCII tree only
python -m static_tf.viz config/seeker.yaml --viz    # + interactive 3D plot
python -m static_tf.viz config/seeker.yaml --viz \
    --changeset config/vehicle-002.yaml \
    --axis-scale 0.05
```

---

## Config File

Sensor extrinsics are declared in YAML. See [`config/seeker.yaml`](config/seeker.yaml) for a complete example.

```yaml
metadata:
  vehicle: "seeker-class"
  reference_frame: "body"          # root frame of the tree
  last_calibrated: "UNCALIBRATED"

sensors:
  cam_fwd:
    parent_frame: "body"
    T_body_sensor:
      translation: [0.0375, 0.0, 0.055]   # metres, [x, y, z]
      quaternion:  [1.0, 0.0, 0.0, 0.0]   # Hamilton [w, x, y, z], unit norm
```

**Transform convention:** `T_body_sensor` maps a point from the sensor frame into the parent (body) frame:

```
p_body = T_body_sensor * p_sensor
```

**Coordinate frame:** body follows NED — x-forward, y-starboard, z-down.

**Quaternion convention:** Hamilton, scalar-first `[w, x, y, z]`. Must be unit norm (|q| = 1 within 1×10⁻⁶).

### Changeset files

A changeset overrides only the keys it specifies; all other values are inherited from the base config:

```yaml
# config/vehicle-002.yaml
sensors:
  nucleus1000:
    T_body_sensor:
      translation: [0.0, 0.0, 0.062]
      quaternion:  [1.0, 0.0, 0.0, 0.0]
```

> **Note:** Changeset merging is Python-only. For C++, produce a merged file externally (e.g., using the Python loader) and pass it to `load_from_yaml`.

---

## API Reference

### `StaticTfTree`

Available in both C++ (`static_tf::StaticTfTree`) and Python (`static_tf.tree.StaticTfTree`).

| Method | Description |
|--------|-------------|
| `register_transform(parent, child, T)` | Register a parent→child edge. Throws on duplicate child or cycle. |
| `lookup(target, source)` | Returns `T_target_source` — maps points from `source` to `target`. |
| `lookup_rotation(target, source)` | 3×3 rotation component of `T_target_source`. |
| `lookup_translation(target, source)` | 3-vector translation component of `T_target_source`. |
| `has_frame(frame)` | Returns true if `frame` is registered (including root). |
| `frames()` | List of all registered child frame ids (excludes root). |
| `root()` | The root frame id. |

All lookup methods support inverse queries automatically (e.g., `lookup("cam_fwd", "body")`).

### `load_from_yaml` (C++)

```cpp
#include "static_tf/sensor_frame_loader.hpp"
static_tf::StaticTfTree tree = static_tf::load_from_yaml("/path/to/config.yaml");
```

Throws `std::runtime_error` on file/parse errors, `std::invalid_argument` on semantic errors (missing fields, non-unit quaternion, duplicate frame).

### `load_tree` / `load_config` (Python)

```python
from static_tf.loader import load_tree, load_config

tree = load_tree("base.yaml")
tree = load_tree("base.yaml", changeset_path="override.yaml")

cfg  = load_config("base.yaml", changeset_path="override.yaml")  # raw dict
```

---

## Running Tests

### C++ (via colcon)

```bash
colcon test --packages-select static_tf
colcon test-result --verbose
```

### C++ (standalone CTest)

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

### Python

```bash
pytest test/python/ -v
```

---

## Repository Layout

```
static_tf/
├── config/
│   └── seeker.yaml                  # Seeker AUV base config
├── include/static_tf/
│   ├── static_tf_tree.hpp           # C++ transform tree (header-only)
│   └── sensor_frame_loader.hpp      # C++ YAML loader interface
├── src/
│   └── sensor_frame_loader.cpp      # C++ YAML loader implementation
├── python/static_tf/
│   ├── tree.py                      # Python transform tree
│   ├── loader.py                    # Python YAML loader + changeset merge
│   └── viz.py                       # 3D visualisation tool
├── test/
│   ├── cpp/test_static_tf_tree.cpp  # C++ unit tests (Google Test)
│   └── python/test_static_tf_tree.py # Python unit tests (pytest)
├── CMakeLists.txt
├── package.xml
├── DESIGN_REQUIREMENTS.md
└── IMPLEMENTATION.md
```

---

## License

MIT. See package.xml for maintainer contact.
