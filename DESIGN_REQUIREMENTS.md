# Design Requirements: static_tf

## 1. Project Purpose

`static_tf` is a lightweight, zero-runtime-overhead static transform tree library for managing sensor frame extrinsics in autonomous systems. It is designed for SLAM, navigation, and perception pipelines where all sensor-to-body transforms are fixed at initialisation and do not change during operation.

The library is primarily developed for the **Seeker-class Autonomous Underwater Vehicle (AUV)** and targets systems where:

- Sensor extrinsics are determined offline (from CAD, measurement, or calibration).
- A single authoritative YAML config file captures all frame relationships.
- Multiple deployments may require per-vehicle overrides to a shared base configuration.

---

## 2. Capabilities

### 2.1 Transform Tree Management

- Maintains a directed acyclic graph (DAG) of coordinate frames rooted at a single reference frame (typically `body`).
- Registers parent–child edges, each carrying a full 6-DOF rigid body transform (rotation + translation).
- Composes transforms along the tree path to answer arbitrary frame-to-frame queries.
- Enforces tree invariants at registration time: no duplicate edges, no cycles, no registration of the root frame as a child.

### 2.2 Transform Lookup

| Method | Returns |
|--------|---------|
| `lookup(target, source)` | Full 4×4 homogeneous transform `T_target_source` |
| `lookup_rotation(target, source)` | 3×3 rotation matrix component |
| `lookup_translation(target, source)` | 3-vector translation component |

Inverse lookups (e.g., `lookup("nucleus1000", "body")`) are supported automatically by inverting the composed transform.

### 2.3 YAML-Based Configuration

- All sensor extrinsics are declared in a YAML config file — no code changes are needed to update calibration values.
- The C++ loader validates the config at load time (file existence, required fields, quaternion unit-norm) and raises typed exceptions on failure.
- The Python loader additionally supports **changeset merging**: a base config and a deployment-specific override YAML are deep-merged before the tree is built. Lists (e.g., `translation`, `quaternion`) are replaced wholesale; map keys are merged recursively.

### 2.4 Dual Language Support (C++ and Python)

Both implementations expose an identical public API and share the same YAML config format:

| Feature | C++ | Python |
|---------|-----|--------|
| Core tree | `StaticTfTree` (header-only) | `StaticTfTree` (`tree.py`) |
| YAML loader | `SensorFrameLoader` (`sensor_frame_loader.cpp`) | `load_tree()` / `load_config()` (`loader.py`) |
| Changeset merge | Not supported (merge externally) | `load_config(path, changeset=path)` |
| Math backend | Eigen3 | NumPy + SciPy |

### 2.5 3D Visualisation

An interactive visualisation tool (`python/static_tf/viz.py`) renders the transform tree as a 3D matplotlib figure:

- RGB triads (x=red, y=green, z=blue) drawn at each frame origin.
- Dashed arrows from each parent frame to each child frame.
- ASCII tree printed to stdout alongside the 3D plot.
- CLI flags: `--viz` (enable plot), `--changeset <path>` (load override), `--axis-scale <float>` (resize triads).

### 2.6 Build and Integration

- CMake package with `find_package(static_tf)` support via exported CMake config.
- ROS 2–compatible `package.xml` (format 3, `ament_cmake`).
- C++ tests via Google Test; Python tests via pytest. Both suites are enabled by default and exercise identical behaviour to guard against Python/C++ divergence.

---

## 3. Config File Structure

All sensor extrinsics are declared in a YAML file following this schema:

```yaml
metadata:
  vehicle: "<vehicle-identifier>"       # Human-readable vehicle name / serial number
  reference_frame: "<frame-name>"       # Root frame of the transform tree (default: "body")
  last_calibrated: "<date | UNCALIBRATED>"  # ISO 8601 date or "UNCALIBRATED"

sensors:
  <sensor_name>:                        # Unique identifier for this sensor / frame
    parent_frame: "<frame-name>"        # Parent frame in the tree (must already exist or be the root)
    T_body_sensor:                      # Rigid body transform from <sensor_name> frame to parent_frame
      translation: [x, y, z]           # Position of sensor origin in parent frame, metres
      quaternion: [w, x, y, z]         # Orientation of sensor frame in parent frame, Hamilton convention
                                        # Must be unit-norm (|q| = 1 within 1 × 10⁻⁶ tolerance)
  # ... additional sensors
```

### 3.1 Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metadata.vehicle` | string | Yes | Vehicle identifier for traceability |
| `metadata.reference_frame` | string | Yes | Name of the root frame (must appear as `parent_frame` in at least one sensor, or be implicit) |
| `metadata.last_calibrated` | string | Yes | Date of last calibration in ISO 8601, or the literal `"UNCALIBRATED"` |
| `sensors.<name>` | map | Yes (≥1 sensor) | One entry per sensor/frame |
| `sensors.<name>.parent_frame` | string | Yes | Name of the parent frame; must be the root or another declared sensor |
| `sensors.<name>.T_body_sensor.translation` | list[float, 3] | Yes | Translation [x, y, z] in metres |
| `sensors.<name>.T_body_sensor.quaternion` | list[float, 4] | Yes | Quaternion [w, x, y, z] in Hamilton convention |

### 3.2 Transform Convention

The transform `T_body_sensor` maps a point expressed in the **sensor frame** to the **parent frame**:

```
p_parent = T_parent_sensor * p_sensor
```

Equivalently, it encodes the pose of the sensor frame **as seen from** the parent frame. All lookups follow the same convention:

```
p_target = T_target_source * p_source
```

### 3.3 Quaternion Convention

Quaternions use the **Hamilton convention** with scalar-first ordering: `[w, x, y, z]`.
All quaternions must satisfy `|q|² = 1` within a tolerance of `1 × 10⁻⁶`. The loader raises an error on violation.

### 3.4 Coordinate Frame Convention

The reference body frame follows **NED (North-East-Down)**:

| Axis | Direction |
|------|-----------|
| x | Forward (bow) |
| y | Starboard |
| z | Down |

### 3.5 Changeset (Override) Files

A changeset YAML has the same schema as the base config but may omit any fields that do not change. Only the fields present in the changeset are merged into the base; absent fields retain their base values.

```yaml
# changeset-vehicle-002.yaml — override nucleus1000 calibration only
sensors:
  nucleus1000:
    T_body_sensor:
      translation: [0.0, 0.0, 0.062]
      quaternion: [1.0, 0.0, 0.0, 0.0]
```

### 3.6 Confidence Annotations (Convention)

Comments in the YAML should annotate the confidence level of each transform value:

| Tag | Meaning |
|-----|---------|
| `[MEASURED]` | Physically measured or CAD-derived with high confidence |
| `[ESTIMATED]` | Rough estimate, not yet calibrated |
| `[UNKNOWN]` | Placeholder; must be replaced before production use |

---

## 4. Design Constraints and Non-Goals

| Constraint | Rationale |
|------------|-----------|
| **Static transforms only** | All transforms are fixed at init; dynamic/time-varying transforms are out of scope |
| **Tree topology enforced** | The frame graph must be a tree (no multiple parents, no cycles). General DAG traversal is not supported |
| **No ROS tf dependency** | The library is ROS 2–compatible at the packaging level but does not depend on `tf2` or any ROS middleware |
| **No memory allocation after init** | Post-construction lookups are read-only; safe to call from real-time threads |
| **Single root frame** | Exactly one root frame per tree instance |
| **No dynamic registration** | Frames cannot be added or removed after the tree is built |

---

## 5. Error Handling

| Condition | C++ | Python |
|-----------|-----|--------|
| Config file not found | `std::runtime_error` | `FileNotFoundError` |
| YAML parse failure | `std::runtime_error` | `ruamel.yaml` parse exception |
| Missing required field | `std::invalid_argument` | `KeyError` / `ValueError` |
| Non-unit quaternion | `std::invalid_argument` | `ValueError` |
| Unknown frame in lookup | `std::invalid_argument` | `KeyError` |
| Duplicate frame registration | `std::invalid_argument` | `ValueError` |
| Root frame registered as child | `std::invalid_argument` | `ValueError` |

---

## 6. Example Config: Seeker AUV

```yaml
# config/seeker.yaml
metadata:
  vehicle: "seeker-class"
  reference_frame: "body"
  last_calibrated: "UNCALIBRATED"

sensors:
  nucleus1000:
    parent_frame: "body"
    T_body_sensor:
      translation: [0.0, 0.0, 0.06]   # [ESTIMATED]
      quaternion: [1.0, 0.0, 0.0, 0.0]

  xsens_ahrs:
    parent_frame: "body"
    T_body_sensor:
      translation: [0.0, 0.0, 0.0]    # [UNKNOWN]
      quaternion: [1.0, 0.0, 0.0, 0.0]

  cam_fwd:
    parent_frame: "body"
    T_body_sensor:
      translation: [0.0375, 0.0, 0.055]  # [MEASURED]
      quaternion: [1.0, 0.0, 0.0, 0.0]

  cam_aft:
    parent_frame: "body"
    T_body_sensor:
      translation: [-0.0375, 0.0, 0.055] # [MEASURED]
      quaternion: [1.0, 0.0, 0.0, 0.0]

  subsonus_usbl:
    parent_frame: "body"
    T_body_sensor:
      translation: [0.0, 0.0, -0.3]   # [ESTIMATED] — mast height
      quaternion: [1.0, 0.0, 0.0, 0.0]

  gps:
    parent_frame: "body"
    T_body_sensor:
      translation: [0.0, 0.0, -0.32]  # [ESTIMATED]
      quaternion: [1.0, 0.0, 0.0, 0.0]
```

---

## 7. Dependencies

### C++

| Dependency | Role | Scope |
|------------|------|-------|
| Eigen3 ≥ 3.3 | Linear algebra (transforms, quaternions) | Public (re-exported to consumers) |
| yaml-cpp | YAML parsing | Private |
| Google Test | Unit testing | Test only |

### Python

| Dependency | Role |
|------------|------|
| numpy | Array math and transform composition |
| scipy | Quaternion-to-rotation-matrix conversion |
| ruamel.yaml | YAML loading with changeset merge support |
| matplotlib | 3D visualisation (optional) |
| pytest | Unit testing |
