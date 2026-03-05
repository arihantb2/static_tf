#pragma once

#include <string>
#include "static_tf/static_tf_tree.hpp"

namespace static_tf
{

/// Load a StaticTfTree from a YAML config file.
///
/// Expected format:
/// @code
/// metadata:
///   reference_frame: body   # root frame id
///
/// sensors:
///   nucleus1000:
///     parent_frame: body
///     T_body_sensor:
///       translation: [x, y, z]
///       quaternion:  [w, x, y, z]
///   cam_left:
///     parent_frame: body
///     T_body_sensor:
///       translation: [x, y, z]
///       quaternion:  [w, x, y, z]
/// @endcode
///
/// The merged config (base + changeset) produced by the Python loader
/// is the intended input. This function only reads a single flat file.
///
/// @throws std::runtime_error on file not found or malformed YAML.
/// @throws std::invalid_argument on duplicate frame registration.
StaticTfTree load_from_yaml(const std::string& yaml_path);

}  // namespace static_tf
