#include "static_tf/sensor_frame_loader.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace static_tf
{

StaticTfTree load_from_yaml(const std::string& yaml_path)
{
    YAML::Node cfg;
    try
    {
        cfg = YAML::LoadFile(yaml_path);
    }
    catch (const YAML::BadFile& e)
    {
        throw std::runtime_error("Could not open config file: " + yaml_path);
    }
    catch (const YAML::ParserException& e)
    {
        throw std::runtime_error("YAML parse error in " + yaml_path + ": " + e.what());
    }

    // Root frame from metadata, default "body"
    std::string root = "body";
    if (cfg["metadata"] && cfg["metadata"]["reference_frame"])
    {
        root = cfg["metadata"]["reference_frame"].as<std::string>();
    }

    StaticTfTree tree(root);

    if (!cfg["sensors"])
    {
        throw std::runtime_error("Config missing 'sensors' key: " + yaml_path);
    }

    for (const auto& entry : cfg["sensors"])
    {
        const std::string name = entry.first.as<std::string>();
        const YAML::Node& s = entry.second;

        if (!s["parent_frame"])
        {
            throw std::runtime_error("Sensor '" + name + "' missing 'parent_frame'");
        }
        if (!s["T_body_sensor"])
        {
            throw std::runtime_error("Sensor '" + name + "' missing 'T_body_sensor'");
        }

        const std::string parent = s["parent_frame"].as<std::string>();
        const YAML::Node& T_node = s["T_body_sensor"];

        auto t_vec = T_node["translation"].as<std::vector<double>>();
        auto q_vec = T_node["quaternion"].as<std::vector<double>>();  // [w, x, y, z]

        if (t_vec.size() != 3)
        {
            throw std::runtime_error("Sensor '" + name + "': translation must have 3 elements");
        }
        if (q_vec.size() != 4)
        {
            throw std::runtime_error("Sensor '" + name + "': quaternion must have 4 elements [w,x,y,z]");
        }

        const Eigen::Vector3d t(t_vec[0], t_vec[1], t_vec[2]);
        const Eigen::Quaterniond q(q_vec[0], q_vec[1], q_vec[2], q_vec[3]);  // Eigen: (w,x,y,z)

        if (std::abs(q.norm() - 1.0) > 1e-4)
        {
            throw std::runtime_error("Sensor '" + name +
                                     "': quaternion is not unit norm (norm=" + std::to_string(q.norm()) + ")");
        }

        tree.register_transform(parent, name, t, q);
    }

    return tree;
}

}  // namespace static_tf
