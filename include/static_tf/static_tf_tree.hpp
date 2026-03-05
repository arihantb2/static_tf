#pragma once

#include <Eigen/Geometry>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace static_tf
{

/// Lightweight static transform tree for sensor frame management.
///
/// Stores T_parent_child per edge and resolves T_target_source for any
/// pair of registered frames by composing transforms through the common root.
///
/// Convention:
///   T_parent_child maps a point expressed in child frame into parent frame:
///     p_parent = T_parent_child * p_child
///
///   lookup("target", "source") returns T_target_source:
///     p_target = T_target_source * p_source
///
/// The tree is a DAG rooted at a single root frame (default: "body").
/// All frames must be reachable from root — no disconnected subtrees.
/// Thread-safe after construction (all queries are read-only).
class StaticTfTree
{
public:
    using FrameId = std::string;
    using Transform = Eigen::Isometry3d;

    explicit StaticTfTree(FrameId root = "body") : root_(std::move(root)) {}

    // ---------------------------------------------------------------------------
    // Registration
    // ---------------------------------------------------------------------------

    /// Register a transform from a translation and quaternion.
    /// @param parent     Parent frame id.
    /// @param child      Child frame id. Must be unique — re-registration throws.
    /// @param translation  [x, y, z] in metres.
    /// @param q          Unit quaternion (w, x, y, z) — Eigen constructor order.
    void register_transform(const FrameId& parent, const FrameId& child, const Eigen::Vector3d& translation,
                            const Eigen::Quaterniond& q)
    {
        Transform T = Transform::Identity();
        T.rotate(q.normalized());
        T.pretranslate(translation);
        register_transform(parent, child, T);
    }

    /// Register a transform directly as an Isometry3d.
    void register_transform(const FrameId& parent, const FrameId& child, const Transform& T_parent_child)
    {
        if (edges_.count(child))
        {
            throw std::invalid_argument("Frame already registered: " + child);
        }
        if (child == root_)
        {
            throw std::invalid_argument("Cannot register root frame as child: " + child);
        }
        edges_[child] = {parent, T_parent_child};
    }

    // ---------------------------------------------------------------------------
    // Query
    // ---------------------------------------------------------------------------

    /// Returns T_target_source as a 4x4 Isometry3d.
    /// Throws std::invalid_argument if either frame is unknown or unreachable.
    Transform lookup(const FrameId& target, const FrameId& source) const
    {
        if (target == source)
        {
            return Transform::Identity();
        }
        // T_target_source = T_root_target^{-1} * T_root_source
        return chain_to_root(target).inverse() * chain_to_root(source);
    }

    /// Convenience: returns the 3x3 rotation matrix R_target_source.
    Eigen::Matrix3d lookup_rotation(const FrameId& target, const FrameId& source) const
    {
        return lookup(target, source).rotation();
    }

    /// Convenience: returns the translation component of T_target_source.
    Eigen::Vector3d lookup_translation(const FrameId& target, const FrameId& source) const
    {
        return lookup(target, source).translation();
    }

    bool has_frame(const FrameId& frame) const { return frame == root_ || edges_.count(frame) > 0; }

    const FrameId& root() const { return root_; }

    /// Returns all registered child frame ids (does not include root).
    std::vector<FrameId> frames() const
    {
        std::vector<FrameId> result;
        result.reserve(edges_.size());
        for (const auto& [child, _] : edges_)
        {
            result.push_back(child);
        }
        return result;
    }

private:
    struct Edge
    {
        FrameId parent;
        Transform T_parent_child;
    };

    FrameId root_;
    std::unordered_map<FrameId, Edge> edges_;

    // Returns T_root_frame: the composed transform from frame up to root.
    //   T_root_frame satisfies: p_root = T_root_frame * p_frame
    //
    // Derivation:
    //   If edges_[X] = {P, T_P_X}, then p_P = T_P_X * p_X.
    //   Composing upward: T_root_X = T_root_P * T_P_X
    //   Iteratively prepend at each step.
    Transform chain_to_root(const FrameId& start) const
    {
        if (!has_frame(start))
        {
            throw std::invalid_argument("Unknown frame: " + start);
        }

        Transform T = Transform::Identity();
        FrameId current = start;

        // Guard against cycles — max depth bounded by number of registered frames.
        const size_t max_depth = edges_.size() + 1;
        size_t depth = 0;

        while (current != root_)
        {
            auto it = edges_.find(current);
            if (it == edges_.end())
            {
                throw std::invalid_argument("Frame '" + current + "' is not reachable from root '" + root_ +
                                            "'. Check parent_frame in config.");
            }
            // T_root_current = T_root_parent * T_parent_current
            T = it->second.T_parent_child * T;
            current = it->second.parent;

            if (++depth > max_depth)
            {
                throw std::invalid_argument("Cycle detected in transform tree at frame: " + current);
            }
        }
        return T;
    }
};

}  // namespace static_tf