#include <gtest/gtest.h>
#include <cmath>
#include <fstream>

#include "static_tf/sensor_frame_loader.hpp"
#include "static_tf/static_tf_tree.hpp"

using namespace static_tf;
using Transform = StaticTfTree::Transform;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void assert_transform_approx(const Transform& A, const Transform& B, double tol = 1e-9)
{
    EXPECT_TRUE(A.matrix().isApprox(B.matrix(), tol)) << "Expected:\n" << B.matrix() << "\nGot:\n" << A.matrix();
}

static Transform make_T(Eigen::Vector3d t, Eigen::Quaterniond q)
{
    Transform T = Transform::Identity();
    T.rotate(q.normalized());
    T.pretranslate(t);
    return T;
}

// Minimal YAML config written to a temp file for loader tests.
static std::string write_temp_config(const std::string& content)
{
    std::string path = "/tmp/static_tf_test_config.yaml";
    std::ofstream f(path);
    f << content;
    return path;
}

// ---------------------------------------------------------------------------
// StaticTfTree unit tests
// ---------------------------------------------------------------------------

class StaticTfTreeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Star topology: all sensors direct children of "body"
        //   cam_left:  t=[-0.0375, 0, 0.11], R=I
        //   cam_right: t=[ 0.0375, 0, 0.11], R=I
        //   nucleus:   t=[ 0.134,  0, 0.085], R=I
        tree_ = StaticTfTree("body");

        tree_.register_transform("body", "cam_left", Eigen::Vector3d(-0.0375, 0.0, 0.11),
                                 Eigen::Quaterniond::Identity());

        tree_.register_transform("body", "cam_right", Eigen::Vector3d(0.0375, 0.0, 0.11),
                                 Eigen::Quaterniond::Identity());

        tree_.register_transform("body", "nucleus1000", Eigen::Vector3d(0.134, 0.0, 0.085),
                                 Eigen::Quaterniond::Identity());
    }

    StaticTfTree tree_{"body"};
};

TEST_F(StaticTfTreeTest, IdentityOnSameFrame)
{
    assert_transform_approx(tree_.lookup("body", "body"), Transform::Identity());
    assert_transform_approx(tree_.lookup("cam_left", "cam_left"), Transform::Identity());
}

TEST_F(StaticTfTreeTest, LookupBodyToSensor)
{
    // T_body_cam_left should have translation [-0.0375, 0, 0.11]
    auto T = tree_.lookup("body", "cam_left");
    EXPECT_NEAR(T.translation().x(), -0.0375, 1e-9);
    EXPECT_NEAR(T.translation().y(), 0.0, 1e-9);
    EXPECT_NEAR(T.translation().z(), 0.11, 1e-9);
    EXPECT_TRUE(T.rotation().isApprox(Eigen::Matrix3d::Identity(), 1e-9));
}

TEST_F(StaticTfTreeTest, LookupSensorToBody_IsInverse)
{
    // T_cam_body = inv(T_body_cam)
    auto T_body_cam = tree_.lookup("body", "cam_left");
    auto T_cam_body = tree_.lookup("cam_left", "body");
    auto product = T_body_cam * T_cam_body;
    assert_transform_approx(product, Transform::Identity());
}

TEST_F(StaticTfTreeTest, StereoBáseline_ChainThroughRoot)
{
    // T_left_right = inv(T_body_left) * T_body_right
    auto T_body_left = tree_.lookup("body", "cam_left");
    auto T_body_right = tree_.lookup("body", "cam_right");
    auto T_left_right_expected = T_body_left.inverse() * T_body_right;
    auto T_left_right = tree_.lookup("cam_left", "cam_right");

    assert_transform_approx(T_left_right, T_left_right_expected);
}

TEST_F(StaticTfTreeTest, StereoBáseline_Magnitude)
{
    // Baseline between cam_left and cam_right must be 75 mm
    auto T = tree_.lookup("cam_left", "cam_right");
    double baseline = T.translation().norm();
    EXPECT_NEAR(baseline, 0.075, 1e-6);
}

TEST_F(StaticTfTreeTest, RotatedFrame)
{
    // Register a sensor with a 90-degree yaw rotation
    Eigen::Quaterniond q_90yaw(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
    tree_.register_transform("body", "rotated_sensor", Eigen::Vector3d(0.1, 0.0, 0.0), q_90yaw);

    auto T = tree_.lookup("body", "rotated_sensor");
    // A point along +X in sensor frame should appear along +Y in body frame after 90-deg yaw
    Eigen::Vector3d p_sensor(1.0, 0.0, 0.0);
    Eigen::Vector3d p_body = T * p_sensor;
    // Translation adds 0.1 in X, rotation maps sensor-X to body-Y
    EXPECT_NEAR(p_body.x(), 0.1, 1e-9);
    EXPECT_NEAR(p_body.y(), 1.0, 1e-9);
    EXPECT_NEAR(p_body.z(), 0.0, 1e-9);
}

TEST_F(StaticTfTreeTest, UnknownFrameThrows)
{
    EXPECT_THROW(tree_.lookup("body", "nonexistent"), std::invalid_argument);
    EXPECT_THROW(tree_.lookup("nonexistent", "body"), std::invalid_argument);
}

TEST_F(StaticTfTreeTest, DuplicateRegistrationThrows)
{
    EXPECT_THROW(tree_.register_transform("body", "cam_left", Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()),
                 std::invalid_argument);
}

TEST_F(StaticTfTreeTest, RootAsChildThrows)
{
    EXPECT_THROW(tree_.register_transform("body", "body", Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()),
                 std::invalid_argument);
}

TEST_F(StaticTfTreeTest, HasFrame)
{
    EXPECT_TRUE(tree_.has_frame("body"));
    EXPECT_TRUE(tree_.has_frame("cam_left"));
    EXPECT_FALSE(tree_.has_frame("nonexistent"));
}

// ---------------------------------------------------------------------------
// Loader tests
// ---------------------------------------------------------------------------

TEST(LoaderTest, LoadsFromYaml)
{
    const std::string yaml = R"(
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
)";
    auto path = write_temp_config(yaml);
    auto tree = load_from_yaml(path);

    EXPECT_TRUE(tree.has_frame("cam_left"));
    EXPECT_TRUE(tree.has_frame("cam_right"));

    auto T = tree.lookup("cam_left", "cam_right");
    EXPECT_NEAR(T.translation().norm(), 0.075, 1e-6);
}

TEST(LoaderTest, MissingFileThrows)
{
    EXPECT_THROW(load_from_yaml("/tmp/does_not_exist.yaml"), std::runtime_error);
}

TEST(LoaderTest, NonUnitQuaternionThrows)
{
    const std::string yaml = R"(
metadata:
  reference_frame: body
sensors:
  cam_left:
    parent_frame: body
    T_body_sensor:
      translation: [0.0, 0.0, 0.0]
      quaternion:  [2.0, 0.0, 0.0, 0.0]
)";
    auto path = write_temp_config(yaml);
    EXPECT_THROW(load_from_yaml(path), std::runtime_error);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
