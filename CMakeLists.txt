cmake_minimum_required(VERSION 3.16)
project(static_tf VERSION 1.0.0 LANGUAGES CXX)

# -----------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------
option(BUILD_TESTING "Build tests" ON)

# -----------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# -----------------------------------------------------------------------
# C++ library
# sensor_frame_loader.cpp is the only translation unit; static_tf_tree.hpp
# is header-only (Eigen only, no yaml-cpp leak into public headers).
# -----------------------------------------------------------------------
add_library(static_tf SHARED src/sensor_frame_loader.cpp)
add_library(static_tf::static_tf ALIAS static_tf)

target_include_directories(static_tf
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(static_tf
  PUBLIC  Eigen3::Eigen
  PRIVATE yaml-cpp             # yaml-cpp is an impl detail; not exposed in public headers
)

target_compile_features(static_tf PUBLIC cxx_std_17)

set_target_properties(static_tf PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# -----------------------------------------------------------------------
# Install headers
# -----------------------------------------------------------------------
install(
  DIRECTORY include/
  DESTINATION include
)

# -----------------------------------------------------------------------
# Install library and export targets
# -----------------------------------------------------------------------
install(
  TARGETS static_tf
  EXPORT  static_tf_targets
  LIBRARY  DESTINATION lib
  ARCHIVE  DESTINATION lib
  RUNTIME  DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  EXPORT static_tf_targets
  FILE      static_tfTargets.cmake
  NAMESPACE static_tf::
  DESTINATION share/${PROJECT_NAME}/cmake
)

# -----------------------------------------------------------------------
# CMake package config — allows downstream: find_package(static_tf)
# -----------------------------------------------------------------------
include(CMakePackageConfigHelpers)

configure_package_config_file(
  cmake/static_tfConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/static_tfConfig.cmake
  INSTALL_DESTINATION share/${PROJECT_NAME}/cmake
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/static_tfConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/static_tfConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/static_tfConfigVersion.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

# -----------------------------------------------------------------------
# Python package
# Installs python/static_tf/ as the 'static_tf' Python package.
# After colcon build, importable as: from static_tf.tree import StaticTfTree
# -----------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR python/${PROJECT_NAME}
)

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  # C++ tests
  ament_add_gtest(test_static_tf_tree
    test/cpp/test_static_tf_tree.cpp
  )
  target_link_libraries(test_static_tf_tree static_tf)

  # Python tests
  ament_add_pytest_test(test_static_tf_python
    test/python
    PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
  )
endif()

# -----------------------------------------------------------------------
# Ament exports — downstream packages get Eigen3 transitively
# -----------------------------------------------------------------------
ament_export_targets(static_tf_targets HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module Eigen3)

ament_package()