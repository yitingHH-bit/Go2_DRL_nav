cmake_minimum_required(VERSION 3.5)
project(pointcloud_to_laserscan)

add_compile_options(-std=c++17 -O3)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(tf2_ros        REQUIRED)
find_package(nav_msgs    REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen      REQUIRED)
find_package(std_msgs REQUIRED)

# PCL 依赖
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

# laserscan -> pointcloud
ament_auto_add_library(laserscan_to_pointcloud SHARED
  src/laserscan_to_pointcloud_node.cpp)
rclcpp_components_register_node(laserscan_to_pointcloud
  PLUGIN "pointcloud_to_laserscan::LaserScanToPointCloudNode"
  EXECUTABLE laserscan_to_pointcloud_node)

# pointcloud -> laserscan
ament_auto_add_library(pointcloud_to_laserscan SHARED
  src/pointcloud_to_laserscan_node.cpp)
rclcpp_components_register_node(pointcloud_to_laserscan
  PLUGIN "pointcloud_to_laserscan::PointCloudToLaserScanNode"
  EXECUTABLE pointcloud_to_laserscan_node)

# pointcloud -> pointcloud_filter_node
ament_auto_add_library(pointcloud_filter SHARED
  src/PointCloudFilterNode.cpp)
rclcpp_components_register_node(pointcloud_filter
  PLUGIN "pointcloud_to_laserscan::PointCloudFilterNode"
  EXECUTABLE pointcloud_filter_node)


# pointcloud -> PointCloudCropNode
ament_auto_add_library(pointcloud_crop SHARED
  src/PointCloudCropNode.cpp)
rclcpp_components_register_node(pointcloud_crop
  PLUGIN "pointcloud_to_laserscan::PointCloudCropNode"
  EXECUTABLE PointCloudCropNode)

# pointcloud -> PointCloudToScanNode
ament_auto_add_library(pointcloud_to_scan SHARED
  src/pointcloud_to_scan_node.cpp)
rclcpp_components_register_node(pointcloud_to_scan
  PLUGIN "pointcloud_to_laserscan::PointCloudToScanNode"
  EXECUTABLE PointCloudToScanNode)


# statistical outlier removal 点云过滤器节点

# 链接 PCL 库
ament_target_dependencies(pointcloud_filter_node
  pcl_conversions
  PCL
  tf2_eigen
  nav_msgs
  std_msgs
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    launch
)
