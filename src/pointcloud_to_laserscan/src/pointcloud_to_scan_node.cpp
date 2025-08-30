#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cmath>
#include <vector>

using std::placeholders::_1;

namespace pointcloud_to_laserscan
{

class PointCloudToScanNode : public rclcpp::Node
{
public:
  explicit PointCloudToScanNode(const rclcpp::NodeOptions & options)
  : Node("pointcloud_to_scan", options)
  {
    init();
  }

  PointCloudToScanNode()
  : Node("pointcloud_to_scan")
  {
    init();
  }

private:
  void init()
  {
    // 声明并读取参数
    this->declare_parameter<int>("num_scan_samples", 580);
    this->declare_parameter<double>("lidar_distance_cap", 5.0);
    this->declare_parameter<int>("min_points_per_bin", 2);

    num_scan_samples_ = this->get_parameter("num_scan_samples").as_int();
    lidar_distance_cap_ = this->get_parameter("lidar_distance_cap").as_double();
    min_points_per_bin_ = this->get_parameter("min_points_per_bin").as_int();

    scan_ranges_.assign(num_scan_samples_, lidar_distance_cap_);
    counts_.assign(num_scan_samples_, 0);

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input_cloud", 10,
      std::bind(&PointCloudToScanNode::cloudCallback, this, _1));

    pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "processed_scan", 10);

    RCLCPP_INFO(this->get_logger(),
      "PointCloudToScanNode 初始化完成: samples=%d, cap=%.2f", 
      num_scan_samples_, lidar_distance_cap_);
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 重置
    std::fill(scan_ranges_.begin(), scan_ranges_.end(), lidar_distance_cap_);
    std::fill(counts_.begin(), counts_.end(), 0);

    const double angle_min = -M_PI;
    const double angle_max = M_PI;
    const double angle_span = angle_max - angle_min;
    const double min_valid_distance = 0.02;  // 过滤距离过近的点（如雷达本体误点）

    // 遍历点云
    for (sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x"),
         it_y(*msg, "y");
         it_x != it_x.end(); ++it_x, ++it_y)
    {
      float x = *it_x;
      float y = *it_y;
      double r = std::hypot(x, y);

      // 增加过滤：小于 2cm 的点全部忽略，不参与统计
      if (r < min_valid_distance) continue;
      if (r > lidar_distance_cap_) continue;

      double angle = std::atan2(y, x);
      if (angle < angle_min || angle > angle_max) continue;

      int idx = static_cast<int>((angle - angle_min) / angle_span * num_scan_samples_);
      idx = std::min(std::max(idx, 0), num_scan_samples_ - 1);
      scan_ranges_[idx] = std::min(scan_ranges_[idx], r);
      counts_[idx]++;
    }

    // 构造并发布 Float32MultiArray
    std_msgs::msg::Float32MultiArray out_msg;
    out_msg.data.reserve(num_scan_samples_);
    for (int i = 0; i < num_scan_samples_; ++i) {
      double dist = scan_ranges_[i];
      if (counts_[i] < min_points_per_bin_) {
        dist = lidar_distance_cap_;
      }
      // 归一化时 clip 到 [min_valid_distance, lidar_distance_cap_]，防止0点
      double norm = std::clamp((dist < min_valid_distance ? min_valid_distance : dist) / lidar_distance_cap_, 0.0, 1.0);
      out_msg.data.push_back(static_cast<float>(norm));
    }

    pub_->publish(out_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  int num_scan_samples_;
  double lidar_distance_cap_;
  int min_points_per_bin_;
  std::vector<double> scan_ranges_;
  std::vector<int> counts_;
};

}  // namespace pointcloud_to_laserscan

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_to_laserscan::PointCloudToScanNode)