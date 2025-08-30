#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class PointCloudCropNode : public rclcpp::Node
{
public:
  explicit PointCloudCropNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
    : Node("robotbody_crop", options)
  {
    // 机器人参数
    declare_parameter<std::string>("input_cloud", "/velodyne_points");
    declare_parameter<std::string>("output_cloud", "/cropped_cloud");
    declare_parameter<double>("robot_length", 0.70);         // 机器人总长(m)
    declare_parameter<double>("robot_width", 0.37);          // 机器人总宽(m)
    declare_parameter<double>("lidar_offset_front", 0.14);   // 雷达距前端距离(m)

    get_parameter("input_cloud", input_topic_);
    get_parameter("output_cloud", output_topic_);
    get_parameter("robot_length", robot_length_);
    get_parameter("robot_width", robot_width_);
    get_parameter("lidar_offset_front", lidar_offset_front_);

    half_l_ = robot_length_ / 2.0;
    half_w_ = robot_width_ / 2.0;
    lidar_x_ = half_l_ - lidar_offset_front_;  // 雷达相对机器人中心的x偏移

    // 订阅和发布
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PointCloudCropNode::cloudCallback, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);

    RCLCPP_INFO(get_logger(),
      "RobotBodyCropNode ready. input=%s output=%s crop_box[L=%.2f W=%.2f lidar_x=%.2f]",
      input_topic_.c_str(), output_topic_.c_str(), robot_length_, robot_width_, lidar_x_);
  }

private:
  // 判断点是否在机器人矩形体内（雷达frame下，机器人中心 = (-lidar_x_, 0)）
  bool is_in_body(double x, double y) const {
    double px = x + lidar_x_;
    return (px >= -half_l_ && px <= half_l_ && y >= -half_w_ && y <= half_w_);
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud_in);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_out->reserve(cloud_in->size());

    size_t removed = 0;
    for (const auto &pt : cloud_in->points) {
      if (!is_in_body(pt.x, pt.y)) {
        cloud_out->push_back(pt);
      } else {
        removed++;
      }
    }

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*cloud_out, out_msg);
    out_msg.header = msg->header;
    pub_->publish(out_msg);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
      "[robotbody_crop] in=%zu, out=%zu, removed=%zu", cloud_in->size(), cloud_out->size(), removed);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  std::string input_topic_, output_topic_;
  double robot_length_, robot_width_, lidar_offset_front_;
  double half_l_, half_w_, lidar_x_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(PointCloudCropNode)
