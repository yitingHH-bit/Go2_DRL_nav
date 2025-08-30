// pointcloud_filter_node.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace pointcloud_to_laserscan
{

using std::placeholders::_1;

class PointCloudFilterNode : public rclcpp::Node
{
public:
  // 1) 带 NodeOptions 的构造函数，供组件加载时使用
  explicit PointCloudFilterNode(const rclcpp::NodeOptions & options)
  : Node("pointcloud_filter", options)
  {
    init();
  }

  // 2) 无参构造函数，供直接运行（如果你想用 main() 也能跑）
  PointCloudFilterNode()
  : Node("pointcloud_filter")
  {
    init();
  }

private:
  void init()
  {
    // 声明并读取参数
    this->declare_parameter<int>("mean_k", 20);
    this->declare_parameter<double>("stddev_mul_thresh", 1.0);

    int mean_k;
    double stddev_mul_thresh;
    this->get_parameter("mean_k", mean_k);
    this->get_parameter("stddev_mul_thresh", stddev_mul_thresh);

    // 订阅原始点云
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input_cloud", 10,
      std::bind(&PointCloudFilterNode::cloudCallback, this, _1));

    // 发布滤波后的点云
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", 10);

    // 配置滤波器
    sor_.setMeanK(mean_k);
    sor_.setStddevMulThresh(stddev_mul_thresh);

    RCLCPP_INFO(this->get_logger(),
      "PointCloudFilterNode 初始化完成：mean_k=%d, stddev_mul_thresh=%.2f",
      mean_k, stddev_mul_thresh);
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 转成 PCL 点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pc);

    // 统计离群点滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    sor_.setInputCloud(pc);
    sor_.filter(*filtered);

    // 转回 ROS 消息并发布
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*filtered, out);
    out.header = msg->header;
    pub_->publish(out);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   pub_;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ>                sor_;
};

}  // namespace pointcloud_to_laserscan

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_to_laserscan::PointCloudFilterNode)
