from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            name='scanner', default_value='scanner',
            description='Namespace for sample topics'
        ),
        Node(
            package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            remappings=[
                #/utlidar/cloud
                #/lio_sam_ros2/mapping/cloud_registered
                #/lio_sam_ros2/deskew/cloud_deskewed    cloud_deskewed
                ('cloud_in', '/notGround_pointCloud'),  # 输入点云主题
                ('scan', '/scan')  # 输出激光扫描主题
            ],
            parameters=[{
                  'target_frame': 'utlidar_lidar', # base_link   utlidar_lidar   odom  # 修正拼写错误，确保与雷达帧一致utlidar_lidar
              
                  'transform_tolerance': 0.01,
                  'queue_size': 10 ,
                  # 只保留 LiDAR ±10cm 高度的点

                  'angle_min': -3.14,# 1.57  3.14   0.785
                  'angle_max': 3.14,
                  'angle_increment': 0.00215,#0.0043,
                  'scan_time': 0.3333,
                  'range_min': 0.2,  
                  'range_max': 7.0,
                  'use_inf': True,
                  'inf_epsilon': 1.0,
                  'qos_overrides./scan.publisher.reliability': 'reliable'
            }],
            name='pointcloud_to_laserscan'
        )
    ])
