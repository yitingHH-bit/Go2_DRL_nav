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
                ('cloud_in', '/utlidar/cloud'),  # 输入点云主题
                ('scan', '/scan')  # 输出激光扫描主题
            ],
            parameters=[{
                'target_frame': 'base_link',  # 修正拼写错误，确保与雷达帧一致utlidar_lidar
                # 'transform_tolerance': 0.01,  # TF 变换的容忍时间
                # 'min_height': -0.3,  # 点云的最小高度
                # 'max_height': 0.3,  # 点云的最大高度
                # 'angle_min': -3.14159,  # 激光扫描的最小角度 (-180°)
                # 'angle_max': 3.14159,  # 激光扫描的最大角度 (180°)
                # 'angle_increment': 0.0043,  # 激光扫描的角度分辨率
                # 'scan_time': 0.3333,  # 扫描时间
                # 'range_min': 0.9,  # 激光扫描的最小范围，与雷达配置一致
                # 'range_max': 100.0,  # 激光扫描的最大范围，与雷达配置一致
                # 'use_inf': True,  # 是否使用无穷远值表示超出范围的点
                # 'inf_epsilon': 1.0  # 无穷远值的偏移量
                  'transform_tolerance': 0.01,
                  'queue_size': 10 ,
                  # 只保留 LiDAR ±10cm 高度的点
                #   'min_height': -1.0,
                #   'max_height': 1.0,
                  'min_height': -0.3 ,        # 切片范围（±10cm），提取平面数据
                  'max_height': 1.5,
                  'angle_min': -3.14,
                  'angle_max': 3.14,
                  'angle_increment': 0.0043,
                  'scan_time': 0.3333,
                  'range_min': 0.1,
                  'range_max': 5.0,
                  'use_inf': True,
                  'inf_epsilon': 1.0,
                  'qos_overrides./scan.publisher.reliability': 'reliable'
            }],
            name='pointcloud_to_laserscan'
        )
    ])
