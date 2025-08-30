from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pointcloud_to_laserscan',
            executable='PointCloudToScanNode',
            name='pointcloud_to_scan',
            output='screen',
            parameters=[
                # 点云分箱与滤波参数
                {'num_scan_samples': 580},
                {'lidar_distance_cap': 6.0},
                {'min_points_per_bin': 2},
                # 话题名称（也可以用 remappings 而不是参数）
                # {'input_cloud': '/filtered/pointcloud'},
                # {'processed_scan': '/processed_scan'},
            ],
            # 如果你更习惯用 remap 方式，也可以这样写： notGround_pointCloud
            remappings=[
                ('input_cloud', '/notGround_pointCloud'), #/filtered/pointcloud
                ('processed_scan', '/processed_scan'),
            ],
        ),
    ])
