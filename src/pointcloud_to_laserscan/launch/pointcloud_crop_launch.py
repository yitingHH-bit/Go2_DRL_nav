from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pointcloud_to_laserscan',
            executable='PointCloudCropNode',
            name='PointCloudCropNode',
            output='screen',
            parameters=[
                {'input_cloud': '/velodyne_points'},
                {'output_cloud': '/cropped_cloud'},
                {'robot_length': 0.70},
                {'robot_width': 0.37},
                {'lidar_offset_front': 0.14}
            ]
        )
    ])
