from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='pointcloud_filter_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='pointcloud_to_laserscan',
                plugin='pointcloud_to_laserscan::PointCloudFilterNode',
                name='pointcloud_filter',
                remappings=[
                    ('input_cloud', '/points'),
                    ('filtered_cloud', '/filtered/pointcloud') #/filtered/pointcloud
                ],
                parameters=[{
                    'mean_k': 30,
                    'stddev_mul_thresh': 1.5,
                }],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([container])
