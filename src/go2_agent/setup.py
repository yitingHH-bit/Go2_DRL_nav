from setuptools import setup,find_packages
from glob import glob
package_name = 'go2_agent'

setup(
    name=package_name,
    version='1.0.0',
    maintainer_email='kaolalaotongxue@robotis.com',
    keywords=['ROS', 'ROS2', 'examples', 'rclpy'],
    license='Apache-2.0',
    install_requires=['setuptools'],
    packages=find_packages(), 
    description=(
        'Go2 TD3 CNN'
    ),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*')),
        (f'share/{package_name}/urdf', glob('urdf/*')),
        (f'share/{package_name}/params', glob('params/*')),
        (f'share/{package_name}/scripts', glob('scripts/*')),
    ],
    entry_points={
        'console_scripts': [
            'go2_pose_command = go2_agent.go2_pose_command:main',
            'model_infer_receive= go2_agent.agent_3_receive:main',
            'drl_agent_3_service = go2_agent.agent_3_service_node:main',
        ],
    },
)
