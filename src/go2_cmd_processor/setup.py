from setuptools import setup

package_name = 'go2_cmd_processor'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unitree',
    maintainer_email='unitree@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sport_ctrl = go2_cmd_processor.sport_ctrl:main',
            'go2_service_client= go2_cmd_processor.go2_service_client:main',
            'keyboard_teleop= go2_cmd_processor.keyboard_teleop:main',
        ],
    },
)
