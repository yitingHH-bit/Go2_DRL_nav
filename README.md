#
# we downsample 3d lidar date to  580 points  feed in the model ,first start the lidar node to deal the data ,
# modifying your odom topic and  position topic , 

## 1st
#source install/setup.bash && ros2 launch pointcloud_to_laserscan pointcloud_to_scan_launch.py   
## 2st
only compile the go2_agent file .
colcon build --symlink-install --packages-select go2_agent	
source install/setup.bash
## 3st
cd ~/drl/Go2_DRL_nav/src/go2_agent
python3 -m pip install -e .  
Execution catalogue:Go2_DRL_nav/src/go2_agent/go2_agent
MODEL_ROOT=/home/jack/drl/Go2_DRL_nav/src/go2_agent/go2_agent/model python3 -m gunicorn go2_agent.agent_3_service_fastbin:app   -b 0.0.0.0:5000 --workers 1 --threads 2 --keep-alive 60
 

Execution catalogue:Go2_DRL_nav/
source install/setup.bash && ros2 run go2_agent model_infer_receive



## reference for gazebo traning :
TD3: https://github.com/sfujim/TD3.git 
Adapted Training  for mode: https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2  
Adapted Training for mode: : https://github.com/tomasvr/turtlebot3_drlnav

