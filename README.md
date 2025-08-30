#this is model inference and deployment code for robot Go2:
## base condition
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-xacro
sudo apt install ros-humble-robot-localization
sudo apt install ros-humble-ros2-controllers
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-velodyne
sudo apt install ros-humble-velodyne-gazebo-plugins
sudo apt-get install ros-humble-velodyne-description

## simulation: gazebo with Go2 robot (the model and sensor from:https://github.com/fishros/simdog.git )
demo:https://youtu.be/Cff0wIKKi_c


# Go2_DRL_nav — Quick Start

This project downsamples the **3D LiDAR** stream to **580 points** as the policy input.  
Before running, make sure your **ODOM** and **Laser/Scan** topics match your setup.

---

## 1) Start the LiDAR → LaserScan pipeline

```bash
source install/setup.bash
  
## 2st step
only compile the go2_agent file .
colcon build --symlink-install --packages-select go2_agent	
source install/setup.bash
## 3st step
cd ~/drl/Go2_DRL_nav/src/go2_agent
python3 -m pip install -e .  
Execution catalogue:Go2_DRL_nav/src/go2_agent/go2_agent
MODEL_ROOT=/home/jack/drl/Go2_DRL_nav/src/go2_agent/go2_agent/model python3 -m gunicorn go2_agent.agent_3_service_fastbin:app   -b 0.0.0.0:5000 --workers 1 --threads 2 --keep-alive 60
 

Execution catalogue:Go2_DRL_nav/
source install/setup.bash && ros2 run go2_agent model_infer_receive

Deployment reference:
https://github.com/unitreerobotics/unitree_sdk2.git 
https://github.com/Glowing-Torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot

## 4th step send your commamd to Go2(this is connected with go2 sdk )
commmand format: ros2 topic pub --once /goal_pose geometry_msgs/msg/Pose "{position: {x: $1, y: $2, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}}"  
python3 Go2CmdVelNode.py
python3 cmd_vel_forwarder.py


##
## reference for gazebo traning :
TD3: https://github.com/sfujim/TD3.git 
TD3:https://hrl.boyuai.com/chapter/3/%E7%9B%AE%E6%A0%87%E5%AF%BC%E5%90%91%E7%9A%84%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/  english version 
Adapted Training framework  for mode: https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2  
Adapted Training framework  for mode: https://github.com/tomasvr/turtlebot3_drlnav
Adapted Training framework  for mode: https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM 
