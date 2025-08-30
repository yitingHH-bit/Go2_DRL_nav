#	ssh -X unitree@192.168.123.18
#	ssh -X unitree@192.168.50.19   #
#	cd /unitree/module/graph_pid_ws  && ./0_unitree_slam.sh 
# 	cd /unitree/lib/unitree_slam/build && ./demo_xt16 eth0 



only compile the go2_agent file .
colcon build --symlink-install --packages-select go2_agent	
source install/setup.bash

cd ~/drl/Go2_DRL_nav/src/go2_agent
python3 -m pip install -e .  


Execution catalogue:Go2_DRL_nav/src/go2_agent/go2_agent

MODEL_ROOT=/home/jack/drl/Go2_DRL_nav/src/go2_agent/go2_agent/model python3 -m gunicorn go2_agent.agent_3_service_fastbin:app   -b 0.0.0.0:5000 --workers 1 --threads 2 --keep-alive 60
 

Execution catalogue:Go2_DRL_nav/
source install/setup.bash && ros2 run go2_agent model_infer_receive

check the file:
ls -lh ~/drl/Go2_DRL_nav/src/go2_agent/go2_agent/model/td3_21_stage_7/stage7_agent_ep21000.pt


reference 
for  traning gazebo medel :
https://github.com/reiniscimurs/DRL-robot-navigation?tab=readme-ov-file


for model deployment: 
https://github.com/eppl-erau-db/go2_rl_ws.git  

https://aexport GO2_AGENT_MODEL_ROOT=~/drl/Go2_DRL_nav/src/go2_agent/go2_agent/modelrxiv.org/pdf/2103.07119	Goal-Driven Autonomous Exploration Through Deep Reinforcement
Learning

ROS2 adapted from: https://github.com/tomasvr/turtlebot3_drlnav
TD3 adapted from: https://github.com/reiniscimurs/DRL-robot-navigation
SAC adapted from: https://github.com/denisyarats/pytorch_sac

Sir,



Sorry to interrupt you at the weekend. I saw the LinkedIn post you shared. There is an item (dual-stack) which is actually just a training and evaluation system. However, the model is trained and deployed separately. I am concerned that this will confuse the readers, so I have modified the abstract to make it clearer.


cd ~/drl/Go2_DRL_nav

# 如果还没初始化过 git：
git init

# 建议设置一次全局签名（换成你的名字和邮箱）
git config --global user.name  "yitingHH-bit"
git config --global user.email "kaolalaotongxue@gmail.com"

https://github.com/yitingHH-bit/Go2_DRL_nav.git 

git remote add origin git@github.com:yitingHH-bit/Go2_DRL_nav.git
'
We present a dual-stack reinforcement learning system for quadruped navigation on the Unitree.
evaluated systems for quadruped navigation on the Unitree
Go2 robot, with two RL models trained separately on different pipelines:
Pipeline A is a LiDAR-centric TD3 policy trained in Gazebo with
simple velocity-target inputs; Pipeline B uses two-stage training.
setup in Isaac Lab/Isaac Sim. In Stage 1, we trained a low-level
joint-space locomotion controller (velocity tracking) on rough
terrains with heavy domain randomisation. In Stage 2, we freeze
the low-level controller and train a high-level navigation policy.
(PPO) to output (vx, vy, ω) at 5 Hz.
'


Br.

Jiancai Hou


https://github.com/sfujim/TD3.git   算法引用

这篇论文的https://arxiv.org/pdf/2103.07119  的这个章节，也可以作为参考帮助模型和全局规划结合使用，III. GOAL-DRIVEN AUTONOMOUS
EXPLORATION
To achieve autonomous navigation and exploration in an
unknown environment, we propose a navigation structure
that consists of two parts: global navigation with optimal
waypoint selection from POI and mapping; a deep reinforcement learning-based local navigation. Points of interest are
extracted from the environment and an optimal waypoint is
selected following the evaluation criteria. At every step, a
waypoint is given to the neural network in the form of polar
coordinates, relative to the robot’s location and heading. An
action is calculated based on the sensor data and executed
towards the waypoint. Mapping is performed while moving
between waypoints towards the global goal.
