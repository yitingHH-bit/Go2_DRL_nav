import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from .off_policy import OffPolicyAgent, Network

LINEAR = 0
ANGULAR = 1
POLICY_NOISE            = 0.2
POLICY_NOISE_CLIP       = 0.5
POLICY_UPDATE_FREQUENCY = 1
  
"""
  CNNTD3 agent
  This network takes as input a state composed of laser scan data, 
  It processes the scan through a 1D CNN stack and embeds the other
    inputs before merging all features through fully connected layers to output a continuous
    action vector.
"""
class NoisePackage(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period=600000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t=0):
        ou_state = self.evolve_state()
        decaying = float(float(t) / self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state


def init_linear(layer, gain=1.0, small=False):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    if small:
        layer.weight.data.mul_(0.01)

class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__(name)
        self.num_scan = 580
        self.extra_inputs = 4

        self.scan_conv = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=8, stride=4),  # 580 -> ~145
    nn.ReLU(inplace=True),
    nn.Conv1d(16, 32, kernel_size=4, stride=2), # ~145 -> ~71
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool1d(8)                     # -> 长度8
)

        # BN -> LN
#         self.scan_conv = nn.Sequential(
#     nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
#     nn.GroupNorm(1, 32),
#     nn.ReLU(),
#     nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
#     nn.GroupNorm(1, 64),
#     nn.ReLU(),
#     nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
#     nn.GroupNorm(1, 128),
#     nn.ReLU(),
#     nn.Conv1d(128, 64, kernel_size=1),   # 瓶颈
#     nn.GroupNorm(1, 64),
#     nn.ReLU(),
#     nn.AdaptiveAvgPool1d(8)              # 池化到 8
# )




        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.num_scan)
            dummy_output = self.scan_conv(dummy_input)
            self.feature_size = dummy_output.view(1, -1).shape[1]

        self.input_dim = self.feature_size + self.extra_inputs

        self.fa1 = nn.Linear(self.input_dim, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)

        # 初始化
        init_linear(self.fa1, gain=nn.init.calculate_gain('relu'))
        init_linear(self.fa2, gain=nn.init.calculate_gain('relu'))
        init_linear(self.fa3, small=True)

    def forward(self, states, visualize=False):
        assert states.shape[1] == self.num_scan + self.extra_inputs
        scan = states[:, :self.num_scan].unsqueeze(1)
        extra = states[:, self.num_scan:self.num_scan + self.extra_inputs]

        x = self.scan_conv(scan)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, extra], dim=1)

        x = F.relu(self.fa1(x))
        x = F.relu(self.fa2(x))
        action = torch.tanh(self.fa3(x))
        return action


class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__(name)
        self.num_scan = 580
        self.extra_inputs = 4

        self.scan_conv = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=8, stride=4),  # 580 -> ~145
    nn.ReLU(inplace=True),
    nn.Conv1d(16, 32, kernel_size=4, stride=2), # ~145 -> ~71
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool1d(8)                     # -> 长度8
)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.num_scan)
            dummy_output = self.scan_conv(dummy_input)
            self.feature_size = dummy_output.view(1, -1).shape[1]

        self.input_dim = self.feature_size + self.extra_inputs + action_size

        self.l1 = nn.Linear(self.input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.l4 = nn.Linear(self.input_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

        for l in [self.l1, self.l2, self.l4, self.l5]:
            init_linear(l, gain=nn.init.calculate_gain('relu'))
        init_linear(self.l3, small=True)
        init_linear(self.l6, small=True)

    def extract_features(self, states):
        assert states.shape[1] == self.num_scan + self.extra_inputs
        scan = states[:, :self.num_scan].unsqueeze(1)
        extra = states[:, self.num_scan:self.num_scan + self.extra_inputs]
        x = self.scan_conv(scan)
        x = torch.flatten(x, start_dim=1)
        return x, extra

    def forward(self, states, actions):
        scan_feat, extra = self.extract_features(states)
        xu = torch.cat([scan_feat, extra, actions], dim=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1_forward_for_actor(self, states, actions):
        scan_feat, extra = self.extract_features(states)
        xu = torch.cat([scan_feat, extra, actions], dim=1)
        x = F.relu(self.l1(xu))
        x = F.relu(self.l2(x))
        return self.l3(x)



class TD3(OffPolicyAgent):
    def __init__(self, device, sim_speed):
        super().__init__(device, sim_speed)

        self.noise = NoisePackage(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
        self.policy_noise = POLICY_NOISE
        self.noise_clip = POLICY_NOISE_CLIP
        self.policy_freq = POLICY_UPDATE_FREQUENCY
        self.last_actor_loss = 0

        #生成连续动作

        self.actor = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)
        
        #评估动作的价值 
        self.critic = self.create_network(Critic, 'critic')
        self.critic_target = self.create_network(Critic, 'target_critic')
        self.critic_optimizer = self.create_optimizer(self.critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

    def get_action(self, state, is_training, step, visualize=False):
        state = torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0).to(self.device)
        action = self.actor(state, visualize)
        if is_training:
            noise = torch.from_numpy(copy.deepcopy(self.noise.get_noise(step))).to(self.device)
            action = torch.clamp(action + noise, -1.0, 1.0)
        return [float(a) for a in action.detach().cpu().squeeze(0).tolist()]

    #def get_action_random(self):
        #return [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0)] * self.action_size
    def get_action_random(self):
        return np.clip(np.random.uniform(-1.0, 1.0, size=self.action_size), -1.0, 1.0).tolist()
     
    def train(self, state, action, reward, state_next, done):
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # get action for actor net work 
        action_next = (self.actor_target(state_next) + noise).clamp(-1.0, 1.0)
        # Q1 Q2 we get from output(actor) of actor_network 
        #计算目标 Q 值，并对 Q 网络进行更新。具体来说，使用两个 Q 网络来计算 Q 值，并更新最小的 Q 值网络
        
        # 计算  目标 Q 值  Q_target
        Q1_value, Q2_value= self.critic_target(state_next, action_next)
        Q_mini_next = torch.min(Q1_value, Q2_value) #over estimate value 
        Q_value_target = reward + (1 - done) * self.discount_factor * Q_mini_next
        
        # 更新 Q 网络
        Q1_Buffer, Q2_Buffer = self.critic(state, action) #  form buffer take real value(state,act ) give to Q_net
        #loss_critic = MSE(Q1, Q_target) + MSE(Q2, Q_target) 更新 critic 参数，使得 Q 值逼近目标值
        loss_critic = self.loss_function(Q1_Buffer, Q_value_target) + self.loss_function(Q2_Buffer, Q_value_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()

        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0)
        self.critic_optimizer.step()

        if self.iteration % self.policy_freq == 0:
            #  critic Q1 function as loss function , input  state and actor network actor , 
            #update the actor network  
            loss_actor = -self.critic.Q1_forward_for_actor(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.last_actor_loss = loss_actor.detach().cpu()

        q_value = 0
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss, q_value]
