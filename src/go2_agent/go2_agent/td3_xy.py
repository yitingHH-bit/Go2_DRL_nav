#!/usr/bin/env python3
from typing import Tuple

import numpy as np
import copy
import os
import time                         
import wandb                        # (pip install wandb)

import torch
import torch.nn.functional as F
import torch.nn as nn
from ..common.ounoise_2 import OUNoise
from .off_policy import OffPolicyAgent, Network
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  
POLICY_NOISE            = 0.2
POLICY_NOISE_CLIP       = 0.5
POLICY_UPDATE_FREQUENCY = 1

NUM_SCAN_SAMPLES= 580 

stage = "td3_29"  

if stage == "td3_7":
    SPEED_LINEAR_MAX    = 0.35
    SPEED_LINEAR_MAX_Y  = 0.20
    SPEED_ANGULAR_MAX   = 0.4
elif stage == "td3_29":
    SPEED_LINEAR_MAX    = 0.38
    SPEED_LINEAR_MAX_Y  = 0.25
    SPEED_ANGULAR_MAX   = 0.50
elif stage == "td3_7":
    SPEED_LINEAR_MAX            = 0.4  
    SPEED_ANGULAR_MAX           = 0.8  
    SPEED_LINEAR_MAX_Y  = 0.0
else:
    #
    SPEED_LINEAR_MAX    = 0.4
    SPEED_LINEAR_MAX_Y  = 0.25
    SPEED_ANGULAR_MAX   = 0.5

def scale_action(a_tanh: torch.Tensor) -> torch.Tensor:
    vx = a_tanh[..., 0] * SPEED_LINEAR_MAX
    vy = a_tanh[..., 1] * SPEED_LINEAR_MAX_Y
    omega = a_tanh[..., 2] * SPEED_ANGULAR_MAX
    return torch.stack([vx, vy, omega], dim=-1)
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
class MultiScaleConv1d(Network):
    
    def __init__(self,
                 name: str,
                 in_channels: int = 1,
                 mid_channels: int = 12,    
                 out_channels: int = 24,    
                 pool_sizes: Tuple[int,int,int] = (24,12,6), 
                 num_heads: int = 1,        
                 se_reduction: int = 8,     
                 use_lstm: bool = False,     
                 lstm_hidden: int = 16,      
                 bidirectional: bool = True):
        super().__init__(name)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 5, padding=2),
            nn.BatchNorm1d(mid_channels), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU())
        self.se1 = SEBlock1d(name+"_se1", mid_channels, se_reduction)
        self.se2 = SEBlock1d(name+"_se2", out_channels, se_reduction)

        self.attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size = out_channels,
                hidden_size= lstm_hidden,
                num_layers = 1,
                batch_first= True,
                bidirectional= bidirectional)
            lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        else:
            lstm_out_dim = 0

        self.pools = nn.ModuleList([nn.AdaptiveMaxPool1d(ps) for ps in pool_sizes])

        self.out_dim = (mid_channels * pool_sizes[0] +
                        out_channels * (pool_sizes[1] + pool_sizes[2]) +
                        lstm_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      
        f1 = self.se1(self.conv1(x))                          
        f2 = self.se2(self.conv2(f1))                        

        # ---------- Attention ----------
        f2 = self.attn(f2.transpose(1,2),  
                       f2.transpose(1,2),  
                       f2.transpose(1,2))[0].transpose(1,2) 

        features = []

        for i, pool in enumerate(self.pools):
            if i == 0:
                features.append(pool(f1).flatten(1))         
            else:
                features.append(pool(f2).flatten(1))         

        # LSTM 
        if self.use_lstm:
            seq = f2.transpose(1,2)          
            _, (h_n, _) = self.lstm(seq)     
            features.append(h_n[-1])         

        # 
        return torch.cat(features, dim=1).flatten(1)
class SEBlock1d(Network):
    def __init__(self, name: str, channels: int, reduction: int = 16):
        super().__init__(name)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x) 
        return x * w
    
class Actor(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        self.num_scan = state_size - 5
        self.extra_inputs = 5
        self.scan_conv = MultiScaleConv1d(name+"_msc")
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.num_scan)
            dummy_feat = self.scan_conv(dummy_input)
            if dummy_feat.dim() == 3:
                dummy_feat = dummy_feat.flatten(1)
            self._true_conv_out_dim = dummy_feat.shape[1]
        print("[DEBUG] true scan_conv out_dim by dummy:", self._true_conv_out_dim)
        self.fa1 = nn.Linear(self._true_conv_out_dim + self.extra_inputs, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)
        self.apply(super().init_weights)

    def forward(self, s: torch.Tensor, *_):
        scan  = s[:, :self.num_scan].unsqueeze(1)
        extra = s[:, self.num_scan:]
        feat  = self.scan_conv(scan)
        if feat.dim() == 3:
            feat = feat.flatten(1)
        x = torch.cat([feat, extra], dim=1)
        x = F.relu(self.fa1(x))
        x = F.relu(self.fa2(x))
        return torch.tanh(self.fa3(x))


class Critic(Network):
    def __init__(self, name, state_size, action_size, hidden_size):
        super().__init__(name)
        self.num_scan = state_size - 5
        self.extra_inputs = 5
        self.action_size = action_size

        self.scan_conv = MultiScaleConv1d(name + "_msc")
        
        with torch.no_grad():
            dummy_s = torch.zeros(1, self.num_scan + self.extra_inputs)
            dummy_a = torch.zeros(1, action_size)
            scan = dummy_s[:, :self.num_scan].unsqueeze(1)
            extra = dummy_s[:, self.num_scan:]
            feat = self.scan_conv(scan)
            if feat.dim() == 3:
                feat = feat.flatten(1)
            xu = torch.cat([feat, extra, dummy_a], dim=1)
            self._true_input_dim = xu.shape[1]
        print("[DEBUG] Critic true input_dim by dummy:", self._true_input_dim)

        self.l1 = nn.Linear(self._true_input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        self.l4 = nn.Linear(self._true_input_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)
        self.apply(super().init_weights)


    def extract_features(self, states):
        scan = states[:, :self.num_scan].unsqueeze(1)
        extra = states[:, self.num_scan:]
        feat = self.scan_conv(scan).flatten(1)
        return feat, extra

    def forward(self, states, actions):
        feat, extra = self.extract_features(states)
        xu = torch.cat([feat, extra, actions], dim=1)
        q1 = self.l3(torch.relu(self.l2(torch.relu(self.l1(xu)))))
        q2 = self.l6(torch.relu(self.l5(torch.relu(self.l4(xu)))))
        return q1, q2

    def Q1_forward_for_actor(self, states, actions):
        feat, extra = self.extract_features(states)
        xu = torch.cat([feat, extra, actions], dim=1)
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(xu)))))

class TD3(OffPolicyAgent):
    def __init__(self, device, sim_speed, hidden_size=512):
        super().__init__(device, sim_speed, action_size=3)
        self.noise = NoisePackage(3, max_sigma=0.1, min_sigma=0.1, decay_period=8e6)
        self.policy_noise = POLICY_NOISE
        self.noise_clip = POLICY_NOISE_CLIP
        self.policy_freq = POLICY_UPDATE_FREQUENCY
        self.last_actor_loss = 0
        self.prev_action = np.zeros(3)  # 初始化前一动作
        
        self.beta = 1  # 平滑系数，越小越慢，自己试   
        print("Init TD3, beta ", self.beta) 
        print("Init hidden_size ", hidden_size) 
        # networks  
        self.actor        = self.create_network(Actor, 'actor')
        self.actor_target = self.create_network(Actor, 'target_actor')
        self.actor_optimizer = self.create_optimizer(self.actor)

        self.critic       = self.create_network(Critic, 'critic')
       
        self.critic_optimizer = self.create_optimizer(self.critic)
        self.critic_target= self.create_network(Critic, 'target_critic')

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        if os.getenv("WANDB_DISABLED", "false").lower() not in ("1", "true"): 
            wandb.init(
                project="go2_td3",              
                name   = f"run_{int(time.time())}",
                config = dict(
                    lr_actor  = self.actor_optimizer.param_groups[0]['lr'],
                    lr_critic = self.critic_optimizer.param_groups[0]['lr'],
                    hidden    = hidden_size,
                    scan_len  = NUM_SCAN_SAMPLES,
                    use_lstm  = True,
                    num_heads = 0         
                )
            )
            self._log_to_wandb = True
        else:
            self._log_to_wandb = False

    def reset_prev_action(self):
        self.prev_action = np.zeros(3)


    #--------------------------------------------------
    def get_action(self, state, is_training, step, visualize=False):
      s_t = torch.from_numpy(np.asarray(state, np.float32)).unsqueeze(0).to(self.device)
   
      a_tanh = self.actor(s_t, visualize)  

      if is_training:
          n = torch.from_numpy(self.noise.get_noise(step).astype(np.float32)).to(self.device)
          a_tanh = torch.clamp(a_tanh + n, -1.0, 1.0)
    
      # 
      a_np = a_tanh.squeeze(0).detach().cpu().numpy()  # [3,]
    
      #
      if not hasattr(self, 'beta'):
          self.beta = 1.0  # 
      smoothed = self.prev_action + self.beta * (a_np - self.prev_action)
      self.prev_action = smoothed  #
    
      return smoothed.tolist()

    
    def get_action_random(self):
        # φ ∈ [0,2π)
        phi = np.random.uniform(0, 2*np.pi)
        #  ∈ [0, SPEED_LINEAR_MAX]
        speed = np.random.uniform(0, SPEED_LINEAR_MAX)
        vx = speed * np.cos(phi)
        vy = speed * np.sin(phi)
        #  ∈ [−SPEED_ANGULAR_MAX, SPEED_ANGULAR_MAX]
        omega = np.random.uniform(-SPEED_ANGULAR_MAX, SPEED_ANGULAR_MAX)
        return [vx, vy, omega]
    
    def train(self, state, action, reward, state_next, done):
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
       
        action_next = (self.actor_target(state_next) + noise).clamp(-1.0, 1.0)
      
        Q1_value, Q2_value= self.critic_target(state_next, action_next)
        Q_mini_next = torch.min(Q1_value, Q2_value) #over estimate value 
        Q_value_target = reward + (1 - done) * self.discount_factor * Q_mini_next

        Q1_Buffer, Q2_Buffer = self.critic(state, action) #  form buffer take real value(state,act ) give to Q_net
        
        loss_critic = self.loss_function(Q1_Buffer, Q_value_target) + self.loss_function(Q2_Buffer, Q_value_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()

        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0)
        self.critic_optimizer.step()

        if self.iteration % self.policy_freq == 0:
            #  critic Q1 function as loss function , input  state and actor network actor , 
            loss_actor = -self.critic.Q1_forward_for_actor(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()   
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor, self.tau)
            self.soft_update(self.critic_target, self.critic, self.tau)
            self.last_actor_loss = loss_actor.detach().cpu()

        q_value = Q_value_target.mean().detach().cpu()
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss, q_value]
