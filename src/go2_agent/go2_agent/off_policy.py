#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refactored off-policy agent base with renamed symbols to avoid collisions
with public repositories. This file also exposes backward-compatibility
aliases so older code importing legacy names continues to work.

Key changes:
- New constant names: LIDAR_SAMPLES, RES_* family.
- Renamed attributes for device/rates/shapes/hparams/etc.
- Helper methods renamed (make_net/make_optim, hard_sync/soft_sync).
- Backward-compat aliases mirror old attributes/methods.

If you want a *clean* version without any aliases, let me know and I’ll
strip the compatibility layer.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

# ===== New constants (with legacy aliases below) =====
LIDAR_SAMPLES        = 580

RES_UNKNOWN          = 0
RES_SUCCESS          = 1
RES_COLLISION_WALL   = 2
RES_COLLISION_OBS    = 3
RES_TIMEOUT          = 4
RES_TUMBLE           = 5
RES_COUNT            = 6

# ----- Legacy aliases (keep old code working) -----
NUM_SCAN_SAMPLES   = LIDAR_SAMPLES
UNKNOWN            = RES_UNKNOWN
SUCCESS            = RES_SUCCESS
COLLISION_WALL     = RES_COLLISION_WALL
COLLISION_OBSTACLE = RES_COLLISION_OBS
TIMEOUT            = RES_TIMEOUT
TUMBLE             = RES_TUMBLE
RESULTS_NUM        = RES_COUNT


class OffPolicyAgent(ABC):
    """
    Off-policy RL base class (renamed attributes).
    Backward-compat aliases are provided after the new attributes are set.
    """

    def __init__(self, device, simulation_speed):
        # --- Core runtime / device ---
        self.dev        = device
        self.sim_rate   = simulation_speed

        # --- Model/IO dimensions ---
        self.obs_dim      = 580 + 4      # lidar + extra features
        self.act_dim      = 2
        self.hidden_width = 512
        self.input_dim    = self.obs_dim

        # --- Hyperparameters ---
        self.batch_sz   = 128
        self.replay_cap = 1_000_000
        self.gamma      = 0.99
        self.lr         = 0.003
        self.polyak_tau = 0.003

        # --- Misc training settings ---
        self.dt         = 0.01
        self.loss_fn    = F.smooth_l1_loss

        # ε-greedy (if your subclass uses it)
        self.eps_greedy = 0.01
        self.eps_decay  = 0.9995
        self.eps_min    = 0.05

        # Execution options
        self.allow_reverse = False
        self.use_stacking  = False
        self.stack_len     = 3
        self.frame_skip    = 4
        if self.use_stacking:
            self.input_dim *= self.stack_len

        # Runtime bookkeeping
        self.nets  = []
        self.iters = 0

        self.state_size   = self.obs_dim
        self.action_size  = self.act_dim
        self.hidden_size  = self.hidden_width
        self.input_size   = self.input_dim
        # Hyperparams
        self.batch_size       = self.batch_sz
        self.buffer_size      = self.replay_cap
        self.discount_factor  = self.gamma
        self.learning_rate    = self.lr
        self.tau              = self.polyak_tau
        # Misc
        self.step_time        = self.dt
        self.loss_function    = self.loss_fn
        self.epsilon          = self.eps_greedy
        self.epsilon_decay    = self.eps_decay
        self.epsilon_minimum  = self.eps_min
        self.backward_enabled = self.allow_reverse
        self.stacking_enabled = self.use_stacking
        self.stack_depth      = self.stack_len
        self.networks         = self.nets
        self.iteration        = self.iters
        self.device           = self.dev              # keep legacy ctor name
        self.simulation_speed = self.sim_rate

    # 
    @abstractmethod
    def train(self, *args, **kwargs):
        """Implement one optimization step for a given batch."""
        pass

    @abstractmethod
    def get_action(self, *args, **kwargs):
        """Return an action for a given state (deterministic or stochastic)."""
        pass

    @abstractmethod
    def get_action_random(self, *args, **kwargs):
        """Return a random action (for exploration/warmup)."""
        pass

    #
    def _train_step(self, replaybuffer):
        s, a, r, ns, d = replaybuffer.sample(self.batch_sz)

        s  = torch.from_numpy(s).to(self.dev)
        a  = torch.from_numpy(a).to(self.dev)
        r  = torch.from_numpy(r).to(self.dev)
        ns = torch.from_numpy(ns).to(self.dev)
        d  = torch.from_numpy(d).to(self.dev)

        result = self.train(s, a, r, ns, d)

        # Update counters and ε-greedy
        self.iters += 1
        self.iteration = self.iters  # keep legacy mirror
        if self.eps_greedy and self.eps_greedy > self.eps_min:
            self.eps_greedy *= self.eps_decay
            self.epsilon = self.eps_greedy  # keep legacy mirror
        return result

    # ===== Builders =====
    def make_net(self, net_type, name):
        """Create a network instance and track it."""
        net = net_type(name, self.input_dim, self.act_dim, self.hidden_width).to(self.dev)
        self.nets.append(net)
        self.networks = self.nets  # legacy mirror
        return net

    def make_optim(self, net):
        """Create an optimizer for a given network."""
        return torch.optim.AdamW(net.parameters(), self.lr)

    # ===== Parameter syncing (hard/polyak) =====
    def hard_sync(self, target, source):
        """Copy parameters from source to target (hard update)."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)

    def soft_sync(self, target, source, tau=None):
        """Polyak averaging: target ← (1-τ)*target + τ*source."""
        tau = self.polyak_tau if tau is None else tau
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

    def dump_config(self):
        """Return a human-readable string of the config values."""
        cfg = ""
        for k, v in self.__dict__.items():
            if k not in ['actor', 'actor_target', 'critic', 'critic_target']:
                cfg += f"{k} = {v}\n"
        return cfg

    #
    def _train(self, replaybuffer):
        return self._train_step(replaybuffer)

    def create_network(self, type, name):
        return self.make_net(type, name)

    def create_optimizer(self, network):
        return self.make_optim(network)

    def hard_update(self, target, source):
        return self.hard_sync(target, source)

    def soft_update(self, target, source, tau):
        return self.soft_sync(target, source, tau)


class Network(torch.nn.Module, ABC):
    def __init__(self, name, visual=None):
        super().__init__()
        self.name = name
        self.visual = visual
        self.iteration = 0

    @abstractmethod
    def forward():
        pass
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

