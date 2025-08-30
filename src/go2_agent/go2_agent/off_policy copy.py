#!/usr/bin/env python3
 
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as torchf

NUM_SCAN_SAMPLES = 580
UNKNOWN = 0
SUCCESS = 1 
COLLISION_WALL = 2
COLLISION_OBSTACLE = 3
TIMEOUT = 4
TUMBLE = 5
RESULTS_NUM = 6 

class OffPolicyAgent(ABC):
    def __init__(self, device, simulation_speed):

        self.device = device
        self.simulation_speed   = simulation_speed

        # Network structure
        self.state_size         = 580 + 4 
        self.action_size        = 2
        self.hidden_size        = 512
        self.input_size         = self.state_size
        # Hyperparameters
        self.batch_size         = 128
        self.buffer_size        = 1000000
        self.discount_factor    = 0.99
        self.learning_rate      = 0.003
        self.tau                = 0.003
        # Other parameters
        self.step_time          = 0.01
        self.loss_function      = torchf.smooth_l1_loss
        self.epsilon            = 0.01 
        self.epsilon_decay      = 0.9995
        self.epsilon_minimum    = 0.05
        self.backward_enabled   = False
        self.stacking_enabled   = False #ENABLE_STACKING
        self.stack_depth        = 3
        self.frame_skip         = 4
        if self.stacking_enabled:
            self.input_size *= self.stack_depth

        self.networks = []
        self.iteration = 0

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def get_action():
        pass

    @abstractmethod
    def get_action_random():
        pass

    def _train(self, replaybuffer):
        batch = replaybuffer.sample(self.batch_size)
        sample_s, sample_a, sample_r, sample_ns, sample_d = batch
        sample_s = torch.from_numpy(sample_s).to(self.device)
        sample_a = torch.from_numpy(sample_a).to(self.device)
        sample_r = torch.from_numpy(sample_r).to(self.device)
        sample_ns = torch.from_numpy(sample_ns).to(self.device)
        sample_d = torch.from_numpy(sample_d).to(self.device)
        result = self.train(sample_s, sample_a, sample_r, sample_ns, sample_d)
        self.iteration += 1
        if self.epsilon and self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        return result

    def create_network(self, type, name):
        network = type(name, self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.networks.append(network)
        return network

    def create_optimizer(self, network):
        return torch.optim.AdamW(network.parameters(), self.learning_rate)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_model_configuration(self):
        configuration = ""
        for attribute, value in self.__dict__.items():
            if attribute not in ['actor', 'actor_target', 'critic', 'critic_target']:
                configuration += f"{attribute} = {value}\n"
        return configuration


class Network(torch.nn.Module, ABC):
    def __init__(self, name, visual=None):
        super(Network, self).__init__()
        self.name = name
        self.visual = visual
        self.iteration = 0

    @abstractmethod
    def forward():
        pass

    def init_weights(n, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)