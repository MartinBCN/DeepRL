from typing import Tuple
import random

from torch import Tensor
from torch.optim import Adam

import torch
import torch.nn.functional as F
from reinforcement_learning.agents.base_agent import BaseDiscrete
from reinforcement_learning.model.models import DQN, DuelingDQN

random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FixedQTargetAgent(BaseDiscrete):
    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type,  buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995,
                 lr: float = 5e-4) -> None:
        super(FixedQTargetAgent, self).__init__(state_size, action_size, buffer_type, buffer_size, batch_size,
                                                gamma, tau,
                                                update_every, eps_start, eps_end, eps_decay, lr)
        # Q-Network
        self.q_network_target = DQN(state_size, action_size).to(device)

    def learn(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> float:
        """
        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            tuple of (s, a, r, s', done) tuples
        Returns
        -------
        loss: float
            Loss is returned for book-keeping
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_network, self.q_network_target, self.tau)

        return float(loss.detach().cpu().numpy())


class DoubleQAgent(BaseDiscrete):
    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type, buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, lr: float = 5e-4, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995) -> None:
        super(DoubleQAgent, self).__init__(state_size, action_size, buffer_type, buffer_size, batch_size,
                                           gamma, tau, update_every, eps_start, eps_end, eps_decay, lr)
        # Q-Network
        self.q_network_target = DQN(state_size, action_size).to(device)
        self.optimizer_target = Adam(self.q_network_target.parameters(), lr=lr)

    def learn(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> float:
        """
        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            tuple of (s, a, r, s', done) tuples
        Returns
        -------
        loss: float
            Loss is returned for book-keeping
        """

        states, actions, rewards, next_states, dones = experiences

        q_targets_next_1 = self.q_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets_next_2 = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets_next = torch.min(q_targets_next_1, q_targets_next_2)

        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_expected_1 = self.q_network(states).gather(1, actions)
        q_expected_2 = self.q_network_target(states).gather(1, actions)

        # Compute loss
        loss_1 = F.mse_loss(q_expected_1, q_targets)
        loss_2 = F.mse_loss(q_expected_2, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss_1.backward()
        self.optimizer.step()

        self.optimizer_target.zero_grad()
        loss_2.backward()
        self.optimizer_target.step()

        return (float(loss_1.detach().cpu().numpy()) + float(loss_2.detach().cpu().numpy())) / 2


class BasicAgent(BaseDiscrete):
    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type, buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995, lr: float = 5e-4, ) -> None:
        super(BasicAgent, self).__init__(state_size, action_size, buffer_type, buffer_size, batch_size,
                                         gamma, tau, update_every, eps_start, eps_end, eps_decay, lr)

    def learn(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> float:
        """
        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            tuple of (s, a, r, s', done) tuples
        Returns
        -------
        loss: float
            Loss is returned for book-keeping
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.q_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu().numpy())


class DuelingBasicAgent(BasicAgent):
    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type, buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, lr: float = 5e-4, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995):
        super(DuelingBasicAgent, self).__init__(state_size, action_size, buffer_type, buffer_size, batch_size,
                                                gamma, tau, update_every, eps_start, eps_end, eps_decay, lr)
        # Q-Network
        self.q_network = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)


class FixedTargetDuelingAgent(FixedQTargetAgent):
    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type, buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, lr: float = 5e-4, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995):
        super(FixedTargetDuelingAgent, self).__init__(state_size, action_size, buffer_type, buffer_size, batch_size,
                                                      gamma, tau, update_every, eps_start, eps_end, eps_decay, lr)

        self.q_network_target = DuelingDQN(state_size, action_size).to(device)
        self.q_network = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
