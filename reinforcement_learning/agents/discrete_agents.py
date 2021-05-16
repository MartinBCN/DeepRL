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
device = 'cpu'


class FixedQTargetAgent(BaseDiscrete):
    def __init__(self, state_size: int, action_size: int, agent_config) -> None:
        super(FixedQTargetAgent, self).__init__(state_size, action_size, agent_config)
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
        self.soft_update(self.q_network, self.q_network_target)

        return float(loss.detach().cpu().numpy())


class DoubleQAgent(BaseDiscrete):
    def __init__(self, state_size: int, action_size: int, agent_config: dict) -> None:
        super(DoubleQAgent, self).__init__(state_size, action_size, agent_config=agent_config)
        # Q-Network
        self.q_network_target = DQN(state_size, action_size).to(device)
        self.optimizer_target = Adam(self.q_network_target.parameters(), lr=agent_config['lr'])

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
    def __init__(self, state_size: int, action_size: int, agent_config) -> None:
        super(BasicAgent, self).__init__(state_size, action_size, agent_config=agent_config)

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
    def __init__(self, state_size: int, action_size: int, agent_config: dict):
        super(DuelingBasicAgent, self).__init__(state_size, action_size, agent_config=agent_config)
        # Q-Network
        self.q_network = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = Adam(self.q_network.parameters(), lr=agent_config['q_network']['lr'])


class FixedTargetDuelingAgent(FixedQTargetAgent):
    def __init__(self, state_size: int, action_size: int, agent_config: dict):
        super(FixedTargetDuelingAgent, self).__init__(state_size, action_size, agent_config=agent_config)

        self.q_network_target = DuelingDQN(state_size, action_size).to(device)
        self.q_network = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = Adam(self.q_network.parameters(), lr=agent_config['q_network']['lr'])
