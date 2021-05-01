from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union, Dict

import numpy as np
import random

from torch import Tensor
from torch.optim import Adam
import torch
import torch.nn as nn

from reinforcement_learning.model.models import DQN, Actor
from reinforcement_learning.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from reinforcement_learning.utils.noise import OUNoise

random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER = {'RB': ReplayBuffer, "PRB": PrioritizedReplayBuffer}


class BaseAgent(ABC):
    """
    Interacts with and learns from the environment.

    Parameters
    ----------
    state_size: int
        dimension of each state
    action_size: int
        dimension of each action
    """

    def __init__(self, state_size: int, action_size: int, agent_config: dict) -> None:
        self.state_size = state_size
        self.action_size = action_size

        self.config = agent_config

        # Epsilon
        self.eps = agent_config['epsilon']['eps_start']
        self.eps_end = agent_config['epsilon']['eps_end']
        self.eps_decay = agent_config['epsilon']['eps_decay']

        self.gamma = agent_config['gamma']
        self.tau = agent_config['tau']

        # Replay memory
        self.batch_size = agent_config['batch_size']
        buffer_size = agent_config['buffer']['buffer_size']
        buffer = BUFFER[agent_config['buffer']['type']]
        self.memory = buffer(action_size, buffer_size, self.batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.update_every = agent_config['update_every']

        self.training_mode = True

    def step_epsilon(self):
        """
        decrease epsilon

        Returns
        -------

        """
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

    def eval(self):
        self.training_mode = False

    def train(self):
        self.training_mode = True

    def epsilon(self):
        if self.training_mode:
            return self.eps
        else:
            return 0.

    def step(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> Dict[str, float]:
        """
        Add a new tuple to the memory and execute the training step after the defined number of time steps

        Parameters
        ----------
        state: np.array
        action: int
        reward: float
        next_state: np.array
        done: bool

        Returns
        -------
        Dict[str, float]
            Loss is returned for book-keeping. To allow for more than one we return a dictionary
        """
        loss = None
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                loss = {'Loss': loss}
        return loss

    @abstractmethod
    def act(self, state: np.array) -> np.array:
        """
        Returns actions for given state as per current policy

        Parameters
        ----------
        state: np.array

        Returns
        -------
        action: np.array
        """

    @abstractmethod
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

    @staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
        local_model: nn.Module
            weights will be copied from
        target_model: nn.Module
            weights will be copied to
        tau: float
            interpolation parameter

        Returns
        -------

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @abstractmethod
    def save(self, path_name: Union[str, Path]) -> None:
        """
        Save model

        Parameters
        ----------
        path_name: Union[str, Path]

        Returns
        -------
        None
        """


class BaseContinuous(BaseAgent, ABC):

    def __init__(self, state_size: int, action_size: int, agent_config: dict) -> None:
        super(BaseContinuous, self).__init__(state_size, action_size, agent_config)

        # Actor
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.optimizer_actor = Adam(self.actor_local.parameters(), lr=agent_config['actor']['lr'])

        # Noise process
        self.use_noise = agent_config.get('use_noise', True)
        self.noise = OUNoise(action_size)

    def act(self, state: np.array) -> np.array:
        """
        Returns actions for given state as per current policy

        Parameters
        ----------
        state: np.array

        Returns
        -------
        action: np.array
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.use_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def save(self, path_name: Union[str, Path]) -> None:
        """
        Save model

        Parameters
        ----------
        path_name: Union[str, Path]

        Returns
        -------
        None
        """
        if type(path_name) is str:
            path_name = Path(path_name)
        path_name.mkdir(exist_ok=True, parents=True)

        file_name = path_name / 'actor.pt'
        torch.save(self.actor_local.state_dict(), file_name)


class BaseDiscrete(BaseAgent, ABC):

    def __init__(self, state_size: int, action_size: int, agent_config: dict) -> None:
        super(BaseDiscrete, self).__init__(state_size, action_size, agent_config)

        # Q-Network
        self.q_network = DQN(state_size, action_size).to(device)
        self.optimizer = Adam(self.q_network.parameters(), lr=agent_config['q_network']['lr'])

    def act(self, state: np.array) -> np.array:
        """
        Returns actions for given state as per current policy

        Parameters
        ----------
        state: np.array

        Returns
        -------
        action: np.array
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon():
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def save(self, file_name: Union[str, Path]) -> None:
        """
        Save model

        Parameters
        ----------
        file_name: Union[str, Path]

        Returns
        -------
        None
        """
        if type(file_name) is str:
            file_name = Path(file_name)
        file_name.parents[0].mkdir(exist_ok=True, parents=True)
        torch.save(self.q_network.state_dict(), file_name)
