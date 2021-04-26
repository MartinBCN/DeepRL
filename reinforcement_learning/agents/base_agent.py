from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import random

from torch import Tensor
from torch.optim import Adam

import torch
import torch.nn.functional as F
import torch.nn as nn

from reinforcement_learning.model.models import DQN, Actor
from reinforcement_learning.utils.buffer import ReplayBuffer
from reinforcement_learning.utils.noise import OUNoise

random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAgent(ABC):
    """
    Interacts with and learns from the environment.

    Parameters
    ----------
    state_size: int
        dimension of each state
    action_size: int
        dimension of each action
    buffer_size: int = int(1e5)
        replay buffer size
    batch_size: int = 64
        mini-batch size
    gamma: float = 0.99
        discount factor
    tau: float = 1e-3
        for soft update of target parameters
    update_every: int = 4
        how often to update the network
    """

    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type,  buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995) -> None:
        self.state_size = state_size
        self.action_size = action_size

        # Epsilon
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.gamma = gamma
        self.tau = tau

        # Replay memory
        self.batch_size = batch_size
        self.memory = buffer_type(action_size, buffer_size, batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.update_every = update_every

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

    def step(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> None:
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
        loss: float
            Loss is returned for book-keeping
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

    # @abstractmethod
    # def state_to_action(self, state: Tensor) -> Tensor:
    #     """
    #     State-to-action step
    #
    #     Parameters
    #     ----------
    #     state: Tensor
    #
    #     Returns
    #     -------
    #     action: Tensor
    #     """

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

    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type,  buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995,
                 lr_actor: float = 1e-4
                 ) -> None:
        super(BaseContinuous, self).__init__(state_size, action_size, buffer_type, buffer_size, batch_size, gamma, tau,
                                             update_every, eps_start, eps_end, eps_decay)

        # Actor
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.optimizer_actor = Adam(self.actor_local.parameters(), lr=lr_actor)

        # Noise process
        self.noise = OUNoise(action_size)

    def act(self, state: np.array, use_noise: bool = True) -> np.array:
        """
        Returns actions for given state as per current policy

        Parameters
        ----------
        state: np.array
        use_noise: bool, default = True

        Returns
        -------
        action: np.array
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if use_noise:
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

    def __init__(self, state_size: int, action_size: int,
                 buffer_type: type,  buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, update_every: int = 4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995,
                 lr: float = 5e-4
                 ) -> None:
        super(BaseDiscrete, self).__init__(state_size, action_size, buffer_type, buffer_size, batch_size, gamma, tau,
                                           update_every, eps_start, eps_end, eps_decay)

        # Q-Network
        self.q_network = DQN(state_size, action_size).to(device)
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)

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
