from typing import Tuple
import numpy as np
import random
from collections import namedtuple, deque
import torch
from torch import Tensor

from reinforcement_learning.utils.sum_tree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(42)


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.

    Parameters
    ----------
    action_size: int
    buffer_size: int
    batch_size: int
    """

    def __init__(self, action_size: int, buffer_size: int, batch_size: int) -> None:
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> None:
        """
        Add a new experience to memory.

        Parameters
        ----------
        state: np.array
        action: int
        reward: float
        next_state: np.array
        done: bool

        Returns
        -------
        None
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Randomly sample a batch of experiences from memory.

        Returns
        -------
        states: Tensor
        actions: Tensor
        rewards: Tensor
        next_states: Tensor
        dones: Tensor
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Length of the internal memory

        Returns
        -------
        int
        """
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size: int, buffer_size: int, batch_size: int) -> None:
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size)
        self.memory = SumTree(capacity=buffer_size)

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        experiences = []
        for _ in range(self.batch_size):
            s = random.random() * self.memory.total()
            experience = self.memory.get(sampled_value=s)
            experiences.append(experience)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def add(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> None:
        """
        Add a new experience to memory.

        Parameters
        ----------
        state: np.array
        action: int
        reward: float
        next_state: np.array
        done: bool

        Returns
        -------
        None
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.add(priority=reward + 1e-1, data=e)
