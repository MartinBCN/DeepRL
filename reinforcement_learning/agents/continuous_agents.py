from pathlib import Path
from typing import Tuple, Union

import numpy as np
import random

from torch import Tensor
from torch.optim import Adam

from model import Actor, Critic
from reinforcement_learning.utils.buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.nn as nn

from reinforcement_learning.utils.noise import OUNoise

random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
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
    lr_actor: float = 5e-4
        Learning rate actor
    lr_critic: float = 1e-4
        Learning rate critic
    update_every: int = 4
        how often to update the network
    """

    def __init__(self, state_size: int, action_size: int, buffer_size: int = int(1e5), batch_size: int = 64,
                 gamma: float = 0.99, tau: float = 1e-3, update_every: int = 4,
                 lr_actor: float = 5e-4, lr_critic: float = 1e-4,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995) -> None:
        self.state_size = state_size
        self.action_size = action_size

        # Actor
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.optimizer_actor = Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.optimizer_critic = Adam(self.critic_local.parameters(), lr=lr_critic)

        # Epsilon
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.gamma = gamma
        self.tau = tau

        # Noise process
        self.noise = OUNoise(action_size)

        # Replay memory
        self.batch_size = batch_size
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
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

    def step(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool)\
            -> Tuple[float, float]:
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
        actor_loss, critic_loss = None, None
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                actor_loss, critic_loss = self.learn(experiences)
        return actor_loss, critic_loss

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

    def learn(self, experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tuple[float, float]:
        """
        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            tuple of (s, a, r, s', done) tuples
        Returns
        -------
        actor_loss: float
            Loss is returned for book-keeping
        critic_loss: float
            Loss is returned for book-keeping
        """

        states, actions, rewards, next_states, dones = experiences

        # --- critic part ---
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        return float(actor_loss.detach().cpu().numpy()), float(critic_loss.detach().cpu().numpy())

    def soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
        local_model: nn.Module
            weights will be copied from
        target_model: nn.Module
            weights will be copied to

        Returns
        -------

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save model

        Parameters
        ----------
        directory: Union[str, Path]

        Returns
        -------
        None
        """
        if type(directory) is str:
            directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        file_name = directory / 'actor.pt'
        torch.save(self.actor_local.state_dict(), file_name)

        file_name = directory / 'critic.pt'
        torch.save(self.critic_local.state_dict(), file_name)
