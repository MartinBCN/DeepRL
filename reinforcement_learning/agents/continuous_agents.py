from pathlib import Path
from typing import Tuple, Union, Dict

import numpy as np
import random

from torch import Tensor, optim
from torch.optim import Adam

from reinforcement_learning.agents.base_agent import BaseContinuous
from reinforcement_learning.model.models import Actor, Critic
import torch
import torch.nn.functional as F
random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class DDPG(BaseContinuous):
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
        super(DDPG, self).__init__(state_size, action_size, agent_config=agent_config)
        self.state_size = state_size
        self.action_size = action_size

        lr = agent_config['model_params']['lr']
        hidden_layers = agent_config['model_params']['hidden_layers']
        dropout = agent_config['model_params']['dropout']
        norm = agent_config['model_params']['norm']
        act_fn = agent_config['model_params']['act_fn']

        # Critic Network (w/ Target Network)
        action_layer = agent_config['model_params']['action_layer']
        weight_decay = agent_config['model_params']['weight_decay']
        self.critic_local = Critic(action_size=action_size, state_size=state_size,
                                   hidden_layers=hidden_layers, action_layer=action_layer,
                                   dropout=dropout, norm=norm, act_fn=act_fn
                                   ).to(device)
        self.critic_target = Critic(action_size=action_size, state_size=state_size,
                                    hidden_layers=hidden_layers, action_layer=action_layer,
                                    dropout=dropout, norm=norm, act_fn=act_fn
                                    ).to(device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr,
                                           weight_decay=weight_decay)

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
            -> Dict[str, float]:
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
        loss = {}
        # Save experience in replay memory
        experience = state, action, reward, next_state, done

        self.memory.add(experience, 1.)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                actor_loss, critic_loss = self.learn(experiences)
                loss = {'ActorLoss': actor_loss, 'CriticLoss': critic_loss}
        return loss

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
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        return float(actor_loss.detach().cpu().numpy()), float(critic_loss.detach().cpu().numpy())

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
