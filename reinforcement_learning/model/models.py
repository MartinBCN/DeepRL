from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
torch.manual_seed(42)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int, act_fn: callable, norm: List,
                 hidden_layers: List[int], dropout: float):
        """
        Actor (Policy) Model.

        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        hidden_layers: List[int]
            Hidden layer sizes
        norm: bool
            Use layer normalization or not
        """
        super(Actor, self).__init__()

        # logger.debug('Parameter: %s', params)

        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(42)
        self.act_fn = act_fn

        inputs = [state_size] + hidden_layers.copy()[:-1]
        outputs = hidden_layers.copy()
        self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in zip(inputs, outputs)])

        # Add a variable number of more hidden layers
        self.output = nn.Linear(hidden_layers[-1], self.action_size)
        self.dropout = nn.Dropout(p=dropout)

        if norm:
            self.norms = [nn.LayerNorm(linear.in_features) for linear in self.hidden_layers]
        else:
            self.norms = []

        self.reset_parameters()

    def forward(self, state: Tensor) -> Tensor:
        """
        Build a network that maps state -> action values.

        Parameters
        ----------
        state: Tensor
            State tensor, shape [batch_size, state_size]
        Returns
        -------
        action: Tensor
            State tensor, shape [batch_size, action_size]
        """
        x = state
        for i, linear in enumerate(self.hidden_layers):
            if self.norms:
                x = self.norms[i](x)
            x = linear(x)
            x = self.act_fn(x)
            x = self.dropout(x)

        x = self.output(x)
        return torch.tanh(x)

    def reset_parameters(self):
        for linear in self.hidden_layers:
            linear.weight.data.uniform_(*hidden_init(linear))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int], act_fn: callable,
                 action_layer: int = None,
                 norm: bool = False, dropout: float = None):
        """
        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        hidden_layers: List[int]
            Hidden layer sizes
        action_layer: int
            Layer at which action is concatenated
        norm: bool
            Use layer normalization or not
        """
        super(Critic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(42)
        self.act_fn = act_fn
        self.action_layer = action_layer

        hidden_layers = hidden_layers.copy()

        self.norms = []
        if norm:
            self.norms.append(nn.LayerNorm(self.state_size))
            for hidden_neurons in hidden_layers:
                self.norms.append(nn.LayerNorm(hidden_neurons))

        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        hidden_layers[self.action_layer - 1] += self.action_size
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Build a network that maps state, action -> Q values.

        Parameters
        ----------
        state: Tensor
            State tensor, shape [batch_size, state_size]
        action: Tensor
            State tensor, shape [batch_size, action_size]
        Returns
        -------
        q: Tensor
            Q-Value for state-action pair
        """
        x = state
        for i, linear in enumerate(self.hidden_layers):
            if self.norms:
                x = self.norms[i](x)
            if i == self.action_layer:
                x = torch.cat((x, action), dim=1)
            x = linear(x)
            x = self.act_fn(x)
            x = self.dropout(x)

        if self.norms:
            x = self.norms[i + 1](x)
        x = self.output(x)
        return x

    def reset_parameters(self):
        for linear in self.hidden_layers:
            linear.weight.data.uniform_(*hidden_init(linear))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


class DQN(nn.Module):
    """
    Actor (Policy) Model.

    Parameters
    ----------
    state_size: int
        Dimension of each state
    action_size: int
        Dimension of each action
    fc1_units: int
        Number of nodes in first hidden layer
    fc2_units: int
        Number of nodes in second hidden layer
    """

    def __init__(self, state_size: int, action_size: int, fc1_units: int = 64, fc2_units: int = 64) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.main = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def reset_parameters(self):
        for linear in self.hidden_layers:
            linear.weight.data.uniform_(*hidden_init(linear))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: Tensor) -> Tensor:
        """
        Build a network that maps state -> action values.

        Parameters
        ----------
        state: Tensor
            State tensor, shape [batch_size, state_size]
        Returns
        -------
        action: Tensor
            State tensor, shape [batch_size, action_size]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN based on https://arxiv.org/abs/1511.06581

    Parameters
    ----------
    state_size: int
        Dimension of each state
    action_size: int
        Dimension of each action
    fc1_units: int, default = 64
        Number of nodes in first hidden layer
    fc2_units: int, default = 64
        Number of nodes in second hidden layer
    value_layer_units: int, default = 64
        Number of nodes in value layer
    """

    def __init__(self, state_size: int, action_size: int, fc1_units: int = 64, fc2_units: int = 64,
                 value_layer_units: int = 64) -> None:
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(fc2_units, value_layer_units),
            nn.ReLU(),
            nn.Linear(value_layer_units, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(fc2_units, value_layer_units),
            nn.ReLU(),
            nn.Linear(value_layer_units, action_size)
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        Build a network that maps state -> action values.

        Parameters
        ----------
        state: Tensor
            State tensor, shape [batch_size, state_size]
        Returns
        -------
        action: Tensor
            State tensor, shape [batch_size, action_size]
        """
        features = self.feature_layer(state)
        values = self.value_layer(features)
        advantage = self.advantage_layer(features)
        action = values + (advantage - advantage.mean())

        return action
