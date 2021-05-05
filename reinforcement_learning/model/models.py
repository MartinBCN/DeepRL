from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
torch.manual_seed(42)


class Actor(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = None, batch_norm: bool = False,
                 dropout: float = None):
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
        batch_norm: bool
            Use batch normalization or not
        """
        super(Actor, self).__init__()

        if hidden_layers is None:
            hidden_layers = [400, 300]

        outputs = hidden_layers.copy()
        inputs = [state_size] + hidden_layers[:-1]

        self.linear_layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in zip(inputs, outputs)])
        if batch_norm:
            self.batch_norm = [nn.LayerNorm(output) for output in outputs]
        else:
            self.batch_norm = []
        self.activation_hidden = nn.ReLU()

        self.final_layer = nn.Linear(hidden_layers[-1], action_size)
        self.activation_final = torch.tanh

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(p=dropout)

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

        for i, linear in enumerate(self.linear_layers):
            x = linear(x)
            x = self.activation_hidden(x)

            if self.batch_norm:
                x = self.batch_norm[i](x)

            if self.dropout is not None:
                x = self.dropout(x)

        x = self.final_layer(x)
        x = self.activation_final(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int], action_layer: int = None,
                 batch_norm: bool = False, dropout: float = None):
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
        batch_norm: bool
            Use batch normalization or not
        """
        super(Critic, self).__init__()

        if action_layer is None:
            self.action_layer = len(hidden_layers) - 1
        else:
            self.action_layer = action_layer

        outputs = hidden_layers.copy()
        inputs = [state_size] + hidden_layers[:-1].copy()
        inputs[self.action_layer] += action_size

        self.linear_layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in zip(inputs, outputs)])
        if batch_norm:
            self.batch_norm = [nn.LayerNorm(output) for output in outputs]
        else:
            self.batch_norm = []
        self.activation_hidden = nn.ReLU()

        self.final_layer = nn.Linear(hidden_layers[-1], 1)

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(p=dropout)

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
        for i, linear in enumerate(self.linear_layers):
            if i == self.action_layer:
                x = torch.cat((x, action), dim=1)
            x = linear(x)
            x = self.activation_hidden(x)

            if self.batch_norm:
                x = self.batch_norm[i](x)

            if self.dropout is not None:
                x = self.dropout(x)

        return self.final_layer(x)


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
