import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
torch.manual_seed(42)


class Actor(nn.Module):
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
    def __init__(self, state_size: int, action_size: int, fc1_units: int = 256, fc2_units: int = 128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

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
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size: int, action_size: int, fc1_units: int = 256, fc2_units: int = 128):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

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

        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
