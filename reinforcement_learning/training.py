import json
from collections import deque
from pathlib import Path
from typing import Union, Tuple
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment

import matplotlib.pyplot as plt

from reinforcement_learning.agents.base_agent import BaseAgent

plt.style.use('ggplot')


class Trainer:
    """
    Training of DQN

    Parameters
    ----------
    env: UnityEnvironment
        Environment
    brain: int
        Number of brain to be used
    max_t: int, default = 1000
        maximum number of time steps per episode
    score_solved: int, default = 30
        Score after a given number of episodes that is the criterion for solving
    n_episode_solved: int, default = 100
        Number of episodes over which the score_solved needs to be averaged
    log_file: Path
        Path for logging
    """
    def __init__(self, env: UnityEnvironment, brain: int = 0, max_t: int = 1000,
                 score_solved: int = 30, n_episode_solved: int = 100, log_file: Union[Path, str] = None):

        self.max_t = max_t

        # Environment
        self.env = env
        self.brain_name = env.brain_names[brain]
        _ = env.reset(train_mode=True)[self.brain_name]

        # Book-keeping
        self.solved = False
        self.logger = {'batch': {}, 'epoch': {'scores_window': deque(maxlen=n_episode_solved)}}

        if type(log_file) is str:
            log_file = Path(log_file)
        log_file.parents[0].mkdir(parents=True, exist_ok=True)
        self.log_file = log_file

        # Scoring
        self.score_solved = score_solved
        self.n_episode_solved = n_episode_solved

    def get_sizes(self) -> Tuple[int, int]:
        """
        Get state size and action size

        Returns
        -------
        state_size: int
        action_size: int
        """
        brain = self.env.brains[self.brain_name]
        state_size = brain.vector_observation_space_size
        action_size = brain.vector_action_space_size
        return state_size, action_size

    def log_batch(self, name: str, value: Union[int, float]):
        if name in self.logger['batch'].keys():
            self.logger['batch'][name].append(value)
        else:
            self.logger['batch'][name] = [value]

    def log_epoch(self, name: str, value: Union[int, float]):
        if name in self.logger['epoch'].keys():
            self.logger['epoch'][name].append(value)
        else:
            self.logger['epoch'][name] = [value]

    def train(self, agent: BaseAgent, n_episodes: int) -> None:
        """
        Main training loop

        Parameters
        ----------
        agent: Agent
        n_episodes: int

        Returns
        -------
        None
        """

        for i_episode in range(1, n_episodes + 1):

            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state

            score = 0
            episode_loss = 0

            for t in range(self.max_t):
                action = agent.act(state)
                env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                loss = agent.step(state, action, reward, next_state, done)
                if loss is not None:
                    self.log_batch('loss', loss)
                    episode_loss += loss
                state = next_state
                score += reward
                if done:
                    break

            # Decrease epsilon
            agent.step_epsilon()

            # Book-keeping
            self.log_epoch('score', score)
            self.log_epoch('score_window', score)
            self.log_epoch('loss', episode_loss)

            mean_score = np.mean(self.logger['epoch']['score_window'])
            log_str = f'\rEpisode {i_episode}\tAverage Score: {mean_score:.2f}'
            self.log_epoch('Mean Score', float(mean_score))
            print(log_str, end="")
            if i_episode % 100 == 0:
                print(log_str)

            # Solving Criterion
            if mean_score >= self.score_solved:  # Criterion in Rubric
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {mean_score:.2f}')
                self.solved = True
                break

            self.save_logs()

    def plot(self, file_name: Union[str, Path] = None) -> None:
        """
        Create analysis plot

        Parameters
        ----------
        file_name: Union[str, Path], default = None

        Returns
        -------
        None
        """

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # --- Scores ---
        scores = self.logger['score']
        x = np.arange(len(scores))
        axes[0].plot(x, scores)
        y = np.convolve(scores, np.ones(100) / 100, mode='same')
        axes[0].plot(x, y)

        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Episode #')

        # --- Losses ---
        losses = self.logger['loss']
        x = np.arange(len(losses))
        axes[1].plot(x, losses)
        y = np.convolve(losses, np.ones(100) / 100, mode='same')
        axes[1].plot(x, y)

        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Episode #')

        if file_name is None:
            fig.show()
        else:
            if type(file_name) is str:
                file_name = Path(file_name)
            file_name.parents[0].mkdir(parents=True, exist_ok=True)
            fig.savefig(file_name)

    def save_logs(self):
        """
        Save the training logs as json

        Returns
        -------

        """

        # Remove dequeue before serialising
        to_file = {'batch': self.logger['batch'],
                   'epoch': {k: v for k, v in self.logger['epoch'].items() if k != 'scores_window'}}
        with open(self.log_file, 'w') as f:
            json.dump(to_file, f)


class ActorCriticTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(ActorCriticTrainer, self).__init__(*args, **kwargs)

    def train(self, agent: BaseAgent, n_episodes: int) -> None:
        """
        Main training loop

        Parameters
        ----------
        agent: Agent
        n_episodes: int

        Returns
        -------
        None
        """

        for i_episode in range(1, n_episodes + 1):

            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state

            score = 0
            episode_loss_actor = 0
            episode_loss_critic = 0

            for t in range(self.max_t):
                action = agent.act(state)
                env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                loss_actor, loss_critic = agent.step(state, action, reward, next_state, done)
                if loss_actor is not None:
                    self.log_batch('loss_batch_actor', loss_actor)
                    self.log_batch('loss_batch_critic', loss_critic)
                    episode_loss_actor += loss_actor
                    episode_loss_critic += loss_critic
                state = next_state
                score += reward
                if done:
                    break

            # Decrease epsilon
            agent.step_epsilon()

            # Book-keeping
            self.log_epoch('score', score)
            self.log_epoch('score_window', score)
            self.log_epoch('loss_actor', episode_loss_actor)
            self.log_epoch('loss_critic', episode_loss_critic)

            mean_score = np.mean(self.logger['epoch']['score_window'])
            log_str = f'\rEpisode {i_episode}\tAverage Score: {mean_score:.2f}'
            self.log_epoch('Mean Score', float(mean_score))
            print(log_str, end="")
            if i_episode % 100 == 0:
                print(log_str)

            # Solving Criterion
            if mean_score >= self.score_solved:  # Criterion in Rubric
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {mean_score:.2f}')
                self.solved = True
                break

            self.save_logs()
