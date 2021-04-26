from pathlib import Path

from unityagents import UnityEnvironment
import os
from reinforcement_learning.agents.discrete_agents import BasicAgent, FixedQTargetAgent, DoubleQAgent, \
    DuelingBasicAgent, FixedTargetDuelingAgent
from reinforcement_learning.training import Trainer
from reinforcement_learning.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer

AGENTS = {'basic': BasicAgent, 'fixed_target': FixedQTargetAgent, 'double_q': DoubleQAgent,
          'dueling_basic': DuelingBasicAgent, 'fixed_target_dueling': FixedTargetDuelingAgent}


def dqn(agent_type: str, name: str = None, n_episodes: int = 2000, max_t: int = 1000, eps_start: float = 1.0,
        eps_end: float = 0.01, eps_decay: float = 0.995, buffer_size: int = int(1e5), batch_size: int = 64,
        gamma: float = 0.99, tau: float = 1e-3, lr: float = 5e-4, update_every: int = 4) -> None:
    """
    Training of DQN

    Parameters
    ----------
    agent_type: str
        Define the type of agent being used
     name: str, default = None
        Name for storing
    n_episodes: int, default = 2000
        maximum number of training episodes
    max_t: int, default = 1000
        maximum number of time steps per episode
    eps_start: float, default = 1.0
        starting value of epsilon, for epsilon-greedy action selection
    eps_end: float = 0.01
        minimum value of epsilon
    eps_decay: float = 0.995
        multiplicative factor (per episode) for decreasing epsilon
    buffer_size: int = int(1e5)
        replay buffer size
    batch_size: int = 64
        mini-batch size
    gamma: float = 0.99
        discount factor
    tau: float = 1e-3
        for soft update of target parameters
    lr: float = 5e-4
        Learning rate
    update_every: int = 4
        how often to update the network

    Returns
    -------
    None
    """
    env = UnityEnvironment('data/Banana_Linux/Banana.x86_64', no_graphics=True)

    # There is a version conflict between the Torch version that came with the Unity Environment
    # and Tensorboard. Resolving this may be complicated -> drop Tensorboard for the time being
    # writer = SummaryWriter(f'runs/{agent_type}')

    if name is None:
        name = agent_type

    log_file = Path(os.environ.get('LOG_DIR', 'runs')) / f'{name}.json'

    trainer = Trainer(env,
                      brain=0,
                      max_t=max_t,
                      score_solved=30,
                      n_episode_solved=100,
                      log_file=log_file)

    state_size, action_size = trainer.get_sizes()
    assert agent_type in AGENTS.keys(), f'agent_type needs to be choice of {AGENTS.keys()}'

    agent = AGENTS[agent_type]
    agent = agent(state_size=state_size,
                  action_size=action_size,
                  buffer_type=ReplayBuffer,
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  eps_start=eps_start,
                  eps_end=eps_end,
                  eps_decay=eps_decay,
                  gamma=gamma,
                  tau=tau,
                  lr=lr,
                  update_every=update_every
    )

    trainer.train(agent, n_episodes=n_episodes)
    fn = Path(os.environ.get('FIG_DIR', '_includes')) / f'{name}.png'
    trainer.plot(fn)

    trainer.save_logs()

    if trainer.solved:
        model_dir = os.environ.get('MODEL_DIR', 'models')
        fn = Path(model_dir) / f'{name}.pt'
        agent.save(fn)


if __name__ == '__main__':
    # for agent in AGENTS.keys():
    #     dqn(agent_type=agent, n_episodes=2500)

    dqn(agent_type='fixed_target_dueling', n_episodes=2000)
