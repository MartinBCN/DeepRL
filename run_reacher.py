from pathlib import Path
import os
from unityagents import UnityEnvironment
from reinforcement_learning.agents.continuous_agents import DDPG
from reinforcement_learning.training import Trainer
from reinforcement_learning.utils.buffer import ReplayBuffer

AGENTS = {'DDPG': DDPG}


def main(agent_type: str, n_episodes: int = 2000, max_t: int = 1000, eps_start: float = 1.0,
         eps_end: float = 0.01, eps_decay: float = 0.995, buffer_size: int = int(1e5), batch_size: int = 64,
         gamma: float = 0.99, tau: float = 1e-3, update_every: int = 4,
         lr_actor: float = 1e-4, lr_critic: float = 1e-3) -> None:
    """
    Training of Reacher

    Parameters
    ----------
    agent_type: str
        Define the type of agent being used
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
    update_every: int = 4
        how often to update the network
    lr_actor: float = 5e-4
        Learning rate
    lr_critic: float = 1e-4
        Learning rate

    Returns
    -------
    None
    """

    # select this option to load version 1 (with a single agent) of the environment
    env = UnityEnvironment(file_name='data/Reacher/Reacher.x86_64', no_graphics=True)

    # There is a version conflict between the Torch version that came with the Unity Environment
    # and Tensorboard. Resolving this may be complicated -> drop Tensorboard for the time being
    # writer = SummaryWriter(f'runs/{agent_type}')

    log_file = Path(os.environ.get('LOG_DIR', 'runs')) / f'{agent_type}.json'

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
                  update_every=update_every,
                  lr_actor=lr_actor,
                  lr_critic=lr_critic
    )

    trainer.train(agent, n_episodes=n_episodes)
    fn = Path(os.environ.get('FIG_DIR', '_includes')) / f'{agent_type}.png'
    trainer.plot(fn)

    if trainer.solved:
        model_dir = os.environ.get('MODEL_DIR', 'models')
        p = Path(model_dir) / f'{agent_type}'
        agent.save(p)


if __name__ == '__main__':
    # for agent in AGENTS.keys():
    #     dqn(agent_type=agent, n_episodes=2500)

    main(agent_type='DDPG', n_episodes=2000)
