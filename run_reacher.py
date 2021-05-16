import json
from pathlib import Path
import os
from unityagents import UnityEnvironment
from reinforcement_learning.agents.continuous_agents import DDPG
from reinforcement_learning.training import Trainer
import torch.nn.functional as F

from reinforcement_learning.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer

AGENTS = {'DDPG': DDPG}


def main(agent_config: dict, n_episodes: int = 2000, max_t: int = 1000) -> None:
    """
    Training of Reacher

    Parameters
    ----------
    n_episodes: int, default = 2000
        maximum number of training episode
    max_t: int, default = 1000
        maximum number of time steps per episode
    agent_config: dict

    Returns
    -------
    None
    """

    # select this option to load version 1 (with a single agent) of the environment
    env = UnityEnvironment(file_name='data/Reacher/Reacher.x86_64', no_graphics=True)

    # There is a version conflict between the Torch version that came with the Unity Environment
    # and Tensorboard. Resolving this may be complicated -> drop Tensorboard for the time being
    # writer = SummaryWriter(f'runs/{agent_type}')

    # --- Path for logging ---
    path = Path('runs/reacher')
    paths = list(path.glob('run*'))

    if paths:
        last = [p.name for p in paths]
        last = max(last)
        last = int(last.replace('run', ''))
        path /= f'run{last + 1:03d}'
    else:
        path /= 'run001'

    path.mkdir()

    log_file = path / f'logs.json'
    conf_file = path / 'config.json'
    # with open(conf_file, 'w') as file:
    #     json.dump(agent_config, file)

    trainer = Trainer(env,
                      brain=0,
                      max_t=max_t,
                      score_solved=30,
                      n_episode_solved=100,
                      log_file=log_file)

    state_size, action_size = trainer.get_sizes()

    agent = agent_config['agent_type'](state_size=state_size,
                                       action_size=action_size,
                                       agent_config=agent_config
                                       )

    trainer.train(agent, n_episodes=n_episodes)

    if trainer.solved:
        model_dir = os.environ.get('MODEL_DIR', 'models')
        p = Path(model_dir) / f'{path.name}'
        agent.save(p)


if __name__ == '__main__':
    config = {
        'agent_type': DDPG,
        'buffer': {'type': ReplayBuffer, 'buffer_size': int(1e6), 'batch_size': 128},
        'epsilon': {'eps_start': 1.0, 'eps_end': 0.01, 'eps_decay': 0.995},
        'gamma': 0.99, 'tau': 1e-3,
        'update_every': 5,
        'num_agents': 1,
        'add_noise': True,
        'gradient_clipping': True,
        'model_params': {
            'norm': True,
            'lr': 0.001,
            'hidden_layers': [512, 256],
            'dropout': 0.05,
            'act_fn': F.leaky_relu,
            'action_layer': 1,
            'weight_decay': 0.01
        },
        'noise_params': {
            'mu': 0.,
            'theta': 0.15,
            'sigma': 0.2,
        }
    }
    main(agent_config=config, n_episodes=2000)
