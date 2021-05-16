from pathlib import Path

from unityagents import UnityEnvironment
import os
from reinforcement_learning.agents.discrete_agents import BasicAgent, FixedQTargetAgent, DoubleQAgent, \
    DuelingBasicAgent, FixedTargetDuelingAgent
from reinforcement_learning.training import Trainer
from reinforcement_learning.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer

AGENTS = {'basic': BasicAgent, 'fixed_target': FixedQTargetAgent, 'double_q': DoubleQAgent,
          'dueling_basic': DuelingBasicAgent, 'fixed_target_dueling': FixedTargetDuelingAgent}


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
    env = UnityEnvironment(file_name='data/Banana_Linux/Banana.x86_64', no_graphics=True)

    # There is a version conflict between the Torch version that came with the Unity Environment
    # and Tensorboard. Resolving this may be complicated -> drop Tensorboard for the time being
    # writer = SummaryWriter(f'runs/{agent_type}')

    # --- Path for logging ---
    path = Path('runs/banana')
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
                      score_solved=13,
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
        'agent_type': FixedQTargetAgent,
        'buffer': {'type': ReplayBuffer, 'buffer_size': int(1e5), 'batch_size': 64},
        'epsilon': {'eps_start': 1.0, 'eps_end': 0.01, 'eps_decay': 0.995},
        'gamma': 0.99, 'tau': 1e-3,
        'update_every': 4,
        'num_agents': 1,
        'q_network': {
            'lr': 0.0005,
        },
    }
    main(agent_config=config, n_episodes=2000)

