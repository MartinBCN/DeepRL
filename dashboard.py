import pandas as pd
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


p = Path('runs/reacher')

batch = pd.DataFrame()
epoch = pd.DataFrame()

configs = pd.DataFrame()

score = pd.DataFrame()
loss = pd.DataFrame()

for path in p.glob('run*'):
    run = path.name

    # --- Configs ---
    with open(path / 'config.json', 'r') as data:
        config = json.load(data)
    next_configs = {'Run': [run], 'Hidden Layer': [config['actor']['hidden_layers']],
                    'Layer Norm': [config['actor']['batch_norm']], 'Dropout': [config['actor']['dropout']]}
    configs = pd.concat([configs, pd.DataFrame(next_configs)])

    # --- Logs ---
    try:
        with open(path / 'logs.json', 'r') as data:
            logs = json.load(data)
        next_batch = pd.DataFrame(logs['batch'])
        next_batch['Batch'] = np.arange(len(next_batch))

        next_epoch = pd.DataFrame(logs['epoch'])
        next_epoch['Epoch'] = np.arange(len(next_epoch))

        next_epoch['ScoreRolling'] = next_epoch['score'].rolling(100).mean()

        next_epoch['Run'] = run

        score[run] = next_epoch['Mean Score']
        loss[run] = next_epoch['score'].rolling(100).mean()

        batch = pd.concat([batch, next_batch])
        epoch = pd.concat([epoch, next_epoch])
    except:
        pass


print(configs)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for col in score.columns:
    axes[0].plot(score[col], label=col)
    axes[0].legend()
for col in loss.columns:
    axes[1].plot(loss[col], label=col)
    axes[1].legend()

plt.savefig('figures/analysis.png')
