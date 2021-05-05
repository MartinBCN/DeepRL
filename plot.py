import streamlit as st
st.set_page_config(page_title='RL Reacher', page_icon=None, layout='wide')
import pandas as pd
import json
from pathlib import Path
import numpy as np
import plotly_express as px

p = Path('runs/reacher')

batch = pd.DataFrame()
epoch = pd.DataFrame()

configs = pd.DataFrame()

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

        batch = pd.concat([batch, next_batch])
        epoch = pd.concat([epoch, next_epoch])
    except:
        pass

col1, col2 = st.beta_columns(2)

fig = px.line(epoch, x="Epoch", y="Mean Score", color='Run')
fig.write_html('figures/mean_score.html')
col1.write(fig)

fig = px.line(epoch, x="Epoch", y="score", color='Run')
col2.write(fig)
