import streamlit as st
st.set_page_config(page_title='RL Reacher', page_icon=None, layout='wide')
import pandas as pd
import json
from pathlib import Path
import numpy as np
import plotly_express as px

p = Path('runs')

batch = pd.DataFrame()
epoch = pd.DataFrame()


for file in p.glob('*.json'):

    with open(file, 'r') as data:
        data = json.load(data)

    next_batch = pd.DataFrame(data['batch'])
    next_batch['Batch'] = np.arange(len(next_batch))
    next_batch['Type'] = file.stem

    next_epoch = pd.DataFrame(data['epoch'])
    next_epoch['Epoch'] = np.arange(len(next_epoch))
    next_epoch['Type'] = file.stem

    next_epoch['ScoreRolling'] = next_epoch['score'].rolling(100).mean()

    batch = pd.concat([batch, next_batch])
    epoch = pd.concat([epoch, next_epoch])

col1, col2 = st.beta_columns(2)

fig = px.line(epoch, x="Epoch", y="Mean Score", color='Type')
col1.write(fig)

fig = px.line(epoch, x="Epoch", y="score", color='Type')
col2.write(fig)
