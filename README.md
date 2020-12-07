## Marketing Analytics


```python color: "#3572A5"
import dash
from dash.dependencies import Input, Output, State, MATCH 
import dash_table 
import dash_core_components as dcc 
import dash_html_components as html 
import plotly.express as px 
import pandas as pd 
import numpy as np

nsample = [10,25,100]
nsimulation = 1000
mean_list = []
std_list = []
for nsize in nsample:
    sample_mean = []
    for times in range(1,nsimulation + 1):
        sample = np.random.normal(loc = 0, scale = 1, size = nsize)
        mean = np.mean(sample)
        sample_mean.append(mean)
    trial_mean = np.mean(sample_mean)
    trial_std = np.std(sample_mean)
    mean_list.append(trial_mean)
    std_list.append(trial_std)
print('Means of the vectors:',mean_list)
print('Standard Deviations  of the vectors:',std_list)
```
