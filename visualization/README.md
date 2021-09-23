# Visualization for weather forecast

## Requirements:
- python 3.6
- python modules: 
in the visualization directory, run:
```python
pip install -r requirements.txt
pip install -e .
```

## Usage
```python
from visualization.visualize import history
csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
xlabel = 'Date time'
ylabel = ''
title = ''
save_path = 'his2.png'
plot_cols = ['T (degC)', 'p (mbar)']
history(csv_path, save_path, xlabel, ylabel, title, plot_cols)
```
See _visualize.py_ for detail.