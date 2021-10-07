import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import datetime
import numpy as np
from typing import List, Union
import random
import math

# RGBs = [(random.random(), random.random(), random.random()) for _ in range(20)]
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
cmap = get_cmap(15)

def load_data(csv_path):
    if isinstance(csv_path, str):
        df = pd.read_csv(csv_path)
        cd = {}
        for col in df.columns:
            cd[col] = col.replace(' ', '')
        df = df.rename(columns=cd)
    else:
        df = load(csv_path)
    return df

def history(csv_path: Union[str, List[dict]], save_path: str,
            xlabel: str, ylabel: str, title: str) :
    """
    Hien thi toan bo du lieu theo tung truong.

    Parameters
    ----------
    csv_path : path to data, or list of dict.
    save_path : path to save image.
    plot_cols : column(s) to show.
    """
    # mpl.rcParams['figure.figsize'] = (18, 16)
    df = load_data(csv_path)
    date_time = pd.to_datetime(df.pop('DateTime'), format='%d.%m.%Y %H:%M:%S')
    for i, col in enumerate(df.columns):
        plot_features = df[col]
        plot_features.index = date_time
        _ = plot_features.plot(subplots=True, color=cmap(i))
        _finish(os.path.join(save_path, col[: col.find('(')] + '.pdf'), xlabel, ylabel, col)

def draw_predict(csv_path: Union[str, List[dict]], predict_path: Union[str, List[dict]], save_path: str,
            xlabel: str, ylabel: str, title: str):
    truth_df = load_data(csv_path)
    pred_df = load_data(predict_path)
    date_time = pd.to_datetime(truth_df.pop('DateTime'), format='%d.%m.%Y %H:%M:%S')
    pred_df.pop('DateTime')
    for i, col in enumerate(truth_df.columns):
        fig, ax1 = plt.subplots()
        ax1.plot(date_time, truth_df[col], label='ground_truth')
        ax1.plot(date_time, pred_df[col], label='predict')
        ax1.legend()
        fig.autofmt_xdate()
        _finish(os.path.join(save_path, col[: col.find('(')] + '.pdf'), xlabel, ylabel, col)

def percen_err(csv_path: Union[str, List[dict]], predict_path: Union[str, List[dict]], save_path: str,
            xlabel: str, ylabel: str, title: str):
    truth_df = load_data(csv_path)
    pred_df = load_data(predict_path)
    truth_df.pop('DateTime')
    pred_df.pop('DateTime')
    # dif = []
    # for col in truth_df.columns:
    #     dif.append(abs(truth_df[col].sum() - pred_df[col].sum()) / abs(truth_df[col].sum()) / len(truth_df) * 100)
    # plt.bar(truth_df.columns, dif)

    dif_df = abs(truth_df - pred_df)
    abstruth_df = abs(truth_df)
    meandif = []
    percdif = []
    stddif = []
    for col in truth_df.columns:
        meandif.append(round(dif_df[col].mean(), 2))
        percen_df = dif_df[col] / abstruth_df[col].mean() * 100
        # percdif.append(percen_df.mean())
        # stddif.append(percen_df.std())
        percdif.append(math.log(percen_df.mean()) )
        stddif.append(math.log(percen_df.std()))

    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'xx-large',
              'figure.figsize': (15, 10),
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'xx-large',
              'ytick.labelsize': 'xx-large'}
    pylab.rcParams.update(params)

    bars = plt.bar(truth_df.columns, percdif, yerr=stddif, align='center', alpha=0.5, ecolor='black', capsize=10)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .05, meandif[i], fontsize=15)
    plt.xticks(rotation=25)
    _finish(save_path + '/err2.pdf', xlabel, ylabel, title)

def predict(date_time: np.ndarray, ground_truth: np.ndarray, predict_values: np.ndarray,
            save_path: str, xlabel: str, ylabel: str, title: str):
    """
    Bieu do duong bieu dien gia tri du doan va gia tri thuc te.

    Parameters
    ----------
    date_time : date time.
    ground_truth : true values.
    predict_values : predicted values.
    save_path : path to save image.
    """
    # plt.clf()
    plt.plot(date_time, ground_truth, label='ground_truth')
    plt.plot(date_time, predict_values, label='predict')
    plt.legend()
    _finish(save_path, xlabel, ylabel, title)

def predict_dif(date_time: np.ndarray, ground_truth: np.ndarray, predict_values: np.ndarray,
                save_path: str, xlabel: str, ylabel: str, title: str):
    """
    Bieu do cot bieu dien khac biet giua gia tri du doan va gia tri thuc te.

    Parameters
    ----------
    date_time : date time.
    ground_truth : true values.
    predict_values : predicted values.
    save_path : path to save image.
    """
    # plt.clf()
    dif = abs(ground_truth - predict_values)
    plt.bar(date_time, dif, width=0.01)
    _finish(save_path, xlabel, ylabel, title)

def mean_yy(csv_path: str, save_path: str, plot_cols: List[str],
            xlabel: str, ylabel: str, title: str):
    """
    Trung binh gia tri cua (cac) truong theo tung nam.

    Parameters
    ----------
    csv_path : path to data.
    save_path : path so save image.
    plot_cols : column(s) to show.
    """
    df = pd.read_csv(csv_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    first_yy = 2009
    last_yy = 2016
    df_mean = pd.DataFrame(columns=df.columns)
    for yy in range(first_yy, last_yy + 1):
        df_mean = df_mean.append(df[(df['Date Time'] >= datetime.datetime(yy, 1, 1)) &
                                    (df['Date Time'] < datetime.datetime(yy + 1, 1, 1))].mean(), ignore_index=True)
    yeas = np.arange(first_yy, last_yy + 1)
    plot_features = df_mean[plot_cols]
    plot_features.index = yeas
    _ = plot_features.plot(subplots=True)
    _finish(save_path, xlabel, ylabel, title)

def meanNstd_yy(csv_path: Union[str, List[dict]], save_path: str, plot_cols: List[str],
            xlabel: str, ylabel: str, title: str):
    """
    Trung binh gia tri cua (cac) truong theo tung nam.

    Parameters
    ----------
    csv_path : path to data.
    save_path : path so save image.
    plot_cols : column(s) to show.
    """
    df = load_data(csv_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d.%m.%Y %H:%M:%S')
    # df = df[plot_cols]
    #
    first_yy = 2009
    last_yy = 2016
    df_mean = pd.DataFrame(columns=df.columns)
    # df_std = pd.DataFrame(columns=df.columns)
    df_max = pd.DataFrame(columns=df.columns)
    df_min = pd.DataFrame(columns=df.columns)
    for yy in range(first_yy, last_yy + 1):
        t = df[(df['DateTime'] >= datetime.datetime(yy, 1, 1)) &
                                    (df['DateTime'] < datetime.datetime(yy + 1, 1, 1))]
        df_mean = df_mean.append(t.mean(), ignore_index=True)
        # df_std = df_std.append(t.std(), ignore_index=True)
        df_max = df_max.append(t.max(), ignore_index=True)
        df_min = df_min.append(t.min(), ignore_index=True)
    years = np.arange(first_yy, last_yy + 1)
    df.pop('DateTime')
    for col in df.columns:
        # plt.errorbar(years, df_mean[col].to_numpy(), df_std[col].to_numpy())
        plt.plot(years, df_max[col].to_numpy(), label='Max')
        plt.plot(years, df_mean[col].to_numpy(), label='Average')
        plt.plot(years, df_min[col].to_numpy(), label='Min')
        plt.legend()
        _finish(os.path.join(save_path, col[: col.find('(')] + '.pdf'), xlabel, ylabel, col)





def correlation(csv_path: str, save_path: str,
            xlabel: str, ylabel: str, title: str):
    """
    Draw correlation matrix.

    Parameters
    ----------
    csv_path : path to data.
    save_path : path so save image.
    """
    df = pd.read_csv(csv_path)
    plt.imshow(df.corr())
    plt.colorbar()
    _finish(save_path, xlabel, ylabel, title)

def wind(csv_path: str, save_path: str, title: str):
    """
    Visualize "win".
    """
    df = load_data(csv_path)

    wv = df['wv(m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max.wv(m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame.

    # plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
    # plt.colorbar()
    # plt.xlabel('Wind Direction [deg]')
    # plt.ylabel('Wind Velocity [m/s]')

    wv = df.pop('wv(m/s)')
    max_wv = df.pop('max.wv(m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd(deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
    plt.colorbar()
    xlabel = 'Wind X [m/s]'
    ylabel = 'Wind Y [m/s]'
    ax = plt.gca()
    ax.axis('tight')
    _finish(save_path, xlabel, ylabel, title)

def _finish(save_path, xlabel, ylabel, title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)
    plt.clf()

def load(data: List[dict]):
    # preprocess data
    for i in range(len(data)):
        for k in data[i].keys():
            if k == 'DateTime':
                data[i][k] = pd.Timestamp(data[i][k], unit='s')
            else:
                data[i][k] = float(data[i][k])
    return pd.DataFrame(data)

