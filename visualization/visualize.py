import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import datetime
import numpy as np
from typing import List


def history(csv_path: str, save_path: str,
            xlabel: str, ylabel: str, title: str,
            plot_cols: List[str]) :
    """
    Hien thi toan bo du lieu theo tung truong.

    Parameters
    ----------
    csv_path : path to data.
    save_path : path to save image.
    plot_cols : column(s) to show.
    """
    mpl.rcParams['figure.figsize'] = (18, 16)
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)
    _finish(save_path, xlabel, ylabel, title)

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
    plt.clf()
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
    plt.clf()
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

def _finish(save_path, xlabel, ylabel, title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)

def test_history():
    csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    xlabel = 'Date time'
    ylabel = ''
    title = ''
    save_path = 'his2.png'
    plot_cols = ['T (degC)', 'p (mbar)']
    history(csv_path, save_path, xlabel, ylabel, title, plot_cols)

def test_predict():
    csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')[:200].to_numpy()
    y = df['T (degC)'][:200].to_numpy()
    yhat = df['T (degC)'][200:400].to_numpy()
    save_path = 'predict.png'
    xlabel = 'Date time'
    ylabel = 'T (degC)'
    title = 'Prediction'
    predict(date_time, y, yhat, save_path, xlabel, ylabel, title)

def test_predict_dif():
    csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')[:200].to_numpy()
    y = df['T (degC)'][:200].to_numpy()
    yhat = df['T (degC)'][200:400].to_numpy()
    save_path = 'predict-dif.png'
    xlabel = 'Date time'
    ylabel = 'Difference'
    title = 'Evaluate prediction'
    predict_dif(date_time, y, yhat, save_path, xlabel, ylabel, title)

def test_mean_yy():
    csv_path = '/home/thulx/Master/BI/jena_climate_2009_2016.csv'
    save_path = 'mean_yy2.png'
    plot_cols = ['T (degC)', 'p (mbar)']
    xlabel = 'Date time'
    ylabel = ''
    title = ''
    mean_yy(csv_path, save_path, plot_cols, xlabel, ylabel, title)



if __name__ == '__main__':
    # test_history()
    test_mean_yy()
    # test_predict_dif()
    test_predict()
