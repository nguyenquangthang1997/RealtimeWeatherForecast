import os
import datetime

# import IPython
# import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf




def visualize_TPR(df, date_time, num_visual=480):
    print("Visualizing TPR...")
    plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    plt.savefig("figures/TPR_all.png")
    plt.close()

    plot_features = df[plot_cols][:num_visual]
    plot_features.index = date_time[:num_visual]
    _ = plot_features.plot(subplots=True)

    plt.savefig("figures/TPR_{}days.png".format(num_visual))
    plt.close()


def visualize_date_in_cyclic_form(df):
    plt.plot(np.array(df['Day sin'])[:25])
    plt.plot(np.array(df['Day cos'])[:25])
    plt.xlabel('Time [h]')
    plt.title('Time of day signal')
    plt.savefig("Cyclic_date_info.png")
    plt.close()


def visualize_wind_data(df):
    print("Visualizing wind information...")
    plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.xlabel('Wind Direction [deg]')
    plt.ylabel('Wind Velocity [m/s]')
    plt.savefig("Wind_data.png")
    plt.close()

    plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
    plt.colorbar()
    plt.xlabel('Wind X [m/s]')
    plt.ylabel('Wind Y [m/s]')
    ax = plt.gca()
    ax.axis('tight')
    plt.savefig("Wind_vector.png")
    plt.close()


def visualize_column_distribution(df, train_mean, train_std):
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.savefig("figures/column_distribution.png")