import os
import datetime

# import IPython
# import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from Visual_data import visualize_date_in_cyclic_form, visualize_TPR, visualize_wind_data
from copy import deepcopy

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df=None, val_df=None, test_df=None,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # import pdb; pdb.set_trace()

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
  
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)
        return ds


    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3, name=""):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.savefig("figures/output_{}.png".format(name))
        plt.close()

    

def preprocess_wind(df):
    """
    FIRST
    In the data, we have some noisy for wind velocity; like -9999 values
    We will replace it by zeros.
    """
    print("Removing noisy data")
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    print("Converting wind direction and velocity into wind vector")

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)')*np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv*np.cos(wd_rad)
    df['max Wy'] = max_wv*np.sin(wd_rad)

    # wv = df['wv (m/s)']
    # max_wv = df['max. wv (m/s)']

    # # Convert to radians.
    # wd_rad = df['wd (deg)']*np.pi / 180

    # # Calculate the wind x and y components.
    # df['Wx'] = wv*np.cos(wd_rad)
    # df['Wy'] = wv*np.sin(wd_rad)

    # # Calculate the max wind x and y components.
    # df['max Wx'] = max_wv*np.cos(wd_rad)
    # df['max Wy'] = max_wv*np.sin(wd_rad)

    return df


def preprocess_time(date_time, df):
    """
    Time is a very useful info, but not in string form.
    """
    # Convert it to seconds 
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    # Time is a cyclic info, so we use cosine to model it
    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df 


def split_data(df, df_ori):
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    test_df_ori = df_ori[int(n*0.9) + 47:]

    num_features = df.shape[1]

    # normalize data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df, num_features, train_mean, train_std, test_df_ori



def compile_and_fit(model, window, patience=2):
    MAX_EPOCHS = 20

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


# class FeedBack(tf.keras.Model):
#     def __init__(self, units, out_steps):
#         super().__init__()
#         self.out_steps = out_steps
#         self.units = units
#         self.lstm_cell = tf.keras.layers.LSTMCell(units)
#         # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
#         self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
#         self.dense = tf.keras.layers.Dense(num_features)

#     def warmup(self, inputs):
#         # inputs.shape => (batch, time, features)
#         # x.shape => (batch, lstm_units)
#         # import pdb; pdb.set_trace()
#         print(inputs.shape)
#         x, *state = self.lstm_rnn(inputs)

#         # predictions.shape => (batch, features)
#         prediction = self.dense(x)
#         return prediction, state

#     def call(self, inputs, training=None):
#         # Use a TensorArray to capture dynamically unrolled outputs.
#         predictions = []
#         # print(inputs.shape)
#         # Initialize the LSTM state.
#         prediction, state = self.warmup(inputs)

#         # Insert the first prediction.
#         predictions.append(prediction)

#         # Run the rest of the prediction steps.
#         for n in range(1, self.out_steps):
#             # Use the last prediction as input.
#             x = prediction
#             # Execute one lstm step.
#             x, state = self.lstm_cell(x, states=state,
#                                     training=training)
#             # Convert the lstm output to a prediction.
#             prediction = self.dense(x)
#             # Add the prediction to the output.
#             predictions.append(prediction)

#         # predictions.shape => (time, batch, features)
#         predictions = tf.stack(predictions)
#         # predictions.shape => (batch, time, features)
#         predictions = tf.transpose(predictions, [1, 0, 2])
#         return predictions


# def 
# def create_model():
#     multi_lstm_model = tf.keras.Sequential([
#         # Shape [batch, time, features] => [batch, lstm_units].
#         # Adding more `lstm_units` just overfits more quickly.
#         tf.keras.layers.LSTM(32, return_sequences=False),
#         # Shape => [batch, out_steps*features].
#         tf.keras.layers.Dense(OUT_STEPS*num_features,
#                             kernel_initializer=tf.initializers.zeros()),
#         # Shape => [batch, out_steps, features].
#         tf.keras.layers.Reshape([OUT_STEPS, num_features])
#     ])

#     multi_lstm_model.compile(loss=tf.losses.MeanSquaredError(),
#                     optimizer=tf.optimizers.Adam(),
#                     metrics=[tf.metrics.MeanAbsoluteError()])

#     return multi_lstm_model
    


def create_df_from_list_of_dict(input):
    df = pd.DataFrame(input)
    return df 
    # keys = input.keys()


def one_time_call():
    csv_path = 'jena_climate_2009_2016.csv'
    df = pd.read_csv(csv_path)
    df = df[5::6] # subsampling hours

    ori_df = deepcopy(df)

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    print("Preprocess wind velocity")
    df = preprocess_wind(df)

    print("Preprocess time: To cosine")
    df = preprocess_time(date_time, df)


    train_df, val_df, test_df, num_features, train_mean, train_std, test_df_ori = split_data(df, ori_df)

    OUT_STEPS = 24
    # num_features = df.shape[1] # check current functions
    multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)

    multi_lstm_model = tf.keras.models.load_model('saved_model/lstm')

    return ori_df, df, multi_window, multi_lstm_model, train_mean, train_std, test_df_ori


def make_prediction(current_47_hours_before=[{'Date Time': '04.11.2015 12:00:00'}], ori_df=None, df=None, multi_window=None, multi_lstm_model=None, train_mean=None, train_std=None, test_df_ori=None):
    """
    current_47_days_before: list of dict
    """
    if current_47_hours_before is not None: 
        start_date = current_47_hours_before[0]['Date Time']
        idx = ori_df.index[ori_df['Date Time'] == start_date][0]
        real_index = (ori_df.index == idx).nonzero()[0][0]

    if current_47_hours_before is not None:
        to_test_df = df.iloc[real_index: real_index + 48]
    else:
        to_test_df = df.iloc[-49:]

    # multi_window.test_df = to_test_df

    output =  multi_lstm_model.predict(multi_window.test)
    output = output * train_std.to_numpy() + train_mean.to_numpy()
    # import pdb; pdb.set_trace()
    keys = df.keys().tolist()
    list_out = list()

    for i in range(len(output)):
        dict_out = dict() 
        dict_out['Date Time'] = test_df_ori.iloc[i]['Date Time']

        for j in range(len(keys)):
            this_key = keys[j]
            dict_out[this_key] = output[i][0][j]
            if j == len(keys) - 5:
                break 
        list_out.append(dict_out)

    # import pdb; pdb.set_trace()
    
    out = pd.DataFrame(list_out)
    out.to_csv('predict.csv', index=False)    

    # for j in range(24):
    #     dict_out = dict()
    #     for i in range(len(keys)):
    #         this_key = keys[i]
    #         dict_out[this_key] = output[j][i]
    #         if i == len(keys) - 5:
    #             break
    #     list_out.append(dict_out)

    return list_out


# def preprocess_new_df(new_df, old_df):
#     date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
#     df = preprocess_wind(df)
#     print("Preprocess time: To cosine")
#     df = preprocess_time(date_time, df)




if __name__ == "__main__":

    # csv_path = 'jena_climate_2009_2016.csv'
    # df = pd.read_csv(csv_path)
    # df = df[5::6] # subsampling hours
    # df.to_csv('preprocessed_data.csv', index=False)

    ori_df, df, multi_window, multi_lstm_model, train_mean, train_std, test_df_ori = one_time_call()

    # import pdb; pdb.set_trace()

    for i in range(1):
        out = make_prediction(ori_df=ori_df, df=df, multi_window=multi_window, multi_lstm_model=multi_lstm_model, train_mean=train_mean, train_std=train_std, test_df_ori=test_df_ori)
        # print(out)


    # mpl.rcParams['figure.figsize'] = (8, 6)
    # mpl.rcParams['axes.grid'] = False

    # csv_path = 'jena_climate_2009_2016.csv'
    # df = pd.read_csv(csv_path)
    # df = df[5::6]
    # date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # print("Preprocess wind velocity")
    # df = preprocess_wind(df)

    # print("Preprocess time: To cosine")
    # df = preprocess_time(date_time, df)

    # train_df, val_df, test_df, num_features = split_data(df)

    # linear = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=1)
    #     ])

    # MAX_EPOCHS = 20

    # single_step_window = WindowGenerator(
    #         input_width=1, label_width=1, shift=1,
    #         label_columns=['T (degC)'], train_df=train_df, val_df=val_df, test_df=test_df)
    

    
    # history = compile_and_fit(linear, single_step_window)
    # val_performance = dict()
    # performance = dict()

    # val_performance['Linear'] = linear.evaluate(single_step_window.val)
    # performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

    # OUT_STEPS = 24
    # num_features = df.shape[1] # check current functions
    # multi_window = WindowGenerator(input_width=24,
    #                            label_width=OUT_STEPS,
    #                            shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)


    # class MultiStepLastBaseline(tf.keras.Model):
    #     def call(self, inputs):
    #         return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

    # last_baseline = MultiStepLastBaseline()
    # last_baseline.compile(loss=tf.losses.MeanSquaredError(),
    #                     metrics=[tf.metrics.MeanAbsoluteError()])

    # multi_val_performance = {}
    # multi_performance = {}

    # multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
    # multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)


    # class RepeatBaseline(tf.keras.Model):
    #     def call(self, inputs):
    #         return inputs

    # repeat_baseline = RepeatBaseline()
    # repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
    #                         metrics=[tf.metrics.MeanAbsoluteError()])

    # multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
    # multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
    # multi_window.plot(repeat_baseline, name='repeat_baseline')



    # multi_linear_model = tf.keras.Sequential([
    #     # Take the last time-step.
    #     # Shape [batch, time, features] => [batch, 1, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    #     # Shape => [batch, 1, out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS*num_features,
    #                         kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])

    # history = compile_and_fit(multi_linear_model, multi_window)
    # multi_window.plot(multi_linear_model, name='multi_linear_model')


    # multi_lstm_model = tf.keras.models.load_model('saved_model/lstm')

    # multi_lstm_model = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, lstm_units].
    #     # Adding more `lstm_units` just overfits more quickly.
    #     tf.keras.layers.LSTM(32, return_sequences=False),
    #     # Shape => [batch, out_steps*features].
    #     tf.keras.layers.Dense(OUT_STEPS*num_features,
    #                         kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features].
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])

    # history = compile_and_fit(multi_lstm_model, multi_window)
    # multi_window.plot(multi_lstm_model, name='multi_lstm_model')

    # model = create_model()



    # IPython.display.clear_output()

    # multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    # multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    # import pdb; pdb.set_trace()
    # multi_window.plot(multi_lstm_model, name='multi_lstm_model')










    # feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)


    # import pdb; pdb.set_trace()


    # history = compile_and_fit(feedback_model, multi_window)

    # prediction, state = feedback_model.warmup(k.example[0])
    # history = compile_and_fit(feedback_model, multi_window)


    # # multi_val_performance = {}
    # # multi_performance = {}

    # # multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
    # # multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
    # # multi_window.plot(feedback_model)
    # # multi_window.plot(multi_lstm_model)


