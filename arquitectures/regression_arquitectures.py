from math import sqrt, ceil
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

class Cnn1dA(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 1D, intercaladas con capas max pooling.
    Finaliza con flatten, dense y función linear de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 3,
            activation="relu", input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.MeanSquaredError())


class Cnn1dB(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 1D, intercaladas con capas max pooling.
    Incorpora una capa Dropout con rate 0.25.
    Finaliza con flatten, dense y función linear de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.MeanSquaredError())

class Cnn1dC(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.
    Incorpora una capa Dropout con rate 0.25.
    Finaliza con flatten, dense y función linear de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.MeanSquaredError())

class Cnn1dD(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.
    Incorpora una capa Dropout con rate 0.25.
    Finaliza con flatten, seis capas dense y función linear de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 3,
            activation="relu", input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=16, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=8, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=4, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=2, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.MeanSquaredError())

class Cnn2dA(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 2D, intercaladas con capas max pooling.

    Finaliza con flatten, dense y función linear de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.MeanSquaredError())

class Cnn2dB(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, dense y función linear de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.MeanSquaredError())

class Cnn2dC(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, dense y función lineal de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.MeanSquaredError())

class Cnn2dD(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.
    
    Finaliza con flatten, seis capas dense y función linear de salida.
    """
    def __init__(self, x_train):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=16, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=8, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=4, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=2, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=1))
        self.compile(optimizer="Adam",
              loss = tf.keras.losses.MeanSquaredError())

class Regressors:
    """Regression models using CNN Arquitectures"""
    def __init__(self, x_train, y_train, x_test, y_test, mode = "1D"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.mode = mode
        if self.mode == "1D":
            self.cnn_a = Cnn1dA(self.x_train)
            self.cnn_b = Cnn1dB(self.x_train)
            self.cnn_c = Cnn1dC(self.x_train)
            self.cnn_d = Cnn1dD(self.x_train)
        if self.mode == "2D":
            dim = self.x_train.shape[1]
            sq_dim = sqrt(dim)
            square_side = ceil(sq_dim)
            resized_x_train = np.resize(self.x_train, (self.x_train.shape[0], square_side*square_side))
            resized_x_test = np.resize(self.x_test, (self.x_test.shape[0], square_side*square_side))
            self.x_train = np.reshape(resized_x_train, (-1, square_side, square_side))
            self.x_test = np.reshape(resized_x_test, (-1, square_side, square_side))
            self.cnn_a = Cnn2dA(self.x_train)
            self.cnn_b = Cnn2dB(self.x_train)
            self.cnn_c = Cnn2dC(self.x_train)
            self.cnn_d = Cnn2dD(self.x_train)

    def fit_models(self, epochs, verbose):
        """Fit all models"""
        self.cnn_a.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)
        self.cnn_b.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)
        self.cnn_c.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)
        self.cnn_d.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)

    def get_performance_metrics(self):
        """
        Returns regression performance metrics.

        Mse, Mae and R2.
        """
        final_metrics = pd.DataFrame()
        for model_name, model in zip(
            ["CNN A", "CNN B", "CNN C", "CNN D"],
            [self.cnn_a, self.cnn_b, self.cnn_c, self.cnn_d]):
            y_train_predicted = model.predict(self.x_train)
            y_test_predicted = model.predict(self.x_test)
            train_metrics = [
                "training",
                mean_squared_error(y_true = self.y_train, y_pred = y_train_predicted),
                mean_absolute_error(y_true = self.y_train, y_pred = y_train_predicted),
                r2_score(y_true = self.y_train, y_pred = y_train_predicted)]
            test_metrics = [
                "testing",
                mean_squared_error(y_true = self.y_test, y_pred = y_test_predicted),
                mean_absolute_error(y_true = self.y_test, y_pred = y_test_predicted),
                r2_score(y_true = self.y_test, y_pred = y_test_predicted)
                ]
            columns = ["data", "mse", "mae", "r2"]
            metrics = pd.DataFrame(columns = columns)
            metrics.loc[0] = train_metrics
            metrics.loc[1] = test_metrics
            metrics["arquitecture"] = f"{model_name}-{self.mode}"
            final_metrics = final_metrics.append(metrics, ignore_index=True)
        return final_metrics

    def save_models(self, folder, prefix = ""):
        """
        Save models in .h5 format, in 'folder' location
        """
        self.cnn_a.save(f"{folder}/{prefix}CNN_A-{self.mode}.h5")
        self.cnn_b.save(f"{folder}/{prefix}CNN_B-{self.mode}.h5")
        self.cnn_c.save(f"{folder}/{prefix}CNN_C-{self.mode}.h5")
        self.cnn_d.save(f"{folder}/{prefix}CNN_D-{self.mode}.h5")