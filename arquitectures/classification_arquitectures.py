from math import sqrt, ceil
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, accuracy_score, recall_score,
    f1_score, matthews_corrcoef
)

class Cnn1dA(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 1D, intercaladas con capas max pooling.
    Finaliza con flatten, dense y función softmax de salida.
    """
    def __init__(self, x_train, labels):
        super().__init__()
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 3,
            activation="relu", input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Cnn1dB(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 1D, intercaladas con capas max pooling.
    Incorpora una capa Dropout con rate 0.25.
    Finaliza con flatten, dense y función softmax de salida.
    """
    def __init__(self, x_train, labels):
        super().__init__()
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Cnn1dC(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.
    Incorpora una capa Dropout con rate 0.25.
    Finaliza con flatten, dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels):
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
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Cnn1dD(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.
    Incorpora una capa Dropout con rate 0.25.
    Finaliza con flatten, cuatro capas dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels):
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
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Cnn2dA(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 2D, intercaladas con capas max pooling.

    Finaliza con flatten, dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Cnn2dB(tf.keras.models.Sequential):
    """
    Utiliza Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size= 3, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Cnn2dC(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels):
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
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Cnn2dD(tf.keras.models.Sequential):
    """
    Utiliza Dobles Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.
    
    Finaliza con flatten, seis capas dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels):
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
        self.add(tf.keras.layers.Dense(units=len(labels), activation="softmax"))
        self.compile(optimizer="Adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy())
            
class Classifiers:
    """Classifiers using CNN Arquitectures"""
    def __init__(self, x_train, y_train, x_test, y_test, labels, mode = "1D"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.mode = mode
        if self.mode == "1D":
            self.cnn_a = Cnn1dA(self.x_train, labels)
            self.cnn_b = Cnn1dB(self.x_train, labels)
            self.cnn_c = Cnn1dC(self.x_train, labels)
            self.cnn_d = Cnn1dD(self.x_train, labels)
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
        Returns classification performance metrics.

        Accuracy, recall, precision, f1_score, mcc.
        """
        final_metrics = pd.DataFrame()
        for model_name, model in zip(
            ["CNN A", "CNN B", "CNN C", "CNN D"],
            [self.cnn_a, self.cnn_b, self.cnn_c, self.cnn_d]):
            y_train_predicted = np.argmax(model.predict(self.x_train), axis = 1)
            y_test_predicted = np.argmax(model.predict(self.x_test), axis = 1)
            train_metrics = [
                "training",
                accuracy_score(y_true = self.y_train, y_pred = y_train_predicted),
                recall_score(y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                precision_score(y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                f1_score(y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                matthews_corrcoef(y_true = self.y_train, y_pred = y_train_predicted)]
            test_metrics = [
                "testing",
                accuracy_score(y_true = self.y_test, y_pred = y_test_predicted),
                recall_score(y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                precision_score(y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                f1_score(y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                matthews_corrcoef(y_true = self.y_test, y_pred = y_test_predicted)]

            columns = ["data", "accuracy", "recall", "precision", "f1_score", "mcc"]
            metrics = pd.DataFrame(columns = columns)
            metrics.loc[0] = train_metrics
            metrics.loc[1] = test_metrics
            metrics["arquitecture"] = f"{model_name}-{self.mode}"
            final_metrics = final_metrics.append(metrics, ignore_index=True)
        return final_metrics