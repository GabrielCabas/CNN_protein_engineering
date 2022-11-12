"""All CNN arquitectures"""
from math import ceil, sqrt
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    precision_score, accuracy_score, recall_score,
    f1_score, matthews_corrcoef, mean_squared_error,
    mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
)
from scipy.stats import (kendalltau, pearsonr, spearmanr)
from keras.utils.layer_utils import count_params

class CnnA(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Convoluciones 1D, intercaladas con capas max pooling.

    Finaliza con flatten y dense.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2,
            activation="relu", input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))

        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))

        self.add(tf.keras.layers.Flatten())
        
        self.add(tf.keras.layers.Dense(units=64,
            activation="tanh"))
        
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class CnnB(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Convoluciones 1D, intercaladas con capas max pooling.

    Incorpora una capas Dropout con rate 0.25.

    Finaliza con flatten, y dense.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class CnnC(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten y dense.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
    
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class CnnD(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, y dense dividiendo las neuronas a la mitad por cada capa.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())

        unit = 64
        while unit > len(labels):
            self.add(tf.keras.layers.Dense(units=unit, activation="tanh"))
            unit = int(unit / 2)

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class CnnE(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Convoluciones 2D, intercaladas con capas max pooling.

    Finaliza con flatten y dense.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 2, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))

        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

        self.add(tf.keras.layers.Flatten())
        
        self.add(tf.keras.layers.Dense(units=64,
            activation="tanh"))
        
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class CnnF(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 2, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.2))

        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.1))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class CnnG(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Dobles Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 2, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.2))

        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.1))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=32, activation="tanh"))
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class CnnH(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Dobles Convoluciones 2D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, seis capas dense y función sigmoid de salida.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 2, activation="relu",
            input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        self.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.2))

        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 4,
            activation="relu"))
        self.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 4,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        self.add(tf.keras.layers.Dropout(0.1))

        self.add(tf.keras.layers.Flatten())

        unit = 64
        while unit > len(labels):
            self.add(tf.keras.layers.Dense(units=unit, activation="tanh"))
            unit = int(unit / 2)

        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())

class Models:
    """Organize CNN objects, train and validation process"""
    def __init__(self, x_train, y_train, x_test, y_test, labels, mode, arquitecture):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.labels = labels
        self.mode = mode
        self.arquitecture = arquitecture

        if self.arquitecture in ("E", "F", "G", "H"):
            self.x_train, self.x_test = self.__reshape()

        if self.arquitecture == "A":
            self.cnn = CnnA(x_train=self.x_train, labels = self.labels, mode=self.mode)
        elif self.arquitecture == "B":
            self.cnn = CnnB(x_train=self.x_train, labels = self.labels, mode=self.mode)
        elif self.arquitecture == "C":
            self.cnn = CnnC(x_train=self.x_train, labels = self.labels, mode=self.mode)
        elif self.arquitecture == "D":
            self.cnn = CnnD(x_train=self.x_train, labels = self.labels, mode=self.mode)
        elif self.arquitecture == "E" and self.x_train.shape[1] >= 20:
            self.cnn = CnnE(x_train=self.x_train, labels = self.labels, mode=self.mode)
        elif self.arquitecture == "F" and self.x_train.shape[1] >= 20:
            self.cnn = CnnF(x_train=self.x_train, labels = self.labels, mode=self.mode)
        elif self.arquitecture == "G" and self.x_train.shape[1] >= 20:
            self.cnn = CnnG(x_train=self.x_train, labels = self.labels, mode=self.mode)
        elif self.arquitecture == "H" and self.x_train.shape[1] >= 20:
            self.cnn = CnnH(x_train=self.x_train, labels = self.labels, mode=self.mode)
        else:
            print("Wrong arquitecture for this dataset")
            exit()

    def __reshape(self):
        dim = self.x_train.shape[1]
        sq_dim = sqrt(dim)
        square_side = ceil(sq_dim)
        resized_x_train = np.resize(self.x_train, (self.x_train.shape[0], square_side*square_side))
        resized_x_test = np.resize(self.x_test, (self.x_test.shape[0], square_side*square_side))
        squared_x_train = np.reshape(resized_x_train, (-1, square_side, square_side))
        squared_x_test = np.reshape(resized_x_test, (-1, square_side, square_side))
        return squared_x_train, squared_x_test

    def fit_models(self, epochs, verbose):
        """Fit model"""
        self.cnn.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)

    def save_model(self, folder, prefix = ""):
        """
        Save model in .h5 format, in 'folder' location
        """
        self.cnn.save(f"{folder}/{prefix}-{self.arquitecture}-{self.mode}.h5")

    def get_metrics(self):
        """
        Returns classification performance metrics.

        Accuracy, recall, precision, f1_score, mcc.
        """
        trainable_count = count_params(self.cnn.trainable_weights)
        non_trainable_count = count_params(self.cnn.non_trainable_weights)
        result = {}
        result["arquitecture"] = self.arquitecture
        result["trainable_params"] = trainable_count
        result["non_trainable_params"] = non_trainable_count
        if self.mode == "binary":
            y_train_predicted = np.round_(self.cnn.predict(self.x_train))
            y_test_score = self.cnn.predict(self.x_test)
            y_test_predicted = np.round_(y_test_score)
        if self.mode == "classification":
            y_train_predicted = np.argmax(self.cnn.predict(self.x_train), axis = 1)
            y_test_score = self.cnn.predict(self.x_test)
            y_test_predicted = np.argmax(y_test_score, axis = 1)
        if self.mode == "regression":
            y_train_predicted = self.cnn.predict(self.x_train)
            y_test_predicted = self.cnn.predict(self.x_test)
        if self.mode in ("binary", "classification"):
            result["labels"] = self.labels.tolist()
            train_metrics = {
                "accuracy": accuracy_score(y_true = self.y_train, y_pred = y_train_predicted),
                "recall": recall_score(
                    y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                "precision": precision_score(
                    y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                "f1_score": f1_score(
                    y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
                "mcc": matthews_corrcoef(y_true = self.y_train, y_pred = y_train_predicted),
                "confusion_matrix": confusion_matrix(
                    y_true = self.y_train, y_pred = y_train_predicted).tolist()
            }
            test_metrics = {
                "accuracy": accuracy_score(y_true = self.y_test, y_pred = y_test_predicted),
                "recall": recall_score(
                    y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                "precision": precision_score(
                    y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                "f1_score": f1_score(
                    y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
                "mcc": matthews_corrcoef(y_true = self.y_test, y_pred = y_test_predicted),
                "confusion_matrix": confusion_matrix(
                    y_true = self.y_test, y_pred = y_test_predicted).tolist()
            }
            if self.mode == "binary":
                test_metrics["roc_auc_score"] = roc_auc_score(
                    y_true = self.y_test, y_score = y_test_score, average="micro")
            else:
                test_metrics["roc_auc_score"] = roc_auc_score(
                    y_true = self.y_test, y_score = y_test_score, multi_class = 'ovr')
        else:
            train_metrics = {
                "mse": mean_squared_error(y_true = self.y_train, y_pred = y_train_predicted),
                "mae": mean_absolute_error(y_true = self.y_train, y_pred = y_train_predicted),
                "r2_score": r2_score(y_true = self.y_train, y_pred = y_train_predicted),
                "kendalltau": kendalltau(self.y_train, y_train_predicted),
                "pearsonr": pearsonr(self.y_train, y_train_predicted),
                "spearmanr": spearmanr(self.y_train, y_train_predicted)
            }
            test_metrics = {
                "mse": mean_squared_error(y_true = self.y_test, y_pred = y_test_predicted),
                "mae": mean_absolute_error(y_true = self.y_test, y_pred = y_test_predicted),
                "r2": r2_score(y_true = self.y_test, y_pred = y_test_predicted),
                "kendalltau": kendalltau(self.y_test, y_test_predicted),
                "pearsonr": pearsonr(self.y_test, y_test_predicted),
                "spearmanr": spearmanr(self.y_test, y_test_predicted)
            }
        result["train_metrics"] = train_metrics
        result["test_metrics"] = test_metrics
        return result
