"""
python3 main.py ../Encoded-Datasets AMP_binary.csv pca 20
"""
import time
import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from arquitectures import Models


#Constants
MODELS_FOLDER = "models"
DATASET_PATH = os.path.realpath(sys.argv[1])
EPOCHS = int(sys.argv[2])
ARQUITECTURE = sys.argv[3]
OUTPUT_PATH = sys.argv[4]

DF_NAME = os.path.basename(DATASET_PATH)
MAIN_DF_NAME = DF_NAME.split("-")[1]
ENCODING = DF_NAME.split("-")[0]
PREPROCESSING = DATASET_PATH.split("/")[-2]
DATA_TYPE = DATASET_PATH.split("/")[-3]


#AUX FUNCTIONS
def create_folder(folder):
    """Create folder"""
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

def df_to_data(path):
    """Takes a path, reads the csv file and return data splitted"""
    datasets_list = pd.read_csv("datasets_list.csv")
    task_name = datasets_list[datasets_list["dataset"] == MAIN_DF_NAME]["task"].values[0]
    data = pd.read_csv(path)
    if DATA_TYPE == "Protein":
        extra_columns = ["id", "target"]
    elif DATA_TYPE == "Protein-Ligand":
        extra_columns = ["id_seq", "id_smiles", "target"]
    elif DATA_TYPE == "Protein-Protein":
        extra_columns = ["id_0", "id_1", "target"]
    if task_name in ("classification", "binary"):
        data = balance_data(data)
    x_values = data.drop(extra_columns, axis = 1).values
    labels_list = data.target.unique()
    replace_dict = {label: index for index, label in enumerate(labels_list)}
    data.target.replace(replace_dict, inplace=True)
    y_values = data.target.to_numpy()
    return train_test_split(x_values, y_values), labels_list, task_name

def balance_data(data):
    """Sample the dataframe with n elements by label"""
    count_labels = data.target.value_counts()
    min_rows = count_labels.min()
    new_df = []
    for label in count_labels.index:
        new_df.append(data[data.target == label].sample(n = min_rows))
    return pd.concat(new_df)

#MAIN
create_folder(MODELS_FOLDER)

time_inicio = time.time()
(x_train, x_test, y_train, y_test), labels, task = df_to_data(DATASET_PATH)
df_name = DATA_TYPE.replace(".csv", "")
models = Models(x_train, y_train, x_test, y_test, labels, task, ARQUITECTURE)
models.fit_models(EPOCHS, 1)
metrics = models.get_metrics()
time_fin = time.time()
delta_time = round(time_fin - time_inicio, 4)
metrics["total_time"] = delta_time
metrics["dataset"] = MAIN_DF_NAME
metrics["encoding"] = ENCODING
metrics["preprocessing"] = PREPROCESSING
metrics["data_type"] = DATA_TYPE
metrics["epochs"] = EPOCHS
with open(OUTPUT_PATH, mode = "w", encoding = "utf-8") as file:
    json.dump(metrics, file)
print(pd.json_normalize(metrics))