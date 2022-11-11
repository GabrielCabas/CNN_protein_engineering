"""
python3 main.py ../Encoded-Datasets graph_dG.csv pca 20
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from arquitectures.binary_arquitectures import BinaryClassifiers
from arquitectures.classification_arquitectures import Classifiers
from arquitectures.regression_arquitectures import Regressors

#Constants
MODELS_FOLDER = "models"
DATASETS_MAIN_FOLDER = sys.argv[1]
DATASET = sys.argv[2]
PREPROCESSING = sys.argv[3]
EPOCHS = int(sys.argv[4])

#AUX FUNCTIONS
def create_folder(folder):
    """Create folder"""
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

def df_to_data(path):
    """Takes a path, reads the csv file and return data splitted"""
    data = pd.read_csv(path)
    if df_type == "Protein":
        extra_columns = ["id", "target"]
    elif df_type == "Protein-Ligand":
        extra_columns = ["id_seq", "id_smiles", "target"]
    elif df_type == "Protein-Protein":
        extra_columns = ["id_0", "id_1", "target"]
    x_values = data.drop(extra_columns, axis = 1).values
    labels_list = data.target.unique()
    replace_dict = {label: index for index, label in enumerate(labels_list)}
    data.target.replace(replace_dict, inplace=True)
    y_values = data.target.to_numpy()
    return train_test_split(x_values, y_values), labels_list

#MAIN
final_results = pd.DataFrame(
    columns = ["df_type", "preprocessing", "df_name", "arquitecture", "epochs", "data"]
)

datasets_list = pd.read_csv("datasets_list.csv")
task = datasets_list[datasets_list["dataset"] == DATASET]["task"].values[0]
df_type = datasets_list[datasets_list["dataset"] == DATASET]["folder"].values[0]

create_folder(MODELS_FOLDER)

dfs = [a for a in os.listdir(f'{DATASETS_MAIN_FOLDER}/{df_type}/{PREPROCESSING}/')
    if DATASET in a]

for df_name in dfs[0:5]:
    (X_train, X_test, y_train, y_test), labels = df_to_data(
        f'{DATASETS_MAIN_FOLDER}/{df_type}/{PREPROCESSING}/{df_name}'
    )
    df_name = df_name.replace(".csv", "")
    if task == "binary":
        model_1d = BinaryClassifiers(X_train, y_train, X_test, y_test)
        model_2d = BinaryClassifiers(X_train, y_train, X_test, y_test, mode = "2D")

    elif task == "classification":
        model_1d = Classifiers(X_train, y_train, X_test, y_test, labels)
        model_2d = Classifiers(X_train, y_train, X_test, y_test, labels, mode="2D")

    elif task == "regression":
        model_1d = Regressors(X_train, y_train, X_test, y_test)
        model_2d = Regressors(X_train, y_train, X_test, y_test, mode="2D")

    for clf in [model_1d, model_2d]:
        clf.fit_models(EPOCHS, verbose = 1)
        clf.save_models(MODELS_FOLDER, f"{df_type}_{PREPROCESSING}_{df_name}_{EPOCHS}")
        metrics = clf.get_performance_metrics()
        metrics["df_type"] = df_type
        metrics["preprocessing"] = PREPROCESSING
        metrics["df_name"] = df_name
        metrics["epochs"] = EPOCHS
        final_results = final_results.append(metrics, ignore_index=True)
final_results.to_csv(f"{DATASET}_{PREPROCESSING}_performance_results.csv", index=False)
