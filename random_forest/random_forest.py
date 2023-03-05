from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from geodesic_calculations import get_distance_and_bearing

def test():
    x_directory = "./datasets/erlangen_dataset_gridlen20.csv"
    y_directory = "./datasets/erlangen_dataset_gridlen20_label.csv"

    x_test_directory = "./datasets/erlangen_test_dataset_gridlen20.csv"
    y_test_directory = "./datasets/erlangen_test_dataset_gridlen20_label.csv"

    num_features = 12

    x_train_df = pd.read_csv(x_directory)
    y_train_df = pd.read_csv(y_directory)

    x_test_df = pd.read_csv(x_test_directory)
    y_test_df = pd.read_csv(y_test_directory)

    x_train = x_train_df.iloc[:, :num_features].to_numpy()
    y_train = y_train_df.iloc[:, 2].to_numpy()

    x_test = x_test_df.iloc[:, :num_features].to_numpy()
    y_test = y_test_df.iloc[:, 2].to_numpy()

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(predictions)
    print(y_test)
    print(y_test == predictions)
    print("Accuracy: ", np.mean(y_test == predictions))
    return predictions, y_test






