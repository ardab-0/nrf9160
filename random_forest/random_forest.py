from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from geodesic_calculations import get_distance_and_bearing


class RandomForest:
    def __init__(self, x_directory, y_directory, x_test_directory, y_test_directory, num_features = 12):
        self.x_directory = x_directory
        self.y_directory = y_directory
        self.x_test_directory = x_test_directory
        self.y_test_directory = y_test_directory
        self.num_features = num_features



    def fit(self):
        x_train_df = pd.read_csv(self.x_directory)
        y_train_df = pd.read_csv(self.y_directory)

        x_train = x_train_df.iloc[:, :self.num_features].to_numpy()
        y_train = y_train_df.iloc[:, 2].to_numpy()

        self.clf = RandomForestClassifier()
        self.clf.fit(x_train, y_train)


    def test(self):

        x_test_df = pd.read_csv(self.x_test_directory)
        y_test_df = pd.read_csv(self.y_test_directory)

        x_test = x_test_df.iloc[:, :self.num_features].to_numpy()
        y_test = y_test_df.iloc[:, 2].to_numpy()


        predictions = self.clf.predict(x_test)
        print(predictions)
        print(y_test)
        print(y_test == predictions)
        print("Accuracy: ", np.mean(y_test == predictions))
        return predictions, y_test






