import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_log_error as RMSLE
from sklearn.metrics import accuracy_score
import xgboost as xgb

# test solving issue
class ML:
    def __init__(self):
        super().__init__()
        self.readData()

    def readData(self):
        dataset = pd.read_csv('train.csv')
        dataset.dropna(subset=['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea',
                               'GarageYrBlt', 'GarageArea'], inplace=True)
        x = dataset[['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea',
                     'GarageYrBlt', 'GarageArea']]
        y = dataset[['SalePrice']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1 / 3, random_state=7)
        self.linearRegression(x_train, x_test, y_train, y_test)
        self.gradientBoost(x_train, x_test, y_train, y_test)

    def linearRegression(self, x_train, x_test, y_train, y_test):
        reg = LassoCV()
        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        res = RMSLE(y_test, y_pred)
        print(res)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

    def gradientBoost(self, x_train, x_test, y_train, y_test):
        model = xgb.XGBClassifier()
        model.fit(x_train, y_train.values.ravel())
        y_pred = model.predict(x_test)
        res = RMSLE(y_test, y_pred)
        print(res)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

# def testF():
#     dataset = pd.read_csv('train.csv')
#     # split data into X and y
#     X = dataset[['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea',
#                      'GarageYrBlt', 'GarageArea']]
#     Y = dataset[['SalePrice']]
#     # split data into train and test sets
#     seed = 7
#     test_size = 0.33
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#     # fit model no training data
#     model = xgb.XGBClassifier()
#     model.fit(X_train, y_train)
#     # make predictions for test data
#     y_pred = model.predict(X_test)
#     predictions = [round(value) for value in y_pred]
#     # evaluate predictions
#     accuracy = accuracy_score(y_test, predictions)

def main():
    work = ML()
    # testF()

if __name__ == "__main__":
    main()