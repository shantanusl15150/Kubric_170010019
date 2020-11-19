import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area):
    df = pd.read_csv('TRAIN_DATA_URL')
    df = df.T
    a = list(df[0])
    price = a[1:]
    price
    b = df.index
    area = list(b[1:])
    X_train = np.array(area).reshape(-1, 1)
    Y_train = np.array(price).reshape(-1, 1)
    reg = LinearRegression.fit(X_train, Y_train)
    res = reg.predict(area)
    return res
    


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys())).reshape(-1, 1)
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
