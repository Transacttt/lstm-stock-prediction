import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae
