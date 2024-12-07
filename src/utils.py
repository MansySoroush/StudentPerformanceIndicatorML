import os
import sys

import numpy as np 
import pandas as pd
import dill

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            search_cv = RandomizedSearchCV(estimator=model,
                                        param_distributions=param,
                                        n_iter=100,
                                        cv=3,
                                        verbose=2,
                                        random_state=42,
                                        n_jobs=-1)
            search_cv.fit(X_train, y_train)

            model.set_params(**search_cv.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = evaluate_model(y_train, y_train_pred)
            test_metrics = evaluate_model(y_test, y_test_pred)

            report[list(models.keys())[i]] = {
                "params": search_cv.best_params_,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(true, predicted):
    metrics = {
        "mae": normalize(mean_absolute_error(true, predicted), 0, 1),
        "mse": normalize(mean_squared_error(true, predicted), 0, 1),
        "rmse": normalize(np.sqrt(mean_squared_error(true, predicted)), 0, 1),
        "r2_square": r2_score(true, predicted),
    }
    return metrics

def weighted_score(metrics, weights):
    weighted_sum = (
        weights["mae"] * (1 / (1 + np.log(1 + metrics["mae"]))) +  # Invert MAE: lower is better
        weights["mse"] * (1 / (1 + np.log(metrics["mse"]))) +  # Invert MSE
        weights["rmse"] * (1 / np.log((1 + metrics["rmse"]))) +  # Invert RMSE
        weights["r2_square"] * metrics["r2_square"]  # Higher R² is better
    )
    return weighted_sum

def normalize(metric, min_val, max_val):
    return (metric - min_val) / (max_val - min_val)
