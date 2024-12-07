import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models, weighted_score
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    weights = {
        "mae": 0.2,
        "mse": 0.2,
        "rmse": 0.2,
        "r2_square": 0.4
    }


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K_Neighbors Regressor": KNeighborsRegressor(),
            }


            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['auto','sqrt','log2'],
                    # 'max_depth':[1,2,3,4,5,10,15,20,25],
                },
                "Random Forest":{
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    # 'bootstrap': [True, False],
                    # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    # 'min_samples_leaf': [1, 2, 4],
                    # 'min_samples_split': [2, 8, 15, 20],
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['auto', 'sqrt','log2', None],
                },
                "Gradient Boosting":{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001, .2],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'min_samples_split': [2, 8, 15, 20],
                    # 'criterion':['squared_error', 'friedman_mse', 'mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    # 'max_depth': [5, 8, 10, 15, None],
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256],
                    # 'max_depth': [5, 8, 12, 20, 30],
                    # 'colsample_bytree': [0.5, 0.8, 1, 0.3, 0.4]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                    # 'loss':['linear','square','exponential'],
                },
                "K_Neighbors Regressor": {
                'n_neighbors': [2, 3, 10, 20, 40, 50],
                'weights': ['uniform','distance']
                }              
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train,
                                                X_test = X_test, y_test = y_test,
                                                models = models, params = params)
            
            # Rank models by combined score
            scores = {
                name: weighted_score(model_info["test_metrics"], ModelTrainerConfig.weights) for name, model_info in model_report.items()
            }

            best_model_name = max(scores, key=scores.get)
            best_model_score = scores[best_model_name]

            best_model = models[best_model_name]
            
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"------------Results-----------")
            logging.info(f"Selected Model: {best_model_name}")
            logging.info(f"Best Params: {model_report[best_model_name]['params']}")
            logging.info(f"Test Metrics: {model_report[best_model_name]['test_metrics']}")
            logging.info(f"Score: {best_model_score}")
            logging.info(f"------------------------------")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square           
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))

