# This file is primarily focused on model training within the larger application.

import os 
import sys
from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostClassifier, 
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer :
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('splitting trainging and test input data'.title())

            xtrain, ytrain, xtest, ytest = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'K-Neighbors' : KNeighborsRegressor(),
                'CatBoosting' : CatBoostRegressor(verbose=False),
                'AdaBoosting' : AdaBoostClassifier()
            }

            params = {
                'Random Forest' : {
                    'n_estimators' : [8, 16, 32, 64, 128, 256],
                    'max_features' : ['sqrt', 'log2', None]
                },
                'Decision Tree' : {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Gradient Boosting' : {
                    'learning_rate' : [.1, .01, .05, .001],
                    'subsample' : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                },
                'K-Neighbors' : {
                    'n_neighbors':[5,7,9,11]
                },
                'CatBoosting' : {
                    'depth' : [6, 8, 10],
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'iterations' : [30, 50, 100]
                },
                'AdaBoosting' : {
                    'learning_rate':[0.1, 0.01, 0.5, 0.001],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                }
            }

            model_report : list = evaluate_model(xtrain=xtrain, 
                                                 ytrain=ytrain, 
                                                 xtest=xtest, 
                                                 ytest=ytest,
                                                 models=models,
                                                 parameters=params)
            
            model_report = pd.DataFrame(model_report, columns=['model', 'r2_score'])

            model_report = model_report.sort_values(by='r2_score')

            best_model_name = model_report.iloc[0]['model']
            best_model = models[best_model_name]

            if model_report.iloc[0]['r2_score'] < 0.6 :
                raise CustomException('No Best Model Found')
            
            logging.info('Best Model found on both training and testing dataset'.title)

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            

            predicted = best_model.predict(xtest)

            r2score = r2_score(ytest, predicted)
            return r2score
        

        except Exception as e:
            raise CustomException(e, sys)