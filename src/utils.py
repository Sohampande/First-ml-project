# the one place where we store all the common items for the project 

import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try : 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(xtrain, ytrain, xtest, ytest, models, parameters):
    ''' 
    This function is used to evaluate all train all the models and 
         return their r2 score. 

         Input : It requires training data(xtrain), training target(ytrain) and 
               : similarly the testing data(xtest) and target(ytest). It also 
               : requires all the models that are going to be tested.

         Ouput : It will return a dictionary with key as name of the model and 
               : item will be <training r2 score>
     '''
    try:
        report = []

        for key in models:
            model = models[key]
            param = parameters[key]

            grid_search = GridSearchCV(model, param, cv=3)
            grid_search.fit(xtrain, ytrain)

            model.set_params(**grid_search.best_params_)
            model.fit(xtrain, ytrain)

            ypredicted_test = model.predict(xtest)
            ypredicted_train = model.predict(xtrain)

            test_model_score = r2_score(ytest, ypredicted_test)

            report.append((key, test_model_score))

            return report

    except Exception as e:
        raise CustomException(e, sys)
