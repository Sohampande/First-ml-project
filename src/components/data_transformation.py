import os
import sys

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

'''
 The main idea here is to prepare the data for modeling. This involoves : 
       - Feature Selection 
       - Data Cleaning
       - Making New Databases 
       - etc
 '''
@dataclass
class DataTransformationConfig:
    '''
    This class makes the file object and stores it at the location 
                  artifact/preprocessor.pkl
    '''
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # main idea here is to create all the pickle files, like converting numerical into categorical, etc.
    def get_data_transformer_object(self):
        '''
            This method builds and returns a columntransformer that knows how to preprocess both numeric
            and categorical columns.
        '''
        try:
            numcols = ['writing_score', 'reading_score']
            catcols = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            numpipe = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            catpipe = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical Columns : {catcols}')
            logging.info(f'Numerical Columns{numcols}')

            preprocessor = ColumnTransformer(
                [
                    ('numpipeline', numpipe, numcols),
                    ('catpipeline', catpipe, catcols)
                ]
            )

            logging.info('Numerical Columns Encoding are preprocessed'.title())
            logging.info('Categorical Columns Encoding Completed'.title())

            return preprocessor


        except Exception as e :
            raise CustomException(sys, e)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # These saves the preprocessing object in the desired location. Note we store the preprocessing 
            # object not the preprocessed data. This is important because our pipeline function, which is 
            # used to fit models, requires preprocesed object.
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
