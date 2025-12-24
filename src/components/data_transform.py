import sys
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.data_injestion import DataInjestion



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import pandas as pd 
import numpy as np 

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
     preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tansformer_object(self):
        try:
            numerical_cols = ["writing_score","reading_score"]
            categorical_cols = ["lunch","gender","race_ethnicity","parental_level_of_education","test_preparation_course"]
            num_pipeline = Pipeline(
                steps = [
                    ("scaler", StandardScaler(strategy = "median")),
                    ("imputer",SimpleImputer())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [("imputer",SimpleImputer(strategy= "most_frequent")),
                         "one_hot_encoder",OneHotEncoder()]
            )

            logging.info("numerical cols scaling completed")
            logging.info("categorical cols encoding completed")

            preprocessor = ColumnTransformer(
                steps = [("num_pipeline",num_pipeline,numerical_cols),
                         ("cat_pipeline",cat_pipeline,categorical_cols)]
            )

            logging.info("attached both the pipeline together with columnTransformer")
        except Exception as e :
            pass


    
    #lets initialise thge object we created for transforming data

    def initialise_data_transformer(self,train_path,test_path):

        try:
                df_train = pd.read_csv(train_path)
                df_test = pd.read_csv(test_path)
                logging.info("read the data for training and testing")

            
                data_transformer_obj= self.get_data_tansformer_object()
                logging.info("got the data transformer object")
                
                target_col_name = "math_score"
                numerical_cols = numerical_cols

                target_train_features_arr = df_train[target_col_name]
                input_train_features_arr = df_train.drop(columns = [target_col_name],axis = 1)
                
                target_train_features_arr = df_test[target_col_name]
                input_test_featues_arr = df_test.drop(columns = [target_col_name],axis = 1)
                

                logging.info("got the data for training and testing ")

                input_feature_train = data_transformer_obj.fit_transform(input_train_features_arr)
                input_feature_test = data_transformer_obj.transform(input_test_featues_arr)

                train_arr = np.c_[input_train_features_arr,np.array(input_train_features_arr)]
                test_arr = np.c_[input_test_featues_arr,np.array(target_train_features_arr)]

                logging.info("saved the preprocessing object")

                save_object(
                     file_path = self.data_transformation_config.preprocessor_obj_file_path,
                     obj = data_transformer_obj
                )

                return (train_arr,
                       test_arr,
                       self.data_transformation_config.preprocessor_obj_file_path)
                

        except Exception as e : 
             raise CustomException(e,sys)
             



if __name__ == "__main__":
    # Call your function here to see results

    obj_data_injestion = DataInjestion()
    train_data,test_data = obj_data_injestion.initiate_data_injestion()

    obj_data_transformation = DataTransformation()
    obj_data_transformation.initiate_data_transformation(train_data,test_data)

