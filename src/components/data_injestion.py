import os
import sys


import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artificats","train.csv")
    test_data_path = os.path.join("artificats","test.csv")
    raw_data_path = os.path.join("artificats","raw.csv")



class DataInjestion:

    def __init__(self):
        self.injestion_Config = DataIngestionConfig()
    

    def initiate_data_injestion(self):
        
        try:
            df = pd.read_csv("notebook/data/stud.csv")

            os.makedirs(os.path.dirname(self.injestion_Config.train_data_path),exist_ok=True)

            df.to_csv(self.injestion_Config.raw_data_path, index = False, header = True)

            train_set,test_set  = train_test_split(df)

            train_set.to_csv(self.injestion_Config.train_data_path, index = False, header = True)

            test_set.to_csv(self.injestion_Config.test_data_path, index = False, header = True)

            logging.info("data injestion has occured successfully")


            return(
            self.injestion_Config.train_data_path,
            self.injestion_Config.train_data_path
            )
        
    
        except Exception as e : 
             raise CustomException(e,sys)






