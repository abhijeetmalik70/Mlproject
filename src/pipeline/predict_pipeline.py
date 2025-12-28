
import sys
import os

import pandas as pd 
from  src.utils import load_object


from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            logging.info("making the function for the prediction ")
            model_path=os.path.join("artificats","model.pkl")
            preprocessor_path=os.path.join('artificats','preprocessor.pkl')
            model = load_object(model_path)
            print(type(model)) 
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            logging.info("did the prediction using scaled data ")
            return pred

        except Exception as e :
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
       
        reading_score,
        writing_score):
        try : 
                    self.gender = gender
                    self.race_ethnicity = race_ethnicity
                    self.parental_level_of_education = parental_level_of_education
                    self.lunch = lunch
                    
                    self.reading_score = reading_score
                    self.writing_score = writing_score
                    self.test_preparation_course = test_preparation_course
        except Exception as e :
           raise CustomException(e,sys)

        
    def get_data_as_dataframe(self):
        try : 
                logging.info("gettig the data in dataframe")
                dict = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch], 
                "test_preparation_course" : [self.test_preparation_course],
                
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
                }
                logging.info("returned the data got form html with the form of data frame")
                return pd.DataFrame(dict)

    
        except Exception as e :
            raise CustomException(e,sys)
    