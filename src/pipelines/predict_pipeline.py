import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

from dataclasses import dataclass

@dataclass
class PredictPipelineConfig:
    preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")
    model_path=os.path.join("artifacts","model.pkl")

class PredictPipeline:
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self,features):
        try:
            logging.info("Before Loading")

            model=load_object(file_path=self.predict_pipeline_config.model_path)
            preprocessor=load_object(file_path=self.predict_pipeline_config.preprocessor_path)

            logging.info("After Loading")

            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)

            logging.info(f"y_pred: {pred}")

            return pred
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    def __str__(self):
        return f"gender={self.gender}, race_ethnicity = {self.race_ethnicity}, parental_level_of_education = {self.parental_level_of_education},\nlunch = {self.lunch}, test_preparation_course = {self.test_preparation_course}, reading_score = {self.reading_score},\nwriting_score = {self.writing_score})"


