import sys
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass

#--------------------Predict the Output--------------------------->
    def predict(self, feature):
        try:
           model_path = 'artifacts/model.pkl'
           preprocessor_path = 'artifacts/preprocessor.pkl'
   
           model = load_object(file_path= model_path)
           preprocessor = load_object(file_path=preprocessor_path) 
           feature_scale = preprocessor.transform(feature)
           resulte = model.predict(feature_scale)
           return resulte
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
            self,
            year,   gdp,  social,   life, freedom, generosity,corruption 
    ):
        self.Year = year
        self.GDP = gdp
        self.SocialSupport = social
        self.Life = life
        self.Freedom = freedom
        self.Generosity = generosity
        self.Corruption = corruption 

# ------------Get the Data into DataFrame-------------------------------------->
    def get_data_frame(self):
        try:
            custom_datafram ={
            "Year": self.Year,
            "GDP per Capita": self.GDP,         
            "SocialSupport": self.SocialSupport,
            "LifeExpectancy": self.Life,     
            "Freedom": self.Freedom,
            "Generosity": self.Generosity,
            "Corruption": self.Corruption
            }
            return pd.DataFrame(custom_datafram, index=[0])
        except Exception as e:
            raise CustomException(e, sys)