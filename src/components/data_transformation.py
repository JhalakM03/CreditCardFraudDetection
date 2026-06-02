import os
import sys
import pandas as pd
import numpy as np

from src.logger import logger
from src.exception import Custom_Exception
from utils import save_object
from src.config import DataTrandformationConfig

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class DataTransformation:
    def __init__ (self):
        self.transformation_config = DataTrandformationConfig()

    def get_preprocessor(self):

        try:
            logger.info("Creating preprocessing pipeline")
            numeric_features=['Time','Amount']
            numeric_transformer = StandardScaler()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                ],
                remainder="passthrough"
            )
            return preprocessor
        
        except Exception as e:
            raise Custom_Exception(e,sys)



    def initiate_data_transformation(self, train_path, test_path):
        try:
            logger.info("Data Transformation started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = ['Class']

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(column=[target_column],axis=1)
            y_test = test_df[target_column]

            logger.info("Splitted features and target")

            preprocessor = self.get_preprocessor()

            Xtrain_transformed = preprocessor.fit_transform(X_train)
            Xtest_transformed = preprocessor.transform(X_test)

            save_object(
                self.transformation_config.preprocessor_obj_file_path,preprocessor
            )
            logger.info("Saved preprocessor")

            return Xtrain_transformed,Xtest_transformed,y_train,y_test
        except Exception as e:
            raise Custom_Exception(e,sys)
        