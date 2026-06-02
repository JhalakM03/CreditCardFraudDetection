import sys
import os
import pandas as pd
from src.logger import logger
from src.exception import Custom_Exception
from sklearn.model_selection import train_test_split
from src.config import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig
    
    def initiate_data_ingestion(self):
        try:
            logger.info("Starting Data ingestion")

            df=pd.read_csv(os.path.join('notebooks/data',"creditcard.csv"))
            logger.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logger.info("Train Test split started")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logger.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise Custom_Exception(e,sys)
        