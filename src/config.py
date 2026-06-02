from dataclasses import dataclass

@dataclass

class DataIngestionConfig:
    train_data_path:str = "artifact/train.csv"
    test_data_path:str = "artifact/test.csv"
    raw_data_path:str = "artifact/raw.csv"

class DataTrandformationConfig:
    preprocessor_obj_file_path:str = "artifact/preprocessor.pkl"

class ModelTrainerConfig:
    trained_model_file_path:str = "artifact/model.pkl"