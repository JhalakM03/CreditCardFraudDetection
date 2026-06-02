from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipelines.training_pipeline import TrainingPipeline

if __name__=="__main__":

    ingestion=DataIngestion()
    train_path,test_path= ingestion.initiate_data_ingestion()

    transformation=DataTransformation()
    X_train,X_test,y_train,y_test=transformation.initiate_data_transformation(train_path,test_path)

    model_trainer=ModelTrainer()
    results_manual=model_trainer.initiate_model_trainer(X_train,X_test,y_train,y_test)

    pipeline=TrainingPipeline()
    results_pipeline=pipeline.strat_training()

    print("Final Model results:", results_pipeline)

