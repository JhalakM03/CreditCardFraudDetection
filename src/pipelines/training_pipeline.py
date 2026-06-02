import sys
from src.logger import logger
from src.exception import Custom_Exception

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:

    def strat_training(self):
        logger.info("Training pipeline started")
        try:
            ingestion=DataIngestion()
            train_path,test_path=ingestion.initiate_data_ingestion()
            logger.ingo(f"Dataingestion completed. Train path {train_path}, Test path {test_path}")

            transformation=DataTransformation()
            X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path,test_path)
            logger.info(f"Data transformation completed.")

            trainer=ModelTrainer()
            results=trainer.initiate_model_trainer( X_train, X_test, y_train, y_test)
            logger.info("Model training completed.")
            logger.info(
                f"Best model: {results['best_model_name']} | "
                f"F1: {results['best_f1']:.4f} | "
                f"AUC: {results['best_auc']:.4f}"
            )
            logger.info(f"Trained model saved at: {results['model_path']}")

            logger.info("Training pipeline finished successfully")
            return results
        except Exception as e:
            raise Custom_Exception(e,sys)