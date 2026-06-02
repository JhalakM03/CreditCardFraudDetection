import os
import sys
import pandas as pd
import numpy as np

from src.config import ModelTrainerConfig
from src.exception import Custom_Exception
from src.logger import logger
from utils import save_object

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, precision_score,f1_score,recall_score

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def resample_with_smote(self, X_train,y_train):
        try:
            logger.info("Applying SMOTE to training data")
            smote=SMOTE(sampling_strategy='minority', random_state=42)
            X_resampled,y_resampled = smote.fit_resample(X_train, y_train)

            logger.info(
                f"After smote, training shape:{X_resampled.shape},"
                f"class distribution:{np.bincount(y_resampled)}"        
            )

            return X_resampled,y_resampled

        except Exception as e:
            raise Custom_Exception(e,sys)
    
    def evaluate_model(self,model, X_train, y_train,X_test,y_test):
        try:
            logger.info("Training model: {model.__class__.__name__}")
            model.fit(X_train,y_train)

            y_proba=model.predict_proba(X_test)[:,-1]
            y_pred=model.predict(X_test)

            auc=roc_auc_score(y_test,y_proba)
            precision=precision_score(y_test,y_pred, zero_division=0)
            recall=recall_score(y_test,y_pred, zero_division=0) 
            f1=f1_score(y_test,y_pred, zero_division=0)

            logger.info(
                f"{model.__class__.__name__} - "
                f"AUC: {auc:.4f}, Precision:{precision:.4f},"
                f"Recall: {recall:.4f},F1:{f1:.4f}"
            )

            return {
                "model":model,
                "auc":auc,
                "recall":recall,
                "precision":precision,
                "f1":f1
            }

        except Exception as e:
            raise Custom_Exception(e,sys)
             
    def initiate_model_trainer(self, X_train,y_train,X_test,y_test):
        try:
            logger.info("Starting model training with SMOTE")
            X_train_res, y_train_res = self.resample_with_smote(X_train,y_train)

            models = {
                "LogisticRegression_SMOTE": LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1
                ),
                "RandomForest_SMOTE": RandomForestClassifier(
                    max_depth=None,
                    n_estimators=200,
                    n_jobs=-1,
                    random_state=42
                )
            }

            best_model_name = None
            best_model = None
            best_f1 = -np.inf
            best_auc = -np.inf

            results={}

            for name, model in models.items():
                logger.info("Evaluating model:{name}")
                metrics = self.evaluate_model(self,model,X_train_res,y_train_res,X_test,y_test)
                results[name]=metrics

                if(
                     metrics['f1']>best_f1
                     or(
                         np.isclose(metrics['f1'],best_f1)
                         and metrics["auc"]>best_auc
                    )
                 ):
                    best_f1=metrics['f1']
                    best_auc=metrics['auc']
                    best_model=metrics['model']
                    best_model_name=name
            
            logger.info(
                f"Best Model: {best_model_name}"
                f"(F1: {best_f1:.4f}, AUC:{best_auc:.4f})"
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,best_model
            )

            logger.info(
                f"Saved best model to: {self.model_trainer_config.trained_model_file_path}"
            )

            return{
                "best_model_name": best_model_name,
                "best_f1": best_f1,
                "best_auc": best_auc,
                "all_results": results,
                "model_path": self.model_trainer_config.trained_model_file_path,
            }
        except Exception as e:
            raise Custom_Exception(e,sys)
        