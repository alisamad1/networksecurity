from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelEvaluationConfig
import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            return metrics
        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in evaluate_model: {str(e)}", error_details=sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            # Load the test data
            test_data = np.load(self.model_evaluation_config.test_data_path)
            X_test = test_data[:, :-1]  # All columns except the last one
            y_test = test_data[:, -1]   # Last column is the target

            # Load the trained model
            model = joblib.load(self.model_trainer_artifact.model_file_path)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            metrics = self.evaluate_model(y_test, y_pred)

            # Save evaluation metrics
            os.makedirs(os.path.dirname(self.model_evaluation_config.evaluation_dir), exist_ok=True)
            metrics_file_path = os.path.join(self.model_evaluation_config.evaluation_dir, "metrics.json")
            with open(metrics_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            # Check if model meets minimum accuracy requirement
            is_model_accepted = metrics['accuracy'] >= self.model_evaluation_config.min_accuracy

            # Create evaluation artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                model_evaluation_file_path=metrics_file_path,
                is_model_accepted=is_model_accepted,
                model_file_path=self.model_trainer_artifact.model_file_path,
                metrics_file_path=metrics_file_path
            )

            logging.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
            return model_evaluation_artifact

        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in initiate_model_evaluation: {str(e)}", error_details=sys) 