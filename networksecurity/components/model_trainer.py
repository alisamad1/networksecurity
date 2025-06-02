from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train the model using the training data
        """
        try:
            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                max_depth=self.model_trainer_config.max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def evaluate_model(self, model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model using the test data
        """
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            return metrics

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate the model training process
        """
        try:
            logging.info("Starting model training")
            
            # Load the transformed data
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)
            
            # Split features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            # Train the model
            model = self.train_model(X_train, y_train)
            
            # Evaluate the model
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Save the model
            os.makedirs(self.model_trainer_config.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_trainer_config.model_dir, self.model_trainer_config.model_name)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save the metrics
            metrics_path = os.path.join(self.model_trainer_config.model_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                model_file_path=model_path,
                metrics_file_path=metrics_path
            )
            
            logging.info("Model training completed")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) 