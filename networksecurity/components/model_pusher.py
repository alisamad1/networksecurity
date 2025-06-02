from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from networksecurity.entity.config_entity import ModelPusherConfig
import os
import sys
import shutil
import json

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiate the model pushing process
        """
        try:
            logging.info("Starting model pushing")
            
            # Check if model is accepted
            if not self.model_evaluation_artifact.is_model_accepted:
                raise Exception("Model is not accepted for production")
            
            # Create the model pusher directory
            os.makedirs(self.model_pusher_config.model_pusher_dir, exist_ok=True)
            
            # Copy the model file
            model_file_path = self.model_evaluation_artifact.model_file_path
            model_pusher_path = os.path.join(self.model_pusher_config.model_pusher_dir, 'model.pkl')
            shutil.copy2(model_file_path, model_pusher_path)
            
            # Copy the metrics file
            metrics_file_path = self.model_evaluation_artifact.metrics_file_path
            metrics_pusher_path = os.path.join(self.model_pusher_config.model_pusher_dir, 'metrics.json')
            shutil.copy2(metrics_file_path, metrics_pusher_path)
            
            # Create model pusher artifact
            model_pusher_artifact = ModelPusherArtifact(
                model_pusher_dir=self.model_pusher_config.model_pusher_dir,
                model_file_path=model_pusher_path,
                metrics_file_path=metrics_pusher_path
            )
            
            logging.info("Model pushing completed")
            return model_pusher_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) 