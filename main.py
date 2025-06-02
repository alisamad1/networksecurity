from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.components.model_evaluation import ModelEvaluation
from networksecurity.components.model_pusher import ModelPusher
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    TrainingPipelineConfig
)

import sys

if __name__ == "__main__":
    try:
        # Initialize training pipeline config
        training_pipeline_config = TrainingPipelineConfig()
        
        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initate_data_ingestion()
        
        # Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        
        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config, data_validation_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        # Model Training
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        # Model Evaluation
        model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation = ModelEvaluation(model_evaluation_config, model_trainer_artifact)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
        
        # Model Pusher
        model_pusher_config = ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config, model_evaluation_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()
        
        logging.info("Training pipeline completed successfully")

    except Exception as e:
        raise NetworkSecurityException(error_message=str(e), error_details=sys)
