from dataclasses import dataclass
from typing import Optional
import os
from datetime import datetime
from networksecurity.constant import training_pipeline

# Optional: Uncomment these if you want to verify the constants are loaded correctly
# print(training_pipeline.PIPELINE_NAME)
# print(training_pipeline.ARTIFACT_DIR)

@dataclass
class TrainingPipelineConfig:
    artifact_dir: str = os.path.join(os.getcwd(), "artifacts", datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig
    data_ingestion_dir: str = os.path.join(os.getcwd(), "artifacts", "data_ingestion")
    feature_store_file_path: str = os.path.join(os.getcwd(), "artifacts", "data_ingestion", "feature_store", "phisingData.csv")
    training_file_path: str = os.path.join(os.getcwd(), "artifacts", "data_ingestion", "ingested", "train.csv")
    testing_file_path: str = os.path.join(os.getcwd(), "artifacts", "data_ingestion", "ingested", "test.csv")
    train_test_split_ratio: float = 0.2
    collection_name: str = "NetworkData"
    database_name: str = "ALI_SAMAD"

@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig
    schema_file_path: str = os.path.join(os.getcwd(), "networksecurity", "constant", "schema.json")
    report_file_path: str = os.path.join(os.getcwd(), "artifacts", "data_validation", "drift_report.json")

@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig
    target_column: str = "Result"
    transformer_dir: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformer")
    imputer_path: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformer", "imputer.pkl")
    scaler_path: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformer", "scaler.pkl")
    transformed_train_dir: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformed", "train")
    transformed_test_dir: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformed", "test")
    transformed_train_path: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformed", "train", "train.npy")
    transformed_test_path: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformed", "test", "test.npy")

@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig
    model_dir: str = os.path.join(os.getcwd(), "artifacts", "model_trainer")
    model_name: str = "model.pkl"
    n_estimators: int = 100
    max_depth: int = 10

@dataclass
class ModelEvaluationConfig:
    training_pipeline_config: TrainingPipelineConfig
    evaluation_dir: str = os.path.join(os.getcwd(), "artifacts", "model_evaluation")
    test_data_path: str = os.path.join(os.getcwd(), "artifacts", "data_transformation", "transformed", "test", "test.npy")
    min_accuracy: float = 0.8

@dataclass
class ModelPusherConfig:
    training_pipeline_config: TrainingPipelineConfig
    model_pusher_dir: str = os.path.join(os.getcwd(), "artifacts", "model_pusher")
