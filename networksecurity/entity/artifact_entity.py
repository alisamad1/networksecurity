from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: Optional[str]
    invalid_test_file_path: Optional[str]
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ModelTrainerArtifact:
    model_file_path: str
    metrics_file_path: str

@dataclass
class ModelEvaluationArtifact:
    model_evaluation_file_path: str
    is_model_accepted: bool
    model_file_path: str
    metrics_file_path: str

@dataclass
class ModelPusherArtifact:
    model_pusher_dir: str
    model_file_path: str
    metrics_file_path: str