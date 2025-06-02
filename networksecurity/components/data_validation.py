from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from networksecurity.entity.config_entity import DataValidationConfig
import os
import sys
import pandas as pd
import json
from typing import Dict, List
import numpy as np
from sklearn.model_selection import train_test_split

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_path = self.data_validation_config.schema_file_path
            self.report_path = self.data_validation_config.report_file_path
        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)

    def validate_schema(self, dataframe: pd.DataFrame) -> bool:
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
            
            # Get schema columns
            schema_columns = [col['name'] for col in schema['columns']]
            
            # Get dataframe columns
            df_columns = dataframe.columns.tolist()
            
            # Check if all schema columns are present in dataframe
            missing_columns = [col for col in schema_columns if col not in df_columns]
            if missing_columns:
                logging.error(f"Missing columns in dataframe: {missing_columns}")
                return False
            
            # Check if all dataframe columns are in schema
            extra_columns = [col for col in df_columns if col not in schema_columns]
            if extra_columns:
                logging.warning(f"Extra columns in dataframe: {extra_columns}")
            
            # Validate data types
            for col in schema_columns:
                if col in df_columns:
                    expected_type = schema['columns'][schema_columns.index(col)]['type']
                    if expected_type == 'numeric':
                        if not pd.api.types.is_numeric_dtype(dataframe[col]):
                            logging.error(f"Column {col} should be numeric but is {dataframe[col].dtype}")
                            return False
                    elif expected_type == 'categorical':
                        if not pd.api.types.is_object_dtype(dataframe[col]):
                            logging.error(f"Column {col} should be categorical but is {dataframe[col].dtype}")
                            return False
            
            return True
        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in validate_schema: {str(e)}", error_details=sys)

    def check_data_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        try:
            drift_report = {}
            
            # Compare basic statistics
            for column in train_df.columns:
                if pd.api.types.is_numeric_dtype(train_df[column]):
                    train_mean = train_df[column].mean()
                    test_mean = test_df[column].mean()
                    train_std = train_df[column].std()
                    test_std = test_df[column].std()
                    
                    # Calculate drift score (simple mean difference)
                    drift_score = abs(train_mean - test_mean) / (train_std + 1e-6)
                    
                    drift_report[column] = {
                        'train_mean': train_mean,
                        'test_mean': test_mean,
                        'train_std': train_std,
                        'test_std': test_std,
                        'drift_score': drift_score,
                        'has_drift': drift_score > 0.1  # Threshold for drift detection
                    }
                else:
                    # For categorical columns, compare value distributions
                    train_dist = train_df[column].value_counts(normalize=True)
                    test_dist = test_df[column].value_counts(normalize=True)
                    
                    # Calculate drift score (simple distribution difference)
                    drift_score = sum(abs(train_dist.get(k, 0) - test_dist.get(k, 0)) for k in set(train_dist.index) | set(test_dist.index))
                    
                    drift_report[column] = {
                        'train_distribution': train_dist.to_dict(),
                        'test_distribution': test_dist.to_dict(),
                        'drift_score': drift_score,
                        'has_drift': drift_score > 0.1  # Threshold for drift detection
                    }
            
            return drift_report
        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in check_data_drift: {str(e)}", error_details=sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Read training and testing data
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Validate schema
            schema_validation = self.validate_schema(train_df)
            if not schema_validation:
                raise NetworkSecurityException(error_message="Schema validation failed", error_details=sys)
            
            # Check for data drift
            drift_report = self.check_data_drift(train_df, test_df)
            
            # Save drift report
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, 'w') as f:
                json.dump(drift_report, f, indent=4)
            
            # Create validation artifact
            validation_artifact = DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.report_path
            )
            
            return validation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in initiate_data_validation: {str(e)}", error_details=sys) 