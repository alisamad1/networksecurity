from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from typing import Tuple

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Tuple[SimpleImputer, StandardScaler]:
        """
        Create and return the data transformer objects
        """
        try:
            # Create imputer for handling missing values
            imputer = SimpleImputer(strategy='mean')
            
            # Create scaler for feature scaling
            scaler = StandardScaler()
            
            return imputer, scaler

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiate the data transformation process
        """
        try:
            logging.info("Starting data transformation")
            
            # Read the data
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            
            # Get the target column
            target_column = self.data_transformation_config.target_column
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            # Get transformer objects
            imputer, scaler = self.get_data_transformer_object()
            
            # Transform the data
            input_feature_train_arr = imputer.fit_transform(input_feature_train_df)
            input_feature_test_arr = imputer.transform(input_feature_test_df)
            
            input_feature_train_arr = scaler.fit_transform(input_feature_train_arr)
            input_feature_test_arr = scaler.transform(input_feature_test_arr)
            
            # Save the transformer objects
            os.makedirs(self.data_transformation_config.transformer_dir, exist_ok=True)
            
            with open(self.data_transformation_config.imputer_path, 'wb') as f:
                pickle.dump(imputer, f)
            
            with open(self.data_transformation_config.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save the transformed data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            os.makedirs(self.data_transformation_config.transformed_train_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.transformed_test_dir, exist_ok=True)
            
            np.save(self.data_transformation_config.transformed_train_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_path, test_arr)
            
            # Create transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformer_dir,
                transformed_train_file_path=self.data_transformation_config.transformed_train_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_path
            )
            
            logging.info("Data transformation completed")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) 