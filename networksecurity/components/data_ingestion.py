from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


## configuratioon of the data ingestion config
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataIngestionConfig
import os
import sys
import pandas as pd
import numpy as np
import pymongo
import certifi
from typing import List
from sklearn.model_selection import train_test_split
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()


MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    raise NetworkSecurityException(error_message="MONGO_DB_URL not found in .env file", error_details=sys)

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(error_message=str(e), error_details=sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            # Parse the MongoDB URL and encode username and password
            if '@' in MONGO_DB_URL:
                # Split the URL into parts
                protocol = MONGO_DB_URL.split('://')[0]
                rest = MONGO_DB_URL.split('://')[1]
                auth = rest.split('@')[0]
                host = rest.split('@')[1]
                
                # Encode username and password
                username = quote_plus(auth.split(':')[0])
                password = quote_plus(auth.split(':')[1], safe='')
                
                # Reconstruct the URL
                encoded_url = f"{protocol}://{username}:{password}@{host}"
                logging.info(f"Connecting to MongoDB with encoded URL")
            else:
                encoded_url = MONGO_DB_URL
            
            # Using certifi for SSL certificate verification
            self.mongo_client = pymongo.MongoClient(encoded_url, tlsCAFile=certifi.where())
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis = 1)
            df.replace({"na":np.nan},inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in export_collection_as_dataframe: {str(e)}", error_details=sys)
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in export_data_into_feature_store: {str(e)}", error_details=sys)
    
    def split_data_as_train_test_split(self,dataframe: pd.DataFrame):
        try:
            train_set , test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed Train Test Split on the DataFrame")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion Class")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok= True)
            logging.info(f"Exporting train and test file path")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Exporting train and test file path")
        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in split_data_as_train_test_split: {str(e)}", error_details=sys)

    def initate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataFrame = self.export_collection_as_dataframe()
            dataFrame = self.export_data_into_feature_store(dataFrame)
            self.split_data_as_train_test_split(dataFrame)
            dataIngestionArtifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return dataIngestionArtifact 
        except Exception as e:
            raise NetworkSecurityException(error_message=f"Error in initate_data_ingestion: {str(e)}", error_details=sys)