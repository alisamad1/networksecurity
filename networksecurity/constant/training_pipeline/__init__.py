import os
import sys
import numpy as np
import pandas as pd
'''
DEFINING COMMON CONSTANT VARIABLES FOR TRAINING PIPELINE
'''
TARGET_COLUMN = "Result"
PIPELINE_NAME : str = "NetworkSecurity"
ARTIFACT_DIR : str = "artifacts"
FILE_NAME : str = "phisingData.csv"
TRAIN_FILE_NAME : str = "train.csv"
TEST_FILE_NAME : str = "test.csv"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME : str = "NetworkData"
DATA_INGESTION_DATABASE_NAME : str = "ALI_SAMAD"
DATA_INGESTION_DIR_NAME  : str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR : str = "feature_store"
DATA_INGESTION_INGESTED_DIR : str = "ingested"
DATA_INGESTION_TRAIN_TEST_RATIO : float = 0.2
