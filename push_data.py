import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    raise Exception("MONGO_DB_URL not found in .env file")
print(f"MONGO_DB_URL: {MONGO_DB_URL}")
import certifi
ca = certifi.where()
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def cv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = json.loads(data.to_json(orient='records'))
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def insert_data_mongodb(self, records, database, collection):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            db = self.mongo_client[database]
            col = db[collection]
            col.insert_many(records)
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
if __name__ == "__main__":
    FILE_PATH = "Network_Data/phisingData.csv"  # Use forward slash or raw string
    database = "ALI_SAMAD"
    collection = "NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.cv_to_json_convertor(file_path=FILE_PATH)
    print(f"Total records extracted: {len(records)}")
    no_of_records = networkobj.insert_data_mongodb(records, database, collection)
    print(f"Successfully inserted {no_of_records} records into MongoDB.")
