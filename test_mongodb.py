from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Fixed URI with encoded password
uri = "mongodb+srv://Ali_Samad:1234samad%40A@cluster0.mbf1rjb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
