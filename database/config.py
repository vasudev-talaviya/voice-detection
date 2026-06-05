from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "voiceembedding")

# Create standard MongoClient with a 5-second connection timeout
client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = client[DATABASE_NAME]

def get_db():
    """
    Returns the database instance for database operations.
    """
    return db