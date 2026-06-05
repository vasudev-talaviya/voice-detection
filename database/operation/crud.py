from bson import ObjectId
from database.config import get_db

COLLECTION_NAME = "voice_records"



def save_voice_record(data: dict) -> bool:
    """
    Stores a new voice record in the MongoDB database.

    Args:
        data: dict containing 'name' and 'embedding' fields.

    Returns:
        bool: True if inserted successfully, False otherwise.
    """
    db = get_db()
    collection = db[COLLECTION_NAME]
    try:
        result = collection.insert_one(data)
        return result.acknowledged
    except Exception as e:
        print(f"[Error] Not able to insert data into the database: {e}")
        return False


def fetch_all_embedding():
    """
    Fetches _id, name, and embedding for all records.

    Projecting 'name' here eliminates the N+1 query pattern that would otherwise
    require a separate fetch_record_by_id() call per record just to obtain the name.

    Returns:
        list[dict]: Each dict has '_id', 'name', and 'embedding'.
    """
    db = get_db()
    collection = db[COLLECTION_NAME]
    try:
        return list(collection.find({}, {"_id": 1, "name": 1, "embedding": 1}))
    except Exception as e:
        print(f"[Error] Not able to fetch embeddings from the database: {e}")
        return []


def fetch_record_by_id(record_id: str) -> dict | None:
    """
    Fetches a full voice record from DB using a str(_id).

    Args:
        record_id: str representation of the MongoDB ObjectId.

    Returns:
        dict: Full document (name, embedding, etc.) or None if not found.
    """
    db = get_db()
    collection = db[COLLECTION_NAME]
    try:
        return collection.find_one({"_id": ObjectId(record_id)})
    except Exception as e:
        print(f"[Error] Could not fetch record by id: {e}")
        return None