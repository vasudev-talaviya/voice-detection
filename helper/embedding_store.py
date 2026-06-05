from database.operation.crud import save_voice_record
from database.models.models import VoiceRecordBase


def embedding_db_store(data: dict) -> bool:
    """
    Validate and persist a new voice embedding to the database.

    Args:
        data: dict with 'name' (str) and 'embedding' (List[List[float]]).

    Returns:
        True if stored successfully, False otherwise.
    """
    record = VoiceRecordBase(**data)
    success = save_voice_record(record.model_dump())
    if not success:
        print("[Error] Database did not insert the data.")
    return success