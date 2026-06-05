from helper.embedding_store import embedding_db_store
from model.machine_learning.similarity import cosine_similarity_check
from model.machine_learning.config import REGISTER_THRESHOLD


def register_voice():
    """
    Register a new speaker:
      1. Extract embedding from audio file.
      2. Check for duplicates using cosine similarity (no ML model needed).
      3. If no duplicate, store embedding in database.
    """
    file_path = input("  Enter audio file path : ").strip()
    name      = input("  Enter person name     : ").strip()

    if not file_path or not name:
        print("[Error] File path and name cannot be empty.")
        return

    print(f"\n[INFO] Checking for duplicate voice  "
          f"(cosine similarity, threshold={REGISTER_THRESHOLD}) …")

    is_duplicate, similarity, matched_name, embedding_tensor = cosine_similarity_check(file_path)

    if is_duplicate:
        print(
            f"[Warn] Voice already registered as '{matched_name}' "
            f"(similarity: {similarity:.4f}). Registration aborted."
        )
        return

    print("[INFO] No duplicate detected. Storing new voice embedding …")

    # Convert tensor to nested list [[...]] expected by VoiceRecordBase
    embedding_data = embedding_tensor.cpu().tolist()

    saved = embedding_db_store({"name": name, "embedding": embedding_data})
    if not saved:
        print("[Error] Failed to save voice embedding to database.")
        return

    print(f"[INFO] '{name}' registered successfully.\n")
