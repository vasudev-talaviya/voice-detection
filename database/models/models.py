from pydantic import BaseModel, Field
from typing import List


class VoiceRecordBase(BaseModel):
    """
    Schema for a Voice Record — validated before storing to MongoDB.
    """
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the person whose voice is being stored"
    )
    embedding: List[List[float]] = Field(
        ...,                           # REQUIRED — a record without embedding is invalid
        description="WavLM x-vector speaker embedding trained on VoxCeleb (512-dim)"
    )


