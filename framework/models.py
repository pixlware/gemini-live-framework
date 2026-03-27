import datetime
from enum import Enum
from pydantic import BaseModel
from google.genai import types

class AudioFormat(Enum):
    PCM16 = "pcm16"
    MULAW = "mulaw"

class Role(Enum):
    USER = "user"
    MODEL = "model"

class Data(BaseModel):
    pass

class AudioData(Data):
    data: bytes
    format: AudioFormat
    sample_rate: int

class TextData(Data):
    text: str

class TranscriptData(Data):
    role: Role
    text: str

class EventData(Data):
    event: str
    metadata: dict = {}

class InterruptionData(Data):
    pass

class TurnCompleteData(Data):
    """Emitted with TURN_COMPLETE; ``audio_chunks`` counts model audio frames in that turn."""

    audio_chunks: int = 0

class ToolCallData(Data):
    id: str
    name: str
    args: dict

class ToolCallCancellationData(Data):
    ids: list[str]

class VoiceActivityData(Data):
    role: Role
    voice_activity_type: types.VoiceActivityType

class TranscriptEntry(BaseModel):
    role: Role
    text: str
    timestamp: datetime.datetime
    interrupted: bool = False

class UsageMetadataData(Data):
    prompt_token_count: int = 0
    response_token_count: int = 0
    total_token_count: int = 0
    thoughts_token_count: int = 0
    tool_use_prompt_token_count: int = 0
