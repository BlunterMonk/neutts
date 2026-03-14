from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str
    voice_id: str
    ref_text: str


class CancelEvent(BaseModel):
    event: str = "cancel"


class DoneEvent(BaseModel):
    event: str = "done"
    chunks: int
    duration_s: float


class ErrorEvent(BaseModel):
    event: str = "error"
    detail: str


class HealthResponse(BaseModel):
    status: str
    backbone: str
    device: str
    busy: bool


class VoiceInfo(BaseModel):
    voice_id: str
    filename: str


class VoiceEncodeResponse(BaseModel):
    voice_id: str


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]
