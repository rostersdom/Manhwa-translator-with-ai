import datetime
from typing import Optional, List
from pydantic import BaseModel


class TranslationJobCreate(BaseModel):
    mode: str = "Манхва (Свиток)"
    engine: str = "yandex"
    source_url: Optional[str] = None


class TranslationJobResponse(BaseModel):
    id: str
    status: str
    mode: str
    engine: str
    total_pages: int
    completed_pages: int
    progress: float
    source_url: Optional[str]
    error_message: Optional[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    completed_at: Optional[datetime.datetime]

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    jobs: List[TranslationJobResponse]
    total: int
