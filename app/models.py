import datetime
import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Float
from app.database import Base


def _uuid():
    return str(uuid.uuid4())[:8]


class TranslationJob(Base):
    __tablename__ = "translation_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    status = Column(String, default="pending")
    mode = Column(String, nullable=False)
    engine = Column(String, default="yandex")

    total_pages = Column(Integer, default=0)
    completed_pages = Column(Integer, default=0)
    progress = Column(Float, default=0.0)

    source_url = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    result_html = Column(Text, nullable=True)
    metadata_json = Column(JSON, default=dict)
