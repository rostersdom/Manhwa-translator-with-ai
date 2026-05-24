from fastapi import Header, HTTPException
from app.config import settings


async def verify_api_key(x_api_key: str = Header(None)):
    if not settings.api_key_enabled:
        return True
    if not x_api_key or x_api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True
