import os
import uuid
import json
import zipfile
import tempfile
from io import BytesIO
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import settings
from app.models import TranslationJob
from app.schemas import TranslationJobCreate, TranslationJobResponse, JobListResponse
from app.auth import verify_api_key
from app.tasks import process_translation_job, download_and_translate_url

router = APIRouter(prefix="/api/v1", tags=["translate"])


@router.post("/translate/files", response_model=TranslationJobResponse, dependencies=[Depends(verify_api_key)])
async def translate_files(
    files: List[UploadFile] = File(...),
    mode: str = Form("Манхва (Свиток)"),
    engine: str = Form("yandex"),
    db: Session = Depends(get_db),
):
    if len(files) > settings.max_pages_per_job:
        raise HTTPException(400, f"Max {settings.max_pages_per_job} pages per job")

    os.makedirs(settings.upload_dir, exist_ok=True)
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(settings.upload_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    saved_paths = []
    for f in files:
        ext = os.path.splitext(f.filename or "image.jpg")[1] or ".jpg"
        dest = os.path.join(job_dir, f"{len(saved_paths):04d}{ext}")
        content = await f.read()
        with open(dest, "wb") as fh:
            fh.write(content)
        saved_paths.append(dest)

    job = TranslationJob(
        id=job_id,
        status="pending",
        mode=mode,
        engine=engine,
        total_pages=len(saved_paths),
        metadata_json={"file_paths": saved_paths},
    )
    db.add(job)
    db.commit()

    process_translation_job.delay(job_id)

    return job


@router.post("/translate/url", response_model=TranslationJobResponse, dependencies=[Depends(verify_api_key)])
async def translate_url(
    body: TranslationJobCreate,
    db: Session = Depends(get_db),
):
    if not body.source_url:
        raise HTTPException(400, "source_url is required")

    job = TranslationJob(
        status="pending",
        mode=body.mode or "Манга (Постранично)",
        engine=body.engine or "yandex",
        source_url=body.source_url,
    )
    db.add(job)
    db.commit()

    download_and_translate_url.delay(job.id)

    return job


@router.get("/jobs/{job_id}", response_model=TranslationJobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(TranslationJob)
    if status:
        q = q.filter(TranslationJob.status == status)
    jobs = q.order_by(TranslationJob.created_at.desc()).offset(skip).limit(limit).all()
    total = q.count()
    return JobListResponse(jobs=jobs, total=total)


@router.delete("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def delete_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    import shutil
    file_paths = job.metadata_json.get("file_paths", []) if job.metadata_json else []
    for fp in file_paths:
        try:
            os.remove(fp)
        except Exception:
            pass
    parent = os.path.dirname(file_paths[0]) if file_paths else ""
    try:
        shutil.rmtree(parent)
    except Exception:
        pass
    db.delete(job)
    db.commit()
    return {"ok": True}


@router.get("/jobs/{job_id}/export", dependencies=[Depends(verify_api_key)])
def export_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.result_html:
        raise HTTPException(400, "Job not completed yet")

    import re
    import base64
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        matches = re.findall(r'data:image/png;base64,([^"]+)', job.result_html)
        for i, b64 in enumerate(matches):
            try:
                img_bytes = base64.b64decode(b64)
                zf.writestr(f"page_{i:04d}.png", img_bytes)
            except Exception:
                pass
        meta = {
            "job_id": job.id,
            "mode": job.mode,
            "engine": job.engine,
            "total_pages": len(matches),
            "source_url": job.source_url,
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2, ensure_ascii=False))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=manga_{job_id}.zip"},
    )


@router.post("/jobs/{job_id}/retry", response_model=TranslationJobResponse, dependencies=[Depends(verify_api_key)])
def retry_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    job.status = "pending"
    job.error_message = None
    job.completed_pages = 0
    job.progress = 0.0
    db.commit()

    if job.source_url:
        download_and_translate_url.delay(job.id)
    else:
        process_translation_job.delay(job.id)
    return job
