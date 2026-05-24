import os
import re
import sys
import gc
import base64
import subprocess
import tempfile
import glob
import shutil
import traceback
from io import BytesIO
from celery import Celery

from app.config import settings
from app.database import SessionLocal
from app.models import TranslationJob
from app.ws import manager

celery_app = Celery(
    "manga_translator",
    broker=settings.redis_url,
    backend=settings.redis_url,
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
)

_BACKEND = None


def _get_backend():
    global _BACKEND
    if _BACKEND is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from backend import get_translator
        _BACKEND = get_translator()
    return _BACKEND


def _broadcast(job_id: str, data: dict):
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(manager.broadcast(job_id, data))
        loop.close()
    except Exception:
        pass


def _cleanup_cache():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def download_and_translate_url(self, job_id: str):
    db = SessionLocal()
    try:
        job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
        if not job:
            return
        job.status = "downloading"
        db.commit()

        _broadcast(job_id, {"type": "status", "status": "downloading"})

        temp_dir = tempfile.mkdtemp(prefix="manga_dl_")
        url = job.source_url

        result = subprocess.run(
            [sys.executable, "-m", "gallery_dl", "--directory", temp_dir, url],
            capture_output=True, text=True, timeout=120,
        )

        image_files = []
        if result.returncode == 0:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                image_files.extend(glob.glob(os.path.join(temp_dir, "**", ext), recursive=True))
        else:
            from app.downloader import download_with_playwright
            image_files = download_with_playwright(url, temp_dir)

        from app.utils import filter_manga_images, natural_sort_key
        image_files = filter_manga_images(image_files)
        image_files.sort(key=natural_sort_key)

        if not image_files:
            raise Exception("No valid manga images found")

        job.total_pages = len(image_files)
        job.metadata_json = {**job.metadata_json, "file_paths": image_files}
        db.commit()

        _process_image_files(job, image_files, db)

        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as exc:
        if db.is_active:
            job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(exc)
                db.commit()
        traceback.print_exc()
        raise self.retry(exc=exc)

    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def process_translation_job(self, job_id: str):
    db = SessionLocal()
    try:
        job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
        if not job:
            return

        file_paths = job.metadata_json.get("file_paths", []) if job.metadata_json else []
        _process_image_files(job, file_paths, db)

    except Exception as exc:
        if db.is_active:
            job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(exc)
                db.commit()
        traceback.print_exc()
        raise self.retry(exc=exc)

    finally:
        db.close()


def _process_image_files(job: TranslationJob, file_paths: list, db: SessionLocal):
    from PIL import Image
    import numpy as np

    job.status = "processing"
    db.commit()
    _broadcast(job.id, {"type": "status", "status": "processing"})

    translator = _get_backend()
    from backend import _trim_ram
    _trim_ram()
    total = len(file_paths)

    style = 'width: 100%; max-width: 800px; display: block; margin: 0; padding: 0; border: none; object-fit: contain;'
    if job.mode != "Манхва (Свиток)":
        style = 'width: 100%; max-width: 800px; display: block; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); object-fit: contain;'

    # Stream HTML to temp file instead of accumulating base64 in RAM
    html_path = os.path.join(tempfile.gettempdir(), f"manga_result_{job.id}.html")
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f'<div style="display: flex; flex-direction: column; align-items: center; width: 100%; background-color: #0b0f19;">')

            for idx, path in enumerate(file_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    img_array = np.array(img)
                    img.close()

                    translated = translator.process_image(img_array, engine=job.engine, mode=job.mode)
                    del img_array

                    buffered = BytesIO()
                    translated.save(buffered, format="PNG")
                    b64 = base64.b64encode(buffered.getvalue()).decode()
                    del buffered
                    del translated

                    f.write(f'<img src="data:image/png;base64,{b64}" style="{style}"/>')
                    del b64

                    job.completed_pages = idx + 1
                    job.progress = (idx + 1) / total
                    db.commit()

                    _broadcast(job.id, {
                        "type": "progress",
                        "completed": idx + 1,
                        "total": total,
                        "progress": job.progress,
                    })

                    _cleanup_cache()

                except Exception as e:
                    traceback.print_exc()
                    print(f"  Error processing {path}: {e}")

            f.write('</div>')

        # Read back the HTML string
        with open(html_path, encoding='utf-8') as f:
            job.result_html = f.read()
    finally:
        try:
            os.unlink(html_path)
        except Exception:
            pass
        del html_path

    job.status = "completed"
    job.progress = 1.0
    db.commit()

    _broadcast(job.id, {"type": "completed", "job_id": job.id})
    translator.free_resources()
    _cleanup_cache()
    _trim_ram()
