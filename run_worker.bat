@echo off
call venv\Scripts\activate.bat
echo Starting Celery Worker...
celery -A app.tasks.celery_app worker --loglevel=info --concurrency=1
pause
