@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Manga Translator (API)

:: Auto-detect GPU layers
for /f "delims=" %%i in ('"%~dp0venv\Scripts\python.exe" "%~dp0auto_gpu_layers.py"') do set GPU_LAYERS=%%i
if "%GPU_LAYERS%"=="" set GPU_LAYERS=12

echo ========================================
echo  Manga Translator - FastAPI
echo ========================================
echo Starting llama-server (Gemma 3 4B)...
echo GPU Layers: %GPU_LAYERS% ^| Context: 8192 ^| 4GB VRAM mode

:: Start llama-server
start /B /MIN "" "%~dp0llama\llama-server.exe" ^
    --model "%USERPROFILE%\Downloads\gemma-3-4b-it-qat-UD-Q4_K_XL.gguf" ^
    --port 11435 --n-gpu-layers %GPU_LAYERS% --host 127.0.0.1 ^
    --ctx-size 4096 --temp 0.3 --reasoning off --mlock --threads 8 --cache-ram 0 ^
    --mmproj "%USERPROFILE%\.ollama\models\mmproj-F16.gguf"

timeout /t 3 /nobreak >nul

call venv\Scripts\activate.bat

echo Starting API...
echo Docs: http://127.0.0.1:8000/docs
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

:: Cleanup on exit
echo.
echo Shutting down llama-server...
taskkill /f /im llama-server.exe >nul 2>&1
echo RAM and VRAM released.
echo.
pause
