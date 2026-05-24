@echo off
title llama-server (Gemma 3 4B)
cd /d "%~dp0"

:: Auto-detect GPU layers
for /f "delims=" %%i in ('"%~dp0venv\Scripts\python.exe" "%~dp0auto_gpu_layers.py"') do set GPU_LAYERS=%%i
if "%GPU_LAYERS%"=="" set GPU_LAYERS=12

echo ========================================
echo  llama-server
echo ========================================
echo GPU Layers: %GPU_LAYERS% ^| Context: 8192 ^| 4GB VRAM mode
echo.

"%~dp0llama\llama-server.exe" ^
    --model "%USERPROFILE%\Downloads\gemma-3-4b-it-qat-UD-Q4_K_XL.gguf" ^
    --port 11435 ^
    --n-gpu-layers %GPU_LAYERS% ^
    --host 127.0.0.1 ^
    --ctx-size 4096 ^
    --temp 0.3 ^
    --reasoning off ^
    --mlock ^
    --threads 8 ^
    --cache-ram 0 ^
    --mmproj "%USERPROFILE%\.ollama\models\mmproj-F16.gguf"

echo.
echo llama-server stopped. RAM and VRAM released.
pause
