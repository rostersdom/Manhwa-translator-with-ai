# 📚 AI Manga & Manhwa Translator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GPU: NVIDIA](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GitHub stars](https://img.shields.io/github/stars/rostersdom/Manhwa-translator-with-ai?style=social)](https://github.com/rostersdom/Manhwa-translator-with-ai/stargazers)

> ⭐ **If you find this project useful, please consider giving it a star on GitHub! It helps a lot.** ⭐

*[Read in Russian / Читать на русском](#-ai-manga-translator-на-русском)*

A powerful, fully automated pipeline for translating Manga, Manhwa, and comics using local LLMs (Large Language Models) and VLMs (Vision Language Models). 

This project autonomously detects text bubbles, seamlessly erases original text, translates it (respecting context and Korean honorifics), and meticulously renders the translated text back into the bubbles using comic-style fonts.

## ✨ Key Features

*   **Smart LLM Translation:** Unlike standard translators (Google/Yandex), this pipeline uses local models (Gemma 3, Qwen) to provide literary translations. The AI understands martial arts/Murim terminology (Qi, Dantian, Cultivation) and correctly handles Korean honorifics (Hyung, Noona, Sunbae).
*   **Vision Language Model (VLM) Support:** If a local VLM is active, the system uses its vision capabilities to read and translate text directly from the image in a single pass.
*   **Perfect Inpainting:** Utilizes the **LaMa** (Large Mask Inpainting) model to completely erase original text and redraw the background without artifacts.
*   **Advanced Text Rendering:** The text rendering algorithm mathematically calculates the optimal aspect ratio to fit translated text perfectly inside elliptical bubbles. It automatically adjusts line spacing, applies precise padding, and uses a Cyrillic handwritten font (`Neucha`).
*   **Asynchronous Processing:** Built for heavy workloads using FastAPI, Celery (for GPU background tasks), and Redis.

## ⚙️ Architecture Pipeline

1.  **YOLOv8 (`comic-text-segmenter`)**: Detects all speech bubbles and text boxes on the page. Uses custom NMS filtering to prevent overlapping detections.
2.  **OCR / VLM**: Reads the text. If a local Gemma/Qwen Vision is connected, it handles this. Otherwise, it falls back to `MangaOCR` (for Japanese) or `EasyOCR` (for English/Korean).
3.  **LaMa Inpainter**: Cleans the original text from the image.
4.  **Translation (LLM)**: Translates the OCR text using a local LLM. Includes a post-processing script to fix Korean name transliteration (Kontsevich system).
5.  **Renderer**: Pillow draws the Russian text back onto the image, adapting font size to bubble geometry.

---

## 🚀 Installation

### Requirements
*   Windows / Linux
*   Python 3.10 - 3.12
*   NVIDIA GPU (Min 4GB VRAM) + CUDA Toolkit.

### 1. Clone & Setup
```bash
git clone https://github.com/rostersdom/Manhwa-translator-with-ai.git
cd Manhwa-translator-with-ai

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate # On Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Weights
Before running, you need to download the required models and fonts:
```bash
python download_yolo.py
python download_lama.py
python download_fonts.py
```

### 3. Environment Configuration
Copy the example config:
```bash
cp .env.example .env
```
Edit `.env` if necessary (e.g., set `QWEN_API_URL` if your local `llama.cpp` server runs on a different port).

---

## 🏃‍♂️ Running the Project

The system consists of an API server (FastAPI), a message broker (Redis), and a background worker (Celery).

### Option A: Quick Start (Windows .bat)
Requires [Redis for Windows](https://github.com/tporadowski/redis/releases) or Redis in Docker (`docker run -d -p 6379:6379 redis`):
1. Start your local LLM: `start_llama_server.bat` (requires a downloaded `.gguf` model and llama-server).
2. Open a new terminal and start the API: `run_api.bat`
3. Open another terminal and start the worker: `run_worker.bat`

API will be available at: http://127.0.0.1:8000/docs

### Option B: Docker Compose (Recommended for Linux/Servers)
For production or containerized setup (requires `nvidia-container-toolkit` for GPU access):
```bash
docker-compose up -d --build
```

---

## 🛠 Low-VRAM Optimization (4GB GPU)
The code is heavily optimized for budget GPUs (like RTX 2050/3050 with 4GB VRAM):
* PyTorch memory fragmentation is strictly controlled (`max_split_size_mb:32`).
* Inpainting (LaMa) is forced to run on the CPU (using 6 threads) to free up precious VRAM for YOLO and translation models.
* Dynamic RAM trimming and cache clearing (`gc.collect`, `torch.cuda.empty_cache`) are utilized to prevent Out-Of-Memory errors during long translation jobs.

---
---

<br>

# 🇷🇺 AI Manga Translator (на русском)

Мощный и полностью автоматизированный пайплайн для перевода манги и манхвы на русский язык с использованием локальных нейросетей (LLM/VLM). 

Проект самостоятельно находит текст, очищает баблы от оригинальных надписей, переводит текст (с учетом контекста и корейского этикета) и идеально вписывает его обратно с использованием комиксных шрифтов.

## ✨ Ключевые особенности

*   **Умный перевод (LLM):** В отличие от Google/Yandex, система использует локальные модели (Gemma 3, Qwen) для литературного перевода. ИИ знает про "хёнов", "нун" и терминологию Мурима (Ци, Даньтянь, Культивация).
*   **Продвинутое распознавание (VLM):** При наличии VLM-модели система использует машинное зрение для распознавания и перевода текста за один проход прямо с картинки.
*   **Идеальная очистка (Inpainting):** Используется модель **LaMa** (Large Mask Inpainting) для полного удаления оригинального текста и восстановления фона без артефактов.
*   **Продвинутая вёрстка текста:** Математически рассчитанный рендеринг подбирает идеальные пропорции переноса строк, чтобы текст максимально крупно вписывался в овалы баблов. Автоматически настраиваются отступы (padding) и используется кириллический комиксный шрифт `Neucha`.
*   **Многопоточность:** Обработка страниц происходит асинхронно для высоких нагрузок через Celery и Redis.

## ⚙️ Архитектура (Пайплайн)

1.  **YOLOv8 (`comic-text-segmenter`)**: Находит все баблы и прямоугольники с текстом на странице. Фильтрует дубликаты (NMS).
2.  **OCR / VLM**: Читает текст. Если подключена локальная Gemma/Qwen Vision, она делает это сама. В противном случае работает `MangaOCR` (для японского) или `EasyOCR` (для английского/манхвы).
3.  **LaMa Inpainter**: "Закрашивает" оригинальный текст на изображении.
4.  **Перевод (LLM)**: Переводит текст через локальную LLM. Работает система авто-коррекции корейских имен (перевод английских кальк в правильную систему Концевича).
5.  **Рендеринг**: Библиотека Pillow отрисовывает русский текст обратно на картинку.

---

## 🚀 Установка

### Требования
*   Windows / Linux
*   Python 3.10 - 3.12
*   NVIDIA GPU (Минимум 4 ГБ VRAM) + установленный CUDA Toolkit.

### 1. Установка окружения
```bash
git clone https://github.com/rostersdom/Manhwa-translator-with-ai.git
cd Manhwa-translator-with-ai

# Создаем виртуальное окружение
python -m venv venv
venv\Scripts\activate  # Для Windows
# source venv/bin/activate # Для Linux

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 2. Загрузка весов моделей
Перед первым запуском необходимо скачать веса моделей и шрифты:
```bash
python download_yolo.py
python download_lama.py
python download_fonts.py
```

### 3. Настройка
Скопируйте пример файла конфигурации:
```bash
cp .env.example .env
```
При необходимости отредактируйте `.env` (укажите ваш API ключ или локальный `QWEN_API_URL`).

---

## 🏃‍♂️ Запуск

Система состоит из трех компонентов: API-сервера (FastAPI), брокера сообщений (Redis) и фонового воркера (Celery).

### Вариант А: Быстрый старт (Windows .bat файлы)
Если у вас установлен [Redis для Windows](https://github.com/tporadowski/redis/releases) или Redis запущен в Docker:
1. Запустите локальную LLM: `start_llama_server.bat` (требует скачанной модели `.gguf`).
2. Откройте новый терминал и запустите API: `run_api.bat`
3. Откройте еще один терминал и запустите обработчик: `run_worker.bat`

API будет доступно по адресу: http://127.0.0.1:8000/docs

### Вариант Б: Docker Compose (Рекомендуется для Linux / Серверов)
Для продакшена (требуется установленный `nvidia-container-toolkit` для доступа к GPU):
```bash
docker-compose up -d --build
```
* Поддержите меня: 40817 810 4 4978 4217332 СБЕР
* Если нужно специально для вас с дообучением модели и тд: писать в тг @Sitmors за символическую плату
* Вставка в бабл не идеальна
---

## 🛠 Оптимизация под слабое железо
Код жестко оптимизирован для запуска на GPU с **4 ГБ VRAM** (например, мобильные RTX 2050/3050):
* Память PyTorch защищена от сильной фрагментации (`max_split_size_mb:32`).
* Тяжелый Inpainting (LaMa) принудительно вынесен на процессор (CPU, 6 потоков), чтобы освободить видеопамять для нейросетей перевода и детекции (YOLO).
* Встроена агрессивная очистка ОЗУ и кэша CUDA после обработки каждого бабла, предотвращающая падение скрипта с ошибкой Out-Of-Memory при пакетной обработке глав.
