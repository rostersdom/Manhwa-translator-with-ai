import gradio as gr
import numpy as np
from PIL import Image
from backend import get_translator, _trim_ram
_trim_ram()
import time
import subprocess
import tempfile
import os
import glob
import shutil
import sys
import re
import gc
import base64
import traceback
from io import BytesIO
from status_printer import status
import requests
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
try:
    import torch as _torch
except ImportError:
    _torch = None

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def save_page(img, project_name, page_num):
    if not project_name:
        return
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", project_name.strip())
    if not safe_name:
        return
    page_dir = os.path.join(OUTPUT_DIR, safe_name)
    os.makedirs(page_dir, exist_ok=True)
    path = os.path.join(page_dir, f"{safe_name}{page_num}.png")
    img.save(path, format="PNG")
    status.detail(f"Saved {path}")

def images_to_html(images, mode):
    html_content = '<div style="display: flex; flex-direction: column; align-items: center; width: 100%; background-color: #0b0f19;">'
    style = 'width: 100%; max-width: 800px; display: block; margin: 0; padding: 0; border: none; object-fit: contain;'
    if mode != "Манхва (Свиток)":
        style = 'width: 100%; max-width: 800px; display: block; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); object-fit: contain;'
    for im in images:
        img_str = img_to_base64(im)
        html_content += f'<img src="data:image/png;base64,{img_str}" style="{style}"/>'
    html_content += '</div>'
    return html_content

def process_images(image_files, mode, engine='yandex', project_name='', progress=gr.Progress()):
    if not image_files:
        return ""
    translator = get_translator()
    
    style = 'width: 100%; max-width: 800px; display: block; margin: 0; padding: 0; border: none; object-fit: contain;'
    if mode != "Манхва (Свиток)":
        style = 'width: 100%; max-width: 800px; display: block; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); object-fit: contain;'
    
    html_parts = []
    total_pages = len(image_files)
    status.total_pages = total_pages
    status.header(f"Processing {total_pages} pages ({mode}, {engine})")
    save_to_disk = bool(project_name and project_name.strip())

    def process_one_page(idx, file_path):
        progress((idx + 1) / total_pages, desc=f"Страница {idx+1}/{total_pages}")
        status.page_progress(idx + 1, total_pages)
        img = Image.open(file_path).convert('RGB')
        img_array = np.array(img)
        img.close()
        del img
        try:
            translated = translator.process_image(img_array, engine=engine, mode=mode)
            return translated
        finally:
            del img_array

    if mode == "Манхва (Свиток)":
        max_chunk_height = 4000
        chunks = []
        chunk_images = []
        chunk_height = 0
        first_page = True

        for idx, file_path in enumerate(image_files):
            try:
                translated_img = process_one_page(idx, file_path)
                if save_to_disk:
                    save_page(translated_img, project_name, idx + 1)
                else:
                    if first_page:
                        target_width = translated_img.width
                        first_page = False
                    im = translated_img
                    if im.width != target_width:
                        new_h = int(im.height * (target_width / im.width))
                        im = im.resize((target_width, new_h), Image.Resampling.LANCZOS)

                    if chunk_height + im.height > max_chunk_height and chunk_images:
                        chunk_img = Image.new('RGB', (target_width, chunk_height), (255, 255, 255))
                        y_offset = 0
                        for ci in chunk_images:
                            chunk_img.paste(ci, (0, y_offset))
                            y_offset += ci.height
                            ci.close()
                        chunks.append(chunk_img)
                        chunk_images = [im]
                        chunk_height = im.height
                    else:
                        chunk_images.append(im)
                        chunk_height += im.height

                del translated_img
            except Exception as e:
                status.err(f"Page {idx+1}: {e}")
            finally:
                gc.collect()
                if _torch and _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                _trim_ram()

        if not save_to_disk:
            if chunk_images:
                chunk_img = Image.new('RGB', (target_width, chunk_height), (255, 255, 255))
                y_offset = 0
                for ci in chunk_images:
                    chunk_img.paste(ci, (0, y_offset))
                    y_offset += ci.height
                    ci.close()
                chunks.append(chunk_img)
            del chunk_images
            gc.collect()
            _trim_ram()

            for chunk in chunks:
                html_parts.append(f'<img src="data:image/png;base64,{img_to_base64(chunk)}" style="{style}"/>')
                chunk.close()
            del chunks
            gc.collect()
            _trim_ram()
    else:
        for idx, file_path in enumerate(image_files):
            try:
                translated_img = process_one_page(idx, file_path)
                if save_to_disk:
                    save_page(translated_img, project_name, idx + 1)
                else:
                    html_parts.append(f'<img src="data:image/png;base64,{img_to_base64(translated_img)}" style="{style}"/>')
                del translated_img
            except Exception as e:
                status.err(f"Page {idx+1}: {e}")
            finally:
                gc.collect()
                if _torch and _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                _trim_ram()
    
    if save_to_disk:
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", project_name.strip())
        page_dir = os.path.join(OUTPUT_DIR, safe_name)
        html_parts.append(f'<p style="color: #aaa;">Сохранено {total_pages} страниц в <b>{page_dir}</b></p>')
    
    html_content = f'<div style="display: flex; flex-direction: column; align-items: center; width: 100%; background-color: #0b0f19;">{"".join(html_parts)}</div>'
    html_parts.clear()
    translator.translation_cache.cache.clear()
    _trim_ram()
    total_time = time.time() - status.start_time
    status.summary(total_time, total_pages, 0, 0, 0, 0)
    return html_content

BLACKLIST_KEYWORDS = [
    'avatar', 'logo', 'icon', 'banner', 'advert', 'sponsor',
    'pixel', 'tracker', 'analytics', 'emoji', 'thumbnail',
    'thumb.', 'sprite', 'background', 'footer', 'header', 'navbar',
    'social', 'share', 'like', 'comment', 'reply', 'vote',
    'rating', 'star', 'loader', 'profile', 'userpic',
    'favicon', 'widget', 'overlay', 'popup', 'modal',
    'separator', 'gradient', 'button', 'badge', 'tag',
    'ribbon', 'divider', 'border', 'teaser', 'mini', 'tiny',
    'captcha', 'recaptcha', 'watermark', 'stamp',
    'coin', 'point', 'level', 'rank',
    'blur', 'blurred', 'nsfw', 'adult',
    'follow', 'subscribe', 'bell', 'notification',
    'gravatar', 'wp-post-image', 'ts-post-image',
    'readerarea', 'wewtwt', 'cropped-',
    '-150x150', '-300x', '-150x', '-50x', '-32x32',
    '-180x', '-192x', '-270x', '-60x', '-96x',
]

MANGA_DOMAIN_PATTERNS = [
    'cdn.kingofshojo.com',
    'i.ibb.co',
    'mangadex',
    'mangaplus',
    'cubari',
    'imgur',
    'remanga',
    'mangalib',
]


def is_likely_manga_url(url):
    url_lower = url.lower().split('?')[0]
    if url_lower.endswith(('.gif', '.svg', '.ico')):
        return False
    for kw in BLACKLIST_KEYWORDS:
        if kw in url_lower:
            return False
    if any(domain in url_lower for domain in MANGA_DOMAIN_PATTERNS):
        return True
    return True


def filter_manga_images_by_size(image_paths, min_dimension=100, min_file_kb=10):
    valid = []
    for path in image_paths:
        try:
            size_kb = os.path.getsize(path) / 1024
            if size_kb < min_file_kb:
                print(f"  Removing {path}: too small ({size_kb:.0f}KB)")
                os.remove(path)
                continue
            with Image.open(path) as img:
                w, h = img.size
            if w < min_dimension or h < min_dimension:
                print(f"  Removing {path}: too small ({w}x{h})")
                os.remove(path)
                continue
            valid.append(path)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass
    return valid


def process_url(url, mode, engine='yandex', project_name='', progress=gr.Progress()):
    if not url:
        return ""
    
    temp_dir = tempfile.mkdtemp(prefix="manga_dl_")
    progress(0.05, desc="Скачивание изображений...")
    print(f"Downloading from {url} to {temp_dir}...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "gallery_dl",
            "--directory", temp_dir,
            url
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"gallery-dl error: {result.stderr}")
            print("Попытка скачать через браузер (Playwright) для обхода JS и Lazy Load...")
            
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--disable-web-security', '--disable-features=IsolateOrigins,site-per-process', '--no-sandbox']
                )
                page = browser.new_page()
                page.set_extra_http_headers({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
                
                try:
                    page.goto(url, wait_until='load', timeout=30000)
                except Exception as e:
                    print(f"Navigation error: {e}")
                try:
                    page.wait_for_load_state('networkidle', timeout=15000)
                except Exception:
                    pass
                
                last_height = page.evaluate("document.body.scrollHeight")
                while True:
                    page.keyboard.press("PageDown")
                    page.wait_for_timeout(500)
                    new_height = page.evaluate("document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                
                try:
                    page.wait_for_selector('.readercontent', timeout=10000)
                except Exception:
                    pass
                page.wait_for_timeout(2000)

                data_urls = page.evaluate("""
                    () => {
                        const results = [];

                        function getSrc(el) {
                            return el.getAttribute('src')
                                || el.getAttribute('data-src')
                                || el.getAttribute('data-lazy-src')
                                || el.getAttribute('data-original')
                                || el.getAttribute('data-url');
                        }

                        function isMangaImg(el) {
                            const src = (el.getAttribute('src') || '').toLowerCase();
                            const alt = (el.getAttribute('alt') || '').toLowerCase();
                            if (src.includes('gravatar') || src.includes('avatar') || src.includes('wp-post-image') || src.includes('ts-post-image') || src.includes('wp-content')) return false;
                            if (el.classList.contains('wp-post-image') || el.classList.contains('ts-post-image')) return false;
                            if (src.includes('cdn.kingofshojo.com/king-bucket/')) return true;
                            if (src.includes('i.ibb.co') && (alt.includes('ch ') || alt.match(/^\\d+-\\d+$/) || src.match(/1-2\\.jpg|2-2\\.jpg/))) return true;
                            if (src.includes('/king-bucket/')) return true;
                            if (alt.startsWith('ch ') && el.naturalWidth > 200) return true;
                            return false;
                        }

                        let imgs = [];
                        const reader = document.querySelector('.readercontent, .chapterbody, #readerpage, .reading-content');
                        if (reader) {
                            imgs = Array.from(reader.querySelectorAll('img')).filter(isMangaImg);
                        }
                        if (imgs.length === 0) {
                            imgs = Array.from(document.querySelectorAll('img[decoding="async"]')).filter(isMangaImg);
                        }
                        if (imgs.length === 0) {
                            imgs = Array.from(document.querySelectorAll('img')).filter(isMangaImg);
                        }

                        imgs.forEach((img, i) => {
                            try {
                                const canvas = document.createElement('canvas');
                                canvas.width = img.naturalWidth || img.width;
                                canvas.height = img.naturalHeight || img.height;
                                if (canvas.width < 100 || canvas.height < 100) return;
                                canvas.getContext('2d').drawImage(img, 0, 0);
                                const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
                                results.push(dataUrl);
                            } catch(e) {}
                        });

                        return results;
                    }
                """)
                print(f"  Extracted {len(data_urls)} manga images via canvas")
                
                for i, data_url in enumerate(data_urls):
                    try:
                        header, encoded = data_url.split(',', 1)
                        img_bytes = base64.b64decode(encoded)
                        with open(os.path.join(temp_dir, f"page_{i:03d}.jpg"), 'wb') as f:
                            f.write(img_bytes)
                    except Exception as e:
                        print(f"  Failed to save image {i}: {e}")
                
                browser.close()
        
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
            image_files.extend(glob.glob(os.path.join(temp_dir, '**', ext), recursive=True))
        
        image_files = filter_manga_images_by_size(image_files)
        
        if not image_files:
            raise Exception("Все скачанные изображения отфильтрованы как не-манга. Попробуйте другой источник.")
        
        image_files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)])
        
        print(f"Manga pages after filtering: {len(image_files)}")
        
        return process_images(image_files, mode, engine, project_name, progress)
        
    except Exception as e:
        print(f"Download/Process Error: {e}")
        traceback.print_exc()
        raise gr.Error(str(e))
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

with gr.Blocks(title="Manga & Manhwa Translator") as app:
    gr.Markdown("# 🎌 Manga & Manhwa Translator")
    gr.Markdown("Автоматический перевод манги и манхвы на русский язык. Поддерживает загрузку файлов или скачивание по ссылке (через `gallery-dl`).")
    
    with gr.Tabs():
        with gr.TabItem("Загрузить файлы"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_images = gr.File(file_count="multiple", type="filepath", label="Загрузите страницы (Изображения)")
                    mode_radio_1 = gr.Radio(["Манга (Постранично)", "Манхва (Свиток)"], value="Манхва (Свиток)", label="Режим просмотра")
                    engine_radio_1 = gr.Radio(["yandex", "google", "bing"], value="yandex", label="Переводчик")
                    project_name_1 = gr.Textbox(label="Имя проекта (сохранить в папку)", placeholder="tree", value="")
                    submit_btn_1 = gr.Button("Перевести файлы", variant="primary")
                    
                with gr.Column(scale=2):
                    output_html_1 = gr.HTML(label="Результат перевода")
                    
            submit_btn_1.click(
                fn=process_images,
                inputs=[input_images, mode_radio_1, engine_radio_1, project_name_1],
                outputs=[output_html_1]
            )
            
        with gr.TabItem("Скачать по ссылке"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_url = gr.Textbox(label="Ссылка на главу манги/манхвы", placeholder="https://mangadex.org/chapter/...")
                    mode_radio_2 = gr.Radio(["Манга (Постранично)", "Манхва (Свиток)"], value="Манга (Постранично)", label="Режим просмотра")
                    engine_radio_2 = gr.Radio(["yandex", "google", "bing"], value="yandex", label="Переводчик")
                    project_name_2 = gr.Textbox(label="Имя проекта (сохранить в папку)", placeholder="tree", value="")
                    submit_btn_2 = gr.Button("Скачать и Перевести", variant="primary")
                    gr.Markdown("*Поддерживаются многие сайты (Mangadex, Imgur, Reddit, Danbooru и т.д.), но некоторые защищены от ботов.*")
                    
                with gr.Column(scale=2):
                    output_html_2 = gr.HTML(label="Результат перевода")
                    
            submit_btn_2.click(
                fn=process_url,
                inputs=[input_url, mode_radio_2, engine_radio_2, project_name_2],
                outputs=[output_html_2]
            )

if __name__ == "__main__":
    print("Starting Gradio server...")
    app.launch(server_name="127.0.0.1", inbrowser=True)
