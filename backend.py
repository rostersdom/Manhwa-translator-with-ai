import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import translators as ts
from manga_ocr import MangaOcr
from status_printer import status

try:
    import g4f
except ImportError:
    g4f = None
from ultralytics import YOLO
import onnxruntime as ort
import os
import sys
import gc
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import pyphen
import re
import requests
from io import BytesIO
import zipfile
import json
import traceback
import urllib.request
import urllib.parse
from typing import Optional, List, Tuple

_EASYOCR_READER = None
_QWEN_BASE = os.environ.get('QWEN_API_URL', 'http://127.0.0.1:11435')
_QWEN_READY = None  # None=untested, True/False


def _trim_ram():
    """Aggressively free cached/unused RAM."""
    gc.collect()
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        kernel32.SetProcessWorkingSetSize(ctypes.c_void_p(-1), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:
        pass
    try:
        ctypes.cdll.msvcrt._heapmin()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


# Free any leftover RAM from previous runs
_trim_ram()


def _ping_qwen() -> bool:
    global _QWEN_READY
    if _QWEN_READY is not None:
        return _QWEN_READY
    _QWEN_READY = False
    
    # Просто проверяем, отвечает ли сервер локальной модели
    try:
        r = requests.get(f"{_QWEN_BASE}/v1/models", timeout=5)
        if r.ok:
            _QWEN_READY = True
    except Exception:
        pass
        
    if not _QWEN_READY:
        try:
            r = requests.get(f"{_QWEN_BASE}/health", timeout=5)
            if r.ok:
                _QWEN_READY = True
        except Exception:
            pass

    if _QWEN_READY:
        status.info(f"Local AI Vision OK at {_QWEN_BASE}")
    else:
        status.info("Local AI Vision unavailable — EasyOCR fallback")
    return _QWEN_READY


def _get_context_crop(full_bgr: np.ndarray, bbox: tuple, margins: dict = None) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = full_bgr.shape[:2]
    if margins is None:
        margins = {'up': 250, 'down': 250, 'left': 250, 'right': 250}
    cx1 = max(0, x1 - margins['left'])
    cy1 = max(0, y1 - margins['up'])
    cx2 = min(w, x2 + margins['right'])
    cy2 = min(h, y2 + margins['down'])
    return full_bgr[cy1:cy2, cx1:cx2]

def _detect_tail_direction(orig_bgr: np.ndarray, bbox: tuple) -> str:
    """Detect which side of the bubble the tail protrudes from.
    Returns 'up', 'down', 'left', 'right', or ''.
    """
    x1, y1, x2, y2 = bbox
    h, w = orig_bgr.shape[:2]
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Sample bubble edge colour from just inside each side of the bbox
    edge_colors = []
    inset = 3
    if y2 - y1 > 8:
        edge_colors.append(orig_bgr[y1 + inset, mid_x])
        edge_colors.append(orig_bgr[y2 - inset, mid_x])
    if x2 - x1 > 8:
        edge_colors.append(orig_bgr[mid_y, x1 + inset])
        edge_colors.append(orig_bgr[mid_y, x2 - inset])
    if not edge_colors:
        return ''
    bubble_color = np.mean(edge_colors, axis=0)

    # Scan outward from centre of each edge, count consecutive matching pixels
    scan_max = 150
    results = {}
    for dir_name, sx, sy, dx, dy in [
        ('up',    mid_x, y1,     0, -1),
        ('down',  mid_x, y2,     0,  1),
        ('left',  x1,    mid_y, -1,  0),
        ('right', x2,    mid_y,  1,  0),
    ]:
        px, py = sx, sy
        count = 0
        for _ in range(scan_max):
            px += dx
            py += dy
            if not (0 <= px < w and 0 <= py < h):
                break
            p = orig_bgr[py, px]
            diff = abs(int(p[0]) - int(bubble_color[0])) + \
                   abs(int(p[1]) - int(bubble_color[1])) + \
                   abs(int(p[2]) - int(bubble_color[2]))
            if diff < 60:
                count += 1
            else:
                break
        results[dir_name] = count

    if max(results.values()) < 5:
        return ''
    return max(results, key=results.get)

def _ocr_with_qwen(crop_bgr: np.ndarray, context_bgr: np.ndarray = None, read_only: bool = False) -> tuple:
    """OCR + translate a single bubble via Qwen vision API.
    If context_bgr is provided, shows the speaker next to the bubble.
    If read_only=True, just returns the raw text without translation.
    Returns (text, already_translated).
    """
    import base64
    img_to_send = context_bgr if context_bgr is not None else crop_bgr
    h, w = img_to_send.shape[:2]
    # Upscale tiny crops so Qwen can read them
    MIN_DIM = 128
    if max(h, w) < MIN_DIM:
        scale = MIN_DIM / max(h, w)
        img_to_send = cv2.resize(img_to_send, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, buf = cv2.imencode('.png', img_to_send)
    b64 = base64.b64encode(buf).decode()
    if read_only:
        prompt = "Read the text in this image. Return ONLY the exact text exactly as written, no translation, no explanation."
    elif context_bgr is not None:
        prompt = (
            "This image shows part of a manga/manhwa page. It might be a speech bubble, thought bubble, narration box, or sign. "
            "Read the text, identify the speaker's tone if there is a character, and translate the text to Russian. "
            "CRITICAL RULES FOR KOREAN NAMES AND HONORIFICS: "
            "1. Translate Korean names naturally using standard Russian transliteration rules (e.g. 'Hoon-yeon' -> 'Хунён', NOT 'ХунЙеон'). Do not use hyphens in names unless necessary. "
            "2. Translate honorifics and suffixes conceptually if possible, or use accepted fandom terms: 'oppa' -> 'оппа', 'noona' -> 'нуна', 'hyung' -> 'хён', 'sunbae' -> 'сонбэ'. "
            "3. Grammatically integrate honorifics: decline them properly in Russian (e.g., 'с нуной', 'к хёну') or rephrase naturally (e.g., 'Почему я должен жениться на нуне Хунён?' instead of 'Хун-Ыоне нуна'). "
            "CRITICAL RULES FOR MARTIAL ARTS / MURIM / WUXIA: "
            "Use established Russian terminology: Qi/Ki -> Ци/Ки, Cultivation -> Культивация/Развитие, Dantian -> Даньтянь, Meridians -> Меридианы, Heavenly Demon -> Небесный Демон, Demonic Cult -> Демонический культ. "
            "Translate martial arts techniques and sect names (e.g., Mount Hua -> Хуашань, Tang clan -> клан Тан) to sound epic and traditional. "
            "Return ONLY the Russian translation, nothing else. "
            "NEVER comment on the image, NEVER say you cannot translate, just provide the translation."
        )
    else:
        prompt = (
            "Read the text in this image from a comic/manga and translate it to Russian. "
            "It might be a speech bubble, sign, or narration box. "
            "CRITICAL RULES FOR KOREAN NAMES AND HONORIFICS: "
            "1. Translate Korean names naturally using standard Russian transliteration rules (e.g. 'Hoon-yeon' -> 'Хунён'). "
            "2. Translate honorifics and suffixes: 'oppa' -> 'оппа', 'noona' -> 'нуна', 'hyung' -> 'хён', 'sunbae' -> 'сонбэ'. "
            "3. Grammatically integrate and decline honorifics properly in Russian (e.g., 'с нуной', 'к хёну'). "
            "4. NEVER drop or omit character names from the translation. If a name (like Hoon-yeon) is in the source text, it MUST be in the translated text. "
            "CRITICAL RULES FOR MARTIAL ARTS / MURIM / WUXIA: "
            "Use established Russian terminology: Qi/Ki -> Ци/Ки, Cultivation -> Культивация, Dantian -> Даньтянь, Meridians -> Меридианы, Heavenly Demon -> Небесный Демон, Demonic Cult -> Демонический культ. "
            "Translate martial arts techniques and sect names to sound epic and traditional. "
            "Return ONLY the Russian translation, nothing else. "
            "NEVER comment on the image, NEVER say you cannot translate, just provide the translation."
        )
    try:
        r = requests.post(
            f"{_QWEN_BASE}/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                "max_tokens": 768,
                "temperature": 0,
                "seed": 42,
            },
            timeout=120,
        )
        if not r.ok:
            return ("", False)
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        text = text.strip('"\'«»„“')
        if text:
            return (text, True)
        return ("", False)
    except Exception:
        return ("", False)


# Reduce PyTorch CUDA fragmentation
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:False'

def _get_free_vram() -> float:
    """Return free VRAM in GB, or 0 if unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, _ = torch.cuda.mem_get_info()
        return free / 1024**3
    except Exception:
        return 0.0


# Pre-compiled regex patterns
_RE_CLEAN = re.compile(r'[^\x20-\x7E\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]')
_RE_SPACES = re.compile(r'\s+')
_RE_REPEAT = re.compile(r'(.)\1{3,}')
_RE_LATIN = re.compile(r'[a-zA-Z]')
_RE_SPACE_PUNCT = re.compile(r'\s+([.,!?;:])')
_RE_PAREN_OPEN = re.compile(r'\(\s+')
_RE_PAREN_CLOSE = re.compile(r'\s+\)')
_RE_DASH = re.compile(r'\s*-\s*')
_RE_REPEAT_LONG = re.compile(r'(.)\1{3,}')

# Shared Draw for text measurement (avoids creating 1x1 image each call)
_MEASURE_DRAW = ImageDraw.Draw(Image.new('RGB', (1, 1)))

_TRANSLATION_ENGINE_PRIORITY = ['deepl', 'google', 'yandex', 'bing']

# Character confusion: when two chars are commonly swapped by OCR, we try each.
_OCR_CHAR_SWAPS = {
    '0': 'O', '1': 'l', '5': 'S', '8': 'B',
    'l': 'I', 'I': 'l', 'i': 'l',
    'n': 'u', 'u': 'n',
    'a': 'o', 'o': 'a',
    'e': 'c', 'c': 'e',
    'h': 'n', 'b': 'h',
    'd': 'cl', 'm': 'rn',
}

# Cyrillic homoglyphs: visually identical to Latin letters but different Unicode.
# When OCR misreads a Latin word, it often outputs Cyrillic code points instead.
_OCR_CYRILLIC_HOMOGLYPHS = str.maketrans({
    '\u0430': 'a', '\u0435': 'e', '\u043E': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0456': 'i',
    '\u043F': 'n', '\u0433': 'r', '\u0442': 't', '\u0438': 'u',
    '\u043A': 'k', '\u043C': 'm', '\u0432': 'b', '\u043D': 'h',
    '\u0437': 'z',
})

# Common English words that appear in manhwa, mapped to their known OCR variants
_OCR_WORD_CORRECTIONS = {
    # pronouns
    'you': 'you', 'yuu': 'you', 'yoo': 'you', 'yoл': 'you',
    'your': 'your', 'youn': 'your', 'yolr': 'your',
    'they': 'they', 'rhey': 'they', 'they': 'they',
    'them': 'them', 'rhem': 'them', 'thein': 'them',
    'this': 'this', 'rnis': 'this', 'tnis': 'this',
    'that': 'that', 'rhat': 'that', 'thai': 'that', 'thaf': 'that',
    'what': 'what', 'whaf': 'what', 'whaт': 'what',
    # common verbs
    'have': 'have', 'have': 'have', 'haue': 'have', 'haue': 'have',
    'here': 'here', 'nere': 'here', 'hece': 'here', 'hete': 'here', 'hene': 'here', 'heге': 'here',
    'there': 'there', 'tnere': 'there', 'rhere': 'there', 'rhere': 'there', 'rhere': 'there',
    'where': 'where', 'wnere': 'where', 'wbete': 'where',
    'from': 'from', 'frora': 'from', 'frorn': 'from', 'frоm': 'from',
    'aura': 'aura', 'alra': 'aura', 'aиra': 'aura', 'aига': 'aura', 'аига': 'aura',
    'power': 'power', 'powet': 'power', 'powen': 'power', 'poweг': 'power',
    'fight': 'fight', 'fighr': 'fight', 'fighт': 'fight',
    'right': 'right', 'righr': 'right', 'righт': 'right',
    'night': 'night', 'nighr': 'night', 'nighт': 'night',
    'light': 'light', 'lighr': 'light', 'lighт': 'light',
    'know': 'know', 'knoш': 'know', 'kпош': 'know',
    'think': 'think', 'rhnk': 'think', 'thinк': 'think',
    'want': 'want', 'wanт': 'want', 'шanт': 'want',
    'need': 'need', 'neeo': 'need', 'neеd': 'need',
    'come': 'come', 'coine': 'come', 'cоme': 'come',
    'like': 'like', 'like': 'like', 'lіke': 'like',
    'make': 'make', 'inake': 'make', 'mаke': 'make',
    'take': 'take', 'rake': 'take', 'tаke': 'take',
    # common adjectives/adverbs
    'very': 'very', 'ueгy': 'very', 'ueгу': 'very',
    'much': 'much', 'mиch': 'much', 'mисh': 'much',
    'more': 'more', 'more': 'more', 'moгe': 'more',
    'still': 'still', 'stiil': 'still', 'stil': 'still',
    'just': 'just', 'jиst': 'just', 'jиsт': 'just',
    'only': 'only', 'onlу': 'only', 'опlу': 'only',
    'even': 'even', 'euen': 'even', 'eveп': 'even',
    'well': 'well', 'weil': 'well', 'weIl': 'well',
    'like': 'like', 'lіke': 'like', 'lікe': 'like',
    'little': 'little', 'IittIe': 'little', 'Iittle': 'little',
    'good': 'good', 'good': 'good', 'gooо': 'good',
    'great': 'great', 'greaт': 'great', 'greaт': 'great',
    'strong': 'strong', 'stroпg': 'strong', 'stroпg': 'strong',
    'weak': 'weak', 'шeak': 'weak', 'шеаk': 'weak',
    'real': 'real', 'геаl': 'real', 'reaI': 'real',
    'really': 'really', 'геаIIy': 'really', 'геаlly': 'really',
    # common prepositions/conjunctions
    'with': 'with', 'шith': 'with', 'шiтh': 'with',
    'without': 'without', 'шithout': 'without', 'шiтhoиt': 'without',
    'about': 'about', 'abоиt': 'about', 'aboиt': 'about',
    'because': 'because', 'becаиse': 'because', 'becаuse': 'because',
    # common manhwa tropes
    'never': 'never', 'neveг': 'never', 'neueг': 'never',
    'always': 'always', 'alшays': 'always', 'аlшays': 'always',
    'again': 'again', 'agaiп': 'again', 'agaiп': 'again',
    'enough': 'enough', 'enoиgh': 'enough', 'enoиgh': 'enough',
    'maybe': 'maybe', 'mayЬe': 'maybe', 'mayЬe': 'maybe',
    'please': 'please', 'рlease': 'please', 'рIease': 'please',
    'sorry': 'sorry', 'sоrry': 'sorry', 'sоггу': 'sorry',
    'thank': 'thank', 'thanк': 'thank', 'thanк': 'thank',
    'help': 'help', 'helр': 'help', 'heIp': 'help',
    'listen': 'listen', 'Iisten': 'listen', 'Iisтen': 'listen',
    'wait': 'wait', 'waiт': 'wait', 'шaiт': 'wait',
    'stop': 'stop', 'stoр': 'stop', 'sтop': 'stop',
    'watch': 'watch', 'waтch': 'watch', 'шатch': 'watch',
    'look': 'look', 'Iook': 'look', 'Iооk': 'look',
    'find': 'find', 'fiпd': 'find', 'fiпd': 'find',
    'keep': 'keep', 'keeр': 'keep', 'keер': 'keep',
    'leave': 'leave', 'Ieave': 'leave', 'Ieаue': 'leave',
    'sure': 'sure', 'sиre': 'sure', 'suгe': 'sure',
    'fine': 'fine', 'fiпe': 'fine', 'fiпе': 'fine',
    'die': 'die', 'dіe': 'die', 'dіе': 'die',
    'alive': 'alive', 'aliue': 'alive', 'аlіue': 'alive',
    'dead': 'dead', 'dеаd': 'dead', 'deаd': 'dead',
    'life': 'life', 'Iife': 'life', 'Iіfe': 'life',
    'death': 'death', 'deаth': 'death', 'dеаth': 'death',
    'world': 'world', 'wогld': 'world', 'шогId': 'world',
    'time': 'time', 'tiпe': 'time', 'тime': 'time',
    'thing': 'thing', 'rhing': 'thing', 'rhiпg': 'thing',
    'people': 'people', 'рeople': 'people', 'реорIе': 'people',
    'friend': 'friend', 'frieпd': 'friend', 'frіепd': 'friend',
    'enemy': 'enemy', 'eneшу': 'enemy', 'епету': 'enemy',
    'father': 'father', 'fаther': 'father', 'fathеr': 'father',
    'mother': 'mother', 'mоther': 'mother', 'mothеr': 'mother',
    'brother': 'brother', 'brоther': 'brother', 'brоthеr': 'brother',
    'sister': 'sister', 'sisтer': 'sister', 'sistеr': 'sister',
    'master': 'master', 'masтer': 'master', 'masteг': 'master',
    'young': 'young', 'уoung': 'young', 'уоuпg': 'young',
    'old': 'old', 'oId': 'old', 'оId': 'old',
    'new': 'new', 'пew': 'new', 'пеw': 'new',
    'first': 'first', 'firsт': 'first', 'firѕt': 'first',
    'last': 'last', 'Iast': 'last', 'Iasт': 'last',
}


# Deduplicated dictionary values for fast fuzzy matching
_OCR_WORD_CORRECTIONS_VALUES = list(set(_OCR_WORD_CORRECTIONS.values()))


def _confusion_distance(a: str, b: str) -> int:
    """Levenshtein distance with lower cost for OCR-confused characters."""
    if len(a) > 20 or len(b) > 20:
        return abs(len(a) - len(b)) + 5
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            if ca == cb:
                cost = 0
            elif (ca in _OCR_CHAR_SWAPS and cb in _OCR_CHAR_SWAPS.get(ca, '')) or \
                 (cb in _OCR_CHAR_SWAPS and ca in _OCR_CHAR_SWAPS.get(cb, '')):
                cost = 0.5
            elif ca.isalpha() and cb.isalpha() and ca.lower() == cb.lower():
                cost = 0.6
            else:
                cost = 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j + 1] + cost))
        prev = curr
    return prev[-1]


def correct_ocr_word(word: str) -> str:
    """Return corrected word if a close dictionary match is found, else original."""
    if len(word) <= 2 or word.isdigit() or '-' in word:
        return word
    lower = word.lower()
    # Exact correction lookup
    if lower in _OCR_WORD_CORRECTIONS:
        correction = _OCR_WORD_CORRECTIONS[lower]
        # Preserve original casing pattern
        if word.isupper():
            return correction.upper()
        if word[0].isupper():
            return correction.capitalize()
        return correction
    # Fuzzy correction for unknown tokens (only check words of similar length)
    if not word.isalpha():
        return word
    wlen = len(word)
    best = word
    best_dist = 1.0 if wlen <= 4 else 0.4 * wlen
    lo, hi = max(3, wlen - 2), wlen + 3
    for dict_word in _OCR_WORD_CORRECTIONS_VALUES:
        dwlen = len(dict_word)
        if dwlen < lo or dwlen > hi:
            continue
        d = _confusion_distance(lower, dict_word.lower())
        if d < best_dist:
            best_dist = d
            best = dict_word
    if best != word:
        if word.isupper():
            return best.upper()
        if word[0].isupper():
            return best.capitalize()
        return best
    return word


def _fullwidth_to_halfwidth(text: str) -> str:
    result = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            result.append(chr(cp - 0xFEE0))
        elif 0xFF10 <= cp <= 0xFF19:
            result.append(chr(cp - 0xFEE0))
        else:
            result.append(ch)
    return ''.join(result)


def normalize_ocr_text(text: str) -> str:
    text = _fullwidth_to_halfwidth(text.replace('\n', ' '))
    text = _RE_CLEAN.sub('', text)
    text = _RE_SPACES.sub(' ', text)
    text = _RE_REPEAT.sub(r'\1\1', text).strip()

    tokens = text.split()
    corrected_tokens = []
    for token in tokens:
        if _RE_LATIN.search(token):
            token = token.translate(_OCR_CYRILLIC_HOMOGLYPHS)
            token = correct_ocr_word(token)
        corrected_tokens.append(token)
    return ' '.join(corrected_tokens).strip()


def _fix_korean_names(text: str) -> str:
    # 1. Исправляем CamelCase в кириллице (ХунЙеон -> Хунйеон)
    text = re.sub(r'([а-яё])([А-ЯЁ])', lambda m: m.group(1) + m.group(2).lower(), text)
    
    # 2. Убираем дефисы между частями имени с большой буквы (Хун-Ыон -> Хуныон)
    text = re.sub(r'\b([А-ЯЁ][а-яё]*)-([А-ЯЁ][а-яё]*)\b', lambda m: m.group(1) + m.group(2).lower(), text)
    
    # 3. Убираем дефисы перед типичными корейскими слогами в нижнем регистре (Хун-ыоне -> Хуныоне)
    korean_syllables = [
        'йеон', 'йонг', 'йоон', 'хеон', 'хьюнг', 'хьюн', 'джунг', 'чонг', 'чои', 'баек',
        'сеонг', 'сео', 'еун', 'ыон', 'ыоне', 'сунг', 'йео', 'джае', 'дае', 'тае', 'хае',
        'гванг', 'кванг', 'вонг', 'рьёнг', 'мьёнг', 'ен', 'ён', 'юн', 'хон', 'хён', 'чон',
        'чхве', 'пэк', 'сон', 'со', 'ын', 'ё', 'джэ', 'дэ', 'тэ', 'хэ', 'гван', 'кван', 'вон'
    ]
    for syl in korean_syllables:
        text = re.sub(rf'\b([А-ЯЁ][а-яё]*)-({syl}[а-яё]*)\b', lambda m: m.group(1) + m.group(2).lower(), text, flags=re.IGNORECASE)

    # 4. Заменяем типичную английскую кальку на правильную систему Концевича
    replacements = [
        ('йеон', 'ён'), ('Йеон', 'Ён'),
        ('йонг', 'ён'), ('Йонг', 'Ён'),
        ('йоон', 'юн'), ('Йоон', 'Юн'),
        ('хеон', 'хон'), ('Хеон', 'Хон'),
        ('хьюнг', 'хён'), ('Хьюнг', 'Хён'),
        ('хьюн', 'хён'), ('Хьюн', 'Хён'),
        ('джунг', 'чон'), ('Джунг', 'Чон'),
        ('чонг', 'чон'), ('Чонг', 'Чон'),
        ('чои', 'чхве'), ('Чои', 'Чхве'),
        ('баек', 'пэк'), ('Баек', 'Пэк'),
        ('сеонг', 'сон'), ('Сеонг', 'Сон'),
        ('сео', 'со'), ('Сео', 'Со'),
        ('еун', 'ын'), ('Еун', 'Ын'),
        ('ыон', 'ён'), ('Ыон', 'Ён'),
        ('сунг', 'сон'), ('Сунг', 'Сон'),
        ('йео', 'ё'), ('Йео', 'Ё'),
        ('джае', 'джэ'), ('Джае', 'Джэ'),
        ('дае', 'дэ'), ('Дае', 'Дэ'),
        ('тае', 'тэ'), ('Тае', 'Тэ'),
        ('хае', 'хэ'), ('Хае', 'Хэ'),
        ('гванг', 'гван'), ('Гванг', 'Гван'),
        ('кванг', 'кван'), ('Кванг', 'Кван'),
        ('вонг', 'вон'), ('Вонг', 'Вон'),
        ('геон', 'гон'), ('Геон', 'Гон'),
        ('сыон', 'сон'), ('Сыон', 'Сон'),
        ('сын', 'сон'), ('Сын', 'Сон'), # Часто Geon читают как Сын в левых переводчиках
    ]

    for bad, good in replacements:
        text = text.replace(bad, good)
        
    # Дополнительно удаляем оставшиеся дефисы внутри русских слов с заглавными буквами
    text = re.sub(r'([А-ЯЁ][а-яё]*)-([А-ЯЁ][а-яё]*)', lambda m: m.group(1) + m.group(2).lower(), text)
    
    return text


def postprocess_translation(text: str) -> str:
    text = text.strip()
    if not text:
        return text
        
    text = _fix_korean_names(text)
    
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = _RE_SPACE_PUNCT.sub(r'\1', text)
    text = _RE_PAREN_OPEN.sub('(', text)
    text = _RE_PAREN_CLOSE.sub(')', text)
    text = _RE_DASH.sub('-', text)
    if len(text) > 3 and text[-1] not in '.!?…':
        text += '.'
    return text


def _try_llama(text: str, dst_lang: str = 'ru') -> Optional[str]:
    try:
        from rag_fewshot import rag
        fewshot = rag.get_fewshot(text, top_k=3)
        fewshot_section = f"\n\nEXAMPLE TRANSLATIONS (use similar style):\n{fewshot}\n" if fewshot else ""

        system_prompt = (
            "You are a professional manga/manhwa translator. Translate English to natural, lively Russian.\n\n"
            "CRITICAL RULES FOR KOREAN NAMES AND HONORIFICS:\n"
            "- Translate Korean names naturally using standard Russian rules (e.g. 'Hoon-yeon' -> 'Хунён', NOT 'ХунЙеон').\n"
            "- Translate honorifics: oppa/hyung = 'хён'/'оппа', noona = 'нуна', sunbae = 'сонбэ'.\n"
            "- Decline honorifics properly in Russian (e.g. 'к нуне', 'с хёном').\n"
            "- Name endings: -ya = drop it (소연아 → Соён), -ssi = '-сси', -nim = '-ним'.\n"
            "- Use colloquial Russian that matches the character's personality.\n"
            "- Interjections: 'А?' for 'Huh?', 'Вот чёрт!' for 'Damn!', 'Ого!' for 'Whoa!'.\n"
            "- Keep emphasis: CAPS → bold or spacing, '...' → '...' \n"
            "CRITICAL RULES FOR MARTIAL ARTS / MURIM:\n"
            "- Use established Russian terminology: Qi/Ki -> Ци/Ки, Cultivation -> Культивация, Dantian -> Даньтянь, Meridians -> Меридианы, Heavenly Demon -> Небесный Демон, Demonic Cult -> Демонический культ.\n"
            "- Translate martial arts techniques, sect/clan names (e.g., Mount Hua -> Хуашань, Tang -> Тан), and realms to sound epic and traditional.\n"
            f"{fewshot_section}\n\n"
            "Output ONLY the Russian translation, no explanations."
        )
        body = {
            "model": "qwen", # Use a generic/fallback model name
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "max_tokens": 384,
            "temperature": 0.3,
        }
        r = requests.post(f"{_QWEN_BASE}/v1/chat/completions", json=body, timeout=120)
        if r.ok:
            data = r.json()
            result = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if result and len(result) > 2:
                rag.add_pair(text, result)
                return result
    except Exception as e:
        status.detail(f"Local LLM translation failed: {e}")
    return None


def _try_direct_google(text: str, dst_lang: str = 'ru') -> Optional[str]:
    try:
        params = urllib.parse.urlencode({'client': 'gtx', 'sl': 'auto', 'tl': dst_lang, 'dt': 't', 'q': text[:5000]})
        url = f"https://translate.googleapis.com/translate_a/single?{params}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            result = ''.join(part[0] for part in data[0] if part[0])
            return result.strip() if result else None
    except Exception:
        return None

def translate_fallback(text: str, src_lang: str, dst_lang: str = 'ru', preferred: str = None) -> str:
    result = _try_llama(text, dst_lang)
    if result:
        return result

    engines = _TRANSLATION_ENGINE_PRIORITY
    if preferred and preferred not in engines:
        engines = [preferred] + engines
    elif preferred:
        engines = [preferred] + [e for e in engines if e != preferred]
    last_error = None
    for eng in engines:
        try:
            result = ts.translate_text(text, translator=eng, from_language=src_lang, to_language=dst_lang)
            if result and result.strip():
                return result.strip()
        except Exception as e:
            last_error = e
            continue
    try:
        result = ts.translate_text(text, translator='google', from_language='auto', to_language=dst_lang)
        if result and result.strip():
            return result.strip()
    except Exception as e:
        last_error = e
    result = _try_direct_google(text, dst_lang)
    if result:
        return result
    if g4f is not None:
        try:
            prompt = (
                f"You are a professional translator of Korean manhwa to Russian. "
                f"Translate this English scanlation text into natural, contextual Russian. "
                f"CRITICAL RULES: "
                f"1. Use Korean etiquette and honorifics (e.g., 'хен', 'нуна', 'сонбэ', 'господин', respectful tone where appropriate). "
                f"2. The length of the Russian translation MUST NOT exceed the length of the English text by more than 10%. Use shorter synonyms to fit the bubble. "
                f"3. Output ONLY the translated Russian string. Do not add quotes, notes, or explanations.\n\n"
                f"Text to translate: {text}"
            )
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4o_mini,
                messages=[{"role": "user", "content": prompt}],
                timeout=10
            )
            if response and len(response.strip()) > 0:
                result = response.strip()
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                return result
        except Exception:
            pass
    return text

def _get_easyocr():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import torch
            import easyocr
            use_gpu = False
            if torch.cuda.is_available():
                free_vram = torch.cuda.mem_get_info()[0] / 1024**3
                use_gpu = free_vram > 0.8
                status.detail(f"EasyOCR GPU: {use_gpu} (VRAM {free_vram:.1f}GB)")
            status.info(f"Loading EasyOCR (en, gpu={use_gpu})...")
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=use_gpu)
            status.ok("EasyOCR loaded")
        except Exception as e:
            status.warn(f"EasyOCR unavailable: {e}")
            _EASYOCR_READER = False
    return _EASYOCR_READER if _EASYOCR_READER else None


def _free_easyocr():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None and _EASYOCR_READER is not False:
        try:
            del _EASYOCR_READER
        except Exception:
            pass
        _EASYOCR_READER = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        status.info("EasyOCR freed")


FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

GOOGLE_FONTS = {
    "Neucha.ttf": "https://github.com/google/fonts/raw/main/ofl/neucha/Neucha.ttf",
    "NotoSans.ttf": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf",
    "NotoSansKR.ttf": "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf",
}

FONT_FALLBACK_CHAIN = [
    "Neucha.ttf",
    "NotoSansKR.ttf",
    "NotoSans.ttf",
    "DejaVuSans.ttf",
]


def _download_google_fonts():
    for name, url in GOOGLE_FONTS.items():
        dest = os.path.join(FONT_DIR, name)
        if not os.path.exists(dest):
            try:
                status.info(f"Downloading {name} from Google Fonts...")
                import urllib.request
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as r:
                    with open(dest, 'wb') as f:
                        f.write(r.read())
                status.ok(f"Font saved → {dest}")
            except Exception as e:
                status.warn(f"Font download failed ({name}): {e}")


def resolve_font_path() -> Optional[str]:
    _download_google_fonts()
    for fname in FONT_FALLBACK_CHAIN:
        path = os.path.join(FONT_DIR, fname)
        if os.path.exists(path):
            return path
    mpl_dejavu = os.path.join(
        os.path.dirname(matplotlib.__file__),
        'mpl-data', 'fonts', 'ttf', 'DejaVuSans.ttf'
    )
    if os.path.exists(mpl_dejavu):
        return mpl_dejavu
    return None


class LRUCache:
    def __init__(self, capacity=256):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def contains_japanese(text):
    count = 0
    for ch in text:
        cp = ord(ch)
        if (0x3040 <= cp <= 0x309F or
            0x30A0 <= cp <= 0x30FF or
            0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF):
            count += 1
            if count >= 2:
                return True
    return False


def contains_korean(text):
    for ch in text:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            return True
    return False


def _is_predominantly_cyrillic(t):
    alpha_chars = [ch for ch in t if ch.isalpha()]
    if not alpha_chars:
        return False
    cyrillic_chars = [ch for ch in alpha_chars if 0x0400 <= ord(ch) <= 0x04FF]
    return len(cyrillic_chars) / len(alpha_chars) > 0.5


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    ox1, oy1, ox2, oy2 = box2
    ix1, iy1 = max(x1, ox1), max(y1, oy1)
    ix2, iy2 = min(x2, ox2), min(y2, oy2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (ox2 - ox1) * (oy2 - oy1)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def filter_overlapping_boxes(boxes, iou_threshold=0.5):
    """
    Удаляет меньшие рамки, которые сильно пересекаются с большими.
    В отличие от merge_boxes, эта функция не создает гигантские рамки,
    а просто фильтрует дубликаты, которые YOLO иногда выдает на один бабл.
    """
    if not boxes:
        return []
    # Сортируем от самых больших к самым маленьким
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    kept_boxes = []
    
    for i, box in enumerate(boxes):
        keep = True
        for kept_box in kept_boxes:
            if iou(box, kept_box) > iou_threshold:
                keep = False
                break
        if keep:
            kept_boxes.append(box)
            
    return kept_boxes


def is_vertical_box(box, ratio_threshold=1.3):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return h > w * ratio_threshold



def preprocess_for_ocr(crop_bgr, for_japanese=False):
    if crop_bgr.size == 0:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if for_japanese:
        denoised = cv2.fastNlMeansDenoising(thresh, None, 5, 7, 21)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def detect_text_language_from_image(crop_rgb):
    """
    Quick heuristic to decide if crop contains Japanese/CJK or Latin text.
    Returns 'jp', 'kr', or 'other' (for English/Russian/etc).
    Uses connected component analysis to classify glyph shapes.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps = [c for c in contours if 20 < cv2.contourArea(c) < 20000]

    if not comps:
        return 'other'

    jp_score = 0
    kr_score = 0
    total = 0

    for cnt in comps:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        aspect = w / h if h > 0 else 0

        if aspect > 2.5 or aspect < 0.3:
            continue

        total += 1

        if solidity < 0.6 and h > w * 1.5:
            jp_score += 1
        elif solidity < 0.65 and h > w * 1.3:
            kr_score += 1

        if w > h * 0.8 and w < h * 2.5 and solidity > 0.7:
            kr_score += 0.5

    if total == 0:
        return 'other'

    jp_ratio = jp_score / total
    kr_ratio = kr_score / total

    if jp_ratio > 0.15:
        return 'jp'
    if kr_ratio > 0.20:
        return 'kr'
    return 'other'


def shrink_bbox(x1, y1, x2, y2, img_w, img_h, shrink_ratio=0.08):
    """Shrink bbox from edges proportionally."""
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(4, int(bw * shrink_ratio))
    pad_y = max(4, int(bh * shrink_ratio))
    nx1 = max(0, x1 + pad_x)
    ny1 = max(0, y1 + pad_y)
    nx2 = min(img_w, x2 - pad_x)
    ny2 = min(img_h, y2 - pad_y)
    if nx2 - nx1 < 20 or ny2 - ny1 < 20:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def detect_background_type_deep(crop_bgr):
    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Tiny boxes: just check mean brightness — no histogram needed
    if h < 30 or w < 30:
        return np.mean(gray) < 100

    border_thickness = max(2, min(w, h) // 15)
    border_pixels = np.concatenate([
        gray[:border_thickness, :].ravel(),
        gray[-border_thickness:, :].ravel(),
        gray[:, :border_thickness].ravel(),
        gray[:, -border_thickness:].ravel(),
    ])
    border_mean = np.mean(border_pixels)

    center_rect = gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = np.mean(center_rect)

    # Quick histogram — only dark/light bins, no full hist
    dark_pct = np.sum(gray < 80) / gray.size
    light_pct = np.sum(gray > 180) / gray.size

    score = (
        (border_mean < 100) * 2.0 +
        (dark_pct > light_pct * 1.3) * 1.5 +
        (center_mean < 100) * 1.5
    )
    return score >= 2.5


def detect_text_color(crop_bgr: np.ndarray, is_dark_bg: bool) -> Tuple[int, int, int]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if is_dark_bg:
        _, text_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    else:
        _, text_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    text_mask = cv2.erode(text_mask, np.ones((2, 2), np.uint8), iterations=1)
    if cv2.countNonZero(text_mask) < 5:
        return (255, 255, 255) if is_dark_bg else (0, 0, 0)
    text_pixels = crop_bgr[text_mask > 0]
    median = np.median(text_pixels, axis=0).astype(int)
    return (int(median[2]), int(median[1]), int(median[0]))

def detect_text_style(text: str) -> str:
    upper = sum(1 for c in text if c.isupper())
    total = sum(1 for c in text if c.isalpha())
    upper_ratio = upper / total if total > 0 else 0
    has_repeat = bool(_RE_REPEAT_LONG.search(text))
    stripped = text.strip()
    word_count = len(stripped.split())
    char_count = len(stripped)
    ends_bang = stripped.endswith('!') or stripped.endswith('?')

    is_sfx = False
    is_scream = False

    if char_count < 12 and upper_ratio > 0.8 and has_repeat:
        is_sfx = True
    if char_count < 8 and upper_ratio > 0.8 and word_count <= 2:
        is_sfx = True
    if word_count >= 3 and upper_ratio > 0.8 and ends_bang:
        is_scream = True
    if word_count >= 2 and upper_ratio > 0.8 and char_count > 10:
        is_scream = True

    if is_sfx and not is_scream:
        return "sfx"
    if is_scream:
        return "scream"
    if upper_ratio > 0.8 and word_count <= 2 and char_count < 15:
        return "sfx"
    return "normal"


def render_text_with_outline(
    draw: ImageDraw.Draw,
    pil_img: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    x: float, y: float,
    text_color: Tuple[int, int, int],
    outline_color: Optional[Tuple[int, int, int]] = None,
    outline_width: int = 3,
    anchor: str = "mm",
    align: str = "center",
    style: str = "normal",
    box_width: float = 0,
    box_height: float = 0,
):
    if outline_color is None:
        outline_color = (255, 255, 255) if sum(text_color) < 384 else (0, 0, 0)
        
    # Явно задаем альфа-канал 255 (непрозрачность)
    rgba_text_color = (text_color[0], text_color[1], text_color[2], 255)
    rgba_outline_color = (outline_color[0], outline_color[1], outline_color[2], 255)

    if style == "sfx":
        ow = max(outline_width, 4)
        draw.text(
            (x, y), text, font=font, fill=rgba_text_color,
            stroke_width=ow, stroke_fill=rgba_outline_color,
            anchor="mm", align="center", spacing=0
        )
        for dx, dy in [(2, 2), (-2, -2), (2, -2), (-2, 2)]:
            draw.text(
                (x + dx, y + dy), text, font=font, fill=(0, 0, 0, 60),
                stroke_width=1, stroke_fill=(0, 0, 0, 30),
                anchor="mm", align="center", spacing=0
            )
        draw.text(
            (x, y), text, font=font, fill=rgba_text_color,
            stroke_width=ow, stroke_fill=rgba_outline_color,
            anchor="mm", align="center", spacing=0
        )
        return

    # Для обычного текста убираем объемные тени, но оставляем тонкую обводку (1px) 
    # для контраста, чтобы текст не сливался с баблом при ошибках определения цвета
    draw.text(
        (x, y), text, font=font, fill=rgba_text_color,
        stroke_width=1, stroke_fill=rgba_outline_color,
        anchor=anchor, align=align, spacing=0
    )


def hyphenated_wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    def measure(t):
        bbox = _MEASURE_DRAW.textbbox((0, 0), t, font=font)
        return bbox[2] - bbox[0]

    try:
        dic = pyphen.Pyphen(lang='ru_RU')
    except Exception:
        dic = None

    lines = []
    words = text.split()
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word]) if current_line else word
        if measure(test_line) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
                
            if measure(word) <= max_width:
                current_line = [word]
            else:
                sub_lines = _split_long_word(word, font, max_width, measure, dic)
                if sub_lines:
                    lines.extend(sub_lines[:-1])
                    current_line = [sub_lines[-1]]

    if current_line:
        lines.append(" ".join(current_line))

    return lines if lines else [text]


def _split_long_word(
    word: str, font: ImageFont.FreeTypeFont, max_width: int,
    measure_fn, dic
) -> List[str]:
    """Split a long word using hyphenation if available, else character-by-character."""
    lines = []
    buffer = ""

    if dic:
        syllables = dic.inserted(word).split("-")
        for i, syl in enumerate(syllables):
            is_last = i == len(syllables) - 1
            test = buffer + syl
            test_measure = test if is_last else test + "-"
            if measure_fn(test_measure) > max_width and buffer:
                lines.append(buffer + "-")
                buffer = syl
            else:
                buffer = test
        if buffer:
            lines.append(buffer)
        return lines

    for ch in word:
        test = buffer + ch
        if measure_fn(test + "-") > max_width and buffer:
            lines.append(buffer + "-")
            buffer = ch
        else:
            buffer = test
    if buffer:
        lines.append(buffer)
    return lines


def find_optimal_font(
    text: str, font_path: str, max_width: int, max_height: int,
    min_size: int = 10, max_size: int = 48,
    line_spacing: float = 1.0,
) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
    avail_w, avail_h = max_width, max_height
    if avail_w < 10 or avail_h < 10:
        avail_w, avail_h = max_width, max_height

    best_overall_size = 0
    best_overall_lines = [text]
    best_overall_font = None

    # Пробуем разные ограничения ширины, чтобы скомпоновать текст в форму, близкую к квадрату/кругу
    width_factors = [1.0, 0.9, 0.8, 0.75, 0.65]

    for w_factor in width_factors:
        target_w = int(avail_w * w_factor)
        
        lo, hi = min_size, max_size
        best_size_for_w = 0
        best_lines_for_w = [text]
        best_font_for_w = None
        
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                font = ImageFont.truetype(font_path, mid)
            except Exception:
                font = ImageFont.load_default()
                return font, [text], 12

            lines = hyphenated_wrap(text, font, target_w)
            test_text = "\n".join(lines)
            
            # Измеряем точный многострочный блок с плотным межстрочным интервалом (spacing=0)
            bbox = _MEASURE_DRAW.multiline_textbbox((0, 0), test_text, font=font, spacing=0, align="center")
            actual_w = bbox[2] - bbox[0]
            actual_h = bbox[3] - bbox[1]

            if actual_w <= avail_w and actual_h <= avail_h:
                best_size_for_w = mid
                best_lines_for_w = lines
                best_font_for_w = font
                lo = mid + 1
            else:
                hi = mid - 1
                
        # Выбираем ту конфигурацию (ширина/строки), которая позволяет сделать шрифт крупнее
        if best_size_for_w > best_overall_size:
            best_overall_size = best_size_for_w
            best_overall_lines = best_lines_for_w
            best_overall_font = best_font_for_w

    if best_overall_font is None:
        try:
            font = ImageFont.truetype(font_path, min_size)
        except Exception:
            font = ImageFont.load_default()
        return font, hyphenated_wrap(text, font, avail_w), min_size

    return best_overall_font, best_overall_lines, best_overall_size


class MangaTranslator:
    def __init__(self):
        import torch
        status.header("Initializing Translator")
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            free_vram = _get_free_vram()
            # Only use CUDA if enough VRAM free at init time
            self.device = 'cuda' if free_vram > 1.0 else 'cpu'
            status.info(f"CUDA available ({total_vram:.1f}GB total, {free_vram:.1f}GB free) → device={self.device}")
        else:
            self.device = 'cpu'
            status.info("Device: CPU")

        status.info("Loading YOLO Comic Text Segmenter...")
        yolo_path = os.path.join(os.path.dirname(__file__), "comic-text-segmenter.pt")
        if not os.path.exists(yolo_path):
            raise Exception("YOLO model not found! Run download_yolo.py first.")
        self.detector = YOLO(yolo_path)
        self.detector.to(self.device)
        status.ok("YOLO loaded")

        status.info("Loading MangaOCR...")
        self.mocr = MangaOcr(force_cpu=(self.device == 'cpu'))
        if self.device == 'cpu' and hasattr(self.mocr, 'model'):
            self.mocr.model.to('cpu')
        status.ok("MangaOCR loaded")

        status.info("Loading LaMa ONNX...")
        lama_path = os.path.join(os.path.dirname(__file__), "lama.onnx")
        if not os.path.exists(lama_path):
            status.warn("lama.onnx not found! Run download_lama.py first. Using OpenCV fallback.")
            self.lama_session = None
        else:
            lama_providers = ['CPUExecutionProvider']
            self.lama_session = ort.InferenceSession(lama_path, providers=lama_providers)
            status.ok("LaMa loaded")

        self.translation_cache = LRUCache(capacity=256)

        font_path = resolve_font_path()
        if font_path:
            status.ok(f"Font: {os.path.basename(font_path)}")
        else:
            status.warn("No font found!")
        _trim_ram()
        status.divider()

    def ceil_modulo(self, x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def pad_img_to_modulo(self, img, mod):
        channels, height, width = img.shape
        out_height = self.ceil_modulo(height, mod)
        out_width = self.ceil_modulo(width, mod)
        return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

    def run_lama(self, image, mask, use_gpu=False):
        if self.lama_session is None:
            status.warn("LaMa unavailable, using OpenCV inpaint fallback")
            return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

        # Re-create ONNX session on the right provider if needed
        current_provider = self.lama_session.get_providers()[0]
        desired_provider = 'CUDAExecutionProvider' if use_gpu else 'CPUExecutionProvider'
        if desired_provider != current_provider:
            lama_path = os.path.join(os.path.dirname(__file__), "lama.onnx")
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 6
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.lama_session = ort.InferenceSession(
                    lama_path, providers=[desired_provider, 'CPUExecutionProvider'], sess_options=opts)
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Modify image in-place — caller deletes img_bgr right after
        h_img, w_img = image.shape[:2]
        TILE_SIZE = 512  # cpu tile

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 10:
                continue

            comp_mask = (labels == i).astype(np.uint8) * 255

            pad = max(50, min(max(w, h) // 2, 200))

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            crop_img = image[y1:y2, x1:x2].copy()
            crop_mask = comp_mask[y1:y2, x1:x2]

            ch, cw = crop_img.shape[:2]

            result_bgr = crop_img.copy()

            for ty in range(0, ch, TILE_SIZE):
                for tx in range(0, cw, TILE_SIZE):
                    ty_end = min(ty + TILE_SIZE, ch)
                    tx_end = min(tx + TILE_SIZE, cw)

                    tile_mask = crop_mask[ty:ty_end, tx:tx_end]
                    if cv2.countNonZero(tile_mask) == 0:
                        continue

                    tile_img = result_bgr[ty:ty_end, tx:tx_end].copy()

                    tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
                    img_float = tile_rgb.astype('float32') / 255.0
                    mask_float = tile_mask.astype('float32') / 255.0

                    img_transpose = np.transpose(img_float, (2, 0, 1))
                    mask_expand = np.expand_dims(mask_float, 0)

                    img_padded = self.pad_img_to_modulo(img_transpose, 8)
                    mask_padded = self.pad_img_to_modulo(mask_expand, 8)

                    img_batch = np.expand_dims(img_padded, 0)
                    mask_batch = np.expand_dims(mask_padded, 0)

                    outputs = self.lama_session.run(None, {
                        'image': img_batch.astype(np.float32),
                        'mask': mask_batch.astype(np.float32)
                    })

                    res = outputs[0][0]
                    res = np.transpose(res, (1, 2, 0))
                    res = np.clip(res * 255, 0, 255).astype('uint8')
                    res = res[:tile_rgb.shape[0], :tile_rgb.shape[1], :]
                    res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

                    tile_mask_3d = np.expand_dims(tile_mask, axis=-1) / 255.0
                    result_bgr[ty:ty_end, tx:tx_end] = (res_bgr * tile_mask_3d + tile_img * (1 - tile_mask_3d)).astype(np.uint8)

                    del tile_rgb, img_float, mask_float, img_transpose, mask_expand
                    del img_padded, mask_padded, img_batch, mask_batch
                    del outputs, res, res_bgr, tile_mask_3d, tile_img

            crop_mask_3d = np.expand_dims(crop_mask, axis=-1) / 255.0
            image[y1:y2, x1:x2] = (result_bgr * crop_mask_3d + crop_img * (1 - crop_mask_3d)).astype(np.uint8)

            del comp_mask, crop_img, crop_mask, result_bgr, crop_mask_3d
            # Убраны частые вызовы gc.collect() и empty_cache() в цикле для ускорения инпейнтинга

        del labels, stats, mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image

    def translate_with_chain(self, text, engine='yandex'):
        cached = self.translation_cache.get(text)
        if cached:
            return cached

        cleaned = normalize_ocr_text(text)
        if not cleaned:
            self.translation_cache.put(text, "")
            return ""

        original_has_jp = contains_japanese(cleaned)
        original_has_kr = contains_korean(cleaned)

        if _is_predominantly_cyrillic(cleaned):
            self.translation_cache.put(text, cleaned)
            return cleaned

        try:
            if original_has_jp:
                result = translate_fallback(cleaned, 'ja', 'ru', preferred=engine)
            elif original_has_kr:
                result = translate_fallback(cleaned, 'ko', 'ru', preferred=engine)
            else:
                # English text from OCR is often ALL CAPS or mixed, which confuses translation grammar.
                # Lowercase and capitalize to get better syntax from the translation engine.
                cleaned_lower = cleaned.lower()
                cleaned_lower = cleaned_lower[0].upper() + cleaned_lower[1:] if cleaned_lower else ""
                result = translate_fallback(cleaned_lower, 'en', 'ru', preferred=engine)
                # Convert back to uppercase for comic styling
                result = result.upper()
        except Exception:
            try:
                result = ts.translate_text(cleaned, translator='google', from_language='auto', to_language='ru')
            except Exception:
                result = cleaned

        if (original_has_jp and contains_japanese(result)) or (original_has_kr and contains_korean(result)):
            result = "..."

        result = postprocess_translation(result)

        self.translation_cache.put(text, result)
        return result

    def free_resources(self):
        self.translation_cache.cache.clear()
        if getattr(self.detector, 'model', None) is not None:
            self.detector.model.to('cpu')
        if getattr(self, 'mocr', None) is not None and hasattr(self.mocr, 'model'):
            self.mocr.model.to('cpu')
        _free_easyocr()
        _trim_ram()

    @torch.no_grad()
    def process_image(self, img_array, font_path=None, engine='yandex',
                      outline_width=2, shrink_ratio=0.06, mode=None):
        """
        Main pipeline:
        1. Detect text regions with YOLO
        2. OCR with MangaOCR or EasyOCR
        3. Translate
        4. Inpaint with LaMa
        5. Render translated text with outline/shadow
        """
        is_manhva = mode and mode.startswith("Манхва")
        if is_manhva:
            # Manhva = Korean/English webtoon — skip MangaOCR, use EasyOCR
            self.device = 'cpu'

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        del img_array
        orig_h, orig_w = img_bgr.shape[:2]
        h, w = img_bgr.shape[:2]
        orig_bgr = img_bgr

        # Check actual free VRAM — adapt device per page
        free_vram = _get_free_vram()
        yolo_gpu = torch.cuda.is_available() and free_vram > 0.3
        use_gpu = False  # 4GB GPU shared with Gemma 3 — LaMa stays on CPU
        if not use_gpu and torch.cuda.is_available():
            status.detail(f"VRAM low ({free_vram:.1f}GB), LaMa on CPU, YOLO {'GPU' if yolo_gpu else 'CPU'}")

        # Put YOLO on the right device for this page
        if getattr(self.detector, 'model', None) is not None:
            self.detector.model.to('cuda' if yolo_gpu else 'cpu')

        status.step(1, 4, "YOLO text detection")
        # Увеличиваем imgsz для лучшего нахождения текста на высоких страницах (манхва),
        # слегка корректируем conf и iou для лучшего склеивания и фильтрации мусора.
        results = self.detector(img_bgr, conf=0.22, iou=0.45, verbose=False,
                                device='cuda' if yolo_gpu else 'cpu', imgsz=1024)
        status.timer("YOLO")

        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        text_regions = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            raw_boxes = []
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box[:4])
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)
                box_w = x_max - x_min
                box_h = y_max - y_min
                if box_w < 10 or box_h < 10:
                    continue
                raw_boxes.append((x_min, y_min, x_max, y_max))

                status.info(f"Raw detections: {len(raw_boxes)}")
                # Удаляем вложенные дубликаты рамок, оставляя только уникальные баблы
                merged_boxes = filter_overlapping_boxes(raw_boxes, iou_threshold=0.3)
                status.info(f"Boxes to process: {len(merged_boxes)}")

            # Check Qwen once for the whole page
            qwen_ok = _ping_qwen()
            if qwen_ok:
                status.info("Using Qwen vision for OCR (no MangaOCR/EasyOCR)")

            # Restore MangaOCR to GPU once before all OCR calls (only if Qwen unavailable)
            if not qwen_ok and use_gpu and hasattr(self.mocr, 'model'):
                self.mocr.model.to('cuda')
            elif hasattr(self.mocr, 'model'):
                self.mocr.model.to('cpu')

            # Scale factor from resized → original coordinates
            scale_x = orig_w / w
            scale_y = orig_h / h

            for ii, box in enumerate(merged_boxes):
                try:
                    x_min, y_min, x_max, y_max = box

                    # Расширяем маску пропорционально размеру бабла, 
                    # чтобы старый текст гарантированно закрашивался (устраняет "двоение")
                    bw, bh = x_max - x_min, y_max - y_min
                    px, py = max(2, int(bw * 0.08)), max(2, int(bh * 0.08))
                    mx1, my1 = max(0, x_min - px), max(0, y_min - py)
                    mx2, my2 = min(w, x_max + px), min(h, y_max + py)
                    cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)

                    is_vertical = is_vertical_box(box)

                    # Crop from original-resolution image for better OCR
                    ox_min = int(x_min * scale_x)
                    oy_min = int(y_min * scale_y)
                    ox_max = int(x_max * scale_x)
                    oy_max = int(y_max * scale_y)
                    crop_bgr = orig_bgr[oy_min:oy_max, ox_min:ox_max].copy()
                    if crop_bgr.size == 0:
                        continue

                    if is_vertical:
                        crop_bgr = cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE)

                    text = ""
                    already_translated = False
                    qwen_did_ocr = False
                    if qwen_ok:
                        # Мы передаем только сам бабл (crop_bgr) без контекста (context_bgr=None).
                        # Огромный контекст приводил к тому, что Qwen видела соседние баблы 
                        # и переводила их текст дважды (дублирование текста).
                        text, already_translated = _ocr_with_qwen(crop_bgr, context_bgr=None)
                        if text:
                            qwen_did_ocr = True
                            status.detail(f"[{ii}] Qwen {'(translated)' if already_translated else ''} -> '{text[:50]}'")
                        else:
                            status.detail(f"[{ii}] Qwen empty — skip")

                    if not qwen_did_ocr:
                        # If Qwen was available but returned empty, skip heavy OCR models
                        # EasyOCR (1GB) is only loaded if Qwen was never available
                        if not qwen_ok:
                            # EasyOCR for quick English text (only when Qwen is disabled)
                            easy_text = ""
                            easy_conf = 0.0
                            easy = _get_easyocr()
                            if easy:
                                try:
                                    processed = preprocess_for_ocr(crop_bgr, for_japanese=False)
                                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                                    easy_results = easy.readtext(processed_rgb, detail=1, paragraph=False)
                                    parts = []
                                    confs = []
                                    for r in easy_results:
                                        txt = r[1].strip()
                                        conf = r[2]
                                        if len(txt) > 1 and conf > 0.3:
                                            parts.append(txt)
                                            confs.append(conf)
                                    easy_text = " ".join(parts)
                                    if confs:
                                        easy_conf = sum(confs) / len(confs)
                                    del easy_results, processed, processed_rgb
                                except Exception as e:
                                    status.detail(f"EasyOCR fail in box {ii}: {e}")

                            if easy_conf > 0.3 and len(easy_text) > 1:
                                text = easy_text
                                status.detail(f"[{ii}] EasyOCR (EN) [{easy_conf:.2f}] -> '{easy_text[:50]}'")

                        # MangaOCR fallback for Japanese (only if not manhva)
                        if not text and not is_manhva:
                            if getattr(self, 'mocr', None) is None:
                                try:
                                    from manga_ocr import MangaOcr
                                    self.mocr = MangaOcr(force_cpu=(self.device == 'cpu'))
                                except Exception:
                                    status.warn("MangaOCR re-init failed")
                            try:
                                processed_jp = preprocess_for_ocr(crop_bgr, for_japanese=True)
                                pil_crop = Image.fromarray(cv2.cvtColor(processed_jp, cv2.COLOR_BGR2RGB))
                                mocr_text = self.mocr(pil_crop) or ""
                                mocr_stripped = mocr_text.strip()
                                if mocr_stripped and len(mocr_stripped) > 1:
                                    text = mocr_stripped
                                    status.detail(f"[{ii}] MangaOCR (fallback) -> '{mocr_stripped[:50]}'")
                                del pil_crop
                            except Exception:
                                pass

                        if not text:
                            status.detail(f"[{ii}] No OCR text")

                    text = normalize_ocr_text(text)
                    if not text or len(text) < 3:
                        continue

                    is_dark_bg = detect_background_type_deep(crop_bgr)
                    orig_text_color = detect_text_color(crop_bgr, is_dark_bg)

                    text_regions.append({
                        'bbox': (x_min, y_min, x_max, y_max),
                        'text': text,
                        'is_dark_bg': is_dark_bg,
                        'orig_text_color': orig_text_color,
                        'is_vertical': is_vertical,
                        'already_translated': already_translated,
                    })
                except Exception as e:
                    status.detail(f"[{ii}] OCR error: {e}")
                    continue
                finally:
                    try:
                        del processed, processed_rgb
                    except NameError:
                        pass
                    del crop_bgr

        del orig_bgr

        # Free MangaOCR if it wasn't used (Qwen handled OCR) or manhva mode
        if getattr(self, 'mocr', None) is not None:
            if qwen_ok or is_manhva:
                try:
                    if hasattr(self.mocr, 'model'):
                        del self.mocr.model
                except Exception:
                    pass
                del self.mocr
                self.mocr = None
                gc.collect()
            else:
                # Move to CPU before LaMa (was used for OCR)
                if hasattr(self.mocr, 'model'):
                    self.mocr.model.to('cpu')

        if getattr(self.detector, 'model', None) is not None:
            self.detector.model.to('cpu')

        try:
            del boxes
        except NameError:
            pass
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Увеличиваем дилатацию для надежного перекрытия краев баблов
        mask = cv2.dilate(mask, kernel, iterations=3)

        text_regions.sort(key=lambda r: (r['bbox'][1], -r['bbox'][0]))

        status.step(2, 4, f"Inpainting {len(text_regions)} regions")
        status.timer("Total OCR")
        gc.collect()
        torch.cuda.empty_cache()

        has_text = len(text_regions) > 0
        if not has_text:
            inpainted_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            try:
                inpainted_bgr = self.run_lama(img_bgr, mask, use_gpu=use_gpu)
            except Exception as e:
                status.err(f"LaMa OOM, falling back to OpenCV inpaint: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                inpainted_bgr = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
            inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
            del inpainted_bgr
        del img_bgr, mask
        status.timer("LaMa")

        pil_img = Image.fromarray(inpainted_rgb)
        del inpainted_rgb

        if not has_text:
            status.warn("No text detected — returning original image")
            return pil_img

        draw = ImageDraw.Draw(pil_img)

        if font_path is None:
            font_path = resolve_font_path()

        if font_path is None:
            status.err("No font available — skipping text overlay")
            return pil_img

        status.step(3, 4, f"Translation ({engine})")
        for idx, region in enumerate(text_regions):
            try:
                original_text = region['text']
                if region.get('already_translated', False):
                    ru_text = original_text
                    status.detail(f"[{idx}] Qwen already translated '{ru_text[:50]}'")
                else:
                    ru_text = self.translate_with_chain(original_text, engine=engine)
                # Skip if translation failed (still original language)
                if contains_japanese(ru_text) or contains_korean(ru_text):
                    status.warn(f"[{idx}] Still has JP/KR: '{ru_text[:30]}'")
                    continue

                is_dark_bg = region['is_dark_bg']
                is_vertical = region.get('is_vertical', False)

                style = detect_text_style(original_text)
                x_min, y_min, x_max, y_max = region['bbox']
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Dinamically adjust padding: 15% for normal text (mathematically perfect for inscribed ellipses)
                # and 5% for SFX/screams to make them huge
                actual_shrink = 0.05 if style == "sfx" else 0.15
                sx1, sy1, sx2, sy2 = shrink_bbox(x_min, y_min, x_max, y_max, w, h, actual_shrink)
                render_width = sx2 - sx1
                render_height = sy2 - sy1

                if is_vertical:
                    render_width, render_height = render_height, render_width

                max_font = 72 if style == "sfx" else 56
                font, wrapped_lines, font_size = find_optimal_font(
                    ru_text, font_path, render_width, render_height,
                    min_size=10, max_size=max_font,
                )
                status.detail(f"[{idx}] '{original_text[:30]}' → '{ru_text[:40]}' [{style}] f{font_size} l{len(wrapped_lines)}")

                center_x = sx1 + render_width / 2
                center_y = sy1 + render_height / 2

                text_to_draw = "\n".join(wrapped_lines)

                if style == "sfx":
                    text_color = (255, 255, 255)
                    outline_color = (0, 0, 0)
                else:
                    text_color = orig_text_color
                    luminance = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
                    outline_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

                render_text_with_outline(
                    draw, pil_img, text_to_draw, font,
                    center_x, center_y,
                    text_color=text_color,
                    outline_color=outline_color,
                    style=style,
                    box_width=render_width,
                    box_height=render_height,
                )
            except Exception as e:
                status.err(f"[{idx}] Render error: {e}")
                status.detail(traceback.format_exc().split('\n')[-3])

        status.step(4, 4, "Done")

        try:
            del draw
        except NameError:
            pass
        del text_regions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _trim_ram()

        return pil_img.convert("RGB")


translator_instance = None


def get_translator():
    global translator_instance
    if translator_instance is None:
        translator_instance = MangaTranslator()
    return translator_instance
