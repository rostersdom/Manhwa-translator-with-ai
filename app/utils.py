import os
import re
from PIL import Image


BLACKLIST_KEYWORDS = [
    "avatar", "logo", "icon", "banner", "advert", "sponsor",
    "pixel", "tracker", "analytics", "emoji", "thumbnail",
    "thumb.", "sprite", "background", "footer", "header", "navbar",
    "social", "share", "like", "comment", "reply", "vote",
    "rating", "star", "loader", "profile", "userpic",
    "favicon", "widget", "overlay", "popup", "modal",
    "separator", "gradient", "button", "badge", "tag",
    "ribbon", "divider", "border", "teaser", "mini", "tiny",
    "captcha", "recaptcha", "watermark", "stamp",
    "coin", "point", "level", "rank",
    "blur", "blurred", "nsfw", "adult",
    "follow", "subscribe", "bell", "notification",
    "gravatar", "wp-post-image", "ts-post-image",
    "readerarea", "wewtwt", "cropped-",
    "-150x150", "-300x", "-150x", "-50x", "-32x32",
    "-180x", "-192x", "-270x", "-60x", "-96x",
]


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def filter_manga_images(image_paths: list[str], min_dimension: int = 100, min_file_kb: int = 10) -> list[str]:
    valid = []
    for path in image_paths:
        try:
            size_kb = os.path.getsize(path) / 1024
            if size_kb < min_file_kb:
                os.remove(path)
                continue
            with Image.open(path) as img:
                w, h = img.size
            if w < min_dimension or h < min_dimension:
                os.remove(path)
                continue
            valid.append(path)
        except Exception:
            try:
                os.remove(path)
            except Exception:
                pass
    return valid
