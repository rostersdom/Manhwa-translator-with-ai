import os
import requests
from tqdm import tqdm

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
os.makedirs(FONT_DIR, exist_ok=True)

FONT_SOURCES = {
    "NotoSans-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf",
    "NotoSansSC-Regular.otf": "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/03_NotoSansCJKsc.zip",
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  Already exists: {os.path.basename(dest_path)}")
        return True
    try:
        print(f"  Downloading {os.path.basename(dest_path)}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total, unit='B', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(1024):
                f.write(data)
                bar.update(len(data))
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def download_fonts():
    downloaded = []
    for name, url in FONT_SOURCES.items():
        dest = os.path.join(FONT_DIR, name)
        if name.endswith(".zip"):
            import zipfile
            import io
            extracted_name = name.replace(".zip", ".otf")
            extracted_path = os.path.join(FONT_DIR, extracted_name)
            if os.path.exists(extracted_path):
                print(f"  Already extracted: {extracted_name}")
                downloaded.append(extracted_path)
                continue
            try:
                print(f"  Downloading {name}...")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                otf_files = [f for f in z.namelist() if f.endswith(".otf") and "NotoSansCJKsc" in f and "Regular" in f]
                if otf_files:
                    z.extract(otf_files[0], FONT_DIR)
                    src = os.path.join(FONT_DIR, otf_files[0])
                    if os.path.exists(src):
                        os.rename(src, extracted_path)
                    print(f"  Extracted to {extracted_name}")
                    downloaded.append(extracted_path)
                else:
                    print(f"  No OTF found in zip for {name}")
            except Exception as e:
                print(f"  Failed to download/extract {name}: {e}")
        else:
            if download_file(url, dest):
                downloaded.append(dest)
    return downloaded

if __name__ == "__main__":
    print("Downloading fonts...")
    paths = download_fonts()
    print(f"Downloaded {len(paths)} fonts: {[os.path.basename(p) for p in paths]}")
