import os
import requests
from tqdm import tqdm

def download_yolo():
    model_path = os.path.join(os.path.dirname(__file__), "comic-text-segmenter.pt")
    if os.path.exists(model_path):
        print("YOLO model already exists.")
        return model_path

    url = "https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m/resolve/main/comic-text-segmenter.pt"
    print(f"Downloading YOLO model from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as file, tqdm(
        desc="comic-text-segmenter.pt",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    print(f"Model downloaded to: {model_path}")
    return model_path

if __name__ == "__main__":
    download_yolo()