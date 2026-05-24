import os
import requests
from tqdm import tqdm

def download_lama_onnx():
    model_path = os.path.join(os.path.dirname(__file__), "lama.onnx")
    if os.path.exists(model_path):
        print("Model already exists.")
        return model_path

    url = "https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"
    print(f"Downloading compatible LaMa ONNX model from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as file, tqdm(
        desc=model_path,
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
    download_lama_onnx()