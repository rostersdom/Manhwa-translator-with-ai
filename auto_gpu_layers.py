"""
Auto-detect optimal --n-gpu-layers for llama.cpp based on free VRAM.
Reserves VRAM for LaMa (~1200 MB) + YOLO (~500 MB) + overhead (~300 MB).
"""
import subprocess, re, os, sys

TOTAL_VRAM_GB = 4.0
RESERVE_MB = 1200 + 500 + 300  # LaMa + YOLO + overhead

def get_free_vram_mb() -> int:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        free, total = torch.cuda.mem_get_info()
        return free // (1024 * 1024)
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5, text=True
        )
        return int(out.strip().split('\n')[0])
    except Exception:
        pass
    return 0

def get_layers_for_vram(free_mb: int) -> int:
    if free_mb <= 0:
        return 0
    available = free_mb - RESERVE_MB
    if available <= 0:
        return 0
    layers = int(available / 240)
    return max(0, min(layers, 99))

def main():
    free_mb = get_free_vram_mb()
    if free_mb == 0:
        print("0", flush=True)
        return

    reserved = RESERVE_MB
    if free_mb < reserved:
        sys.stderr.write(
            f"WARNING: Low VRAM ({free_mb}MB free, need {reserved}MB for LaMa+YOLO). "
            f"Using CPU layers only.\n"
        )
        print("0", flush=True)
        return

    layers = get_layers_for_vram(free_mb)
    sys.stderr.write(
        f"VRAM: {free_mb}MB free | "
        f"Reserving {RESERVE_MB}MB (LaMa+YOLO+overhead) | "
        f"GPU layers: {layers}\n"
    )
    print(layers, flush=True)

if __name__ == "__main__":
    main()
