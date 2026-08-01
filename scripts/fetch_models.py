#!/usr/bin/env python3
"""Download the model weights kleincannon needs into ./models and ComfyUI's dirs.

Run after install_deps.sh:  python scripts/fetch_models.py

It downloads:
  * F5TTS-v1-Base-MLX-fp16  -> models/  (primary TTS engine, via speech-swift)
  * FLUX.2-klein-4B GGUF     -> ComfyUI/models/diffusion_models/ (image backend)

Weights are large (several GB). Re-run safely; existing files are skipped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
COMFY = ROOT.parent / "image" / "ComfyUI"   # default; override via --comfy

HF = "https://huggingface.co"


# Minimal resumable downloader (no extra deps).
def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "kleincannon"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = r.headers.get("Content-Length")
        got = 0
        while True:
            block = r.read(chunk)
            if not block:
                break
            f.write(block)
            got += len(block)
            if total:
                sys.stdout.write(f"\r  {dest.name}  {got // 1_048_576}/{int(total)//1_048_576} MB")
                sys.stdout.flush()
    tmp.replace(dest)
    print(f"\n  saved {dest}")


def _hf_download(repo: str, filename: str, dest: Path) -> None:
    # Hugging Face direct file URL (supports resolve/main). Use hf's CDN.
    url = f"{HF}/{repo}/resolve/main/{filename}"
    print(f"[fetch] {repo}/{filename}")
    if dest.exists():
        print(f"  exists, skip: {dest}")
        return
    try:
        _download(url, dest)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {e}")
        print(f"  download manually: {url}")
        print(f"  and place it at:  {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", type=Path, default=COMFY,
                    help="ComfyUI directory (for the FLUX.2-klein GGUF)")
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--gguf", default="flux-2-klein-4b-Q4_K_M.gguf",
                    help="FLUX.2-klein quant to fetch")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)

    if not args.skip_tts:
        # F5TTS MLX bundle (primary TTS engine)
        _hf_download(
            "aufklarer/F5TTS-v1-Base-MLX-fp16", "F5TTS_v1_Base/config.json",
            MODELS / "F5TTS-v1-Base-MLX-fp16" / "config.json",
        )
        # the actual model weights (filename varies by repo; fetch the safetensors)
        _hf_download(
            "aufklarer/F5TTS-v1-Base-MLX-fp16", "F5TTS_v1_Base/model.safetensors",
            MODELS / "F5TTS-v1-Base-MLX-fp16" / "model.safetensors",
        )
        # vocab for F5
        _hf_download(
            "aufklarer/F5TTS-v1-Base-MLX-fp16", "vocab.txt",
            MODELS / "F5TTS-v1-Base-MLX-fp16" / "vocab.txt",
        )

    if not args.skip_images:
        comfy = args.comfy
        dest_dir = comfy / "models" / "diffusion_models"
        _hf_download(
            "unsloth/FLUX.2-klein-4B-GGUF", args.gguf,
            dest_dir / args.gguf,
        )

    print("\nDone. Run `./kc doctor` to verify paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
