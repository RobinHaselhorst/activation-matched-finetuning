"""
Download models from HuggingFace into the local models directory.

Run:
  python 1_setup/download_models.py

Env:
  RAF_MODELS_DIR   where to save the downloads (default: ./models)
  HF_TOKEN         HuggingFace token (needed for gated models)
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))


MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
]


def download_model(repo_id: str):
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(MODELS_DIR, repo_id)
    print(f"Downloading {repo_id} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Done: {repo_id}")


def list_models_dir():
    if not os.path.isdir(MODELS_DIR):
        print(f"{MODELS_DIR} does not exist")
        return
    for root, dirs, files in os.walk(MODELS_DIR):
        level = root.replace(MODELS_DIR, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root) or root}/")
        if level < 2:
            subindent = " " * 2 * (level + 1)
            for file in files:
                size = os.path.getsize(os.path.join(root, file))
                size_mb = size / (1024 * 1024)
                print(f"{subindent}{file} ({size_mb:.1f} MB)")


def main():
    # HuggingFace downloads are I/O-bound; sequential is reliable and avoids
    # hammering one disk with N concurrent writers. Switch to concurrent.futures
    # if you really want parallelism.
    for repo_id in MODELS:
        download_model(repo_id)

    print("\nModels dir contents:")
    list_models_dir()
    print("\nAll models downloaded successfully!")


if __name__ == "__main__":
    main()
