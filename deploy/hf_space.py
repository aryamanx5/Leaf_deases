"""Upload the backend (what the Dockerfile copies) to a Hugging Face Docker Space.

The Space needs a README.md with a YAML header, which the project README does not have, so one is written into a
staging copy. Needs HF_TOKEN (write access) and HF_SPACE ("user/space-name") in the environment.
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", "requirements-api.txt"]
DIRS = ["api", "configs", "models", "results", "src"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "cnn_smoke.pt", "figures")
HEADER = """---
title: Patra API
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

FastAPI backend for the Patra leaf-disease demo (turmeric and black gram). Endpoints: `/api/health`, `/api/analyze`,
`/api/meta`, `/api/code`.
"""


def main() -> None:
    space = os.environ["HF_SPACE"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for f in FILES:
            shutil.copy(ROOT / f, out / f)
        for d in DIRS:
            shutil.copytree(ROOT / d, out / d, ignore=SKIP)
        (out / "README.md").write_text(HEADER)
        api.upload_folder(folder_path=str(out), repo_id=space, repo_type="space",
                          commit_message="Deploy from GitHub Actions")
    print(f"uploaded to https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
