"""
Download runtime model artifacts from Hugging Face Hub.

Run on a fresh machine after installing requirements:
  python scripts/download_models.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


DEFAULT_OWNER = "GabCaiado"
BACKEND_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    env_var: str
    default_repo: str
    local_dir: Path


MODEL_SPECS = [
    ModelSpec(
        name="detector",
        env_var="HF_SLANG_DETECTOR_REPO",
        default_repo=f"{DEFAULT_OWNER}/english-app-slang-detector",
        local_dir=BACKEND_ROOT / "models" / "slang_detector",
    ),
    ModelSpec(
        name="sense-classifier",
        env_var="HF_SLANG_SENSE_CLASSIFIER_REPO",
        default_repo=f"{DEFAULT_OWNER}/english-app-slang-sense-classifier",
        local_dir=BACKEND_ROOT / "models" / "slang_sense_classifier",
    ),
    ModelSpec(
        name="normalizer",
        env_var="HF_SLANG_NORMALIZER_REPO",
        default_repo=f"{DEFAULT_OWNER}/english-app-slang-normalizer-v4-1-small",
        local_dir=BACKEND_ROOT / "models" / "slang_normalizer_v4_1_small",
    ),
]


def selected_specs(names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return MODEL_SPECS
    wanted = set(names)
    specs = [spec for spec in MODEL_SPECS if spec.name in wanted]
    missing = wanted - {spec.name for spec in specs}
    if missing:
        valid = ", ".join(spec.name for spec in MODEL_SPECS)
        raise SystemExit(f"Unknown model name(s): {', '.join(sorted(missing))}. Valid: {valid}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Download runtime models from Hugging Face Hub.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.name for spec in MODEL_SPECS],
        help="Download only selected model(s). Defaults to all.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hub revision, branch, tag, or commit hash. Defaults to HF_MODEL_REVISION or main.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading.")
    args = parser.parse_args()

    load_dotenv(BACKEND_ROOT / ".env")
    revision = args.revision or os.getenv("HF_MODEL_REVISION", "main")
    token = os.getenv("HF_TOKEN")

    for spec in selected_specs(args.only):
        repo_id = os.getenv(spec.env_var, spec.default_repo)
        print(f"{spec.name}: {repo_id} -> {spec.local_dir} @ {revision}")
        if args.dry_run:
            continue

        spec.local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(spec.local_dir),
            token=token,
        )

    print("Done.")


if __name__ == "__main__":
    main()
