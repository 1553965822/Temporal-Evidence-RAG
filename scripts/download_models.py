#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]


MODEL_SPECS = {
    "qwen3": {
        "repo_id": "Qwen/Qwen3-8B",
        "local_dir": ROOT / "models" / "qwen3_8b",
        "env_name": "QWEN3_8B_MODEL_PATH",
        "allow_patterns": ["*.json", "*.model", "*.txt", "*.py", "*.safetensors", "*.bin", "*.tiktoken", "tokenizer.*", "merges.txt", "vocab.json"],
    },
    "minicpm": {
        "repo_id": "openbmb/MiniCPM-2B-sft-bf16",
        "local_dir": ROOT / "models" / "minicpm_2_4b",
        "env_name": "MINICPM_2_4B_MODEL_PATH",
        "allow_patterns": ["*.json", "*.model", "*.txt", "*.py", "*.safetensors", "*.bin", "tokenizer.*"],
    },
    "roberta": {
        "repo_id": "hfl/chinese-roberta-wwm-ext",
        "local_dir": ROOT / "models" / "chinese_roberta_wwm_ext",
        "env_name": "ROBERTA_MODEL_PATH",
        "allow_patterns": ["*.json", "*.model", "*.txt", "*.safetensors", "*.bin", "vocab.txt", "tokenizer.*"],
    },
    "internlm_law": {
        "repo_id": "internlm/internlm2-law-7b",
        "local_dir": ROOT / "models" / "internlm2_law_7b",
        "env_name": "INTERNLM_LAW_7B_MODEL_PATH",
        "allow_patterns": ["*.json", "*.model", "*.txt", "*.py", "*.safetensors", "*.bin", "tokenizer.*"],
    },
}


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def save_env(path: Path, values: dict[str, str]) -> None:
    current = load_env(path)
    current.update(values)
    ordered = [f"{key}={value}" for key, value in current.items()]
    path.write_text("\n".join(ordered) + "\n", encoding="utf-8")


def assert_model_exists(repo_id: str) -> bool:
    api = HfApi()
    try:
        api.model_info(repo_id)
        return True
    except Exception as exc:
        print(f"Model lookup failed for {repo_id}: {exc}")
        return False


def download_one(name: str) -> dict:
    spec = MODEL_SPECS[name]
    repo_id = spec["repo_id"]
    target = spec["local_dir"]
    env_updates: dict[str, str] = {}
    if name == "internlm_law" and not assert_model_exists(repo_id):
        raise RuntimeError(f"Model lookup failed for {repo_id}; no alternative model is downloaded automatically.")
    print(f"Downloading {name}: {repo_id} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=spec.get("allow_patterns"),
    )
    env_updates[spec["env_name"]] = str(target)
    save_env(ROOT / ".env", env_updates)
    return {
        "name": name,
        "repo_id": repo_id,
        "local_dir": str(target),
        "env_updates": list(env_updates.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free/local baseline models from Hugging Face.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=["qwen3", "minicpm", "roberta", "internlm_law"],
    )
    args = parser.parse_args()
    results = []
    for name in args.models:
        try:
            results.append(download_one(name))
        except Exception as exc:
            results.append({"name": name, "status": "failed", "error": str(exc)})
            print(f"Download failed for {name}: {exc}")
    out = ROOT / "outputs" / "model_download_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
