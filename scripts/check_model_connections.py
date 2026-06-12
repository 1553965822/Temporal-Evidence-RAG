#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    env.update({k: v for k, v in os.environ.items() if k not in env})
    return env


def mask(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def check_local_model(name: str, path_value: str | None) -> dict:
    if not path_value:
        return {"name": name, "status": "missing_path"}
    path = Path(path_value)
    if not path.exists():
        return {"name": name, "status": "path_not_found", "path": str(path)}
    config = path / "config.json"
    tokenizer_candidates = list(path.glob("tokenizer*")) + list(path.glob("vocab.*")) + list(path.glob("*.model"))
    weight_candidates = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    result = {
        "name": name,
        "status": "downloaded" if config.exists() and tokenizer_candidates and weight_candidates else "incomplete",
        "path": str(path),
        "has_config": config.exists(),
        "tokenizer_files": len(tokenizer_candidates),
        "weight_files": len(weight_candidates),
    }
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        result["transformers_load"] = "config_tokenizer_ok"
    except Exception as exc:
        result["transformers_load"] = f"failed: {type(exc).__name__}: {exc}"
    return result


def check_modelscope_api(env: dict[str, str]) -> dict:
    if env.get("ENABLE_MODELSCOPE_FREE_API", "false").lower() != "true":
        return {"name": "ModelScope Qwen3-8B API", "status": "disabled"}
    key = env.get("MODELSCOPE_API_KEY") or env.get("OPENAI_COMPATIBLE_API_KEY")
    if not key or key.startswith("replace_"):
        return {"name": "ModelScope Qwen3-8B API", "status": "missing_key"}
    base_url = env.get("OPENAI_COMPATIBLE_BASE_URL", "https://api-inference.modelscope.cn/v1").rstrip("/")
    url = base_url + "/chat/completions"
    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "只回复OK"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        else:
            content = ""
        return {
            "name": "ModelScope Qwen3-8B API",
            "status": "ok" if choices else "unexpected_response",
            "base_url": base_url,
            "key": mask(key),
            "response_preview": content[:80],
            "raw_keys": sorted(body.keys()) if isinstance(body, dict) else [],
            "raw_preview": json.dumps(body, ensure_ascii=False)[:300],
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:300]
        return {"name": "ModelScope Qwen3-8B API", "status": "http_error", "code": exc.code, "detail": detail}
    except Exception as exc:
        return {"name": "ModelScope Qwen3-8B API", "status": "failed", "detail": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    env = load_env(ROOT / ".env")
    results = [
        check_modelscope_api(env),
        check_local_model("Qwen3-8B", env.get("QWEN3_8B_MODEL_PATH")),
        check_local_model("MiniCPM-2.4B", env.get("MINICPM_2_4B_MODEL_PATH")),
        check_local_model("RoBERTa", env.get("ROBERTA_MODEL_PATH")),
        check_local_model("InternLM-Law-7B", env.get("INTERNLM_LAW_7B_MODEL_PATH")),
    ]
    out = ROOT / "outputs" / "model_connection_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    hard_fail = [r for r in results if r["name"] in {"Qwen3-8B", "MiniCPM-2.4B", "RoBERTa"} and r["status"] != "downloaded"]
    if hard_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
