#!/usr/bin/env python
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import torch
import transformers.utils.import_utils as import_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MINICPM_2_4B_MODEL_PATH", ROOT / "models/minicpm_2_4b"))
OUT_PATH = ROOT / "outputs/model_smoke_tests/minicpm_2_4b_smoke.json"


def main() -> None:
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False

    result = {
        "model": "MiniCPM-2.4B",
        "model_path": str(MODEL_PATH),
        "cuda_available": torch.cuda.is_available(),
        "status": "not_started",
    }
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        # Newer Transformers normalizes RoPE metadata in a way this older MiniCPM
        # remote-code implementation does not expect.
        config.rope_scaling = None
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
        prompt = "请判断条款是否存在草原承包合同风险：甲方可随时单方解除合同，乙方不得要求赔偿。只回答有风险或无风险。"
        encoded = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=16,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        result.update(
            {
                "status": "ok",
                "device": str(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "free_vram_gb_after_load": round(torch.cuda.mem_get_info(0)[0] / 1024**3, 3) if torch.cuda.is_available() else None,
                "prompt": prompt,
                "output": text,
            }
        )
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
