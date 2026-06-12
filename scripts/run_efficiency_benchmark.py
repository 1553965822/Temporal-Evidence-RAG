#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_minicpm_rag_evaluation as rag_eval  # noqa: E402


DEFAULT_METHODS = [
    "Expert-only",
    "MiniCPM-SFT",
    "Temporal-RAG + Evidence-RAG",
]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    prompt_method: str
    base_path: Path | None
    adapter_path: Path | None
    loader: str


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    env.update(os.environ)
    return env


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_suffix(value: str) -> str:
    value = value.strip() or time.strftime("%Y%m%d_%H%M%S")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def resolve_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def build_method_specs(args: argparse.Namespace, env: dict[str, str]) -> list[MethodSpec]:
    minicpm_path = first_existing(
        [
            resolve_path(args.minicpm_model_path),
            resolve_path(env.get("MINICPM_2_4B_MODEL_PATH")),
            ROOT / "models" / "minicpm_2_4b",
        ]
    )
    sft_adapter = first_existing(
        [
            resolve_path(args.sft_adapter_path),
            resolve_path(env.get("MINICPM_SFT_ADAPTER_PATH")),
            ROOT / "models" / "minicpm_sft_lora_grassrisk_user",
            ROOT / "models" / "minicpm_sft_lora",
        ]
    )
    evidence_adapter = first_existing(
        [
            resolve_path(args.evidence_adapter_path),
            resolve_path(env.get("EVIDENCE_RAG_ADAPTER_PATH")),
            sft_adapter,
        ]
    )
    expert_path = first_existing(
        [
            resolve_path(args.expert_model_path),
            resolve_path(env.get("EXPERT_13B_MODEL_PATH")),
            resolve_path(env.get("EXPERT_MODEL_PATH")),
        ]
    )

    specs: list[MethodSpec] = []
    for method in args.methods:
        if method == "Expert-only":
            if expert_path:
                specs.append(MethodSpec(method, "MiniCPM-2.4B Direct", expert_path, None, "generic_causal_lm"))
        elif method == "MiniCPM-SFT":
            if minicpm_path and sft_adapter:
                specs.append(MethodSpec(method, "MiniCPM-SFT", minicpm_path, sft_adapter, "minicpm_lora"))
        elif method == "Temporal-RAG + Evidence-RAG":
            if minicpm_path and evidence_adapter:
                specs.append(MethodSpec(method, "Temporal-RAG + Evidence-RAG", minicpm_path, evidence_adapter, "minicpm_lora"))
        else:
            raise ValueError(f"Unknown efficiency method: {method}")
    return specs


def load_generic_causal_lm(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    return tokenizer, model


def load_method_model(spec: MethodSpec):
    if spec.loader == "minicpm_lora":
        assert spec.base_path is not None
        assert spec.adapter_path is not None
        return rag_eval.load_minicpm_with_adapter(spec.base_path, spec.adapter_path)
    if spec.loader == "generic_causal_lm":
        assert spec.base_path is not None
        return load_generic_causal_lm(spec.base_path)
    raise ValueError(f"Unknown loader: {spec.loader}")


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params_b(model) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1e9


def prepare_prompts(args: argparse.Namespace) -> tuple[list[dict], dict]:
    dataset_dir = ROOT / "data" / "processed" / args.dataset
    train_dir = ROOT / "data" / "processed" / (args.train_dataset or args.dataset)
    train_rows = load_jsonl(train_dir / "train.jsonl")
    eval_path = resolve_path(args.eval_file) if args.eval_file else dataset_dir / f"{args.split}.jsonl"
    if not eval_path:
        raise ValueError("Evaluation path cannot be empty.")
    eval_rows = load_jsonl(eval_path)
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    law_kb_path = resolve_path(args.law_kb) if args.law_kb else rag_eval.default_law_kb_path(args.dataset)
    laws = load_jsonl(law_kb_path)
    retriever = rag_eval.TfidfRetriever(laws, train_rows)

    prompt_cache: dict[str, list[dict]] = {}
    for method in args.methods:
        prompt_method = "MiniCPM-SFT" if method == "MiniCPM-SFT" else method
        if method == "Expert-only":
            prompt_method = "MiniCPM-2.4B Direct"
        rows: list[dict] = []
        for sample in eval_rows:
            prompt, retrieved_laws, examples = rag_eval.build_context(prompt_method, sample, retriever, label_style=args.label_style)
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "label": int(sample.get("label", 0)),
                    "prompt": prompt,
                    "retrieved_law_ids": [law.get("law_id") for law in retrieved_laws],
                    "retrieved_law_temporal_states": [law.get("_temporal_state") for law in retrieved_laws],
                    "retrieved_example_ids": [example.get("sample_id") for example in examples],
                    "anchor_date": rag_eval.extract_anchor_date(sample),
                }
            )
        prompt_cache[method] = rows

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_file": str(eval_path),
        "eval_size": len(eval_rows),
        "train_dataset": args.train_dataset or args.dataset,
        "train_size": len(train_rows),
        "law_kb_path": str(law_kb_path),
        "law_kb_size": len(laws),
        "retrieval_time_excluded": True,
        "prompt_prepared_before_timing": True,
    }
    return prompt_cache, metadata


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_once(tokenizer, model, prompt: str, max_input_tokens: int, max_new_tokens: int) -> dict:
    device = model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = int(encoded["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_tokens = int(generated.shape[-1] - input_len)
    text = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()
    return {"input_tokens": input_len, "output_tokens": output_tokens, "output_text": text}


def benchmark_method(
    spec: MethodSpec,
    prompt_rows: list[dict],
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    tokenizer, model = load_method_model(spec)
    try:
        params_b = count_params_b(model)
        warmup_rows = prompt_rows[: min(args.warmup, len(prompt_rows))]
        for row in warmup_rows:
            _ = generate_once(tokenizer, model, row["prompt"], args.max_input_tokens, args.max_new_tokens)
        cuda_sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        timing_rows: list[dict] = []
        started = time.perf_counter()
        for row in prompt_rows:
            cuda_sync()
            item_start = time.perf_counter()
            generation = generate_once(tokenizer, model, row["prompt"], args.max_input_tokens, args.max_new_tokens)
            cuda_sync()
            latency_ms = (time.perf_counter() - item_start) * 1000
            timing_rows.append(
                {
                    "method": spec.method,
                    "sample_id": row.get("sample_id"),
                    "label": row.get("label"),
                    "latency_ms": round(latency_ms, 3),
                    "input_tokens": generation["input_tokens"],
                    "output_tokens": generation["output_tokens"],
                    "output_text": generation["output_text"],
                    "anchor_date": row.get("anchor_date"),
                    "retrieved_law_ids": row.get("retrieved_law_ids", []),
                    "retrieved_law_temporal_states": row.get("retrieved_law_temporal_states", []),
                    "retrieved_example_ids": row.get("retrieved_example_ids", []),
                }
            )
            print(
                json.dumps(
                    {
                        "method": spec.method,
                        "sample_id": row.get("sample_id"),
                        "latency_ms": round(latency_ms, 3),
                    },
                    ensure_ascii=False,
                )
            )
        total_seconds = time.perf_counter() - started
        latencies = [row["latency_ms"] for row in timing_rows]
        peak_mem_gb = 0.0
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        summary = {
            "method": spec.method,
            "params_b": round(params_b, 3),
            "latency_ms_per_clause": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "throughput_clause_per_s": round(len(timing_rows) / total_seconds, 3) if total_seconds > 0 else 0.0,
            "gpu_mem_gb": round(peak_mem_gb, 3) if torch.cuda.is_available() else None,
            "samples": len(timing_rows),
            "base_model_path": str(spec.base_path) if spec.base_path else "",
            "adapter_path": str(spec.adapter_path) if spec.adapter_path else "",
        }
        return summary, timing_rows
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_markdown(payload: dict, path: Path) -> None:
    lines = [f"# Efficiency Benchmark - {payload['metadata']['dataset']}", ""]
    lines.append("Retrieval and prompt construction are completed before timing; the table measures generation-side per-clause inference cost only.")
    lines.append("")
    lines.append("| Method | Params/B | Latency/ms per clause | Throughput/clause/s | GPU Mem/GB | Samples |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["results"]:
        gpu_mem = "-" if row["gpu_mem_gb"] is None else f"{row['gpu_mem_gb']:.3f}"
        lines.append(
            f"| {row['method']} | {row['params_b']:.3f} | {row['latency_ms_per_clause']:.3f} | "
            f"{row['throughput_clause_per_s']:.3f} | {gpu_mem} | {row['samples']} |"
        )
    lines.append("")
    lines.append("## Metadata")
    for key, value in payload["metadata"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real generation-side deployment efficiency for Table 13-style experiments.")
    parser.add_argument("--dataset", default="GrassRisk")
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--law-kb", default="")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--label-style", choices=["numeric", "text"], default="numeric")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--expert-model-path", default="")
    parser.add_argument("--minicpm-model-path", default="")
    parser.add_argument("--sft-adapter-path", default="")
    parser.add_argument("--evidence-adapter-path", default="")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and report loadable rows without loading models.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    args = parser.parse_args()

    env = load_env()
    prompt_cache, metadata = prepare_prompts(args)
    specs = build_method_specs(args, env)

    suffix = safe_suffix(args.output_suffix)
    out_dir = ROOT / "outputs" / "efficiency_benchmark" / args.dataset / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata.update(
        {
            "methods_requested": args.methods,
            "methods_selected": [spec.method for spec in specs],
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "dry_run": args.dry_run,
        }
    )

    if args.dry_run:
        payload = {"metadata": metadata, "results": []}
        (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(payload, out_dir / "results.md")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    summaries: list[dict] = []
    timing_rows: list[dict] = []
    for spec in specs:
        try:
            summary, rows = benchmark_method(spec, prompt_cache[spec.method], args)
        except Exception as exc:
            print(f"{spec.method} benchmark failed, omitting result row: {type(exc).__name__}: {exc}")
            continue
        summaries.append(summary)
        timing_rows.extend(rows)
        metadata["methods_measured"] = [row["method"] for row in summaries]
        write_jsonl(out_dir / "timings.jsonl", timing_rows)
        payload = {"metadata": metadata, "results": summaries}
        (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(payload, out_dir / "results.md")

    payload = {"metadata": metadata, "results": summaries}
    metadata["methods_measured"] = [row["method"] for row in summaries]
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "timings.jsonl", timing_rows)
    write_markdown(payload, out_dir / "results.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
