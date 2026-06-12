#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import transformers.utils.import_utils as import_utils
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sft_prompt(row: dict) -> str:
    if row.get("sft_prompt"):
        return str(row["sft_prompt"])
    clause = row.get("clause_text", "")
    analysis = row.get("gold_legal_analysis") or row.get("analysis") or ""
    evidence = row.get("evidence_summary") or ""
    steps = row.get("review_steps") or {}
    cn_step_keys = [
        "evidence_summary",
        "clause_evidence_alignment",
        "legal_consequence",
        "temporal_consequence",
        "risk_judgement",
        "s1_clause_summary",
        "s2_risk_type",
        "s3_selected_legal_evidence",
        "s4_temporal_alignment",
        "s5_evidence_use",
        "s6_gold_judgement",
    ]
    step_parts = []
    for key in cn_step_keys:
        value = steps.get(key)
        if value:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            step_parts.append(f"{key}: {value}")
    if row.get("evidence_language") == "en" or row.get("legal_kb") == "CUADRiskEnglishLegalKB":
        return (
            "You are a CUAD contract risk review model. Decide whether the clause contains legal or contractual risk.\n"
            "Label definition: 1=risk clause, 0=non-risk clause. The final answer must be one digit only.\n"
            f"Contract time anchor: {row.get('anchor_date')}\n"
            f"Risk type: {row.get('risk_type')}\n"
            f"Risk category: {row.get('risk_category')}\n"
            f"Clause: {clause}\n"
            f"Legal evidence: {row.get('evidence_text') or analysis or evidence}\n"
            f"Evidence-aware review steps: {' | '.join(step_parts) if step_parts else analysis}\n"
            "Answer:"
        )
    return (
        "你是草原承包合同风险审查模型。请根据合同条款判断是否存在法律风险。\n"
        "标签定义：1=有风险，0=无风险。最终答案只能输出一个数字。\n"
        f"合同时间锚点：{row.get('anchor_date')}\n"
        f"待审条款：{clause}\n"
        f"法律证据：{row.get('evidence_text') or analysis or evidence}\n"
        f"证据感知关键步骤：{' | '.join(step_parts) if step_parts else analysis or evidence}\n"
        "答案："
    )


class SftDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = sft_prompt(row)
        answer = "1" if int(row.get("label", 0)) else "0"
        if self.tokenizer.eos_token:
            answer += self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        max_prompt_len = max(1, self.max_length - len(answer_ids))
        prompt_ids = prompt_ids[-max_prompt_len:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    output = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        pad_value = pad_token_id if key == "input_ids" else 0
        if key == "labels":
            pad_value = -100
        tensors = []
        for item in batch:
            value = item[key]
            if value.numel() < max_len:
                value = torch.cat([value, torch.full((max_len - value.numel(),), pad_value, dtype=value.dtype)])
            tensors.append(value)
        output[key] = torch.stack(tensors)
    return output


def load_base_model(model_path: Path):
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    config.rope_scaling = None
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return tokenizer, model


def score_candidate(tokenizer, model, prompt: str, candidate: str, max_input_tokens: int) -> float:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)["input_ids"][0]
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    input_ids = torch.cat([prompt_ids, candidate_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : prompt_ids.numel()] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    return -float(out.loss.item())


def evaluate_label_score(tokenizer, model, rows: list[dict], max_input_tokens: int) -> dict:
    if not rows:
        return {"accuracy": 0.0, "total": 0}
    preds = []
    labels = []
    for row in rows:
        prompt = sft_prompt(row)
        yes = score_candidate(tokenizer, model, prompt, "1", max_input_tokens)
        no = score_candidate(tokenizer, model, prompt, "0", max_input_tokens)
        preds.append(1 if yes >= no else 0)
        labels.append(int(row.get("label", 0)))
    correct = sum(int(a == b) for a, b in zip(preds, labels))
    tp = sum(1 for a, b in zip(preds, labels) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(preds, labels) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(preds, labels) if a == 0 and b == 1)
    precision = tp / (tp + fp) * 100 if tp + fp else 0.0
    recall = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / len(labels) * 100,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(labels),
        "positive": sum(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/SFT train MiniCPM-2.4B for GrassRisk labels.")
    parser.add_argument("--dataset", default="GrassRiskExpandedEval")
    parser.add_argument("--model-path", default=str(ROOT / "models/minicpm_2_4b"))
    parser.add_argument("--output-dir", default=str(ROOT / "models/minicpm_sft_lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means full train split.")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 means full val split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = ROOT / "data/processed" / args.dataset
    train_rows = load_jsonl(data_dir / "train.jsonl")
    val_rows = load_jsonl(data_dir / "val.jsonl")
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples:
        val_rows = val_rows[: args.max_eval_samples]

    tokenizer, model = load_base_model(Path(args.model_path))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataset = SftDataset(train_rows, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    total_updates = math.ceil(len(loader) * args.epochs / args.gradient_accumulation_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    started = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    max_batches = int(math.ceil(len(loader) * args.epochs))
    batch_idx = 0
    while batch_idx < max_batches:
        for batch in loader:
            batch_idx += 1
            if batch_idx > max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                out = model(**batch, use_cache=False)
                loss = out.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu().item()) * args.gradient_accumulation_steps)
            if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == max_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                recent = losses[-args.gradient_accumulation_steps:]
                print(json.dumps({
                    "step": global_step,
                    "total_steps": total_updates,
                    "batch": batch_idx,
                    "loss": round(sum(recent) / len(recent), 4),
                    "elapsed_sec": round(time.time() - started, 1),
                }, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.eval()
    eval_metrics = evaluate_label_score(tokenizer, model, val_rows, args.max_length)
    payload = {
        "dataset": args.dataset,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_positive": sum(int(row.get("label", 0)) for row in train_rows),
        "val_positive": sum(int(row.get("label", 0)) for row in val_rows),
        "base_model_path": str(Path(args.model_path)),
        "adapter_path": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "eval": eval_metrics,
        "seconds": round(time.time() - started, 2),
    }
    write_json(ROOT / "outputs/minicpm_sft/train_metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
