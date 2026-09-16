import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_CONFIG = Path("configs/local_peft_2024plus.json")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ChatStyleDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.rows[index]["messages"]
        prompt_messages = messages[:-1]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        if all(label == -100 for label in labels) and labels:
            labels[-1] = full_ids[-1]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalChat:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local PEFT LoRA adapter for QQ style data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Load data/model and one batch, then exit.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    cache_dir = Path(config["cache_dir"])
    output_dir = Path(config["output_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    data_dir = Path(config["data_dir"])
    train_rows = read_jsonl(data_dir / config["train_file"])
    val_rows = read_jsonl(data_dir / config["val_file"])
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_rows = val_rows[: args.max_val_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        cache_dir=str(cache_dir / "models"),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ChatStyleDataset(train_rows, tokenizer, int(config["max_length"]))
    val_dataset = ChatStyleDataset(val_rows, tokenizer, int(config["max_length"]))

    dtype = resolve_dtype(str(config.get("dtype", "bf16")))
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        cache_dir=str(cache_dir / "models"),
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        target_modules=list(config["lora_target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = DataCollatorForCausalChat(tokenizer.pad_token_id)
    sample_batch = collator([train_dataset[0]])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "train_rows": len(train_rows),
                    "val_rows": len(val_rows),
                    "sample_tokens": int(sample_batch["input_ids"].shape[1]),
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config["num_train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        logging_steps=int(config["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(config["eval_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
