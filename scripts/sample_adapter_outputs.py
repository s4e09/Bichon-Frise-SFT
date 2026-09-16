import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Sample a trained PEFT adapter on style test rows.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", default="D:/bixiong_lora/output/style_lora_2024plus/final_adapter")
    parser.add_argument("--cache-dir", default="D:/bixiong_lora/cache/models")
    parser.add_argument("--test-file", default="E:/TencentFile/QQrecv/qq_style_2024plus/style_test.jsonl")
    parser.add_argument("--answers-file", default="E:/TencentFile/QQrecv/qq_style_2024plus/style_test_answers.jsonl")
    parser.add_argument("--indices", default="0,20,60,120,200")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.test_file))
    answers = {row["id"]: row["output"] for row in read_jsonl(Path(args.answers_file))}
    indices = [int(item) for item in args.indices.split(",") if item.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda")
    model.eval()

    outputs = []
    for index in indices:
        row = rows[index]
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.03,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        outputs.append(
            {
                "index": index,
                "id": row["id"],
                "time": row["time"],
                "context": messages[1]["content"],
                "reference": answers.get(row["id"], ""),
                "model_output": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            }
        )

    text = json.dumps(outputs, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
