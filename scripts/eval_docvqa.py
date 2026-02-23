import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.docvqa_data import DocVQASplit, OCR2Packer, build_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="/home/mossbee/project/deepseek_ocr_2_replication/DeepSeek-OCR-2-HuggingFace")
    p.add_argument("--hf_model_id", type=str, default="", help="Optional Hugging Face model id, e.g. deepseek-ai/DeepSeek-OCR-2")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--hf_token", type=str, default="", help="Optional HF token for gated/private models")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--output_path", type=str, default="/home/mossbee/project/deepseek_ocr_2_replication/outputs/docvqa_eval.json")
    p.add_argument("--max_new_tokens", type=int, default=64)
    return p.parse_args()


def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def anls_score(pred: str, answers: List[str], tau: float = 0.5) -> float:
    pred_n = normalize_text(pred)
    best = 0.0
    for gt in answers:
        gt_n = normalize_text(gt)
        if not gt_n and not pred_n:
            best = max(best, 1.0)
            continue
        d = edit_distance(pred_n, gt_n)
        denom = max(len(pred_n), len(gt_n), 1)
        sim = 1.0 - (d / denom)
        best = max(best, sim if sim >= tau else 0.0)
    return best


def resolve_model_ref(args: argparse.Namespace) -> str:
    return args.hf_model_id.strip() or args.model_path


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ref = resolve_model_ref(args)
    hf_token = args.hf_token.strip() or os.environ.get("HF_TOKEN", None)
    cache_dir = args.cache_dir.strip() or None
    print(f"loading_model={model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=True,
        revision=args.revision,
        cache_dir=cache_dir,
        token=hf_token,
    )
    model = AutoModel.from_pretrained(
        model_ref,
        trust_remote_code=True,
        revision=args.revision,
        cache_dir=cache_dir,
        token=hf_token,
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        _attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    ).to(device)
    model.eval()
    packer = OCR2Packer(tokenizer=tokenizer)
    ds = DocVQASplit(split=args.split, max_samples=args.max_samples)

    preds = []
    with torch.no_grad():
        for i in range(len(ds)):
            ex = ds[i]
            packed = packer.pack(ex["question"], "", ex["image"])
            prompt = build_prompt(ex["question"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = packed["input_ids"][:-1].unsqueeze(0).to(device)
            images_seq_mask = packed["images_seq_mask"][:-1].unsqueeze(0).to(device)
            images_spatial_crop = packed["images_spatial_crop"].unsqueeze(0).to(device)
            images = [(
                packed["image_crop"].to(device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                packed["image_ori"].to(device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=True,
                )
            gen = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
            stop = "<｜end▁of▁sentence｜>"
            if stop in gen:
                gen = gen.split(stop, 1)[0]
            gen = gen.strip()
            score = anls_score(gen, ex["answers"])
            preds.append(
                {
                    "id": i,
                    "question": ex["question"],
                    "prediction": gen,
                    "answers": ex["answers"],
                    "anls": score,
                    "prompt_token_count": len(prompt_ids),
                }
            )
            if (i + 1) % 50 == 0:
                print(f"processed={i + 1}/{len(ds)}")

    mean_anls = sum(x["anls"] for x in preds) / max(len(preds), 1)
    out = {"split": args.split, "num_samples": len(preds), "mean_anls": mean_anls, "predictions": preds}
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean_anls={mean_anls:.4f}")
    print(f"saved={os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()

