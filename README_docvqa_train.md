# DocVQA Fine-Tuning (DeepSeek-OCR-2)

## What this reproduces
- Practical downstream fine-tuning of released `DeepSeek-OCR-2` on DocVQA.
- Distributed PyTorch training (`torchrun`) with decoder-only updates.
- Generation-based validation script with ANLS metric.

## Install
Use the same environment as DeepSeek-OCR-2 plus:
- `datasets`

## Train (DDP)
```bash
cd /home/mossbee/project/deepseek_ocr_2_replication
torchrun --nproc_per_node=8 /home/mossbee/project/deepseek_ocr_2_replication/scripts/train_docvqa_ddp.py \
  --model_path /home/mossbee/project/deepseek_ocr_2_replication/DeepSeek-OCR-2-HuggingFace \
  --output_dir /home/mossbee/project/deepseek_ocr_2_replication/outputs/docvqa_ocr2 \
  --epochs 2 --train_batch_size 1 --grad_accum_steps 8
```

Or load weights directly from Hugging Face:
```bash
torchrun --nproc_per_node=8 /home/mossbee/project/deepseek_ocr_2_replication/scripts/train_docvqa_ddp.py \
  --hf_model_id deepseek-ai/DeepSeek-OCR-2 \
  --output_dir /home/mossbee/project/deepseek_ocr_2_replication/outputs/docvqa_ocr2
```

## Evaluate
```bash
python /home/mossbee/project/deepseek_ocr_2_replication/scripts/eval_docvqa.py \
  --model_path /home/mossbee/project/deepseek_ocr_2_replication/outputs/docvqa_ocr2/final_model \
  --split validation \
  --output_path /home/mossbee/project/deepseek_ocr_2_replication/outputs/docvqa_eval.json
```

Hub-based eval example:
```bash
python /home/mossbee/project/deepseek_ocr_2_replication/scripts/eval_docvqa.py \
  --hf_model_id deepseek-ai/DeepSeek-OCR-2 \
  --split validation \
  --output_path /home/mossbee/project/deepseek_ocr_2_replication/outputs/docvqa_eval.json
```

## Notes
- Uses OCR2-style image token packing (global view + optional dynamic local crops).
- Vision branch is frozen (`sam_model`, `qwen2_model`, `projector`); decoder is trainable.
- For smaller GPUs, reduce `--max_samples`, `--max_new_tokens`, and global batch.
- Optional Hub args: `--revision`, `--cache_dir`, `--hf_token` (or set `HF_TOKEN`).

