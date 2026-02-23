import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.docvqa_data import DocVQACollator, DocVQASplit, OCR2Packer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="/home/mossbee/project/deepseek_ocr_2_replication/DeepSeek-OCR-2-HuggingFace")
    p.add_argument("--hf_model_id", type=str, default="", help="Optional Hugging Face model id, e.g. deepseek-ai/DeepSeek-OCR-2")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--hf_token", type=str, default="", help="Optional HF token for gated/private models")
    p.add_argument("--output_dir", type=str, default="/home/mossbee/project/deepseek_ocr_2_replication/outputs/docvqa_ocr2")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_dist(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_vision_branch(model: torch.nn.Module) -> None:
    model_core = model.model
    for module in [model_core.sam_model, model_core.qwen2_model, model_core.projector]:
        for p in module.parameters():
            p.requires_grad = False
    model_core.view_seperator.requires_grad = False


def resolve_model_ref(args: argparse.Namespace) -> str:
    return args.hf_model_id.strip() or args.model_path


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "images_seq_mask": batch["images_seq_mask"].to(device, non_blocking=True),
        "images_spatial_crop": batch["images_spatial_crop"].to(device, non_blocking=True),
        "images": [],
    }
    for crop, ori in batch["images"]:
        out["images"].append((crop.to(device, dtype=torch.bfloat16), ori.to(device, dtype=torch.bfloat16)))
    return out


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, world_size: int) -> float:
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    for batch in loader:
        b = move_batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=b["input_ids"],
                labels=b["labels"],
                attention_mask=b["attention_mask"],
                images_seq_mask=b["images_seq_mask"],
                images_spatial_crop=b["images_spatial_crop"],
                images=b["images"],
            )
        loss_sum += out.loss.detach() * b["input_ids"].shape[0]
        count += b["input_ids"].shape[0]
    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    return (loss_sum / count.clamp_min(1.0)).item()


def save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    local_rank, rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model_ref = resolve_model_ref(args)
    hf_token = args.hf_token.strip() or os.environ.get("HF_TOKEN", None)
    cache_dir = args.cache_dir.strip() or None
    if is_main:
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
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    freeze_vision_branch(model)
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    packer = OCR2Packer(tokenizer=tokenizer)
    collator = DocVQACollator(packer)
    train_ds = DocVQASplit(split="train", max_samples=args.max_train_samples)
    eval_ds = DocVQASplit(split="validation", max_samples=args.max_eval_samples)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    max_steps = args.epochs * update_steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_steps,
    )

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            b = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=b["input_ids"],
                    labels=b["labels"],
                    attention_mask=b["attention_mask"],
                    images_seq_mask=b["images_seq_mask"],
                    images_spatial_crop=b["images_spatial_crop"],
                    images=b["images"],
                )
                loss = out.loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if is_main and global_step % args.log_every == 0:
                    print(f"epoch={epoch} step={global_step} loss={loss.item() * args.grad_accum_steps:.4f}")
                if is_main and global_step % args.save_every == 0:
                    save_ckpt(out_dir / f"checkpoint_step_{global_step}.pt", model, optimizer, scheduler, global_step, epoch)

        val_loss = evaluate_loss(model, eval_loader, device, world_size)
        if is_main:
            print(f"epoch={epoch} val_loss={val_loss:.4f}")
            save_ckpt(out_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer, scheduler, global_step, epoch)

    if is_main:
        core = model.module if isinstance(model, DDP) else model
        core.save_pretrained(out_dir / "final_model")
        tokenizer.save_pretrained(out_dir / "final_model")

    cleanup_dist(world_size)


if __name__ == "__main__":
    main()

