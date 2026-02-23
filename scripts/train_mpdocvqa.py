"""DDP training for multi-page DocVQA – flat concat baseline (global-only, 256 tok/page).

Approach:
  1. Encode each page through the frozen encoder (SAM → Qwen2 → projector).
  2. Concatenate page visual tokens + text embeddings into `inputs_embeds`.
  3. Call the decoder with `inputs_embeds` (the model's internal vision branch is
     skipped via zero-valued dummy images).
"""

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

try:
    import wandb
except ImportError:
    wandb = None
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = TaskType = get_peft_model = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.mpdocvqa_data import MPDocVQADataset, MultiPageCollator, MultiPagePacker

# ── utilities ────────────────────────────────────────────────────────────────

def setup_dist():
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    r = int(os.environ.get("RANK", "0"))
    w = int(os.environ.get("WORLD_SIZE", "1"))
    if w > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(lr)
    return lr, r, w


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _find_vision_core(model: torch.nn.Module) -> torch.nn.Module:
    q = [model]
    seen: set = set()
    while q:
        c = q.pop(0)
        if c is None:
            continue
        cid = id(c)
        if cid in seen:
            continue
        seen.add(cid)
        if all(hasattr(c, a) for a in ("sam_model", "qwen2_model", "projector", "view_seperator")):
            return c
        q.append(getattr(c, "model", None))
        q.append(getattr(c, "base_model", None))
    raise AttributeError("Cannot locate vision core (sam_model/qwen2_model/projector/view_seperator)")


def freeze_vision_branch(model: torch.nn.Module) -> None:
    vc = _find_vision_core(model)
    for mod in (vc.sam_model, vc.qwen2_model, vc.projector):
        for p in mod.parameters():
            p.requires_grad = False
    vc.view_seperator.requires_grad = False


def maybe_enable_lora(model, args):
    if not args.use_lora:
        return model
    if get_peft_model is None:
        raise RuntimeError("pip install peft")
    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    return get_peft_model(model, cfg)


def save_ckpt(path: Path, model, opt, sched, step, epoch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if isinstance(model, DDP) else model
    torch.save({
        "step": step, "epoch": epoch,
        "model": core.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
    }, path)


# ── multi-page embedding builder ────────────────────────────────────────────

def _get_embed_fn(model):
    m = model.module if isinstance(model, DDP) else model
    return m.get_input_embeddings()


def build_multipage_embeds(batch, embed_fn, vision_core, device):
    """Replace visual placeholders with frozen encoder outputs."""
    input_ids = batch["input_ids"].to(device)
    inputs_embeds = embed_fn(input_ids)  # (B, L, D)
    B = input_ids.shape[0]

    # collect all pages across batch
    all_pages, info = [], []
    for b in range(B):
        for p_idx, (s, e) in enumerate(batch["vis_ranges"][b]):
            all_pages.append(batch["page_tensors"][b][p_idx])
            info.append((b, s, e))

    if not all_pages:
        return inputs_embeds

    all_pages_t = torch.stack(all_pages).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        feats = vision_core.sam_model(all_pages_t)
        feats = vision_core.qwen2_model(feats)
        feats = vision_core.projector(feats)           # (total, 256, D)
    sep = vision_core.view_seperator.detach()            # (D,)
    D = feats.shape[-1]

    for i, (b, s, e) in enumerate(info):
        vis = feats[i].reshape(-1, D)                    # (256, D)
        page_vis = torch.cat([vis, sep.unsqueeze(0)], 0) # (257, D)
        inputs_embeds[b, s:e] = page_vis

    return inputs_embeds


def _make_dummy_images(B, L, device):
    """Zero-valued dummy so the model's vision branch is skipped."""
    z = torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch.bfloat16)
    return (
        [(z, z)] * B,
        torch.zeros(B, L, dtype=torch.bool, device=device),
        torch.ones(B, 2, dtype=torch.long, device=device),
    )


def _forward(model, batch, embed_fn, vision_core, device):
    """Build inputs_embeds + forward through the model."""
    inputs_embeds = build_multipage_embeds(batch, embed_fn, vision_core, device)
    B, L, _ = inputs_embeds.shape
    dummy_imgs, dummy_mask, dummy_crop = _make_dummy_images(B, L, device)
    return model(
        input_ids=batch["input_ids"].to(device),
        inputs_embeds=inputs_embeds,
        labels=batch["labels"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        images=dummy_imgs,
        images_seq_mask=dummy_mask,
        images_spatial_crop=dummy_crop,
    )


# ── arg parser ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--imdb_dir", type=str, required=True, help="MP-DocVQA imdb dir with imdb_{split}.npy")
    p.add_argument("--images_dir", type=str, required=True, help="MP-DocVQA images directory")
    p.add_argument("--img_ext", type=str, default=".jpg")
    p.add_argument("--max_pages", type=int, default=20)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=500)
    # model
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--hf_model_id", type=str, default="", help="e.g. deepseek-ai/DeepSeek-OCR-2")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--hf_token", type=str, default="")
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    # training
    p.add_argument("--output_dir", type=str, default="outputs/mpdocvqa_ocr2")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,"
                           "o_proj,gate_proj,up_proj,down_proj")
    # W&B
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


# ── eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, loader, embed_fn, vision_core, device, world_size):
    model.eval()
    total = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    for batch in loader:
        with torch.autocast("cuda", torch.bfloat16):
            out = _forward(model, batch, embed_fn, vision_core, device)
        bs = batch["input_ids"].shape[0]
        total += out.loss.detach() * bs
        count += bs
    if world_size > 1:
        dist.all_reduce(total)
        dist.all_reduce(count)
    model.train()
    return (total / count.clamp_min(1.0)).item()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    local_rank, rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    out_dir = Path(args.output_dir)

    # W&B
    wb = None
    if is_main and args.wandb_project:
        if wandb is None:
            raise RuntimeError("pip install wandb")
        wb = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity or None,
            name=args.wandb_name or None, mode=args.wandb_mode,
            config=vars(args), dir=str(out_dir),
        )
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # ── model ────────────────────────────────────────────────────────────
    model_ref = args.hf_model_id.strip() or args.model_path
    hf_token = args.hf_token.strip() or os.environ.get("HF_TOKEN")
    cache_dir = args.cache_dir.strip() or None
    if is_main:
        print(f"loading model={model_ref}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, trust_remote_code=True, revision=args.revision,
        cache_dir=cache_dir, token=hf_token,
    )
    model = AutoModel.from_pretrained(
        model_ref, trust_remote_code=True, revision=args.revision,
        cache_dir=cache_dir, token=hf_token, use_safetensors=True,
        torch_dtype=torch.bfloat16, _attn_implementation=args.attn_implementation,
    )
    model = model.to(device)
    model = maybe_enable_lora(model, args)
    freeze_vision_branch(model)

    vision_core = _find_vision_core(model)  # ref before DDP

    if is_main:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"trainable={tr}  total={tot}  pct={100 * tr / max(tot, 1):.4f}%")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    embed_fn = _get_embed_fn(model)

    # ── data ─────────────────────────────────────────────────────────────
    packer = MultiPagePacker(tokenizer=tokenizer)
    collator = MultiPageCollator(packer)

    train_ds = MPDocVQADataset(args.imdb_dir, args.images_dir, "train",
                               args.max_pages, args.max_train_samples, args.img_ext)
    eval_ds = MPDocVQADataset(args.imdb_dir, args.images_dir, "val",
                              args.max_pages, args.max_eval_samples, args.img_ext)
    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_ds, world_size, rank, shuffle=False)
    train_loader = DataLoader(train_ds, args.train_batch_size, sampler=train_sampler,
                              collate_fn=collator, num_workers=args.num_workers,
                              pin_memory=True)
    eval_loader = DataLoader(eval_ds, args.eval_batch_size, sampler=eval_sampler,
                             collate_fn=collator, num_workers=args.num_workers,
                             pin_memory=True)

    # ── optimizer / scheduler ────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.epochs * steps_per_epoch,
    )

    # ── training loop ────────────────────────────────────────────────────
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            with torch.autocast("cuda", torch.bfloat16):
                out = _forward(model, batch, embed_fn, vision_core, device)
                loss = out.loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main and global_step % args.log_every == 0:
                    lv = loss.item() * args.grad_accum_steps
                    print(f"epoch={epoch} step={global_step} loss={lv:.4f}")
                    if wb:
                        wandb.log({"train/loss": lv, "train/lr": scheduler.get_last_lr()[0]},
                                  step=global_step)
                if is_main and global_step % args.save_every == 0:
                    save_ckpt(out_dir / f"ckpt_step_{global_step}.pt",
                              model, optimizer, scheduler, global_step, epoch)

        # end-of-epoch eval
        val_loss = evaluate_loss(model, eval_loader, embed_fn, vision_core, device, world_size)
        if is_main:
            print(f"epoch={epoch} val_loss={val_loss:.4f}")
            if wb:
                wandb.log({"val/loss": val_loss}, step=global_step)
            save_ckpt(out_dir / f"ckpt_epoch_{epoch}.pt",
                      model, optimizer, scheduler, global_step, epoch)

    # ── save final ───────────────────────────────────────────────────────
    if is_main:
        core = model.module if isinstance(model, DDP) else model
        core.save_pretrained(out_dir / "final_model")
        tokenizer.save_pretrained(out_dir / "final_model")
        if wb:
            wandb.finish()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
