import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image, ImageOps


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = 6,
    image_size: int = 768,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    return processed_images, target_aspect_ratio


class ImageTransform:
    def __init__(self) -> None:
        self.mean = (0.5, 0.5, 0.5)
        self.transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=self.mean, std=(0.5, 0.5, 0.5))]
        )

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        return self.transform(pil_img)


def build_prompt(question: str) -> str:
    return f"<image>\nQuestion: {question.strip()}\nAnswer: "


@dataclass
class OCR2Packer:
    tokenizer: object
    image_size: int = 768
    base_size: int = 1024
    crop_mode: bool = True
    min_crops: int = 2
    max_crops: int = 6
    ignore_id: int = -100

    def __post_init__(self) -> None:
        self.patch_size = 16
        self.downsample_ratio = 4
        self.image_token = "<image>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<｜▁pad▁｜>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_transform = ImageTransform()
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prepare_image_tokens(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        image = image.convert("RGB")
        images_crop_list: List[torch.Tensor] = []
        if self.crop_mode and (image.size[0] > self.image_size or image.size[1] > self.image_size):
            crop_images, crop_ratio = dynamic_preprocess(
                image, min_num=self.min_crops, max_num=self.max_crops, image_size=self.image_size
            )
            for crop in crop_images:
                images_crop_list.append(self.image_transform(crop))
        else:
            crop_ratio = (1, 1)

        global_view = ImageOps.pad(
            image,
            (self.base_size, self.base_size),
            color=tuple(int(x * 255) for x in self.image_transform.mean),
        )
        image_ori = self.image_transform(global_view).unsqueeze(0)
        images_spatial_crop = torch.tensor([crop_ratio[0], crop_ratio[1]], dtype=torch.long)
        if images_crop_list:
            image_crop = torch.stack(images_crop_list, dim=0)
        else:
            image_crop = torch.zeros((1, 3, self.base_size, self.base_size))

        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
        image_tokens = ([self.image_token_id] * num_queries_base) * num_queries_base
        image_tokens += [self.image_token_id]
        if crop_ratio[0] > 1 or crop_ratio[1] > 1:
            image_tokens += ([self.image_token_id] * (num_queries * crop_ratio[0])) * (
                num_queries * crop_ratio[1]
            )
        return image_crop, image_ori, images_spatial_crop, len(image_tokens)

    def pack(self, question: str, answer: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        prompt = build_prompt(question)
        image_crop, image_ori, images_spatial_crop, image_token_len = self._prepare_image_tokens(image)
        prompt_text = prompt.split(self.image_token)
        assert len(prompt_text) == 2, "Prompt must contain exactly one <image> token."

        prefix_ids = self._encode(prompt_text[0])
        suffix_ids = self._encode(prompt_text[1])
        answer_ids = self._encode(answer.strip())
        full_ids = [self.bos_id] + prefix_ids + ([self.image_token_id] * image_token_len) + suffix_ids + answer_ids + [self.eos_id]
        prompt_len = len([self.bos_id] + prefix_ids + ([self.image_token_id] * image_token_len) + suffix_ids)
        labels = torch.tensor(full_ids, dtype=torch.long)
        labels[:prompt_len] = self.ignore_id
        labels[labels == self.image_token_id] = self.ignore_id
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        images_seq_mask = input_ids.eq(self.image_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "image_crop": image_crop,
            "image_ori": image_ori,
            "answers": answer,
            "question": question,
        }


class DocVQASplit(torch.utils.data.Dataset):
    def __init__(self, split: str, max_samples: int = -1):
        ds = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        x = self.ds[idx]
        return {
            "question": x["question"],
            "answers": x["answers"],
            "image": x["image"].convert("RGB"),
        }


class DocVQACollator:
    def __init__(self, packer: OCR2Packer):
        self.packer = packer

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        packed = [self.packer.pack(x["question"], x["answers"][0], x["image"]) for x in batch]
        max_len = max(x["input_ids"].shape[0] for x in packed)

        input_ids: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        attention_mask: List[torch.Tensor] = []
        images_seq_mask: List[torch.Tensor] = []
        images: List[Tuple[torch.Tensor, torch.Tensor]] = []
        images_spatial_crop: List[torch.Tensor] = []
        answers: List[List[str]] = []
        questions: List[str] = []
        for sample, raw in zip(packed, batch):
            pad = max_len - sample["input_ids"].shape[0]
            input_ids.append(
                torch.cat(
                    [
                        sample["input_ids"],
                        torch.full((pad,), self.packer.pad_id, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [sample["labels"], torch.full((pad,), self.packer.ignore_id, dtype=torch.long)]
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(sample["input_ids"].shape[0], dtype=torch.long),
                        torch.zeros(pad, dtype=torch.long),
                    ]
                )
            )
            images_seq_mask.append(
                torch.cat(
                    [
                        sample["images_seq_mask"],
                        torch.zeros(pad, dtype=torch.bool),
                    ]
                )
            )
            images.append((sample["image_crop"], sample["image_ori"]))
            images_spatial_crop.append(sample["images_spatial_crop"])
            answers.append(raw["answers"])
            questions.append(raw["question"])

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "images_seq_mask": torch.stack(images_seq_mask, dim=0),
            "images_spatial_crop": torch.stack(images_spatial_crop, dim=0),
            "images": images,
            "answers": answers,
            "questions": questions,
        }

