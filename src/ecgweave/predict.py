"""NPZ推理入口；使用者自行提供获授权的输入与可信权重。"""

import argparse
from importlib.resources import files
import json
from pathlib import Path

import numpy as np
import torch

from .pipeline import load_classifier, load_delineator, validate_signals, waveform_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--delineator", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default=1, type=int)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size须为正整数")
    if args.output.exists():
        raise FileExistsError(f"输出已存在：{args.output}")
    model = load_classifier(args.checkpoint, args.device)
    names = json.loads(files("ecgweave").joinpath("task_names71.json").read_text())
    if model.config.n_classes != len(names):
        raise ValueError("此入口固定使用71类模型及已登记顺序")
    with np.load(args.input, allow_pickle=False) as source:
        signals = torch.from_numpy(source["signals"])
        validate_signals(signals)
        cached = "reference" in source and "reference_support" in source
        if not cached and args.delineator is None:
            parser.error(
                "原波形输入须提供--delineator，或提供对齐reference/reference_support"
            )
        reference = torch.from_numpy(source["reference"]) if cached else None
        support = torch.from_numpy(source["reference_support"]) if cached else None
        sample_support = (
            torch.from_numpy(source["sample_support"])
            if "sample_support" in source
            else None
        )
    if cached and (
        reference.shape != (len(signals), 12, 250, 4)
        or support.shape != (len(signals), 12, 250)
        or reference.dtype != torch.float32
        or support.dtype != torch.bool
    ):
        raise ValueError(
            "reference须float32[B,12,250,4]；reference_support须bool[B,12,250]"
        )
    delineator = load_delineator(args.delineator, args.device) if not cached else None
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(signals), args.batch_size):
            rows = slice(start, start + args.batch_size)
            wave = signals[rows].to(args.device)
            if cached:
                batch = {
                    "signals": wave,
                    "reference": reference[rows].to(args.device),
                    "reference_support": support[rows].to(args.device),
                }
            else:
                valid = (
                    sample_support[rows].to(args.device)
                    if sample_support is not None
                    else None
                )
                batch = waveform_batch(wave, delineator, valid)
            outputs.append(model(batch).float().cpu())
    logits = torch.cat(outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            logits=logits.numpy(),
            probabilities=logits.sigmoid().numpy(),
            task_names=np.asarray(names),
        )
    print(f"已保存{len(logits)}条记录的71类研究输出：{args.output}")


if __name__ == "__main__":
    main()
