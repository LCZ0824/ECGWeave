"""正式尺寸的合成前反向与资源检查，不更新参数、不读取患者数据。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch

from .diagnosis.model import JointMultiresClassifier, ModelConfig as DiagnosisConfig
from .delineation.model import build_segmenter
from .pipeline import pool_reference
from .pretraining.core import LossConfig, ModelConfig
from .pretraining.model import ValidECGMAE, make_valid_mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    torch.backends.cuda.matmul.allow_tf32 = False
    signals = torch.randn(1, 12, 5000, device=device)
    report = {
        "device": str(device),
        "torch": torch.__version__,
        "input_shape": list(signals.shape),
        "patient_data_read": False,
        "optimizer_steps": 0,
        "components": {},
    }
    for name in ("pretraining", "delineation", "diagnosis"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model = {
            "pretraining": lambda: ValidECGMAE(ModelConfig()),
            "delineation": build_segmenter,
            "diagnosis": lambda: JointMultiresClassifier(DiagnosisConfig()),
        }[name]().to(device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            if name == "pretraining":
                valid = torch.ones(1, 250, dtype=torch.bool, device=device)
                mask = make_valid_mask(valid, 0.75, "limbsync_mixed")
                output = model(signals, mask, valid)
                loss = model.loss(
                    output, signals, mask, LossConfig("local_softnorm"), valid
                )
            elif name == "delineation":
                output = model(signals)
                loss = output.float().square().mean()
                reference, reference_support = pool_reference(
                    output.detach().float().softmax(1), signals
                )
            else:
                output = model(
                    {
                        "signals": signals,
                        "reference": reference,
                        "reference_support": reference_support,
                    }
                )
                loss = output.float().square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if (
            not torch.isfinite(output).all()
            or not torch.isfinite(loss)
            or not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
        ):
            raise RuntimeError(f"{name}前反向出现无效输出或梯度")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["components"][name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "output_shape": list(output.shape),
            "loss": float(loss.detach()),
            "seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None,
        }
        del model, output, loss, gradients
    report["status"] = "passed"
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
