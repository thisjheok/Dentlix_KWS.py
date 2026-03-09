import argparse
from dataclasses import fields
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import TinyKWSNet
from train_kws_znorm import CFG, KWSDataset, read_split_file


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained KWS model on test split.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_stageA.pt")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--test-list", type=str, default="splits/test.txt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA requested but unavailable, using CPU.")
        return "cpu"
    return device_arg


def load_cfg_from_checkpoint(ckpt_cfg: Dict) -> CFG:
    valid = {f.name for f in fields(CFG)}
    filtered = {k: v for k, v in ckpt_cfg.items() if k in valid}
    return CFG(**filtered)


@torch.no_grad()
def evaluate_with_confusion(model, loader, device: str, num_classes: int) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        pred = torch.argmax(logits, dim=1)
        loss_sum += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        total += y.size(0)

        for t, p in zip(y.view(-1), pred.view(-1)):
            conf[t.long(), p.long()] += 1

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, conf


def print_per_class_metrics(conf: torch.Tensor, labels: Tuple[str, ...]) -> None:
    print("\nper-class metrics:")
    print("label\tprecision\trecall\tf1\tsupport")
    for i, label in enumerate(labels):
        tp = conf[i, i].item()
        fp = conf[:, i].sum().item() - tp
        fn = conf[i, :].sum().item() - tp
        support = conf[i, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{label}\t{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}\t{support}")


def print_confusion_matrix(conf: torch.Tensor, labels: Tuple[str, ...]) -> None:
    print("\nconfusion matrix (rows=true, cols=pred):")
    header = "\t".join(["true\\pred", *labels])
    print(header)
    for i, label in enumerate(labels):
        row = "\t".join(str(v.item()) for v in conf[i])
        print(f"{label}\t{row}")


def main():
    args = parse_args()
    device = choose_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" not in ckpt:
        raise KeyError("checkpoint must contain key 'model'")

    if "cfg" in ckpt:
        cfg = load_cfg_from_checkpoint(ckpt["cfg"])
    else:
        cfg = CFG()

    labels = tuple(ckpt.get("labels", cfg.labels))
    cfg.root_dir = args.root_dir
    cfg.test_list = args.test_list
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.device = device
    cfg.labels = labels

    test_files = read_split_file(cfg.root_dir, cfg.test_list)
    ds_test = KWSDataset(cfg.root_dir, test_files, cfg.labels, cfg, train=False)
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = TinyKWSNet(num_classes=len(labels)).to(cfg.device)
    model.load_state_dict(ckpt["model"], strict=True)

    test_loss, test_acc, conf = evaluate_with_confusion(model, dl_test, cfg.device, len(labels))
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc * 100:.2f}%")
    print_per_class_metrics(conf, labels)
    print_confusion_matrix(conf, labels)


if __name__ == "__main__":
    main()
