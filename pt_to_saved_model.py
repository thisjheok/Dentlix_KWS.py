import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from models import TinyKWSNet
from train_kws_znorm import CFG


def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError("checkpoint must contain key 'model'")

    labels = tuple(ckpt.get("labels", CFG().labels))
    cfg_dict = ckpt.get("cfg", {})

    cfg = CFG()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    model = TinyKWSNet(num_classes=len(labels))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def export_onnx(model, cfg: CFG, onnx_path: Path):
    # 1s, 16k, n_fft=400, hop=160, center=True => roughly 101 frames.
    # Use 101 as default time axis for this KWS pipeline.
    dummy = torch.randn(1, 1, cfg.n_mels, 101, dtype=torch.float32)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    except ModuleNotFoundError as exc:
        if exc.name != "onnxscript":
            raise
        # Fallback to legacy exporter for environments without onnxscript.
        try:
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
        except Exception as inner_exc:
            raise RuntimeError(
                "ONNX export failed because `onnxscript` is missing and legacy fallback also failed.\n"
                "Install missing deps: pip install onnx onnxscript"
            ) from inner_exc


def run_onnx2tf(onnx_path: Path, saved_model_dir: Path):
    if shutil.which("onnx2tf") is None:
        raise RuntimeError(
            "`onnx2tf` command not found.\n"
            "Install: pip install onnx2tf onnx tensorflow"
        )
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["onnx2tf", "-i", str(onnx_path), "-o", str(saved_model_dir)]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Export PyTorch checkpoint to TensorFlow SavedModel via ONNX.")
    p.add_argument("--checkpoint", type=str, default="saved_model/best_stageA.pt")
    p.add_argument("--onnx-out", type=str, default="saved_model/model.onnx")
    p.add_argument("--saved-model-dir", type=str, default="saved_model")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    onnx_out = Path(args.onnx_out)
    saved_model_dir = Path(args.saved_model_dir)

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(checkpoint)
    export_onnx(model, cfg, onnx_out)
    print(f"[OK] ONNX exported: {onnx_out}")

    run_onnx2tf(onnx_out, saved_model_dir)
    print(f"[OK] SavedModel exported under: {saved_model_dir}")


if __name__ == "__main__":
    main()
