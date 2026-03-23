import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

from train_kws_BN import CFG, KWSDataset, infer_labels, read_split_file


def _shape_compatible(candidate_shape: tuple[int, ...], spec_shape: list[int | None]) -> bool:
    if len(candidate_shape) != len(spec_shape):
        return False
    for got, want in zip(candidate_shape, spec_shape):
        if want is None or want == -1:
            continue
        if got != want:
            return False
    return True


def _format_feature_for_signature(feature, input_shape: list[int | None]) -> np.ndarray:
    c_hw = feature.numpy().astype(np.float32)
    b_chw = feature.unsqueeze(0).numpy().astype(np.float32)
    b_hwc = np.transpose(b_chw, (0, 2, 3, 1))

    for arr in (c_hw, b_chw, b_hwc):
        if _shape_compatible(arr.shape, input_shape):
            return arr

    raise ValueError(
        f"No compatible representative sample shape for input spec {input_shape}. "
        f"Tried {[arr.shape for arr in (c_hw, b_chw, b_hwc)]}"
    )


def representative_dataset(
    cfg: CFG, split_file: str, max_samples: int, input_shape: list[int | None]
) -> Iterator[list[np.ndarray]]:
    files = read_split_file(cfg.root_dir, split_file)
    dataset = KWSDataset(cfg.root_dir, files, cfg.labels, cfg, train=False)

    for idx in range(min(max_samples, len(dataset))):
        feature, _ = dataset[idx]
        yield [_format_feature_for_signature(feature, input_shape)]


def get_saved_model_input_shape(tf_module, saved_model_dir: Path) -> list[int | None]:
    loaded = tf_module.saved_model.load(str(saved_model_dir))
    signatures = loaded.signatures
    if not signatures:
        raise RuntimeError("No signatures found in SavedModel.")

    fn = signatures.get("serving_default") or next(iter(signatures.values()))
    _, kwargs = fn.structured_input_signature
    if not kwargs:
        raise RuntimeError("Cannot find keyword input signature in SavedModel.")

    first_spec = next(iter(kwargs.values()))
    return first_spec.shape.as_list()


def check_saved_model_dir(saved_model_dir: Path) -> None:
    if not saved_model_dir.exists():
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")
    if not (saved_model_dir / "saved_model.pb").exists():
        pt_files = list(saved_model_dir.glob("*.pt"))
        hint = ""
        if pt_files:
            hint = (
                f"\nFound PyTorch checkpoint(s): {[p.name for p in pt_files]}\n"
                "This script converts TensorFlow SavedModel to TFLite. "
                "Convert/export your PyTorch model to TensorFlow SavedModel first."
            )
        raise FileNotFoundError(f"`saved_model.pb` not found in: {saved_model_dir}{hint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to fully INT8 TFLite.")
    parser.add_argument("--saved-model-dir", type=str, default="saved_model")
    parser.add_argument("--output", type=str, default="saved_model/model_int8.tflite")
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--rep-split", type=str, default="splits/train.txt")
    parser.add_argument("--rep-samples", type=int, default=200)
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required. Install with `pip install tensorflow`.") from exc

    cfg = CFG()
    cfg.root_dir = args.root_dir
    cfg.labels = infer_labels(cfg.root_dir, (cfg.train_list, cfg.val_list, cfg.test_list))

    saved_model_dir = Path(args.saved_model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    check_saved_model_dir(saved_model_dir)
    input_shape = get_saved_model_input_shape(tf, saved_model_dir)
    print(f"[INFO] SavedModel input shape: {input_shape}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(cfg, args.rep_split, args.rep_samples, input_shape)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    in_dtype = interpreter.get_input_details()[0]["dtype"]
    out_dtype = interpreter.get_output_details()[0]["dtype"]
    print(f"[OK] Saved INT8 TFLite model to: {output_path}")
    print(f"[INFO] input dtype: {in_dtype}, output dtype: {out_dtype}")


if __name__ == "__main__":
    main()
