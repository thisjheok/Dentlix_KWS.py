import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import soundfile as sf


@dataclass
class VadConfig:
    sr: int = 16000
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    min_speech_ms: float = 200.0
    pad_ms: float = 120.0
    hangover_ms: float = 200.0
    threshold_db_over_noise: float = 10.0


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    ratio = dst_sr / src_sr
    n = int(round(len(x) * ratio))
    if n <= 1:
        return np.zeros((dst_sr,), dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(t_dst, t_src, x).astype(np.float32)


def normalize_peak(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-9
    return (x / m * peak).astype(np.float32)


def pad_or_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    pad = target_len - len(x)
    return np.pad(x, (0, pad), mode="constant")


def rms_db(frames: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)), mode="constant")
    n = 1 + (len(x) - frame_len) // hop_len
    if n <= 0:
        n = 1
    return np.stack([x[i * hop_len : i * hop_len + frame_len] for i in range(n)], axis=0)


def crop_silence_to_1s(
    in_wav: str,
    out_dir: str,
    target_sr: int = 16000,
    chunk_sec: float = 1.0,
    keep_remainder: bool = False,
    prefix: str = "sil",
):
    ensure_dir(out_dir)
    x, sr = sf.read(in_wav, dtype="float32", always_2d=False)
    x = to_mono(x)
    x = resample_linear(x, sr, target_sr)
    x = normalize_peak(x, peak=0.98)

    chunk_len = int(round(target_sr * chunk_sec))
    total = len(x)
    num_full = total // chunk_len
    idx = 0

    for i in range(num_full):
        seg = x[i * chunk_len : (i + 1) * chunk_len]
        out_path = os.path.join(out_dir, f"{prefix}__{idx:06d}__dur-{chunk_sec:.2f}s.wav")
        sf.write(out_path, seg, target_sr, subtype="PCM_16")
        idx += 1

    if keep_remainder:
        rem = x[num_full * chunk_len :]
        if len(rem) > 0:
            rem = pad_or_trim(rem, chunk_len)
            out_path = os.path.join(out_dir, f"{prefix}__{idx:06d}__dur-{chunk_sec:.2f}s.wav")
            sf.write(out_path, rem, target_sr, subtype="PCM_16")


def detect_speech_segments_energy(x: np.ndarray, cfg: VadConfig) -> List[Tuple[int, int]]:
    frame_len = int(round(cfg.sr * cfg.frame_ms / 1000.0))
    hop_len = int(round(cfg.sr * cfg.hop_ms / 1000.0))
    frames = frame_signal(x, frame_len, hop_len)
    db = rms_db(frames)

    k = max(1, int(0.1 * len(db)))
    noise_floor = float(np.mean(np.partition(db, k)[:k]))
    thr = noise_floor + cfg.threshold_db_over_noise
    speech_mask = db > thr

    min_frames = int(math.ceil(cfg.min_speech_ms / cfg.hop_ms))
    hang_frames = int(math.ceil(cfg.hangover_ms / cfg.hop_ms))
    pad_frames = int(math.ceil(cfg.pad_ms / cfg.hop_ms))

    segments = []
    in_seg = False
    seg_start = 0
    hang = 0

    for i, is_sp in enumerate(speech_mask):
        if is_sp:
            if not in_seg:
                in_seg = True
                seg_start = i
            hang = hang_frames
        elif in_seg:
            if hang > 0:
                hang -= 1
            else:
                seg_end = i
                s = max(0, seg_start - pad_frames)
                e = min(len(speech_mask), seg_end + pad_frames)
                if (e - s) >= min_frames:
                    start_sample = s * hop_len
                    end_sample = min(len(x), e * hop_len + frame_len)
                    segments.append((start_sample, end_sample))
                in_seg = False

    if in_seg:
        seg_end = len(speech_mask)
        s = max(0, seg_start - pad_frames)
        e = min(len(speech_mask), seg_end + pad_frames)
        if (e - s) >= min_frames:
            start_sample = s * hop_len
            end_sample = min(len(x), e * hop_len + frame_len)
            segments.append((start_sample, end_sample))

    merged = []
    gap_samples = int(round(cfg.sr * 0.08))
    for s, e in sorted(segments):
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s <= pe + gap_samples:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])

    return [(int(s), int(e)) for s, e in merged]


def crop_unknown_speech_only(
    in_wav: str,
    out_dir: str,
    cfg: VadConfig,
    out_clip_sec: float = 1.0,
    mode: str = "centered",
    prefix: str = "unk",
):
    ensure_dir(out_dir)
    x, sr = sf.read(in_wav, dtype="float32", always_2d=False)
    x = to_mono(x)
    x = resample_linear(x, sr, cfg.sr)
    x = normalize_peak(x, peak=0.98)

    out_len = int(round(cfg.sr * out_clip_sec))
    in_stem = os.path.splitext(os.path.basename(in_wav))[0]
    segments = detect_speech_segments_energy(x, cfg)

    if segments:
        main_s, main_e = max(segments, key=lambda seg: seg[1] - seg[0])
        center = (main_s + main_e) // 2
        start = center - out_len // 2
        end = start + out_len
        if start < 0:
            start = 0
            end = out_len
        if end > len(x):
            end = len(x)
            start = max(0, end - out_len)
        clip = pad_or_trim(x[start:end], out_len)
    else:
        clip = pad_or_trim(x, out_len)

    out_path = os.path.join(out_dir, f"{in_stem}__crop1s.wav")
    sf.write(out_path, clip, cfg.sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silence_in", type=str, default=None, help="long silence wav path")
    ap.add_argument("--silence_out", type=str, default=None, help="output folder for silence clips")

    ap.add_argument("--unknown_in", type=str, default=None, help="unknown wav path")
    ap.add_argument("--unknown_out", type=str, default=None, help="output folder for unknown clips")

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--chunk_sec", type=float, default=1.0)
    ap.add_argument("--keep_remainder", action="store_true")

    ap.add_argument("--vad_thr_db", type=float, default=10.0)
    ap.add_argument("--vad_min_ms", type=float, default=200.0)
    ap.add_argument("--vad_pad_ms", type=float, default=120.0)
    ap.add_argument("--vad_hang_ms", type=float, default=200.0)
    ap.add_argument("--mode", type=str, default="centered", choices=["centered", "segments"])

    args = ap.parse_args()

    if args.silence_in and args.silence_out:
        crop_silence_to_1s(
            in_wav=args.silence_in,
            out_dir=args.silence_out,
            target_sr=args.sr,
            chunk_sec=args.chunk_sec,
            keep_remainder=args.keep_remainder,
            prefix="sil",
        )
        print("[OK] silence crop done.")

    if args.unknown_in and args.unknown_out:
        cfg = VadConfig(
            sr=args.sr,
            min_speech_ms=args.vad_min_ms,
            pad_ms=args.vad_pad_ms,
            hangover_ms=args.vad_hang_ms,
            threshold_db_over_noise=args.vad_thr_db,
        )
        crop_unknown_speech_only(
            in_wav=args.unknown_in,
            out_dir=args.unknown_out,
            cfg=cfg,
            out_clip_sec=args.chunk_sec,
            mode=args.mode,
            prefix="unk",
        )
        print("[OK] unknown speech crop done.")


if __name__ == "__main__":
    main()
