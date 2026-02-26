import os
import glob
import math
import argparse
import re
from dataclasses import dataclass

import numpy as np
import soundfile as sf


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    # soundfile always_2d=True면 (T, C) 형태로 올 수 있어서 axis 주의
    if x.shape[1] <= 8:  # (T, C)일 가능성
        return np.mean(x, axis=1)
    return np.mean(x, axis=0)

def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """간단 선형 보간 리샘플 (MVP용)."""
    if src_sr == dst_sr:
        return x.astype(np.float32)
    ratio = dst_sr / src_sr
    n = int(round(len(x) * ratio))
    if n <= 1:
        return np.zeros((dst_sr,), dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(t_dst, t_src, x).astype(np.float32)

def normalize_peak(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-9)
    return (x / m * peak).astype(np.float32)

def frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)), mode="constant")
    n = 1 + (len(x) - frame_len) // hop_len
    if n <= 0:
        n = 1
    return np.stack([x[i * hop_len : i * hop_len + frame_len] for i in range(n)], axis=0)

def rms(frames: np.ndarray) -> np.ndarray:
    # frames: (N, frame_len)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

def pad_or_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - len(x)), mode="constant")


def find_next_index(out_dir: str, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.wav$")
    max_idx = 0
    for p in glob.glob(os.path.join(out_dir, "*.wav")):
        name = os.path.basename(p)
        m = pattern.match(name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


# -------------------------
# Center crop by energy peak
# -------------------------
@dataclass
class CenterCropCfg:
    target_sr: int = 16000
    out_sec: float = 1.0
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    # peak 찾을 때 너무 찌그러진 한 프레임에만 끌려가지 않게 약간 smoothing
    smooth_win: int = 5  # 프레임 단위 이동평균 창 (홀수 추천)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")


def crop_center_by_energy(wav: np.ndarray, sr: int, cfg: CenterCropCfg) -> np.ndarray:
    """
    wav: mono float32
    return: 1.0s (cfg.out_sec) clip centered around energy peak
    """
    # resample + normalize
    wav = resample_linear(wav, sr, cfg.target_sr)
    wav = normalize_peak(wav, 0.98)

    out_len = int(round(cfg.target_sr * cfg.out_sec))

    # framing
    frame_len = int(round(cfg.target_sr * cfg.frame_ms / 1000.0))
    hop_len = int(round(cfg.target_sr * cfg.hop_ms / 1000.0))
    frames = frame_signal(wav, frame_len, hop_len)
    e = rms(frames)
    e = moving_average(e, cfg.smooth_win)

    # peak frame index
    peak_i = int(np.argmax(e))
    peak_center_sample = peak_i * hop_len + frame_len // 2

    # center crop
    start = peak_center_sample - out_len // 2
    end = start + out_len

    # boundary fix + pad
    if start < 0:
        start = 0
        end = out_len
    if end > len(wav):
        end = len(wav)
        start = max(0, end - out_len)

    clip = wav[start:end]
    clip = pad_or_trim(clip, out_len)
    return clip.astype(np.float32)


def process_keyword_folder(
    in_dir: str,
    out_dir: str,
    cfg: CenterCropCfg,
    glob_pattern: str = "*.wav",
    sequential: bool = False,
    prefix: str = "zoom",
    start_idx: int = 1,
):
    ensure_dir(out_dir)
    paths = sorted(glob.glob(os.path.join(in_dir, glob_pattern)))
    if not paths:
        print(f"[WARN] No wav found: {in_dir}")
        return

    cur_idx = start_idx
    for idx, p in enumerate(paths):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = to_mono(x)

        clip = crop_center_by_energy(x, sr, cfg)

        if sequential:
            out_name = f"{prefix}_{cur_idx:06d}.wav"
            cur_idx += 1
        else:
            base = os.path.splitext(os.path.basename(p))[0]
            out_name = f"{base}__crop1s.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, clip, cfg.target_sr, subtype="PCM_16")

        if (idx + 1) % 100 == 0:
            print(f"[OK] {idx+1}/{len(paths)} saved...")

    print(f"[DONE] saved {len(paths)} clips -> {out_dir}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="키워드 wav 폴더 (raw)")
    ap.add_argument("--out_dir", type=str, required=True, help="1초 크롭 결과 저장 폴더 (processed)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--out_sec", type=float, default=1.0)
    ap.add_argument("--frame_ms", type=float, default=20.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--smooth_win", type=int, default=5)
    ap.add_argument("--sequential", action="store_true", help="prefix_000001.wav 형태로 순번 저장")
    ap.add_argument("--prefix", type=str, default="zoom", help="sequential 저장 시 파일명 prefix")
    ap.add_argument("--start_idx", type=int, default=1, help="sequential 저장 시작 번호")
    ap.add_argument("--continue_numbering", action="store_true", help="out_dir에서 기존 번호 다음부터 저장")
    args = ap.parse_args()

    cfg = CenterCropCfg(
        target_sr=args.sr,
        out_sec=args.out_sec,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        smooth_win=args.smooth_win,
    )
    start_idx = args.start_idx
    if args.sequential and args.continue_numbering:
        start_idx = find_next_index(args.out_dir, args.prefix)

    process_keyword_folder(
        args.in_dir,
        args.out_dir,
        cfg,
        sequential=args.sequential,
        prefix=args.prefix,
        start_idx=start_idx,
    )


if __name__ == "__main__":
    main()
