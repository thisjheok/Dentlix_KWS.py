import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

from models import TinyKWSNet


# -----------------------------
# Config
# -----------------------------
@dataclass
class StreamCfg:
    sr: int = 16000
    win_sec: float = 1.0        # 모델 입력 윈도우 길이(학습 때와 동일)
    hop_sec: float = 0.1        # 100ms hop
    n_fft: int = 400            # 25ms
    hop_length: int = 160       # 10ms
    n_mels: int = 40
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    feature_znorm: bool = False      # train_kws_BN.py와 맞추려면 False

    # Trigger logic
    keyword_class: str = "zoom"      # 네 Stage A 키워드 라벨
    threshold: float = 0.30          # 키워드 확률 임계값
    debounce_n: int = 1              # 연속 N번 이상이면 트리거
    cooldown_sec: float = 0.5        # 트리거 후 쿨다운
    merge_sec: float = 1.0           # 이벤트 후보 병합 간격


# -----------------------------
# Feature extractor (log-mel)
# -----------------------------
class LogMelExtractor(nn.Module):
    def __init__(self, cfg: StreamCfg):
        super().__init__()
        self.feature_znorm = cfg.feature_znorm
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            center=True,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, wav_1d: torch.Tensor) -> torch.Tensor:
        """
        wav_1d: (T,)
        return: (1, 1, n_mels, time)
        """
        wav = wav_1d.unsqueeze(0)  # (1, T)
        m = self.mel(wav)          # (1, n_mels, time)
        x = self.db(m)             # (1, n_mels, time)
        if self.feature_znorm:
            x = (x - x.mean()) / (x.std() + 1e-6)
        return x.unsqueeze(0)      # (1, 1, n_mels, time)


# -----------------------------
# Load audio
# -----------------------------
def load_wav_mono(path: str, target_sr: int) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        # Support both (time, channels) and (channels, time) layout.
        ch_axis = int(np.argmin(x.shape))
        x = np.mean(x, axis=ch_axis)
    if sr != target_sr:
        # torchaudio resample
        tx = torch.tensor(x)
        tx = torchaudio.functional.resample(tx, sr, target_sr)
        x = tx.numpy()
    return x.astype(np.float32)


# -----------------------------
# Streaming inference
# -----------------------------
@torch.no_grad()
def run_stream_kws(
    wav_path: str,
    model: nn.Module,
    labels: List[str],
    cfg: StreamCfg
) -> List[Tuple[float, float]]:
    """
    Returns:
      - triggers: list of (trigger_time_sec, prob)
      - kw_probs: per-window keyword probabilities
    """
    x = load_wav_mono(wav_path, cfg.sr)
    model.eval().to(cfg.device)

    extractor = LogMelExtractor(cfg).to(cfg.device)

    win_len = int(round(cfg.sr * cfg.win_sec))
    hop_len = int(round(cfg.sr * cfg.hop_sec))

    kw_idx = labels.index(cfg.keyword_class)

    cooldown_samples = int(round(cfg.cooldown_sec * cfg.sr))
    next_allowed = 0

    triggers = []
    kw_probs = []
    in_event = False
    consec = 0
    event_max_prob = -1.0
    event_max_start = 0

    for start in range(0, max(1, len(x) - win_len + 1), hop_len):
        end = start + win_len
        if end > len(x):
            break

        # cooldown
        if start < next_allowed:
            continue

        clip = torch.tensor(x[start:end], device=cfg.device)

        feat = extractor(clip)  # (1,1,n_mels,time)
        logits = model(feat)
        prob = torch.softmax(logits, dim=1)[0, kw_idx].item()
        kw_probs.append(prob)

        if prob >= cfg.threshold:
            if not in_event:
                in_event = True
                consec = 1
                event_max_prob = prob
                event_max_start = start
            else:
                consec += 1
                if prob > event_max_prob:
                    event_max_prob = prob
                    event_max_start = start
        else:
            if in_event:
                if consec >= cfg.debounce_n:
                    t = event_max_start / cfg.sr
                    triggers.append((t, event_max_prob))
                    # apply cooldown after finalized event
                    next_allowed = start + cooldown_samples
                in_event = False
                consec = 0

    # finalize trailing event at end-of-file
    if in_event and consec >= cfg.debounce_n:
        t = event_max_start / cfg.sr
        triggers.append((t, event_max_prob))

    # merge nearby candidates: keep only the highest-prob one in each group
    merged = []
    if triggers:
        cur_group = [triggers[0]]
        for t, p in triggers[1:]:
            prev_t = cur_group[-1][0]
            if (t - prev_t) <= cfg.merge_sec:
                cur_group.append((t, p))
            else:
                merged.append(max(cur_group, key=lambda x: x[1]))
                cur_group = [(t, p)]
        merged.append(max(cur_group, key=lambda x: x[1]))

    return merged, kw_probs


# -----------------------------
# Example model loader
# -----------------------------
def load_model_ckpt(ckpt_path: str) -> Tuple[nn.Module, List[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    labels = list(ckpt["labels"])
    model = TinyKWSNet(num_classes=len(labels))
    model.load_state_dict(ckpt["model"], strict=True)
    return model, labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Streaming KWS debug runner")
    ap.add_argument("--wav-path", type=str, default="./practice_data_16k/Zoom_test1_answer_3.wav")
    ap.add_argument("--ckpt-path", type=str, default="./checkpoints/best_stageA.pt")
    ap.add_argument("--keyword", type=str, default="zoom")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--debounce-n", type=int, default=3)
    ap.add_argument("--cooldown-sec", type=float, default=0.9)
    ap.add_argument("--merge-sec", type=float, default=1.0)
    ap.add_argument("--feature-znorm", action="store_true",
                    help="apply per-window z-normalization to log-mel feature (use for znorm-trained model)")
    args = ap.parse_args()

    cfg = StreamCfg(
        keyword_class=args.keyword,
        threshold=args.threshold,
        debounce_n=args.debounce_n,
        cooldown_sec=args.cooldown_sec,
        merge_sec=args.merge_sec,
        feature_znorm=args.feature_znorm,
    )

    wav_path = args.wav_path
    ckpt_path = args.ckpt_path

    model, labels = load_model_ckpt(ckpt_path)

    triggers, kw_probs = run_stream_kws(wav_path, model, labels, cfg)

    if len(kw_probs) > 0:
        arr = np.asarray(kw_probs, dtype=np.float32)
        print("== Prob Stats ==")
        print(f"windows={len(arr)}  mean={arr.mean():.3f}  max={arr.max():.3f}")
        for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
            print(f"p>={th:.1f}: {(arr >= th).sum()}")

    print("== Triggers ==")
    for t, p in triggers:
        print(f"t={t:6.2f}s  prob={p:.3f}")
