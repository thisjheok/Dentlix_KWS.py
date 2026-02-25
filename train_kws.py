import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio


# -----------------------------
# 0) Config
# -----------------------------
@dataclass
class CFG:
    # dataset
    root_dir: str = "." 
    train_list: str = "splits/train.txt"
    val_list: str = "splits/val.txt"
    test_list: str = "splits/test.txt"

    # audio
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    n_fft: int = 400         # 25ms at 16k
    hop_length: int = 160    # 10ms at 16k
    n_mels: int = 40

    # training
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # labels
    labels: Tuple[str, ...] = ("zoom", "unknown", "silence")


cfg = CFG()


# -----------------------------
# 1) Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split_file(root_dir: str, split_relpath: str) -> List[str]:
    """
    returns list of relative audio paths like:
    audio/zoom/A__zoom__spk-...wav
    """
    p = os.path.join(root_dir, split_relpath)
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def pad_or_trim(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    wav: (1, T)
    """
    T = wav.shape[-1]
    if T == target_len:
        return wav
    if T > target_len:
        return wav[..., :target_len]
    # pad
    pad_len = target_len - T
    return torch.nn.functional.pad(wav, (0, pad_len))


# -----------------------------
# 2) Dataset
# -----------------------------
class KWSDataset(Dataset):
    def __init__(self, root_dir: str, file_list: List[str], labels: Tuple[str, ...], cfg: CFG, train: bool):
        self.root_dir = root_dir
        self.file_list = file_list
        self.label_to_id = {lab: i for i, lab in enumerate(labels)}
        self.cfg = cfg
        self.train = train

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            center=True,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        self.target_len = int(cfg.sample_rate * cfg.clip_seconds)

    def _infer_label_from_path(self, relpath: str) -> str:
        # relpath; 두번째 토큰이 라벨이된다. 
        parts = relpath.replace("\\", "/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Bad relpath: {relpath}")
        return parts[1]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        relpath = self.file_list[idx]
        abspath = os.path.join(self.root_dir, relpath)
        wav, sr = torchaudio.load(abspath)  # wav: (C, T)

        # mono + resample
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)

        # length normalize
        wav = pad_or_trim(wav, self.target_len)

        # (선택) 간단한 augmentation: gain, noise (데모용 최소)
        if self.train:
            # random gain
            gain = 10 ** (random.uniform(-6, 3) / 20.0)  # -6dB ~ +3dB
            wav = wav * gain
            # clamp
            wav = torch.clamp(wav, -1.0, 1.0)

        # feature: log-mel (shape: [n_mels, time])
        mel = self.mel(wav)             # (1, n_mels, time)
        logmel = self.amplitude_to_db(mel)  # (1, n_mels, time)

        # normalize per sample
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

        label_str = self._infer_label_from_path(relpath)
        y = self.label_to_id[label_str]

        # return: (1, n_mels, time), label
        return logmel, torch.tensor(y, dtype=torch.long)


# -----------------------------
# 3) Model (tiny DS-CNN-ish)
# -----------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class TinyKWSNet(nn.Module):
    """
    Input: (B, 1, n_mels, time)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableConv(16, 32, stride=2)  # downsample
        self.block2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.block3 = DepthwiseSeparableConv(64, 96, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


# -----------------------------
# 4) Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train():
    set_seed(cfg.seed)

    train_files = read_split_file(cfg.root_dir, cfg.train_list)
    val_files = read_split_file(cfg.root_dir, cfg.val_list)

    ds_train = KWSDataset(cfg.root_dir, train_files, cfg.labels, cfg, train=True)
    ds_val = KWSDataset(cfg.root_dir, val_files, cfg.labels, cfg, train=False)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = TinyKWSNet(num_classes=len(cfg.labels)).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, y in dl_train:
            x, y = x.to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item()

        val_loss, val_acc = evaluate(model, dl_val, cfg.device)
        print(f"[{epoch:02d}/{cfg.epochs}] train_loss={running/len(dl_train):.4f}  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "labels": cfg.labels, "cfg": cfg.__dict__}, "checkpoints/best_stageA.pt")
            print(f"  -> saved best: {best_val_acc*100:.2f}%")

    print("done. best val acc:", best_val_acc)


if __name__ == "__main__":
    train()
