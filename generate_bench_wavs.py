import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import soundfile as sf


DEFAULT_SR = 16000


@dataclass
class ClipEvent:
    label: str
    path: Path
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class BenchScenario:
    set_id: int
    slug: str
    description: str
    keywords: Sequence[str]
    include_unknown: bool
    keyword_range: tuple[int, int]
    unknown_range: tuple[int, int]


SCENARIOS: List[BenchScenario] = [
    BenchScenario(
        set_id=1,
        slug="zoom_silence",
        description="zoom keyword and silence only",
        keywords=("zoom",),
        include_unknown=False,
        keyword_range=(2, 4),
        unknown_range=(0, 0),
    ),
    BenchScenario(
        set_id=2,
        slug="zoom_unknown_silence",
        description="zoom keyword mixed with unknown speech and silence",
        keywords=("zoom",),
        include_unknown=True,
        keyword_range=(2, 4),
        unknown_range=(1, 3),
    ),
    BenchScenario(
        set_id=3,
        slug="reset_silence",
        description="reset keyword and silence only",
        keywords=("reset",),
        include_unknown=False,
        keyword_range=(2, 4),
        unknown_range=(0, 0),
    ),
    BenchScenario(
        set_id=4,
        slug="reset_unknown_silence",
        description="reset keyword mixed with unknown speech and silence",
        keywords=("reset",),
        include_unknown=True,
        keyword_range=(2, 4),
        unknown_range=(1, 3),
    ),
    BenchScenario(
        set_id=5,
        slug="reset_zoom_unknown_silence",
        description="zoom and reset keywords mixed with unknown speech and silence",
        keywords=("zoom", "reset"),
        include_unknown=True,
        keyword_range=(3, 5),
        unknown_range=(1, 3),
    ),
]


def list_wavs(directory: Path) -> List[Path]:
    paths = sorted(directory.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No wav files found in: {directory}")
    return paths


def read_mono_16k(path: Path, target_sr: int = DEFAULT_SR) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resample_linear(audio, sr, target_sr)
    return audio.astype(np.float32)


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32)
    ratio = dst_sr / src_sr
    n = int(round(len(x) * ratio))
    if n <= 1:
        return np.zeros((dst_sr,), dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(t_dst, t_src, x).astype(np.float32)


class ClipPicker:
    def __init__(self, paths: Sequence[Path], rng: random.Random):
        self.paths = list(paths)
        self.rng = rng
        self.pool: List[Path] = []

    def pick(self) -> Path:
        if not self.pool:
            self.pool = self.paths[:]
            self.rng.shuffle(self.pool)
        return self.pool.pop()


class BenchComposer:
    def __init__(
        self,
        bench_dir: Path,
        out_dir: Path,
        seed: int,
        items_per_set: int,
        sr: int = DEFAULT_SR,
    ):
        self.bench_dir = bench_dir
        self.out_dir = out_dir
        self.sr = sr
        self.items_per_set = items_per_set
        self.rng = random.Random(seed)

        self.pickers: Dict[str, ClipPicker] = {
            "zoom": ClipPicker(list_wavs(bench_dir / "zoom_crop"), self.rng),
            "reset": ClipPicker(list_wavs(bench_dir / "reset_crop"), self.rng),
            "unknown": ClipPicker(list_wavs(bench_dir / "unknown_crop"), self.rng),
            "silence": ClipPicker(list_wavs(bench_dir / "silence"), self.rng),
        }

    def make_all(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: List[dict[str, str]] = []
        event_rows: List[dict[str, str]] = []

        for scenario in SCENARIOS:
            for item_idx in range(1, self.items_per_set + 1):
                stem = self.make_stem(scenario, item_idx)
                plan = self.build_random_plan(scenario)
                manifest_row, plan_event_rows = self.render_plan(
                    stem=stem,
                    scenario=scenario,
                    item_idx=item_idx,
                    plan=plan,
                )
                manifest_rows.append(manifest_row)
                event_rows.extend(plan_event_rows)

        self.write_manifest(manifest_rows)
        self.write_ground_truth_events(event_rows)

    def make_stem(self, scenario: BenchScenario, item_idx: int) -> str:
        return f"set{scenario.set_id:02d}_item{item_idx:02d}__{scenario.slug}"

    def pick_clip(self, label: str) -> Path:
        return self.pickers[label].pick()

    def silence_block(self, seconds: int) -> List[Path]:
        return [self.pick_clip("silence") for _ in range(seconds)]

    def random_silence_block(self, min_sec: int = 1, max_sec: int = 3) -> List[Path]:
        return self.silence_block(self.rng.randint(min_sec, max_sec))

    def build_keyword_sequence(self, scenario: BenchScenario) -> List[str]:
        keyword_count = self.rng.randint(*scenario.keyword_range)

        if len(scenario.keywords) == 1:
            return [scenario.keywords[0]] * keyword_count

        sequence = list(scenario.keywords)
        while len(sequence) < keyword_count:
            sequence.append(self.rng.choice(scenario.keywords))
        self.rng.shuffle(sequence)
        return sequence

    def build_non_silence_labels(self, scenario: BenchScenario) -> List[str]:
        labels = self.build_keyword_sequence(scenario)

        if scenario.include_unknown:
            unknown_count = self.rng.randint(*scenario.unknown_range)
            labels.extend(["unknown"] * unknown_count)

        self.rng.shuffle(labels)
        return labels

    def build_random_plan(self, scenario: BenchScenario) -> List[Path]:
        non_silence_labels = self.build_non_silence_labels(scenario)
        plan: List[Path] = []

        plan.extend(self.random_silence_block())
        for label in non_silence_labels:
            plan.append(self.pick_clip(label))
            plan.extend(self.random_silence_block())

        if self.rng.random() < 0.4:
            plan.extend(self.random_silence_block())

        return plan

    def infer_label(self, path: Path) -> str:
        name = path.name.lower()
        if name.startswith("zoom_"):
            return "zoom"
        if name.startswith("reset_"):
            return "reset"
        if name.startswith("unk_"):
            return "unknown"
        if name.startswith("sil__"):
            return "silence"
        return "unknown_label"

    def render_plan(
        self,
        stem: str,
        scenario: BenchScenario,
        item_idx: int,
        plan: Sequence[Path],
    ) -> tuple[dict[str, str], List[dict[str, str]]]:
        audio_parts: List[np.ndarray] = []
        events: List[ClipEvent] = []
        cursor = 0.0

        for path in plan:
            clip = read_mono_16k(path, self.sr)
            duration = len(clip) / self.sr
            label = self.infer_label(path)

            audio_parts.append(clip)
            events.append(
                ClipEvent(
                    label=label,
                    path=path,
                    start_sec=cursor,
                    end_sec=cursor + duration,
                )
            )
            cursor += duration

        mixed = np.concatenate(audio_parts).astype(np.float32)
        wav_path = self.out_dir / f"{stem}.wav"
        txt_path = self.out_dir / f"{stem}.txt"

        sf.write(wav_path, mixed, self.sr, subtype="PCM_16")
        self.write_label_file(
            txt_path=txt_path,
            wav_name=wav_path.name,
            scenario=scenario,
            item_idx=item_idx,
            events=events,
        )
        return (
            self.build_manifest_row(wav_path.name, txt_path.name, scenario, item_idx, events),
            self.build_event_rows(wav_path.name, scenario, item_idx, events),
        )

    def build_manifest_row(
        self,
        wav_name: str,
        txt_name: str,
        scenario: BenchScenario,
        item_idx: int,
        events: Sequence[ClipEvent],
    ) -> dict[str, str]:
        keyword_events = [event for event in events if event.label in {"zoom", "reset"}]
        unknown_events = [event for event in events if event.label == "unknown"]
        silence_events = [event for event in events if event.label == "silence"]
        total_duration = events[-1].end_sec if events else 0.0

        keyword_timeline = ", ".join(
            f"{event.label}@{event.start_sec:.2f}s" for event in keyword_events
        )

        return {
            "set_id": f"{scenario.set_id:02d}",
            "set_slug": scenario.slug,
            "item_id": f"{item_idx:02d}",
            "wav_file": wav_name,
            "label_file": txt_name,
            "total_duration_sec": f"{total_duration:.2f}",
            "keyword_count": str(len(keyword_events)),
            "unknown_count": str(len(unknown_events)),
            "silence_chunks": str(len(silence_events)),
            "keyword_timeline": keyword_timeline,
        }

    def build_event_rows(
        self,
        wav_name: str,
        scenario: BenchScenario,
        item_idx: int,
        events: Sequence[ClipEvent],
    ) -> List[dict[str, str]]:
        rows: List[dict[str, str]] = []
        keyword_index = 0

        for event_index, event in enumerate(events, start=1):
            if event.label in {"zoom", "reset"}:
                keyword_index += 1

            rows.append(
                {
                    "set_id": f"{scenario.set_id:02d}",
                    "set_slug": scenario.slug,
                    "item_id": f"{item_idx:02d}",
                    "wav_file": wav_name,
                    "event_index": str(event_index),
                    "keyword_index": str(keyword_index) if event.label in {"zoom", "reset"} else "",
                    "label": event.label,
                    "start_sec": f"{event.start_sec:.2f}",
                    "end_sec": f"{event.end_sec:.2f}",
                    "duration_sec": f"{event.end_sec - event.start_sec:.2f}",
                    "source_file": event.path.name,
                }
            )

        return rows

    def write_label_file(
        self,
        txt_path: Path,
        wav_name: str,
        scenario: BenchScenario,
        item_idx: int,
        events: Sequence[ClipEvent],
    ):
        keyword_events = [event for event in events if event.label in {"zoom", "reset"}]
        total_duration = events[-1].end_sec if events else 0.0

        lines = [
            f"set_id: {scenario.set_id:02d}",
            f"set_type: {scenario.slug}",
            f"set_description: {scenario.description}",
            f"item_id: {item_idx:02d}",
            f"wav_file: {wav_name}",
            f"sample_rate: {self.sr}",
            f"total_duration_sec: {total_duration:.2f}",
            f"total_events: {len(events)}",
            "",
            "[keyword_summary]",
        ]
        lines.extend(
            f"{idx:02d}. {event.label}\t{event.start_sec:.2f}s\t{event.end_sec:.2f}s\t{event.path.name}"
            for idx, event in enumerate(keyword_events, start=1)
        )
        if not keyword_events:
            lines.append("(no keywords)")

        lines.extend(
            [
                "",
                "[full_timeline]",
                "index\tstart_sec\tend_sec\tlabel\tsource_file",
            ]
        )
        lines.extend(
            f"{idx:02d}\t{event.start_sec:.2f}\t{event.end_sec:.2f}\t{event.label}\t{event.path.name}"
            for idx, event in enumerate(events, start=1)
        )
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_manifest(self, rows: Sequence[dict[str, str]]):
        csv_path = self.out_dir / "bench_manifest.csv"
        fieldnames = [
            "set_id",
            "set_slug",
            "item_id",
            "wav_file",
            "label_file",
            "total_duration_sec",
            "keyword_count",
            "unknown_count",
            "silence_chunks",
            "keyword_timeline",
        ]

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_ground_truth_events(self, rows: Sequence[dict[str, str]]):
        csv_path = self.out_dir / "ground_truth_events.csv"
        fieldnames = [
            "set_id",
            "set_slug",
            "item_id",
            "wav_file",
            "event_index",
            "keyword_index",
            "label",
            "start_sec",
            "end_sec",
            "duration_sec",
            "source_file",
        ]

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate randomized benchmark wav files and aligned label tables from bench clips."
    )
    parser.add_argument("--bench_dir", type=Path, default=Path("bench"))
    parser.add_argument("--out_dir", type=Path, default=Path("bench") / "generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--items_per_set", type=int, default=10)
    parser.add_argument("--sr", type=int, default=DEFAULT_SR)
    args = parser.parse_args()

    composer = BenchComposer(
        bench_dir=args.bench_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        items_per_set=args.items_per_set,
        sr=args.sr,
    )
    composer.make_all()

    total_files = len(SCENARIOS) * args.items_per_set
    print(f"[DONE] Generated {total_files} benchmark wavs in: {args.out_dir}")
    print("[INFO] Naming rule: setXX_itemYY__scenario_slug.wav")
    print("[INFO] Label files: setXX_itemYY__scenario_slug.txt")
    print("[INFO] Manifest file: bench_manifest.csv")
    print("[INFO] Event labels: ground_truth_events.csv")


if __name__ == "__main__":
    main()
