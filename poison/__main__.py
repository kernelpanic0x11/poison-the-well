"""Poison a clip:  uv run python -m poison song.wav --start 30 --duration 3"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .attack import Config, poison
from .figure import render
from .psycho import masking_loss, masking_threshold
from .surrogate import MAX_SECONDS, Surrogate


def load(path: Path, start: float, duration: float) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    frames = int(duration * info.samplerate) if duration > 0 else -1
    audio, sr = sf.read(path, start=int(start * info.samplerate), frames=frames,
                        dtype="float32", always_2d=True)
    return audio, sr


def demo_clip(name: str) -> Path:
    import librosa  # only needed to fetch a freely licensed example clip

    return Path(librosa.example(name))


def mp3_round_trip(wav: Path, bitrate: str = "192k") -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as tmp:
        mp3, back = Path(tmp) / "a.mp3", Path(tmp) / "a.wav"
        for cmd in (["ffmpeg", "-y", "-i", str(wav), "-b:a", bitrate, str(mp3)],
                    ["ffmpeg", "-y", "-i", str(mp3), str(back)]):
            subprocess.run(cmd, check=True, capture_output=True)
        audio, sr = sf.read(back, dtype="float32", always_2d=True)
    return audio, sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", type=Path, help="audio file (omit to use a freely licensed demo clip)")
    ap.add_argument("--demo", default="vibeace", help="librosa example to use when no input is given")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the file")
    ap.add_argument("--duration", type=float, default=3.0, help="seconds to poison, 0 = to the end")
    ap.add_argument("--target", help="AudioSet label to steer the model towards, e.g. Speech")
    ap.add_argument("--eps", type=float, default=Config.eps)
    ap.add_argument("--steps-fool", type=int, default=Config.steps_fool)
    ap.add_argument("--steps-hide", type=int, default=Config.steps_hide)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--device", default="cpu", help="cpu is faster than mps for this model")
    args = ap.parse_args()

    source = args.input or demo_clip(args.demo)
    start = args.start if args.input else (args.start or 10.0)
    audio, sr = load(source, start, args.duration)
    mono = torch.from_numpy(audio.mean(axis=1))
    print(f"{source.name}: {len(mono) / sr:.1f} s at {sr} Hz, surrogate on {args.device}")

    model = Surrogate(args.device)
    cfg = Config(steps_fool=args.steps_fool, steps_hide=args.steps_hide, eps=args.eps, target=args.target)

    window = int(MAX_SECONDS * sr)
    spans = [(i, min(i + window, len(mono))) for i in range(0, len(mono), window)]
    delta = torch.zeros_like(mono)
    for n, (a, b) in enumerate(spans, 1):
        print(f"window {n}/{len(spans)}")
        delta[a:b] = poison(mono[a:b], sr, model, cfg).delta

    poisoned = np.clip(audio + delta.numpy()[:, None], -1.0, 1.0)
    args.out.mkdir(parents=True, exist_ok=True)
    gap = np.zeros((int(0.4 * sr), audio.shape[1]), dtype=np.float32)
    sf.write(args.out / "clean.wav", audio, sr, subtype="PCM_16")
    sf.write(args.out / "poisoned.wav", poisoned, sr, subtype="PCM_16")
    sf.write(args.out / "ab_clean_then_poisoned.wav", np.concatenate([audio, gap, poisoned]), sr, subtype="PCM_16")
    sf.write(args.out / "perturbation_only.wav", delta.numpy(), sr, subtype="PCM_16")

    # Everything below is measured on what was actually written to disk.
    disk, _ = sf.read(args.out / "poisoned.wav", dtype="float32", always_2d=True)
    disk_mono = torch.from_numpy(disk.mean(axis=1))
    mp3, _ = mp3_round_trip(args.out / "poisoned.wav")
    mp3_mono = torch.from_numpy(mp3.mean(axis=1))[: len(mono)]

    a, b = spans[0]
    labels_clean = model.describe(mono[a:b], sr)
    labels_poisoned = model.describe(disk_mono[a:b], sr)
    labels_mp3 = model.describe(mp3_mono[a:b], sr)
    with torch.no_grad():
        e_clean = model.forward(mono[a:b], sr)[1]
        e_poison = model.forward(disk_mono[a:b], sr)[1]
        clean_top = model.label_id(labels_clean[0][0])
        p_after = float(torch.sigmoid(model.forward(disk_mono[a:b], sr)[0][clean_top]))
        p_after_mp3 = float(torch.sigmoid(model.forward(mp3_mono[a:b], sr)[0][clean_top]))

    real_delta = disk_mono - mono
    theta, ref = masking_threshold(mono, sr)
    over_db, masked = masking_loss(real_delta, theta, ref)
    snr = 10 * np.log10(float(mono.pow(2).mean()) / max(float(real_delta.pow(2).mean()), 1e-20))

    report = {
        "source": source.name,
        "start_s": start,
        "seconds": len(mono) / sr,
        "sample_rate": sr,
        "surrogate": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "snr_db": round(snr, 2),
        "peak_perturbation": round(float(real_delta.abs().max()), 5),
        "fraction_of_bins_under_masking_threshold": round(float(masked), 4),
        "mean_db_over_threshold": round(float(over_db), 4),
        "embedding_cosine_clean_vs_poisoned": round(float(torch.cosine_similarity(e_clean, e_poison, dim=0)), 4),
        "clean_top_label": labels_clean[0][0],
        "clean_top_label_prob": {"clean": round(labels_clean[0][1], 4), "poisoned_wav": round(p_after, 4),
                                 "poisoned_after_mp3_192k": round(p_after_mp3, 4)},
        "top_labels": {"clean": labels_clean, "poisoned_wav": labels_poisoned,
                       "poisoned_after_mp3_192k": labels_mp3},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    render(args.out / "figure.png", mono[a:b].numpy(), real_delta[a:b].numpy(), sr,
           labels_clean, labels_poisoned, snr, float(masked))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}/: clean.wav, poisoned.wav, ab_clean_then_poisoned.wav, "
          "perturbation_only.wav, report.json, figure.png")


if __name__ == "__main__":
    main()
