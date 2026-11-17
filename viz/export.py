"""Export what the visuals need from a finished run.

    uv run python -m viz.export out/vibeace

Writes <run>/viz/terrain.npz (hearing threshold and perturbation as dB grids on a
log frequency axis) and <run>/viz/dose.json (model confidence as the perturbation
is scaled from 0% to 100%).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter

from poison.psycho import N_FFT, masking_threshold, power_spectrum, to_db_spl
from poison.surrogate import Surrogate

FREQ_BINS = 96
DOSES = 61


def main(run: Path) -> None:
    clean, sr = sf.read(run / "clean.wav", dtype="float32", always_2d=True)
    poisoned, _ = sf.read(run / "poisoned.wav", dtype="float32", always_2d=True)
    clean = torch.from_numpy(clean.mean(axis=1))
    delta = torch.from_numpy(poisoned.mean(axis=1)) - clean

    theta, ref = masking_threshold(clean, sr)
    grids = {
        "threshold": theta.numpy(),
        "music": to_db_spl(power_spectrum(clean), ref).numpy(),
        "poison": to_db_spl(power_spectrum(delta), ref).numpy(),
    }
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    log_f = np.geomspace(40, sr / 2, FREQ_BINS)
    out = {}
    for name, grid in grids.items():
        resampled = np.stack([np.interp(log_f, freqs, grid[:, t]) for t in range(grid.shape[1])], axis=1)
        # Light smoothing for display only, applied identically to every layer.
        out[name] = gaussian_filter(np.clip(resampled, 0, 96), sigma=0.7).astype(np.float32)
    viz = run / "viz"
    viz.mkdir(exist_ok=True)
    np.savez(viz / "terrain.npz", freqs=log_f, seconds=len(clean) / sr, **out)

    model = Surrogate()
    names = [n for n, _ in model.describe(clean, sr, 3)] + [n for n, _ in model.describe(clean + delta, sr, 3)]
    names = list(dict.fromkeys(names))
    ids = [model.label_id(n) for n in names]
    doses = np.linspace(0, 1, DOSES)
    probs = []
    with torch.no_grad():
        for s in doses:
            logits, _ = model.forward(clean + float(s) * delta, sr)
            probs.append(torch.sigmoid(logits)[ids].tolist())
    snr = 10 * np.log10(float(clean.pow(2).mean()) / float(delta.pow(2).mean()))
    (viz / "dose.json").write_text(json.dumps({
        "labels": names, "clean_labels": names[:3], "doses": doses.tolist(),
        "probs": probs, "snr_db_at_full": snr,
    }, indent=1))
    print(f"wrote {viz}/terrain.npz and dose.json; labels: {names}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
