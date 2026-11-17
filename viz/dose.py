"""Animated verdict: the model's labels flip as the perturbation is dialled in.

    uv run python -m viz.dose out/vibeace      ->  out/vibeace/viz/dose.mp4 (1080x1920)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

SURFACE, INK, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
CLEAN, POISON = "#3987e5", "#d95926"
FPS, RAMP, HOLD_START, HOLD_END = 30, 150, 20, 50


def main(run: Path) -> None:
    data = json.loads((run / "viz" / "dose.json").read_text())
    labels, doses, probs = data["labels"], np.array(data["doses"]), np.array(data["probs"])
    colors = [CLEAN if name in data["clean_labels"] else POISON for name in labels]
    y = np.arange(len(labels))[::-1]

    fig = plt.figure(figsize=(9, 16), dpi=120, facecolor=SURFACE)
    fig.text(0.08, 0.90, "What the model hears", color=INK, fontsize=30, weight="bold")
    status = fig.text(0.08, 0.865, "", color=MUTED, fontsize=16)

    meter = fig.add_axes([0.08, 0.78, 0.84, 0.022], facecolor=GRID)
    meter.set_xlim(0, 1), meter.set_ylim(0, 1), meter.axis("off")
    meter.add_patch(plt.Rectangle((0, 0), 1, 1, color=GRID))
    fill = meter.add_patch(plt.Rectangle((0, 0), 0, 1, color=POISON))
    fig.text(0.08, 0.815, "Perturbation strength", color=MUTED, fontsize=14)
    pct = fig.text(0.92, 0.815, "", color=INK, fontsize=14, ha="right")

    ax = fig.add_axes([0.08, 0.12, 0.84, 0.58], facecolor=SURFACE)
    bars = ax.barh(y, probs[0], color=colors, height=0.42)
    ax.set_xlim(0, 1), ax.set_ylim(-0.6, len(labels) - 0.3), ax.axis("off")
    for yi, name in zip(y, labels):
        ax.text(0, yi + 0.3, name, color=INK, fontsize=18, va="bottom")
    values = [ax.text(0, yi, "", color=INK, fontsize=18, va="center") for yi in y]
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CLEAN), plt.Rectangle((0, 0), 1, 1, color=POISON)],
               labels=["Heard in the original", "Heard after poisoning"], loc="lower left",
               bbox_to_anchor=(0.07, 0.05), frameon=False, labelcolor=INK, fontsize=14, ncols=2)

    def draw(dose: float) -> None:
        p = np.array([np.interp(dose, doses, probs[:, i]) for i in range(len(labels))])
        for bar, text, value in zip(bars, values, p):
            bar.set_width(value)
            text.set_x(value + 0.015), text.set_text(f"{value:.0%}")
        fill.set_width(dose)
        pct.set_text(f"{dose:.0%}")
        if dose < 0.005:
            status.set_text("Original audio, nothing added")
        else:
            level = data["snr_db_at_full"] - 20 * np.log10(dose)
            status.set_text(f"Perturbation sits {level:.0f} dB under the music")

    out = run / "viz" / "dose.mp4"
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "16"])
    with writer.saving(fig, str(out), dpi=120):
        for frame in range(HOLD_START + RAMP + HOLD_END):
            t = np.clip((frame - HOLD_START) / RAMP, 0, 1)
            draw(float(t * t * (3 - 2 * t)))  # ease in and out
            writer.grab_frame(facecolor=SURFACE)
    draw(1.0)
    fig.savefig(run / "viz" / "dose_final.png", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
