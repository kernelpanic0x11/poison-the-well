"""Gradient attack with a psychoacoustic budget.

Stage 1 finds a perturbation that changes what the surrogate hears, bounded only
by a small peak amplitude. Stage 2 keeps the model fooled while pushing the
perturbation under the masking threshold of the music, trading the two off with
an adaptive weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .psycho import masking_loss, masking_threshold, project_under_mask
from .surrogate import Surrogate


@dataclass
class Config:
    steps_fool: int = 100
    steps_hide: int = 500
    eps: float = 0.02  # peak perturbation amplitude (full scale = 1.0)
    lr: float = 1e-3
    lr_final: float = 2e-5  # step size decays to this during the hide stage
    top_k: int = 5  # how many of the clean labels to suppress
    target: str | None = None  # optional AudioSet label to steer towards
    alpha: float = 0.5  # weight of the masking penalty in the hide stage
    logit_ceiling: float = -4.0  # push each clean label under sigmoid(-4), about 2%
    target_floor: float = 2.0  # push the target label over sigmoid(2), about 88%
    max_cosine: float = 0.5  # embedding must be at most this similar to the clean one
    fooled_below: float = 0.10  # clean top label must fall under this probability


@dataclass
class Result:
    delta: torch.Tensor
    fooled: bool
    masked_fraction: float
    history: list[dict] = field(default_factory=list)


def poison(clean: torch.Tensor, sample_rate: int, model: Surrogate, cfg: Config, log=print) -> Result:
    """clean: mono waveform in [-1, 1] at its native sample rate (at most ~10 s)."""
    clean = clean.detach().cpu().float()
    theta, ref_max = masking_threshold(clean, sample_rate)

    with torch.no_grad():
        clean_logits, clean_emb = model.forward(clean, sample_rate)
    clean_logits, clean_emb = clean_logits.cpu(), clean_emb.cpu()
    suppress = torch.topk(clean_logits, cfg.top_k).indices
    top1 = int(suppress[0])
    target = model.label_id(cfg.target) if cfg.target else None

    delta = (1e-4 * torch.randn_like(clean)).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=cfg.lr)
    best: tuple[float, torch.Tensor, float] | None = None  # (mask loss, delta, masked fraction)
    history: list[dict] = []
    total = cfg.steps_fool + cfg.steps_hide

    for step in range(total):
        hiding = step >= cfg.steps_fool
        if hiding:  # small steps at the end, or Adam's jitter is itself audible
            progress = (step - cfg.steps_fool) / max(cfg.steps_hide - 1, 1)
            opt.param_groups[0]["lr"] = cfg.lr * (cfg.lr_final / cfg.lr) ** progress
        logits, emb = model.forward((clean + delta).clamp(-1, 1), sample_rate)
        logits, emb = logits.cpu(), emb.cpu()

        # Hinged objectives: once the model is fooled with some margin these go
        # flat, and all remaining gradient is spent on hiding the perturbation.
        adv = torch.relu(logits[suppress] - cfg.logit_ceiling).mean()
        if target is not None:
            adv = adv + torch.relu(cfg.target_floor - logits[target])
        adv = adv + torch.relu(torch.cosine_similarity(emb, clean_emb, dim=0) - cfg.max_cosine)

        mask, masked = masking_loss(delta, theta, ref_max)
        loss = adv + (cfg.alpha * mask if hiding else 0.0)

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-cfg.eps, cfg.eps)
            # Never let the sum leave the valid sample range.
            delta.copy_((clean + delta).clamp(-1, 1) - clean)

        probs = torch.sigmoid(logits.detach())
        fooled = bool(probs[top1] < cfg.fooled_below)
        if target is not None:
            fooled = fooled and int(probs.argmax()) == target
        if hiding:
            if fooled and (best is None or float(mask) < best[0]):
                best = (float(mask), delta.detach().clone(), float(masked))

        row = {
            "step": step,
            "adv": float(adv),
            "mask_db": float(mask),
            "masked": float(masked),
            "p_clean_top": float(probs[top1]),
            "fooled": fooled,
        }
        history.append(row)
        if step % 25 == 0 or step == total - 1:
            log(
                f"  step {step:4d}  {'hide' if hiding else 'fool'}  "
                f"p({model.labels[top1]})={row['p_clean_top']:.3f}  "
                f"over-threshold={row['mask_db']:.2f} dB  masked={row['masked']:.1%}  "
                f"{'FOOLED' if fooled else ''}"
            )

    if best is None:
        return Result(delta.detach().clone(), False, history[-1]["masked"], history)

    # Final hard clip (the codex's psychoacoustic_clip), kept only if it still fools the model.
    clipped = project_under_mask(best[1], theta, ref_max)
    with torch.no_grad():
        logits, _ = model.forward((clean + clipped).clamp(-1, 1), sample_rate)
        probs = torch.sigmoid(logits.cpu())
    still = bool(probs[top1] < cfg.fooled_below) and (target is None or int(probs.argmax()) == target)
    if still:
        masked = float(masking_loss(clipped, theta, ref_max)[1])
        log(f"  hard clip kept: masked={masked:.2%}")
        return Result(clipped, True, masked, history)
    log("  hard clip undid the attack, keeping the soft result")
    return Result(best[1], True, best[2], history)
