import numpy as np
import pytest
import torch

from poison.psycho import absolute_threshold_db, masking_loss, masking_threshold

SR = 44_100


def tone(freq, amp=0.5, seconds=1.0):
    t = torch.arange(int(SR * seconds)) / SR
    return amp * torch.sin(2 * np.pi * freq * t)


def bin_of(freq):
    return int(round(freq * 2048 / SR))


def test_threshold_is_raised_near_a_masker_and_falls_back_to_quiet_far_away():
    theta, _ = masking_threshold(tone(1000), SR)
    frame = theta[:, theta.shape[1] // 2].numpy()
    ath = absolute_threshold_db(np.fft.rfftfreq(2048, 1 / SR))
    assert frame[bin_of(1100)] > ath[bin_of(1100)] + 30  # hidden next to the tone
    assert frame[bin_of(15000)] == pytest.approx(ath[bin_of(15000)], abs=1.0)  # exposed far away
    assert frame[bin_of(1100)] < 96  # never louder than the masker itself


def test_masking_is_stronger_above_the_masker_than_below():
    theta, _ = masking_threshold(tone(1000), SR)
    frame = theta[:, theta.shape[1] // 2].numpy()
    assert frame[bin_of(1500)] > frame[bin_of(500)]


def test_quiet_perturbation_near_masker_is_free_and_loud_far_one_is_penalised():
    clean = tone(1000)
    theta, ref = masking_threshold(clean, SR)
    hidden, masked = masking_loss(tone(1100, amp=0.0005), theta, ref)
    exposed, _ = masking_loss(tone(4000, amp=0.05), theta, ref)
    assert float(hidden) < 0.01 and float(masked) > 0.99
    assert float(exposed) > float(hidden) + 0.01


def test_masking_loss_is_differentiable():
    clean = tone(1000)
    theta, ref = masking_threshold(clean, SR)
    delta = (0.01 * torch.randn_like(clean)).requires_grad_(True)
    loss, _ = masking_loss(delta, theta, ref)
    loss.backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


@pytest.mark.slow
def test_differentiable_front_end_matches_the_stock_feature_extractor():
    from transformers import ASTFeatureExtractor

    from poison.surrogate import MODEL_ID, MODEL_SR, Surrogate

    wave = 0.3 * torch.randn(MODEL_SR * 3)
    stock = ASTFeatureExtractor.from_pretrained(MODEL_ID)(
        wave.numpy(), sampling_rate=MODEL_SR, return_tensors="pt"
    )["input_values"]
    ours = Surrogate().features(wave, MODEL_SR)
    assert ours.shape == stock.shape
    assert torch.allclose(ours, stock, atol=1e-3)
