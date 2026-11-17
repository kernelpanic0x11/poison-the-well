# Phase two: does the poison survive training?

Phase one shows a trained model mishearing a poisoned clip. It does not show
that a model trained on poisoned music learns worse. Phase two tests that claim
at classifier scale, on a laptop.

## Question

If a model is trained on music carrying a perturbation that sits under the
psychoacoustic masking threshold, is it measurably worse on clean music than the
same model trained on the clean versions?

## Design

1. **Data.** GTZAN: 1,000 clips, 10 genres, 30 s each (about 1.2 GB). Cut into
   3 to 5 s segments. Hold out a clean test split by track, so no song appears
   on both sides.
2. **Victim model.** A small CNN on log-mel spectrograms, trained from scratch.
   Small enough to train in minutes and to craft poison against in seconds per
   clip. The phase one surrogate (AST) takes about 4 minutes per clip, which is
   too slow for a whole training set.
3. **Poison.** Error-minimising noise (Huang et al. 2021, "Unlearnable
   Examples"; the approach HarmonyCloak builds on). This is the opposite sign
   from phase one: lower the training loss so the noise becomes a shortcut and
   the model learns the noise instead of the music. Crafted by alternating a few
   steps of model training with a few steps of noise optimisation, then clipped
   with the existing `project_under_mask`.
4. **Runs.** Same architecture, seed and schedule for each:
   - A: train on clean audio (baseline).
   - B: train on error-minimising poison.
   - C: train on phase one style error-maximising poison (expected to do little;
     included to show why the recipe matters).
   - D: train on random noise shaped to the same masking budget (control, to
     rule out "any noise hurts").
5. **Measure.** Clean test accuracy for each run, averaged over at least three
   seeds, with the spread reported. Also report the perturbation level in dB
   under the music and the share of bins under the masking threshold, measured
   on the files as written.

## What counts as a result

- **Works:** B is well below A and D on clean test accuracy, outside the seed
  spread.
- **Does not work:** B is within the spread of A. That gets reported as is.

## Stress tests, if B works

- Retrain B with standard augmentation (random gain, EQ, MP3 round trip, time
  shift). Published work shows this can weaken unlearnable noise.
- Poison only part of the training set (100%, 50%, 10%).
- Train a different architecture on the same poisoned files, to test transfer.

## Known limits

- This is a classifier, not a generative music model. HarmonyCloak tested
  against generative models, which needs GPU time beyond a laptop.
- The masking budget is tight (about 30 dB under the music in phase one). The
  noise may simply be too quiet to act as a shortcut.
- GTZAN has known label and duplication problems. The split by track limits the
  damage but does not remove it.

## Deliverable for the reel

One chart: clean test accuracy for runs A to D. Built by a single command, like
`viz/build.sh` in phase one.

## Estimated effort

One to two hours to build, under an hour of compute on CPU.
