# Threat model and potential misuse

This project is a research prototype for studying adversarial audio. It adds a
small, psychoacoustically masked perturbation that changes an audio classifier's
output while attempting to preserve the listening experience for a person.

That capability is dual-use. This document records plausible abuse paths so
that experiments, publication, and future development can be evaluated against
them. It describes risks at a threat-model level; it is not an operational guide
for attacking deployed systems.

## Current capability

The implementation performs a white-box attack against one Audio Spectrogram
Transformer trained on AudioSet. It can suppress the clean clip's strongest
labels, optionally steer the model toward a chosen AudioSet label, and constrain
the perturbation using a psychoacoustic masking model. Existing experiments also
measure whether the result survives a 192 kbps MP3 round trip.

The current results demonstrate inference-time evasion of a specific surrogate.
They do not demonstrate cross-model transfer, over-the-air robustness, training
data poisoning, or imperceptibility to human listeners.

## Potential abuse paths

### Classifier evasion

An attacker could adapt the optimization objective to make monitored or
prohibited audio appear benign to automated moderation, indexing, surveillance,
or content-triage systems. Conversely, benign audio could be made to resemble a
prohibited category and cause wrongful filtering or review.

The present AudioSet surrogate is not itself a moderation system. Exploiting a
real service would require knowledge of, or transfer to, its model and complete
preprocessing pipeline.

### Acoustic sensor misdirection

A targeted attack could attempt to suppress or manufacture labels used by
acoustic monitoring systems, such as speech, alarms, breaking glass, gunshots,
machinery faults, or other safety-relevant events. Successful suppression could
hide a real event; successful fabrication could create false alarms or degrade
trust in a monitoring system.

Safety sensors often use specialized models, temporal logic, sensor fusion, and
physical microphones. This repository has not been tested against those
conditions.

### Copyright and catalog interference

The technique could be adapted to interfere with systems used for music
attribution, catalog matching, recommendation, rights management, or royalty
workflows. Possible outcomes include missed matches, incorrect associations, or
manipulated downstream recommendations.

AudioSet classification is different from robust audio fingerprinting. This
risk therefore requires a different surrogate and should not be inferred from
the current demonstration alone.

### Training-set poisoning

An attacker could publish many apparently ordinary recordings whose
machine-readable representation is systematically associated with an unrelated
concept. If those files and their metadata were scraped into a training set,
they might introduce spurious correlations or reduce data quality.

A more advanced version could attempt to implant a backdoor: a model trained on
the poisoned data would behave normally except when a particular acoustic
trigger is present. That would require control over enough training examples,
labels or metadata, and knowledge of the victim's training procedure. The
current code attacks inference only and does not establish that either form of
training poisoning works.

### Synthetic-audio and provenance evasion

With an appropriate surrogate, perturbations could be optimized against
deepfake detectors, synthetic-audio classifiers, provenance systems, or audio
watermark readers. This could allow generated or altered media to evade a
detector while remaining perceptually close to the source.

This repository does not currently target any provenance or watermark system.
Robust watermarking may also use transformations or secret keys that make a
white-box attack substantially harder.

### Semantic chaff

Large numbers of files could be released whose apparent meaning to people
differs from their representation to machine classifiers. Such "semantic
chaff" could degrade automated dataset labeling, search, retrieval, clustering,
or model evaluation when collected at scale.

The likely impact depends on transfer across models and on whether ingestion
pipelines retain or remove the perturbation through transcoding and other
normalization.

### False-alarm denial of service

Adversarial audio could deliberately trigger high-cost classification or review
paths. At sufficient scale, false positives could consume human attention,
increase operational cost, or cause operators to ignore genuine alerts.

This is distinct from conventional computational denial of service: the scarce
resource is downstream trust and review capacity rather than model inference
alone.

### Model and pipeline fingerprinting

An attacker could submit controlled perturbations and compare a service's
responses to local surrogate models. Similar decision-boundary behavior might
reveal the likely model family, checkpoint, preprocessing steps, or version
changes. That information could then improve subsequent evasion attempts.

Confidence scores and detailed labels make fingerprinting easier. Services that
expose only coarse outcomes still leak some information through repeated
queries, but at a lower rate.

## Capability escalation to watch

The highest-risk extension is a transferable attack optimized jointly across
multiple models and realistic transformations. Relevant transformations include
lossy codecs, resampling, filtering, gain changes, background noise, microphone
and speaker response, and room acoustics. Success across both unrelated model
families and physical playback would turn a model-specific demonstration into a
more general audio-evasion capability.

Other material escalations include:

- reliable targeted steering rather than merely suppressing the original label;
- attacks against an actual safety, moderation, provenance, or rights system;
- automatic black-box optimization using only service responses;
- durable effects on models trained from poisoned audio;
- universal perturbations that work across many unrelated recordings; and
- evidence that listeners cannot reliably distinguish clean and modified audio.

## Present limitations

- The attack requires white-box gradients through one known AST model.
- Cross-architecture transfer has not been demonstrated.
- Robustness has been tested through MP3 compression, not physical playback.
- AudioSet labels are generic acoustic categories rather than security policy
  decisions.
- The psychoacoustic threshold is a model, not a human listening test.
- The same mono perturbation is currently applied to every audio channel.
- Long recordings are processed in independent windows, which may introduce
  boundary effects.
- Reporting for a long recording focuses on its first model window.
- The implementation does not embed hidden speech, commands, or arbitrary data.
- The implementation does not currently poison a model's training behavior.

## Defensive evaluation plan

Any expansion of the project should measure both capability and defenses:

1. **Transferability:** test held-out model families that were not used during
   optimization and report failed transfers as prominently as successes.
2. **Transformation robustness:** evaluate codecs, resampling, filtering, level
   changes, common platform transcodes, and benign audio processing.
3. **Physical robustness:** where appropriate and safely controlled, evaluate
   speaker-to-microphone playback across rooms and devices.
4. **Human perception:** run preregistered blind listening tests rather than
   treating a masking-model score as proof of imperceptibility.
5. **Detection:** test spectral consistency checks, denoising, randomized input
   transformations, ensemble disagreement, and dedicated adversarial detectors.
6. **Robust training:** assess whether adversarial training or augmentation
   improves resilience without unacceptable clean-audio regressions.
7. **Pipeline controls:** measure the benefit of rate limits, coarse outputs,
   query monitoring, sensor fusion, and mandatory human review for consequential
   decisions.
8. **Training-data hygiene:** test provenance tracking, duplicate and near-
   duplicate detection, metadata validation, and quarantine of anomalous audio.

## Responsible development and release

- Keep experiments scoped to owned systems, public research models, or systems
  for which explicit authorization has been granted.
- Do not test against production safety, emergency, moderation, rights, or
  provenance services without written authorization.
- Separate research artifacts from deployment tooling and avoid publishing
  turnkey integrations targeting real services.
- Document the surrogate, transforms, query access, datasets, and failure cases
  needed to reproduce every capability claim.
- Coordinate disclosure with affected vendors before publishing a result that
  bypasses a deployed consequential system.
- Reassess release scope if the project gains reliable black-box,
  cross-architecture, universal, training-time, or over-the-air capability.

## Intended use

The intended uses are authorized robustness research, defensive evaluation,
development of detection and mitigation techniques, and clearer communication
about the limitations of machine audio perception. The existence of a plausible
abuse path is not evidence that this repository presently realizes it.
