# MNE-Python ICA for EEG Artifact Removal

Independent Component Analysis (ICA, Hyvärinen 1999) decomposes a
multi-channel EEG recording into statistically independent source
components. It is the de-facto method for removing eye-blink and
heartbeat artifacts before downstream analysis.

## Why ICA, not PCA

PCA decomposes signals into orthogonal components — but neural sources
are not orthogonal in scalp space, they are statistically independent.
ICA's independence assumption matches the physics: the eye, the heart,
and cortical sources fire on uncorrelated schedules.

## The standard workflow

1. Bandpass the raw recording at 0.5-40 Hz to remove DC drift and line
   noise (50/60 Hz).
2. Fit ICA with N components (typically 15-30, less than channel count).
3. Identify artifact components by correlating each ICA source with the
   EOG (eye) channel; reject components with |correlation| > 0.5.
4. Reconstruct the cleaned signal by zeroing out the rejected
   components and inverse-transforming.

## Quality check

Post-ICA, the EOG channel should show minimal residual correlation
with frontal channels (Fp1/Fp2). If it doesn't, the ICA fit was likely
unstable — re-run with a different random seed or more components.
