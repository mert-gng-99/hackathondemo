"""Generate deterministic synthetic MRI fixtures for the Day-3 pipeline tests.

Six 8×8×8 NIfTI volumes split across two simulated sites. Each volume is a
spherical "brain" with isotropic Gaussian noise plus a per-site additive bias
that ComBat is expected to remove. The fixture is committed alongside this
script so test runs are reproducible without re-running.

Channels:
  - Site A: subject_0, subject_1, subject_2 (bias = +0.0 a.u.)
  - Site B: subject_3, subject_4, subject_5 (bias = +5.0 a.u.)

NOTE: byte-determinism of the .nii.gz output is coupled to nibabel==5.2.1
(pinned in requirements.txt) and a fixed nibabel.Nifti1Image header. If the
nibabel pin is upgraded, re-run this script and commit the rebuilt artifacts
alongside the dependency bump.
"""
from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np


SITE_A_BIAS = 0.0
SITE_B_BIAS = 5.0
_SITE_BIAS: dict[str, float] = {"A": SITE_A_BIAS, "B": SITE_B_BIAS}
VOLUME_SHAPE = (8, 8, 8)
SUBJECTS = (
    ("subject_0", "A"),
    ("subject_1", "A"),
    ("subject_2", "A"),
    ("subject_3", "B"),
    ("subject_4", "B"),
    ("subject_5", "B"),
)


def _spherical_brain(rng: np.random.Generator, bias: float) -> np.ndarray:
    """Build an 8×8×8 volume: spherical brain (radius 3) + noise + site bias."""
    d, h, w = VOLUME_SHAPE
    z, y, x = np.indices((d, h, w))
    cz, cy, cx = (d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0
    radius2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    brain_mask = radius2 <= 3.0**2
    # Brain intensity ~10 a.u., background ~0.1 a.u. (so default threshold splits cleanly).
    volume = np.where(brain_mask, 10.0, 0.1).astype(np.float64)
    volume += rng.standard_normal(VOLUME_SHAPE) * 0.5
    volume[brain_mask] += bias
    return volume


def build(out_dir: Path | None = None) -> Path:
    """Generate the MRI fixture and write all 7 artifacts to ``out_dir``.

    The output directory will contain six 8×8×8 NIfTI volumes
    (``subject_0.nii.gz`` through ``subject_5.nii.gz``) and a ``sites.csv``
    file with columns ``subject_id, site``. Files are byte-deterministic
    given a fixed nibabel version (see module-level NOTE).

    Args:
        out_dir: Target directory. Defaults to ``tests/fixtures/mri_sample/``
            resolved relative to this script (so CWD-independent).

    Returns:
        The resolved output directory path.

    Raises:
        KeyError: if ``SUBJECTS`` lists a site label not present in
            ``_SITE_BIAS``. This is a fail-fast guard that prevents a
            silent fall-through when adding a new site without updating
            ``_SITE_BIAS``.
    """
    out = out_dir if out_dir is not None else Path(__file__).parent / "mri_sample"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)
    affine = np.eye(4)

    sites_rows: list[tuple[str, str]] = []
    for subject_id, site in SUBJECTS:
        bias = _SITE_BIAS[site]
        volume = _spherical_brain(rng, bias=bias)
        img = nib.Nifti1Image(volume, affine=affine)
        nib.save(img, out / f"{subject_id}.nii.gz")
        sites_rows.append((subject_id, site))

    with (out / "sites.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", "site"])
        writer.writerows(sites_rows)
    return out


if __name__ == "__main__":
    p = build()
    print(f"Wrote MRI fixture to {p}")
