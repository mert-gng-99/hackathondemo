# ComBat Harmonization for Multi-Site Neuroimaging

ComBat (Johnson et al. 2007, adapted to MRI by Fortin et al. 2017, 2018)
is the de-facto standard for removing scanner / acquisition-site bias
from multi-center neuroimaging studies.

## How it works

ComBat models per-site location (mean) and scale (variance) parameters
using an empirical-Bayes hierarchical framework. It estimates these
parameters jointly across all sites and shrinks them toward a global
prior — small-N sites are pulled toward the global mean, preventing
overfitting.

## Site-gap reduction

A typical demonstration: the per-site mean of a hippocampus volume
feature can vary by 5+ standard deviations across hospitals. ComBat
typically collapses this gap to <0.005 — a 1000x+ reduction — while
preserving within-site biological variance (age, sex, diagnosis).

## When it fails

ComBat requires at least 2 sites with overlapping covariate
distributions. Single-site data, or sites with completely disjoint
populations (e.g., one site only-pediatric, another only-elderly),
produce unreliable harmonization.
