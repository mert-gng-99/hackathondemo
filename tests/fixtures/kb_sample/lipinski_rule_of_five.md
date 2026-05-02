# Lipinski's Rule of Five — BBB Permeability Heuristic

Lipinski's Rule of Five (Lipinski 1997, 2001) is the foundational
medicinal-chemistry rule for predicting whether a small molecule will
cross the blood-brain barrier (BBB) by passive diffusion.

## The four criteria

A molecule is likely BBB-permeable if it satisfies all four:

1. Molecular weight (MW) <= 500 Daltons
2. Octanol-water partition coefficient (logP) <= 5
3. Hydrogen-bond donors <= 5
4. Hydrogen-bond acceptors <= 10

Molecules violating two or more criteria are typically poorly absorbed
or impermeant.

## Why ethanol crosses

Ethanol (CCO) has MW=46 Da, logP=-0.31, 1 H-bond donor, 1 H-bond
acceptor — well within all four thresholds. This explains its rapid
CNS penetration despite hydrophilicity.

## SHAP attribution interpretation

When a Random Forest BBB classifier flags Morgan fingerprint bits with
positive SHAP values toward a "permeable" label, the bit usually
corresponds to a small lipophilic substructure (CH3-, -OCH3-, aromatic
ring) consistent with Lipinski compliance.
