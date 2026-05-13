# Paper 15: 𝒩_esc Recipe — Reproduction Code

**The 𝒩_esc Recipe: A Cross-Regime Observation of Bekenstein-Bound Saturation from the Static Escrow Construction |*U*|/*T***

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20145106-blue)](https://doi.org/10.5281/zenodo.20145106)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC_BY_4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Track: Entropic Bounds](https://img.shields.io/badge/Track-2_·_Entropic_Bounds-8b5cf6)](https://windstorminstitute.org/#track2)

> **Track 2 of the Windstorm Institute — Entropic Bounds in Analog Systems.** Companion code to the paper's §5 lattice verification of the Casini–BW inequality and the boost-generator BW identification (0.087% mean accuracy, Table 3).

---

## Published paper

- **[Windstorm-Institute/nesc-recipe](https://github.com/Windstorm-Institute/nesc-recipe)** — paper PDF, LaTeX source, article HTML
- **Website article:** [windstorminstitute.org/articles/nesc-recipe.html](https://windstorminstitute.org/articles/nesc-recipe.html)
- **Zenodo:** [10.5281/zenodo.20145106](https://doi.org/10.5281/zenodo.20145106)

## Contents

```
experiments/
├── lattice_1d_modular.py    Canonical 1+1D modular-Hamiltonian lattice
│                            computation. Reproduces Table 3 BW identification
│                            at 0.087% mean accuracy across 10 parameter
│                            combinations.
└── lattice_3d_modular.py    Canonical 3+1D companion, providing the
                             cross-dimensional comparison context referenced
                             in §4.3 of the paper.
```

## Quick start

```bash
git clone https://github.com/Windstorm-Labs/nesc-recipe
cd nesc-recipe
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib

# 1+1D — reproduces Table 3 BW identification (~0.087% mean accuracy)
python experiments/lattice_1d_modular.py

# 3+1D — companion run for cross-dimensional comparison
python experiments/lattice_3d_modular.py
```

Both scripts are self-contained: they construct the free scalar Hamiltonian on a 1+1D or 3+1D lattice, compute the bipartition entanglement entropy via correlator-matrix eigenvalue methods, and verify the **Casini–BW inequality** Δ*S*<sub>A</sub> ≤ Δ⟨*K*<sub>A</sub>⟩ across the parameter space.

## What the lattice runs verify

- **Boost-generator BW identification at lattice-discretization precision** (~0.1%). The modular Hamiltonian *K*<sub>A</sub> of the half-space reduced state equals 2π ∫ *x*¹ *T*<sup>00</sup>(*x*) *dx* up to lattice artifacts.
- **Casini–BW inequality verified within max 5.4% saturation** at the Compton scale, across mass-perturbation and Compton-wavepacket protocols.
- **Moment-positivity assumption** empirically validated at 0.98–0.999 across the parameter space.

The lattice work is the **empirical anchor** for §4–§5 of the paper. Theorem 1 (the cross-regime observation that the same recipe |*U*|/*T* produces the Bekenstein form in all three regimes) is conditional on (a) BW, (b) Casini, (c) moment-positivity — all three are verified or validated at lattice precision.

## Hardware

- **Hardware:** Current-generation Nvidia GPU (32 GB VRAM, CUDA), Intel Core Ultra 9 285K, 256 GB RAM
- **Note:** GPU is *optional* for these scripts; CPU is sufficient. GPU acceleration becomes useful for the larger lattice-survey sweeps documented in [Paper 13's Labs repo](https://github.com/Windstorm-Labs/lattice-qft-test).

## Why this code lives in three repos

The same canonical lattice scripts back three related papers:
- [Paper 13](https://github.com/Windstorm-Labs/lattice-qft-test) — the standalone lattice paper that first reported the modular-Hamiltonian content (and the retracted 1/30 prefactor)
- [Paper 14](https://github.com/Windstorm-Labs/escrow-spacetime) — the translation paper that placed the lattice content inside the FGHMV theoretical framework
- [Paper 15](https://github.com/Windstorm-Labs/nesc-recipe) — this paper, formalizing 𝒩<sub>esc</sub> as a cross-regime function

Each repo ships the canonical scripts directly so reproduction doesn't require navigating cross-repo dependencies.

## In the Series

### Track 2 — Entropic Bounds in Analog Systems · 7 papers (Papers 10–16)

| # | Paper | DOI | Labs mirror |
|---|---|---|---|
| 10 | [Phonon Extraction Bound (BEC Analog Gravity)](https://github.com/Windstorm-Institute/phonon-extraction-bound) | [10.5281/zenodo.20014391](https://doi.org/10.5281/zenodo.20014391) | [Labs](https://github.com/Windstorm-Labs/phonon-extraction-bound) |
| 11 | [Gravitational Entropy Escrow](https://github.com/Windstorm-Institute/gravitational-entropy-escrow) *(framework paper)* | [10.5281/zenodo.20032023](https://doi.org/10.5281/zenodo.20032023) | [Labs](https://github.com/Windstorm-Labs/gravitational-entropy-escrow) |
| 12 | [C8 Clarification Note](https://github.com/Windstorm-Institute/c8-clarification-note) | [10.5281/zenodo.20041992](https://doi.org/10.5281/zenodo.20041992) | [Labs](https://github.com/Windstorm-Labs/c8-clarification-note) |
| 13 | [Lattice QFT Test of the Static Escrow Postulate](https://github.com/Windstorm-Institute/lattice-qft-test) | [10.5281/zenodo.20057538](https://doi.org/10.5281/zenodo.20057538) | [Labs](https://github.com/Windstorm-Labs/lattice-qft-test) |
| 14 | [Spacetime as Escrow Bookkeeping](https://github.com/Windstorm-Institute/escrow-spacetime) | [10.5281/zenodo.20126091](https://doi.org/10.5281/zenodo.20126091) | [Labs](https://github.com/Windstorm-Labs/escrow-spacetime) |
| 15 | [The 𝒩<sub>esc</sub> Recipe](https://github.com/Windstorm-Institute/nesc-recipe) *(this paper)* | [10.5281/zenodo.20145106](https://doi.org/10.5281/zenodo.20145106) | [Labs](https://github.com/Windstorm-Labs/nesc-recipe) |
| 16 | [The Compton Corollary](https://github.com/Windstorm-Institute/compton-corollary) | [10.5281/zenodo.20163451](https://doi.org/10.5281/zenodo.20163451) | [Labs](https://github.com/Windstorm-Labs/compton-corollary) |

---

## License

Code: MIT · Data/figures: CC BY 4.0 · Paper text: CC BY 4.0 (see [Windstorm-Institute/nesc-recipe](https://github.com/Windstorm-Institute/nesc-recipe) for the paper)
