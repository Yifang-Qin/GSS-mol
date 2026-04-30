# GSS-Mol: Generative Structure Search for Molecules

GSS-Mol applies energy-guided diffusion sampling to explore molecular conformational landscapes. It combines a learned diffusion prior (EquiformerV2-based score model) with gradient guidance from machine-learning force fields (ANI, UMA, UFF) to generate diverse, low-energy molecular conformations.

This is the molecular (non-periodic) counterpart of GSS, which handles periodic crystal systems. GSS-Mol is demonstrated on aspirin (acetylsalicylic acid), a well-characterized molecule with two primary torsional degrees of freedom.

## Overview

Conformational search aims to find stable 3D arrangements of a molecule. Existing approaches each have limitations:

- **Random Structure Search (RSS)**: perturbs seed structures and relaxes via force minimization. Broad exploration, but low efficiency -- most samples fail to reach valid conformations.
- **Diffusion generative models**: learn the distribution of realistic conformations and sample efficiently, but tend to cluster around the training distribution without physical guarantees.

**GSS-Mol** combines both: at each diffusion denoising step, the update is augmented with energy gradients from an MLFF, biasing samples toward low-energy regions while the diffusion prior maintains conformational diversity.

## Method

Two guidance strategies are implemented:

- **Mi** (M_i-guided): compute forces directly on the current noisy state M_i and blend with the diffusion update via a sigmoid schedule
- **MN** (M_N-guided): estimate the clean structure E[M_N|i], compute forces on this prediction, and apply an additive correction

Both use a hybrid force field: a primary calculator (ANI or UMA) for energetics, plus UFF for bond-length correction. The guidance activates after a configurable number of diffusion steps, controlled by a sigmoid schedule.

## Installation

Requires Python 3.12+. The project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

GPU support (CUDA 11.8 on Linux) is configured automatically via `pyproject.toml`. On macOS, CPU-only PyTorch is installed.

### Prerequisites

GSS-Mol requires a pretrained score model checkpoint and the MD17 aspirin dataset. Additional force-field checkpoints are only needed if the corresponding `--calculator` option is used.

| File | Contents | Required For |
|------|----------|-------------|
| `checkpoints/aspirin_dsm.ckpt` | Pretrained EquiformerV2 score model | Guided sampling |
| `md17/md17_aspirin.npz` | MD17 aspirin dataset (194 MB) | All commands |
| `pretrained_uma/checkpoints/uma-m-1p1.pt` + `pretrained_uma/references/iso_atom_elem_refs.yaml` | UMA-M-1.1 model and atom references | `--calculator UMA` |
| `pretrained_mattersim/mattersim-v1.0.0-5M.pth` | MatterSim 5M checkpoint | `--calculator MatterSim` |

The score model checkpoint is archived on Zenodo (DOI: [10.5281/zenodo.19652325](https://doi.org/10.5281/zenodo.19652325)) together with the CSP checkpoint used by the sister GuidedCSP pipeline.

```bash
# Score model checkpoint (aspirin_dsm.ckpt, ~240 MB)
mkdir -p checkpoints
wget -O checkpoints/aspirin_dsm.ckpt \
  "https://zenodo.org/records/19652325/files/aspirin_dsm.ckpt?download=1"

# MD17 aspirin dataset (~194 MB)
mkdir -p md17
wget -O md17/md17_aspirin.npz \
  "http://www.quantum-machine.org/gdml/data/npz/md17_aspirin.npz"
```

**Force field backends**:
- **ANI** ([TorchANI](https://github.com/aiqm/torchani)): installed via `torchani`; used as the default guidance calculator. No extra download needed.
- **UFF**: provided by RDKit; always used as auxiliary bond correction. No extra download needed.
- **UMA** ([FAIRChem](https://github.com/facebookresearch/fairchem)): installed via `fairchem-core`. Download the `uma-m-1p1.pt` checkpoint and atom references from the [FAIRChem repository](https://github.com/facebookresearch/fairchem) and place them under `pretrained_uma/` as shown in the table above.
- **MatterSim** ([MatterSim](https://github.com/microsoft/mattersim)): installed via `mattersim`. Download `mattersim-v1.0.0-5M.pth` from the [MatterSim pretrained models](https://github.com/microsoft/mattersim/tree/main/pretrained_models) page and place it under `pretrained_mattersim/`.

## Directory Layout

```
gss-mol/
  src/
    sample.py                 # Sampling entry point
    model/
      lit_model.py            # Lit_EquiformerV2 (diffusion + guided sampling)
      guide_utils.py          # Force field guidance dispatch
      equiformer_v2/          # EquiformerV2 backbone (SE(3)-equivariant GNN)
    ani_calculator.py         # ANI force calculator
    uff_calculator.py         # UFF force calculator
    uma_calculator.py         # UMA force calculator
    batch_relaxer.py          # Structure relaxation engine
    rss_sample.py             # RSS baseline
    plot_aspirin.py           # Torsion angle distribution heatmap
    calc_dihedral_energy.py   # Dihedral energy landscape
    utils/                    # Data loading and utilities
  config/
    sample.yaml               # Sampling configuration
  checkpoints/                # Pretrained weights (user-provided)
  scripts/                    # Shell scripts
```

## CLI Usage

Four commands are available after `pip install -e .`. See [CLI-REFERENCE.md](CLI-REFERENCE.md) for full parameter documentation.

### Guided Sampling

```bash
gss-mol-sample -d ANI -n 4096 -o ./sample/aspirin_gss.xyz
```

### Post-Relaxation

```bash
gss-mol-relax -i ./sample/aspirin_gss.xyz -o ./sample/aspirin_relaxed.xyz -d ANI
```

### RSS Baseline

```bash
gss-mol-rss --output-dir ./rss_results --num-samples 1024
```

### Visualization

```bash
gss-mol-plot -i ./sample/aspirin_gss.xyz -o ./plots/dihedral.png
```

## Evaluation

Aspirin's conformational space is characterized by two dihedral angles:
- **phi_Ac** (acetyloxy rotation): atoms [1, 6, 12, 11]
- **phi_COOH** (carboxyl rotation): atoms [0, 5, 10, 7]

`plot_aspirin.py` produces Gaussian-smoothed 2D histograms over this space (Fig. 4b/c in the paper).

## Acknowledgments

The diffusion backbone is based on [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2) (Liao & Smidt, 2023). Energy guidance uses [TorchANI](https://github.com/aiqm/torchani), [UMA/FAIRChem](https://github.com/facebookresearch/fairchem), and [RDKit](https://www.rdkit.org/) UFF.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
