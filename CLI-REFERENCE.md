# GSS-Mol CLI Reference

Four commands are available after `pip install -e .`:

| Command          | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `gss-mol-sample` | Energy-guided diffusion sampling for aspirin conformers        |
| `gss-mol-relax`  | Post-relax sampled conformations with a force field            |
| `gss-mol-rss`    | RSS baseline: random dihedral perturbation + relaxation        |
| `gss-mol-plot`   | Plot torsion angle distribution heatmap (Fig. 4b/c)           |


---

## `gss-mol-sample`

Run energy-guided diffusion sampling for aspirin conformer generation.

```bash
# MN-guided (default)
gss-mol-sample --calculator ANI --sample-num 4096

# Mi-guided with custom guidance schedule
gss-mol-sample --calculator ANI --sampling mi --step-num 750 --guide-scale 0.1
```

Output: `./sample/aspirin_gss.xyz` (default, configurable via `--output` or YAML `save_path`)

| Parameter       | Type  | Default              | Description                                                      |
| --------------- | ----- | -------------------- | ---------------------------------------------------------------- |
| `--conf`, `-c`  | str   | `config/sample.yaml` | Configuration YAML file                                          |
| `--calculator`, `-d` | str | `ANI`             | Guidance force field: `ANI`, `UMA`, `UFF`, `MatterSim`           |
| `--batch-size`, `-b` | int | from YAML (`512`) | Per-batch parallel sample count (overrides YAML)                 |
| `--sample-num`, `-n` | int | `4096`            | Total number of conformations to generate                        |
| `--output`, `-o`     | str | from YAML         | Output XYZ path (overrides YAML `save_path`)                     |
| `--device`      | str   | from YAML (`cuda`)   | Device (overrides YAML)                                          |
| `--guide-scale` | float | `0.1`                | Guidance force scale                                             |
| `--t-mid`       | float | `100`                | Sigmoid schedule midpoint $t_\text{mid}$ (Eq. 2)                |
| `--t-scale`     | float | `150`                | Sigmoid schedule width $t_\text{scale}$ (Eq. 2)                 |
| `--step-num`    | int   | `750`                | Number of trailing diffusion steps with guidance active          |


---

## `gss-mol-relax`

Post-relax sampled conformations to local energy minima using a force field.

```bash
gss-mol-relax --input ./sample/aspirin_gss.xyz --output ./sample/aspirin_relaxed.xyz
gss-mol-relax --input ./sample/aspirin_gss.xyz --output ./sample/aspirin_relaxed.xyz --calculator UMA
```

| Parameter       | Type  | Default  | Description                                                |
| --------------- | ----- | -------- | ---------------------------------------------------------- |
| `--input`, `-i` | str   | required | Input XYZ file path                                       |
| `--output`, `-o`| str   | required | Output XYZ file path                                      |
| `--calculator`, `-d` | str | `ANI` | Force field for relaxation: `ANI`, `UMA`, `UFF`, `MatterSim` |
| `--batch-size`  | int   | auto     | Max atoms per relaxation batch (auto: 4096 for UMA, 8192 otherwise) |
| `--device`      | str   | `cuda`   | Device                                                     |


---

## `gss-mol-rss`

RSS baseline: randomize aspirin dihedral angles, relax with a force field, and filter by dihedral validity.

```bash
gss-mol-rss --output-dir ./rss_results --num-samples 1024
```

Output: `{output-dir}/{output-name}` (default: `rss_structures.xyz`)

| Parameter        | Type  | Default                                     | Description                                      |
| ---------------- | ----- | ------------------------------------------- | ------------------------------------------------ |
| `--output-dir`   | str   | required                                    | Directory to store generated conformations        |
| `--plot-dir`     | str   | same as `output-dir`                        | Directory to store dihedral heatmaps              |
| `--output-name`  | str   | `rss_structures.xyz`                        | Output XYZ filename                               |
| `--calculator`, `-d` | str | `ANI`                                     | Force field for relaxation: `ANI`, `UMA`, `UFF`, `MatterSim` |
| `--num-samples`  | int   | `1024`                                      | Number of conformers to randomize                 |
| `--seed`         | int   | `42`                                        | Random seed                                       |
| `--fmax`         | float | `0.001`                                     | Max force threshold for relaxation (eV/A)         |
| `--max-relax-steps` | int | `1000`                                    | Max relaxation steps                              |
| `--md17-path`    | str   | `./md17/md17_aspirin.npz`                     | Path to MD17 aspirin npz file                   |


---

## `gss-mol-plot`

Plot torsion angle distribution heatmap in ($\phi_\text{Ac}$, $\phi_\text{COOH}$) space (Fig. 4b/c).

```bash
gss-mol-plot --input ./sample/aspirin_gss.xyz --output ./plots/dihedral.png
```

| Parameter       | Type  | Default                  | Description                     |
| --------------- | ----- | ------------------------ | ------------------------------- |
| `--input`, `-i` | str   | required                 | Input XYZ file path             |
| `--output`, `-o`| str   | `./plots/dihedral.png`   | Output plot path                |
| `--sigma`       | float | `1.0`                    | Gaussian smoothing sigma        |
