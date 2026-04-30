"""GSS-Mol CLI entry points.

Each public function corresponds to a console_scripts entry in pyproject.toml:
  gss-mol-sample -> sample()
  gss-mol-relax  -> relax()
  gss-mol-rss    -> rss()
  gss-mol-plot   -> plot()
"""

import argparse
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# gss-mol-sample
# ---------------------------------------------------------------------------
def sample():
    import warnings

    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

    parser = argparse.ArgumentParser(
        prog="gss-mol-sample",
        description="Energy-guided diffusion sampling for aspirin conformers.",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=str,
        default="config/sample.yaml",
        help="Configuration YAML file (default: config/sample.yaml)",
    )
    parser.add_argument(
        "--calculator",
        "-d",
        type=str,
        default="ANI",
        choices=["ANI", "UMA", "UFF", "MatterSim"],
        help="Guidance force field (default: ANI)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Per-batch parallel sample count (overrides YAML)",
    )
    parser.add_argument(
        "--sample-num",
        "-n",
        type=int,
        default=4096,
        help="Total number of conformations to generate (default: 4096)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output XYZ path (overrides YAML save_path)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (overrides YAML, default: cuda)"
    )
    parser.add_argument(
        "--guide-scale", type=float, default=None, help="Guidance force scale (default: 0.1)"
    )
    parser.add_argument(
        "--t-mid", type=float, default=None, help="Sigmoid schedule midpoint (default: 100)"
    )
    parser.add_argument(
        "--t-scale", type=float, default=None, help="Sigmoid schedule scale (default: 150)"
    )
    parser.add_argument(
        "--step-num", type=int, default=None, help="Number of guided steps (default: 750)"
    )
    args = parser.parse_args()

    import torch
    import yaml
    from ase.io.xyz import write_xyz
    from torch_geometric.data import Batch, Data
    from tqdm import tqdm

    from src.calculators import setup_calculator
    from src.model import Lit_EquiformerV2
    from src.uff_calculator import UFFCalculator
    from src.utils.aspirin_bond_util import TransformWithBondInfo
    from src.utils.aspirin_data_with_bond import AspirinDataWithBond

    torch.serialization.add_safe_globals([Data])

    with open(args.conf) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    sample_params = configs["diffusion_params"]["sample_params"]
    if args.batch_size is not None:
        sample_params["batch_size"] = args.batch_size
    if args.device is not None:
        sample_params["device"] = args.device

    guidance_params = configs.get("guidance_params", {})
    if args.guide_scale is not None:
        guidance_params["guide_scale"] = args.guide_scale
    if args.t_mid is not None:
        guidance_params["t_mid"] = args.t_mid
    if args.t_scale is not None:
        guidance_params["t_scale"] = args.t_scale
    if args.step_num is not None:
        guidance_params["step_num"] = args.step_num

    device = sample_params.get("device", "cuda")

    calculator = setup_calculator(args.calculator, device=device)
    uff = UFFCalculator(device=device)

    # Load one aspirin template for topology (atomic numbers, bonds, edges).
    # All randomness comes from the diffusion noise, so a single template suffices.
    dataset = AspirinDataWithBond(
        root=configs["data_params"]["root"],
        transform=TransformWithBondInfo(
            cutoff=configs["data_params"]["max_radius"],
            use_pbc=configs["data_params"]["use_pbc"],
        ),
    )
    template = dataset[0]

    model = Lit_EquiformerV2.load_from_checkpoint(sample_params["ckpt_path"]).to(device)

    sample_num = args.sample_num
    batch_size = sample_params["batch_size"]
    atoms_list = []
    remaining = sample_num
    pbar = tqdm(total=sample_num, desc="Sampled structures", unit="samples")
    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        example = Batch.from_data_list([template] * cur_bs).to(device)
        atoms, _ = model.sample(example, calculator=calculator, uff=uff, **guidance_params)
        atoms_list.extend(atoms)
        pbar.update(cur_bs)
        remaining -= cur_bs
    pbar.close()

    output_path = args.output or sample_params.get("save_path", "./sample/gss_mol_sample.xyz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        write_xyz(f, atoms_list)
    print(f"Saved {len(atoms_list)} structures to {output_path}")


# ---------------------------------------------------------------------------
# gss-mol-relax
# ---------------------------------------------------------------------------
def relax():
    import warnings

    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
    warnings.filterwarnings(
        "ignore", message="cuaev not installed", category=UserWarning, module="torchani.aev"
    )

    parser = argparse.ArgumentParser(
        prog="gss-mol-relax",
        description="Post-relax sampled conformations using a force field.",
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input XYZ file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output XYZ file path")
    parser.add_argument(
        "--calculator",
        "-d",
        type=str,
        default="ANI",
        choices=["ANI", "UMA", "UFF", "MatterSim"],
        help="Force field for relaxation (default: ANI)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Max atoms per relaxation batch (default: auto based on calculator)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    args = parser.parse_args()

    import numpy as np
    from ase.io import read, write

    from src.batch_relaxer import BatchRelaxer
    from src.calculators import setup_calculator

    mlff = setup_calculator(args.calculator, device=args.device)

    if args.batch_size is not None:
        max_natoms_per_batch = args.batch_size
    elif args.calculator == "UMA":
        max_natoms_per_batch = 4096
    else:
        max_natoms_per_batch = 8192

    relaxer = BatchRelaxer(
        potential=mlff,
        optimizer="BFGS",
        filter=None,
        fmax=0.001,
        max_natoms_per_batch=max_natoms_per_batch,
        max_relaxation_step=1000,
    )

    atoms_list = read(args.input, index=":")
    print(f"Relaxing {len(atoms_list)} structures from {args.input}")

    for atoms in atoms_list:
        if args.calculator == "UMA":
            atoms.pbc = False
        else:
            atoms.set_cell(np.eye(3) * 50)
            atoms.pbc = True

    converged_list, unconverged_list = relaxer.relax(atoms_list)

    relaxed_atoms = []
    for atoms in converged_list:
        atoms.calc = None
        relaxed_atoms.append(atoms.copy())

    print(f"Relaxation done, {len(converged_list)}/{len(atoms_list)} converged.")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write(args.output, relaxed_atoms)
    print(f"Saved {len(relaxed_atoms)} structures to {args.output}")


# ---------------------------------------------------------------------------
# gss-mol-rss
# ---------------------------------------------------------------------------
def rss():
    parser = argparse.ArgumentParser(
        prog="gss-mol-rss",
        description="RSS baseline: random dihedral perturbation + force field relaxation.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to store generated conformations"
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to store dihedral heatmaps (default: same as output-dir)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="rss_structures.xyz",
        help="Output XYZ filename (default: rss_structures.xyz)",
    )
    parser.add_argument(
        "--calculator",
        "-d",
        type=str,
        default="ANI",
        choices=["ANI", "UMA", "UFF", "MatterSim"],
        help="Force field for relaxation (default: ANI)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Number of conformers to randomize (default: 1024)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.001,
        help="Max force threshold for relaxation (default: 0.001)",
    )
    parser.add_argument(
        "--max-relax-steps", type=int, default=1000, help="Max relaxation steps (default: 1000)"
    )
    parser.add_argument(
        "--md17-path",
        type=str,
        default="./md17/md17_aspirin.npz",
        help="Path to MD17 aspirin npz file",
    )
    args = parser.parse_args()

    # Delegate to rss_sample.main() with a compatible namespace
    from src.rss_sample import main as rss_main

    # rss_sample.parse_args() returns a Namespace; we build one directly
    rss_main.__wrapped__ = True  # noqa: just a marker to avoid confusion
    # Call rss_sample.main() by overwriting sys.argv so its parse_args() works
    sys.argv = [
        "gss-mol-rss",
        "--md17_path",
        args.md17_path,
        "--output_dir",
        args.output_dir,
        "--output_name",
        args.output_name,
        "--calculator",
        args.calculator,
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--fmax",
        str(args.fmax),
        "--max_relax_steps",
        str(args.max_relax_steps),
    ]
    if args.plot_dir:
        sys.argv += ["--plot_dir", args.plot_dir]
    rss_main()


# ---------------------------------------------------------------------------
# gss-mol-plot
# ---------------------------------------------------------------------------
def plot():
    parser = argparse.ArgumentParser(
        prog="gss-mol-plot",
        description="Plot torsion angle distribution heatmap for aspirin conformers.",
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input XYZ file path")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./plots/dihedral.png",
        help="Output plot path (default: ./plots/dihedral.png)",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Gaussian smoothing sigma (default: 1.0)"
    )
    args = parser.parse_args()

    import numpy as np
    from ase.io import read

    from src.plot_aspirin import calc_phi1_phi2, create_mol_from_pos, plot_heatmap, set_plot_style

    set_plot_style()

    atoms_samples = read(args.input, index=":")
    atomic_numbers = atoms_samples[0].numbers
    pos = np.stack([atoms.positions for atoms in atoms_samples], axis=0)

    phi1_list, phi2_list = [], []
    for i in range(pos.shape[0]):
        try:
            mol = create_mol_from_pos(pos[i], atomic_numbers)
            phi1, phi2 = calc_phi1_phi2(mol)
            phi1_list.append(phi1)
            phi2_list.append(phi2)
        except Exception as e:
            print(f"Error processing mol {i}: {e}")
            phi1_list.append(np.nan)
            phi2_list.append(np.nan)

    phi1_arr = np.array(phi1_list)
    phi2_arr = np.array(phi2_list)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot_heatmap(phi1_arr, phi2_arr, args.output, args.sigma)
    print(f"Saved plot to {args.output}")
