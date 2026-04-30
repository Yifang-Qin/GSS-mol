import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import write
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from mattersim.forcefield.potential import Potential
from rdkit import Chem
from rdkit.Chem import rdchem, rdDetermineBonds, rdMolTransforms
from rdkit.Geometry import Point3D

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ani_calculator import ANICalculator
from src.batch_relaxer import BatchRelaxer
from src.uff_calculator import UFFCalculator
from src.uma_calculator import UMABatchCalculator

PHI1_ATOMS = [1, 6, 12, 11]
PHI2_ATOMS = [0, 5, 10, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSS over Aspirin dihedrals using MD17 seeds.")
    parser.add_argument(
        "--md17_path",
        type=str,
        default="./md17/md17_aspirin.npz",
        help="Path to the MD17 aspirin npz file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store generated conformations.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to store dihedral heatmaps. Defaults to output_dir.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="rss_structures.xyz",
        help="File name for the saved XYZ file under output_dir.",
    )
    parser.add_argument(
        "--calculator",
        type=str,
        default="ANI",
        choices=["ANI", "UFF", "MatterSim", "UMA"],
        help="Force field used during RSS relaxation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Number of MD17 conformers to randomize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and dihedral initialization.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.001,
        help="Max force threshold for BatchRelaxer.",
    )
    parser.add_argument(
        "--max_relax_steps",
        type=int,
        default=1000,
        help="Max relaxation steps inside BatchRelaxer.",
    )
    return parser.parse_args()


def load_md17_conformers(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"MD17 file not found: {npz_path}")
    data = np.load(npz_path)
    positions = data["R"]
    atomic_numbers = data["z"]
    return positions, atomic_numbers


def sample_indices(total: int, sample_size: int, seed: int) -> np.ndarray:
    if total == 0:
        raise ValueError("MD17 dataset is empty.")
    sample_size = min(sample_size, total)
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=sample_size, replace=False)


def build_rdkit_mol(
    atomic_numbers: Sequence[int], coords: np.ndarray, total_charge: int | None = None
) -> Chem.Mol:
    rw_mol = Chem.RWMol()
    for z in atomic_numbers:
        rw_mol.AddAtom(Chem.Atom(int(z)))
    mol = rw_mol.GetMol()

    conformer = rdchem.Conformer(len(atomic_numbers))
    for idx, coord in enumerate(coords):
        conformer.SetAtomPosition(idx, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))
    mol.AddConformer(conformer, assignId=True)

    if total_charge is None:
        rdDetermineBonds.DetermineBonds(mol)
    else:
        rdDetermineBonds.DetermineBonds(mol, charge=int(total_charge))
    Chem.SanitizeMol(mol)
    return mol


def build_rdkit_mol_geometry_only(atomic_numbers: Sequence[int], coords: np.ndarray) -> Chem.Mol:
    """Mimic plot_aspirin.py: only set atoms and coordinates, skip bond perception."""
    rw_mol = Chem.RWMol()
    for z in atomic_numbers:
        rw_mol.AddAtom(Chem.Atom(int(z)))
    mol = rw_mol.GetMol()
    conformer = rdchem.Conformer(len(atomic_numbers))
    for idx, coord in enumerate(coords):
        conformer.SetAtomPosition(idx, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))
    mol.AddConformer(conformer, assignId=True)
    return mol


def calc_phi_angles(mol: Chem.Mol) -> tuple[float, float]:
    conf = mol.GetConformer()
    phi1 = rdMolTransforms.GetDihedralDeg(conf, *PHI1_ATOMS)
    phi2 = rdMolTransforms.GetDihedralDeg(conf, *PHI2_ATOMS)
    return phi1, phi2


def randomize_dihedrals(
    mol: Chem.Mol,
    rng: np.random.Generator,
) -> tuple[Chem.Mol, float, float]:
    conf = mol.GetConformer()
    target_phi1 = rng.uniform(-180.0, 180.0)
    target_phi2 = rng.uniform(-180.0, 180.0)
    rdMolTransforms.SetDihedralDeg(conf, *PHI1_ATOMS, float(target_phi1))
    rdMolTransforms.SetDihedralDeg(conf, *PHI2_ATOMS, float(target_phi2))
    return mol, target_phi1, target_phi2


def mol_to_ase(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer()
    positions = []
    atomic_numbers = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        positions.append([pos.x, pos.y, pos.z])
        atomic_numbers.append(atom.GetAtomicNum())
    return Atoms(numbers=atomic_numbers, positions=positions)


def compute_dihedrals_from_atoms(
    atoms_list: Sequence[Atoms], geometry_only: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    phi1_vals: list[float] = []
    phi2_vals: list[float] = []
    for atoms in atoms_list:
        if geometry_only:
            mol = build_rdkit_mol_geometry_only(atoms.numbers, atoms.positions)
        else:
            mol = build_rdkit_mol(atoms.numbers, atoms.positions)
        phi1, phi2 = calc_phi_angles(mol)
        phi1_vals.append(phi1)
        phi2_vals.append(phi2)
    return np.asarray(phi1_vals), np.asarray(phi2_vals)


def plot_dihedral_heatmap(phi1: np.ndarray, phi2: np.ndarray, out_path: Path, title: str) -> None:
    mask = ~np.isnan(phi1) & ~np.isnan(phi2)
    if not np.any(mask):
        raise ValueError(f"No valid dihedral data for {title}.")
    phi1_filtered = phi1[mask]
    phi2_filtered = phi2[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("black")
    _, _, _, im = ax.hist2d(
        phi1_filtered,
        phi2_filtered,
        bins=100,
        range=[[-180, 180], [-180, 180]],
        cmap="viridis",
        vmin=0,
        density=False,
    )
    ax.set_xlabel(r"Acetoxy rotation $\phi_1$ (degrees)")
    ax.set_ylabel(r"Carboxyl rotation $\phi_2$ (degrees)")
    ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(im, ax=ax, label="Counts")
    plt.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def prepare_atoms_samples(
    coords: np.ndarray,
    atomic_numbers: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    for source_idx in indices:
        mol = build_rdkit_mol(atomic_numbers, coords[source_idx])
        init_phi1, init_phi2 = calc_phi_angles(mol)
        mol, random_phi1, random_phi2 = randomize_dihedrals(mol, rng)
        atoms = mol_to_ase(mol)
        atoms.info["source_idx"] = int(source_idx)
        atoms.info["phi1_init"] = float(init_phi1)
        atoms.info["phi2_init"] = float(init_phi2)
        atoms.info["phi1_random"] = float(random_phi1)
        atoms.info["phi2_random"] = float(random_phi2)
        atoms_list.append(atoms)
    return atoms_list


def setup_calculator(name: str) -> tuple[object, int]:
    if name == "MatterSim":
        mlff = Potential.load(
            load_path="./pretrained_mattersim/mattersim-v1.0.0-5M.pth",
            model_name="m3gnet",
        ).cuda()
        max_natoms = 8192
    elif name == "ANI":
        mlff = ANICalculator(device="cuda")
        max_natoms = 8192
    elif name == "UFF":
        mlff = UFFCalculator(device="cuda")
        max_natoms = 8192
    elif name == "UMA":
        uma_root = "./pretrained_uma"
        checkpoint_path = Path(uma_root) / "checkpoints" / "uma-m-1p1.pt"
        references_dir = Path(uma_root) / "references"
        atom_refs_path = references_dir / "iso_atom_elem_refs.yaml"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"UMA checkpoint not found: {checkpoint_path}")
        if not atom_refs_path.exists():
            raise FileNotFoundError(f"UMA atom references not found: {atom_refs_path}")
        predictor = load_predict_unit(
            str(checkpoint_path),
            inference_settings="default",
            overrides=None,
            device="cuda",
        )
        base_calculator = FAIRChemCalculator(predictor, task_name="omol")
        mlff = UMABatchCalculator(base_calculator, device="cuda")
        max_natoms = 4096
    else:
        raise ValueError(f"Unsupported calculator: {name}")
    return mlff, max_natoms


def run_relaxation(
    atoms_list: list[Atoms],
    calculator_name: str,
    fmax: float,
    max_relaxation_step: int,
) -> tuple[list[Atoms], list[Atoms]]:
    mlff, max_natoms_per_batch = setup_calculator(calculator_name)
    for atoms in atoms_list:
        if calculator_name == "UMA":
            atoms.pbc = False
        else:
            atoms.set_cell(np.eye(3) * 50)
            atoms.pbc = True
    relaxer = BatchRelaxer(
        potential=mlff,
        optimizer="BFGS",
        fmax=fmax,
        max_natoms_per_batch=max_natoms_per_batch,
        max_relaxation_step=max_relaxation_step,
    )
    converged, unconverged = relaxer.relax(atoms_list)
    if unconverged:
        print(f"[RSS] Warning: {len(unconverged)} structures did not converge.")
    return converged, unconverged


"""
usage: python src/rss_sample.py --output_dir sample/rss_sample/test --plot_dir plots/rss
"""


def main():
    args = parse_args()
    npz_path = Path(args.md17_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    coords, atomic_numbers = load_md17_conformers(npz_path)
    sampled_idx = sample_indices(len(coords), args.num_samples, args.seed)
    rng = np.random.default_rng(args.seed)
    atoms_list = prepare_atoms_samples(coords, atomic_numbers, sampled_idx, rng)

    converged_atoms, _ = run_relaxation(
        atoms_list,
        calculator_name=args.calculator,
        fmax=args.fmax,
        max_relaxation_step=args.max_relax_steps,
    )
    if not converged_atoms:
        print("[RSS] No structures converged; skipping output.")
        return
    relaxed_atoms = converged_atoms
    output_path = output_dir / args.output_name
    print(f"[RSS] Saving {len(relaxed_atoms)} structures to {output_path}")
    write(str(output_path), relaxed_atoms)

    phi_init_1 = np.asarray([atoms.info.get("phi1_init", np.nan) for atoms in relaxed_atoms])
    phi_init_2 = np.asarray([atoms.info.get("phi2_init", np.nan) for atoms in relaxed_atoms])
    phi_rand_1 = np.asarray([atoms.info.get("phi1_random", np.nan) for atoms in relaxed_atoms])
    phi_rand_2 = np.asarray([atoms.info.get("phi2_random", np.nan) for atoms in relaxed_atoms])
    phi_relaxed_1, phi_relaxed_2 = compute_dihedrals_from_atoms(relaxed_atoms, geometry_only=True)

    plot_dihedral_heatmap(
        phi_init_1,
        phi_init_2,
        plot_dir / "dihedral_md17_sample.png",
        "MD17 sampled dihedrals",
    )
    plot_dihedral_heatmap(
        phi_rand_1,
        phi_rand_2,
        plot_dir / "dihedral_randomized.png",
        "Randomized dihedrals",
    )
    plot_dihedral_heatmap(
        phi_relaxed_1,
        phi_relaxed_2,
        plot_dir / "dihedral_rss_relaxed.png",
        "RSS relaxed dihedrals",
    )


if __name__ == "__main__":
    main()
