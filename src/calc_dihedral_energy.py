import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from src.uma_calculator import UMABatchCalculator


def create_mol_from_pos(pos, atomic_numbers):
    mol = Chem.RWMol()

    for atomic_number in atomic_numbers:
        atom = Chem.Atom(int(atomic_number))
        mol.AddAtom(atom)

    conf = Chem.Conformer(len(atomic_numbers))
    for i, coord in enumerate(pos):
        conf.SetAtomPosition(i, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))
    mol.AddConformer(conf)

    return mol.GetMol()


def _wrap_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def _ang_diff_deg(a: float, b: float) -> float:
    return _wrap_deg(a - b)


def calc_phi1_phi2(mol):
    # Phi1: Acetoxy group rotation relative to the ring
    phi1_atoms = [1, 6, 12, 11]
    # Phi2: Carboxyl group rotation relative to the ring
    phi2_atoms = [0, 5, 10, 7]

    conf = mol.GetConformer()
    phi1 = float(rdMolTransforms.GetDihedralDeg(conf, *phi1_atoms))
    phi2 = float(rdMolTransforms.GetDihedralDeg(conf, *phi2_atoms))
    return _wrap_deg(phi1), _wrap_deg(phi2)


def load_atoms(sample_path: str):
    path = Path(sample_path)
    if path.is_dir():
        exts = [".xyz", ".extxyz", ".traj", ".pdb", ".mol", ".sdf"]
        files = []
        for ext in exts:
            files.extend(sorted(path.glob(f"*{ext}")))
        if not files:
            raise FileNotFoundError(f"No structure files found in {sample_path}")

        atoms_list = []
        for file_path in files:
            atoms_list.extend(read(str(file_path), index=":"))
        return atoms_list

    if not path.exists():
        raise FileNotFoundError(f"Sample path not found: {sample_path}")

    return read(str(path), index=":")


def get_uma_calculator(device: str = "cuda"):
    uma_path = "./pretrained_uma"
    checkpoint_path = os.path.join(uma_path, "checkpoints", "uma-m-1p1.pt")
    references_dir = os.path.join(uma_path, "references")
    atom_refs_path = os.path.join(references_dir, "iso_atom_elem_refs.yaml")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(atom_refs_path):
        raise FileNotFoundError(f"Atom refs not found: {atom_refs_path}")

    predictor = load_predict_unit(
        checkpoint_path,
        inference_settings="default",
        overrides=None,
        device=device,
    )
    base_calculator = FAIRChemCalculator(predictor, task_name="omol")
    return UMABatchCalculator(base_calculator, device=device)


def calculate_energies(atoms_list, calculator, batch_size: int = 64):
    energies = []
    for i in range(0, len(atoms_list), batch_size):
        batch = atoms_list[i : i + batch_size]
        clean_batch = []
        for atoms in batch:
            atoms_copy = atoms.copy()
            atoms_copy.pbc = False
            clean_batch.append(atoms_copy)

        batch_energies, _, _ = calculator.predict_properties(
            clean_batch, include_forces=False, include_stresses=False
        )
        energies.extend(batch_energies)

    return np.array(energies, dtype=float)


def extract_centers(phi: np.ndarray, threshold_deg: float) -> np.ndarray:
    centers: list[np.ndarray] = []
    for phi1, phi2 in phi:
        if np.isnan(phi1) or np.isnan(phi2):
            continue
        p = np.array([phi1, phi2], dtype=float)

        matched = False
        for c in centers:
            d1 = abs(_ang_diff_deg(p[0], float(c[0])))
            d2 = abs(_ang_diff_deg(p[1], float(c[1])))
            if max(d1, d2) <= threshold_deg:
                matched = True
                break
        if not matched:
            centers.append(p)

    return np.stack(centers, axis=0) if centers else np.zeros((0, 2), dtype=float)


def assign_classes(phi: np.ndarray, centers: np.ndarray) -> np.ndarray:
    labels = np.full((phi.shape[0],), -1, dtype=int)
    for idx, (phi1, phi2) in enumerate(phi):
        if np.isnan(phi1) or np.isnan(phi2):
            continue
        dists = []
        for c1, c2 in centers:
            d1 = _ang_diff_deg(float(phi1), float(c1))
            d2 = _ang_diff_deg(float(phi2), float(c2))
            dists.append((d1 * d1 + d2 * d2) ** 0.5)
        labels[idx] = int(np.argmin(dists))
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Compute dihedral clusters and average UMA energies."
    )
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--threshold_deg", type=float, default=5.0)
    parser.add_argument("--energy_unit", type=str, default="kcal", choices=["eV", "kcal"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading samples from: {args.sample_path}")
    atoms_list = load_atoms(args.sample_path)
    if len(atoms_list) == 0:
        raise ValueError("No structures found in sample_path")

    print(f"Total structures: {len(atoms_list)}")
    atomic_numbers = atoms_list[0].numbers

    phi1_list = []
    phi2_list = []
    for idx, atoms in enumerate(atoms_list):
        try:
            mol = create_mol_from_pos(atoms.positions, atomic_numbers)
            phi1, phi2 = calc_phi1_phi2(mol)
            phi1_list.append(phi1)
            phi2_list.append(phi2)
        except Exception as exc:
            print(f"Warning: failed to compute dihedrals for index {idx}: {exc}")
            phi1_list.append(np.nan)
            phi2_list.append(np.nan)

    phi1_arr = np.array(phi1_list, dtype=float)
    phi2_arr = np.array(phi2_list, dtype=float)

    valid_mask = np.isfinite(phi1_arr) & np.isfinite(phi2_arr)
    if valid_mask.sum() < args.clusters:
        raise ValueError("Not enough valid dihedral samples for clustering")

    phi = np.stack([phi1_arr, phi2_arr], axis=1)

    print("Extracting dihedral centers...")
    centers = extract_centers(phi, threshold_deg=args.threshold_deg)
    if centers.shape[0] != args.clusters:
        raise ValueError(
            f"Expected {args.clusters} centers, got {centers.shape[0]}. "
            f"Try adjusting --threshold_deg."
        )

    labels = assign_classes(phi, centers)

    print("Loading UMA calculator...")
    calculator = get_uma_calculator(device=device)

    print("Calculating energies...")
    energies = calculate_energies(atoms_list, calculator, batch_size=args.batch_size)

    results = []
    for cluster_id in range(args.clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        phi1_center = float(centers[cluster_id][0])
        phi2_center = float(centers[cluster_id][1])
        cluster_energies = energies[indices]
        mean_energy = float(np.mean(cluster_energies))
        std_energy = float(np.std(cluster_energies))
        results.append(
            {
                "cluster_id": int(cluster_id),
                "phi1_deg": phi1_center,
                "phi2_deg": phi2_center,
                "count": len(indices),
                "mean_energy_ev": mean_energy,
                "std_energy_ev": std_energy,
            }
        )

    if not results:
        raise ValueError("No clustering results produced")

    min_energy = min(item["mean_energy_ev"] for item in results)
    for item in results:
        item["relative_energy_ev"] = item["mean_energy_ev"] - min_energy

    if args.energy_unit == "kcal":
        ev_to_kcal = 23.0605
        for item in results:
            item["mean_energy_kcal"] = item["mean_energy_ev"] * ev_to_kcal
            item["std_energy_kcal"] = item["std_energy_ev"] * ev_to_kcal
            item["relative_energy_kcal"] = item["relative_energy_ev"] * ev_to_kcal

    results.sort(key=lambda x: x["mean_energy_ev"])

    header = "cluster_id,phi1_deg,phi2_deg,count,mean_energy_ev,std_energy_ev,relative_energy_ev"
    if args.energy_unit == "kcal":
        header += ",mean_energy_kcal,std_energy_kcal,relative_energy_kcal"
    print("\nCluster results (sorted by mean energy):")
    print(header)
    for item in results:
        line = (
            f"{item['cluster_id']},{item['phi1_deg']:.2f},{item['phi2_deg']:.2f},"
            f"{item['count']},{item['mean_energy_ev']:.6f},{item['std_energy_ev']:.6f},"
            f"{item['relative_energy_ev']:.6f}"
        )
        if args.energy_unit == "kcal":
            line += (
                f",{item['mean_energy_kcal']:.4f},{item['std_energy_kcal']:.4f},"
                f"{item['relative_energy_kcal']:.4f}"
            )
        print(line)

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for item in results:
                line = (
                    f"{item['cluster_id']},{item['phi1_deg']:.6f},{item['phi2_deg']:.6f},"
                    f"{item['count']},{item['mean_energy_ev']:.8f},{item['std_energy_ev']:.8f},"
                    f"{item['relative_energy_ev']:.8f}"
                )
                if args.energy_unit == "kcal":
                    line += (
                        f",{item['mean_energy_kcal']:.6f},{item['std_energy_kcal']:.6f},"
                        f"{item['relative_energy_kcal']:.6f}"
                    )
                f.write(line + "\n")
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
