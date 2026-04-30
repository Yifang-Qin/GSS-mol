import numpy as np
import torch
from ase.calculators.calculator import Calculator
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.utils.aspirin_data_with_bond import AspirinDataWithBond

# Suppress verbose RDKit logs to avoid UFFTYPER warning spam
RDLogger.DisableLog("rdApp.*")

# Unit conversion: RDKit UFF uses kcal/mol and kcal/mol/Å
# We output everything in eV and eV/Å
KCALMOL_TO_EV: float = 0.0433641153087705


def build_graph(elements, pos):
    pos = torch.from_numpy(pos)
    data = Data(atomic_numbers=torch.from_numpy(elements).long(), pos=pos)
    return data


def build_dataloader(atoms_list, batch_size, only_inference=True):
    data_list = [build_graph(atoms.numbers, atoms.positions) for atoms in atoms_list]
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


def _split_batch_to_molecules(graph_batch: Batch) -> list[tuple[np.ndarray, np.ndarray]]:
    batch_indices = graph_batch.batch
    molecules = []
    for i in range(int(batch_indices.max()) + 1):
        mask = batch_indices == i
        pos_i = graph_batch.pos[mask].detach().cpu().numpy()
        z_i = graph_batch.atomic_numbers[mask].detach().cpu().numpy()
        molecules.append((z_i, pos_i))
    return molecules


_ASPIRIN_TEMPLATE_MOL: Chem.Mol | None = None
_MD17_ATOMIC_NUMBERS: np.ndarray | None = None


def _get_aspirin_template_from_md17() -> tuple[Chem.Mol, np.ndarray]:
    """Build aspirin template from MD17 data, ensuring consistent atom ordering."""
    global _ASPIRIN_TEMPLATE_MOL, _MD17_ATOMIC_NUMBERS
    if _ASPIRIN_TEMPLATE_MOL is not None and _MD17_ATOMIC_NUMBERS is not None:
        return _ASPIRIN_TEMPLATE_MOL, _MD17_ATOMIC_NUMBERS

    import os

    # Try multiple possible MD17 paths
    md17_paths = [
        "./md17/md17_aspirin.npz",
    ]

    md17_data = None
    for path in md17_paths:
        if os.path.exists(path):
            md17_data = np.load(path)
            break

    # Build from MD17 data
    energy = md17_data["E"]
    min_idx = np.argmin(energy)
    pos = md17_data["R"][min_idx]
    atomic_numbers = md17_data["z"]

    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Failed to create aspirin molecule from SMILES")
    mol = Chem.AddHs(mol)
    # Molecule creation consistent with AspirinDataWithBond
    # AllChem.EmbedMolecule(mol, randomSeed=42)

    # Set MD17 coordinates
    if not mol.GetNumConformers():
        conf = Chem.Conformer(len(atomic_numbers))
        mol.AddConformer(conf)
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(pos):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    Chem.SanitizeMol(mol)

    _ASPIRIN_TEMPLATE_MOL = mol
    _MD17_ATOMIC_NUMBERS = atomic_numbers.astype(int)
    return mol, _MD17_ATOMIC_NUMBERS


def _build_rdkit_mol(
    atomic_numbers: np.ndarray, positions: np.ndarray, data_reference: Data
) -> Chem.Mol:
    """Build RDKit mol from SMILES topology and write input positions into the conformer.

    Follows the same approach as `AspirinDataWithBond.download`:
    1) Create molecule from SMILES with explicit hydrogens and a conformer;
    2) Match atoms by element to reorder input positions to RDKit atom order;
    3) Write reordered coordinates into the conformer.
    """
    _z_in = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    pos_in = np.asarray(positions, dtype=float).reshape(-1, 3)
    # n = z_in.shape[0]

    # Use stable mol_block from data_reference to ensure consistent atom ordering and topology
    mol_block = getattr(data_reference, "mol_block", None)
    mol: Chem.Mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)

    # Write coordinates into conformer
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(pos_in):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    Chem.SanitizeMol(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol


def _min_distance(positions: np.ndarray) -> float:
    """Compute minimum pairwise atomic distance."""
    pos = positions.reshape(-1, 3)
    n = pos.shape[0]
    if n < 2:
        return float("inf")

    # Compute all pairwise distances
    diff = pos[:, None, :] - pos[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    # Set diagonal to inf to exclude self-distances
    np.fill_diagonal(distances, float("inf"))

    return float(distances.min())


class UFFCalculator(Calculator):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        aspirin_data = AspirinDataWithBond(root="aspirin_fixed")
        self.data_reference = aspirin_data.get(0)

    def predict_properties(self, dataloader, include_forces=True, include_stresses=False):
        energies = []
        forces = []
        stresses = []

        for graph_batch in dataloader:
            mol_specs = _split_batch_to_molecules(graph_batch)
            for z_i, pos_i in mol_specs:
                mol = _build_rdkit_mol(z_i, pos_i, self.data_reference)
                if mol is None:
                    energies.append(float("nan"))
                    forces.append(np.zeros((len(z_i), 3)))
                    # Align with ani_calculator.py: always return zero stresses
                    stresses.append(np.zeros((3, 3)))
                    continue

                ff = AllChem.UFFGetMoleculeForceField(mol)
                grad = np.array(ff.CalcGrad(), dtype=float).reshape(-1, 3)

                # Check geometry validity; optionally pre-relax if atoms are too close
                # if min_dist < 0.9:  # interatomic distance < 0.9 Å
                #     # pre-relax to resolve severe clashes
                #     ff.Minimize(maxIts=50)

                energy = float(ff.CalcEnergy()) * KCALMOL_TO_EV
                energies.append(energy)

                if include_forces:
                    grad = np.array(ff.CalcGrad(), dtype=float).reshape(-1, 3)
                    # force = -dE/dx, convert from kcal/mol/Å to eV/Å
                    force = -grad * KCALMOL_TO_EV

                    # Force clipping: cap max per-atom force norm
                    max_force_per_atom = 10.0  # kcal/mol/Å
                    norms = np.linalg.norm(force, axis=1).max() + 1e-12
                    scale = np.minimum(1.0, max_force_per_atom / norms)
                    force = force * scale

                    forces.append(force)
                else:
                    forces.append(np.zeros((len(z_i), 3)))

                # Align with ani_calculator.py: always return zero stresses
                stresses.append(np.zeros((3, 3)))

        return energies, forces, stresses
