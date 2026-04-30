import numpy as np
import torch
from ase import Atoms
from mattersim.datasets.utils.build import build_dataloader as build_mattersim_dataloader
from mattersim.forcefield.potential import Potential

from src.ani_calculator import ANICalculator
from src.ani_calculator import build_dataloader as build_ani_dataloader
from src.uff_calculator import UFFCalculator
from src.uff_calculator import build_dataloader as build_uff_dataloader
from src.uma_calculator import UMABatchCalculator


def obtain_guidance(
    atomic_number,
    positions,
    batch_ptr,
    calculator: Potential | ANICalculator | UFFCalculator | UMABatchCalculator,
):
    atoms_list = []
    batch_size = int(batch_ptr.max().item()) + 1

    for batch_idx in range(batch_size):
        mask = batch_ptr == batch_idx
        batch_atomic_number = atomic_number[mask]
        batch_positions = positions[mask]

        atoms = obtain_atoms(batch_atomic_number, batch_positions)
        atoms_list.append(atoms)

    if isinstance(calculator, Potential):
        potential_loader = build_mattersim_dataloader(
            atoms_list, batch_size=256, only_inference=True
        )
    elif isinstance(calculator, UFFCalculator):
        potential_loader = build_uff_dataloader(atoms_list, batch_size=256, only_inference=True)
    elif isinstance(calculator, UMABatchCalculator):
        potential_loader = atoms_list
        for atoms in potential_loader:
            atoms.pbc = False
    elif isinstance(calculator, ANICalculator):
        potential_loader = build_ani_dataloader(atoms_list, batch_size=256, only_inference=True)

    with torch.enable_grad():
        predictions = calculator.predict_properties(potential_loader, include_forces=True)
    pred_forces = torch.from_numpy(np.concatenate(predictions[1], axis=0)).to(positions)
    return pred_forces


def obtain_atoms(atomic_number, positions):
    atoms = Atoms(
        numbers=atomic_number.cpu().numpy(),
        positions=positions.cpu().numpy(),
        pbc=True,
        cell=np.eye(3) * 50,
    )
    return atoms
