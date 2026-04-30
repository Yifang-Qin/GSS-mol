from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import ClassVar

import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.stress import voigt_6_to_full_3x3_stress
from fairchem.core import FAIRChemCalculator
from fairchem.core.datasets import data_list_collater
from fairchem.core.units.mlip_unit.api.inference import DEFAULT_CHARGE, DEFAULT_SPIN_OMOL


def build_dataloader(
    atoms_list: Sequence[Atoms],
    batch_size: int | None = None,
    only_inference: bool = True,
) -> list[Atoms]:
    """Mimic other calculators' dataloader builders by returning copies of atoms."""
    return [atoms.copy() for atoms in atoms_list]


class UMABatchCalculator(Calculator):
    """Wrapper that exposes UMA's FAIRChemCalculator through BatchRelaxer interface."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, calculator: FAIRChemCalculator, device: str = "cuda"):
        super().__init__()
        self.calculator = calculator
        self.predictor = calculator.predictor
        self.a2g = calculator.a2g
        self.device = device

    def predict_properties(
        self,
        atoms_batches: Iterable[Atoms],
        include_forces: bool = True,
        include_stresses: bool = True,
    ) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
        atoms_list = list(atoms_batches)
        if not atoms_list:
            return [], [], []

        clean_atoms: list[Atoms] = []
        natoms_list: list[int] = []
        for atoms in atoms_list:
            atoms_copy = atoms.copy()
            atoms_copy.calc = None
            atoms_copy.info.pop("stress", None)
            _ensure_charge_and_spin(atoms_copy)
            clean_atoms.append(atoms_copy)
            natoms_list.append(len(atoms_copy))

        batched_atomic_data = [self.a2g(atoms) for atoms in clean_atoms]
        batch = data_list_collater(batched_atomic_data, otf_graph=True)

        with torch.enable_grad():
            pred = self.predictor.predict(batch)

        energies = _extract_energies(pred, len(clean_atoms))
        forces = _extract_forces(pred, natoms_list, include_forces)
        stresses = _extract_stresses(pred, len(clean_atoms), include_stresses)

        return energies, forces, stresses


def _ensure_charge_and_spin(atoms: Atoms) -> None:
    if "charge" not in atoms.info:
        atoms.info["charge"] = DEFAULT_CHARGE
    if "spin" not in atoms.info:
        atoms.info["spin"] = DEFAULT_SPIN_OMOL


def _extract_energies(pred: dict, batch_size: int) -> list[float]:
    energy_arr = pred.get("energy")
    if energy_arr is None:
        return [0.0] * batch_size
    energy_np = np.atleast_1d(energy_arr.detach().cpu().numpy()).reshape(-1)
    return energy_np.tolist()


def _extract_forces(
    pred: dict, natoms_list: Sequence[int], include_forces: bool
) -> list[np.ndarray]:
    if not include_forces or "forces" not in pred:
        return [np.zeros((n, 3)) for n in natoms_list]
    forces_arr = pred["forces"].detach().cpu().numpy()
    forces_list: list[np.ndarray] = []
    cursor = 0
    for natoms in natoms_list:
        forces_list.append(forces_arr[cursor : cursor + natoms])
        cursor += natoms
    return forces_list


def _extract_stresses(pred: dict, batch_size: int, include_stresses: bool) -> list[np.ndarray]:
    if not include_stresses or "stress" not in pred:
        return [np.zeros((3, 3)) for _ in range(batch_size)]

    raw_stress = pred["stress"].detach().cpu().numpy()
    raw_stress = np.asarray(raw_stress)

    if raw_stress.ndim == 2 and raw_stress.shape == (3, 3):
        raw_stress = raw_stress[None, :, :]
    elif raw_stress.ndim == 1 and raw_stress.size == 6:
        raw_stress = raw_stress[None, :]
    elif raw_stress.ndim == 2 and raw_stress.shape[0] != batch_size:
        raw_stress = raw_stress.reshape(batch_size, -1)

    stresses: list[np.ndarray] = []
    for idx in range(batch_size):
        stress_entry = raw_stress[idx]
        stress_matrix = _to_stress_matrix(stress_entry)
        stresses.append(stress_matrix / units.GPa)
    return stresses


def _to_stress_matrix(stress_entry: np.ndarray) -> np.ndarray:
    stress_entry = np.asarray(stress_entry)
    if stress_entry.ndim == 2 and stress_entry.shape == (3, 3):
        return stress_entry
    if stress_entry.ndim == 1 and stress_entry.size == 6:
        return voigt_6_to_full_3x3_stress(stress_entry)
    if stress_entry.ndim == 2 and stress_entry.shape in {(6, 1), (1, 6)}:
        return voigt_6_to_full_3x3_stress(stress_entry.reshape(6))
    if stress_entry.ndim == 1 and stress_entry.size == 9:
        return stress_entry.reshape(3, 3)
    if stress_entry.ndim == 2 and stress_entry.shape in {(9, 1), (1, 9)}:
        return stress_entry.reshape(3, 3)
    raise ValueError(f"Unsupported stress entry shape: {stress_entry.shape}")
