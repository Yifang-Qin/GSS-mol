"""Force-field calculator factory shared by sampling and relaxation."""

import os
from os.path import join

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from mattersim.forcefield.potential import MatterSimCalculator

from src.ani_calculator import ANICalculator
from src.uff_calculator import UFFCalculator
from src.uma_calculator import UMABatchCalculator


def setup_calculator(name: str, device: str = "cuda"):
    if name == "MatterSim":
        print("Using MatterSim calculator")
        return MatterSimCalculator(
            load_path="./pretrained_mattersim/mattersim-v1.0.0-5M.pth",
            device=device,
        ).potential
    elif name == "ANI":
        print("Using ANI calculator")
        return ANICalculator(device=device)
    elif name == "UFF":
        print("Using UFF calculator")
        return UFFCalculator(device=device)
    elif name == "UMA":
        uma_path = "./pretrained_uma"
        print("Using UMA calculator")
        checkpoint_path = join(uma_path, "checkpoints", "uma-m-1p1.pt")
        references_dir = join(uma_path, "references")
        atom_refs_path = join(references_dir, "iso_atom_elem_refs.yaml")
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
    else:
        raise ValueError(f"Unsupported calculator: {name}")
