import warnings

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="cuaev not installed",
    category=UserWarning,
    module="torchani.aev",
)

import torchani
from ase.calculators.calculator import Calculator
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


def build_graph(elements, pos):
    pos = torch.from_numpy(pos)
    data = Data(atomic_numbers=torch.from_numpy(elements).long(), pos=pos)
    return data


def build_dataloader(atoms_list, batch_size, only_inference=True):
    data_list = [build_graph(atoms.numbers, atoms.positions) for atoms in atoms_list]
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


class ANICalculator(Calculator):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.ani_model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    @torch.enable_grad()
    def predict_properties(self, dataloader, include_forces=True, include_stresses=False):
        """Batch property prediction compatible with BatchRelaxer."""
        energies = []
        forces = []
        stresses = []

        for graph_batch in dataloader:
            graph_batch = graph_batch.to(self.device)
            species_padded, pos_padded = obtain_padded_batch(graph_batch)

            pos_padded.requires_grad_(True)
            energy = self.ani_model((species_padded, pos_padded)).energies
            num_atoms = torch.bincount(graph_batch.batch)
            force_padded = -torch.autograd.grad(energy.sum(), pos_padded)[0]
            force_list = [
                force_padded[i, : num_atoms[i]].detach().cpu().numpy()
                for i in range(len(graph_batch))
            ]
            energies.extend(energy.detach().cpu().numpy().tolist())
            forces.extend(force_list)
            stresses.extend([np.zeros((3, 3)) for _ in range(len(graph_batch))])

        return energies, forces, stresses


def obtain_padded_batch(batch: Batch) -> tuple[Tensor, Tensor]:
    batch_indices = batch.batch
    positions_list = []
    atomic_numbers_list = []
    for i in range(int(batch_indices.max()) + 1):
        mask = batch_indices == i
        positions_list.append(batch.pos[mask])
        atomic_numbers_list.append(batch.atomic_numbers[mask])
    pos_padded = pad_sequence(positions_list, batch_first=True, padding_value=0.0).float()
    species_padded = pad_sequence(atomic_numbers_list, batch_first=True, padding_value=-1)
    return species_padded, pos_padded
