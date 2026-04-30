import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_dense_adj


class TransformWithBondInfo:
    def __init__(self, cutoff, use_pbc):
        self.cutoff = cutoff
        self.use_pbc = use_pbc

    def __call__(self, data):
        data.atomic_numbers = data.z
        if self.use_pbc:
            src, dst, edge_shift = primitive_neighbor_list(
                "ijS",
                pbc=data.pbc,
                cell=data.cell[0].numpy(),
                positions=data.pos.numpy(),
                cutoff=self.cutoff,
            )
            edge_index = torch.from_numpy(np.vstack([src, dst])).long()
            edge_shift = torch.from_numpy(edge_shift).float()
            data.edge_index, data.edge_shift = edge_index, edge_shift

        else:
            original_edge_index = data.edge_index.clone()
            original_edge_attr = data.edge_attr.clone()

            data.edge_index = radius_graph(data.pos, self.cutoff).long()
            adj_matrix = to_dense_adj(
                edge_index=original_edge_index,
                edge_attr=original_edge_attr,
                max_num_nodes=data.pos.shape[0],
            ).squeeze(0)  # [num_node, num_node, num_bond_type]

            data.bond_attr = torch.zeros(data.edge_index.shape[1], adj_matrix.shape[2])

            for i in range(data.edge_index.shape[1]):
                src_node = data.edge_index[0, i].item()
                dst_node = data.edge_index[1, i].item()
                data.bond_attr[i] = adj_matrix[src_node, dst_node]
        return data
