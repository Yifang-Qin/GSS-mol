import os
import os.path as osp
from collections.abc import Callable

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import one_hot
from torch_scatter import scatter


class AspirinDataWithBond(InMemoryDataset):
    r"""Aspirin molecular dataset with bond information.
    If raw files do not exist, creates default conformations from the MD17 dataset.

    Args:
        root (str): Root directory for dataset storage.
        transform (callable, optional): Transform applied to each data object on access.
        pre_transform (callable, optional): Transform applied before saving to disk.
        pre_filter (callable, optional): Filter to decide which data objects to include.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    # Aspirin SMILES string
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["aspirin_data_with_bond.pt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        import numpy as np

        md17_path = "./md17/md17_aspirin.npz"
        if not os.path.exists(md17_path):
            raise FileNotFoundError(f"MD17 dataset not found: {md17_path}")

        data = np.load(md17_path)
        energy = data["E"]
        min_idx = np.argmin(energy)
        pos = data["R"][min_idx]
        pos = torch.tensor(pos, dtype=torch.float)
        atomic_number = data["z"]

        try:
            import rdkit  # noqa: F401
            from rdkit import Chem
            from rdkit.Chem.rdchem import BondType as BT  # noqa: N817
            from rdkit.Chem.rdchem import HybridizationType
        except ImportError:
            raise ImportError(
                "'rdkit' is required to create molecular data from SMILES. "
                "Install via: conda install -c conda-forge rdkit"
            ) from None

        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        if osp.exists(raw_path):
            return

        os.makedirs(self.raw_dir, exist_ok=True)

        # Build molecule and conformer directly from MD17 z/pos, auto-infer bonds
        from rdkit.Chem import rdchem, rdDetermineBonds

        md17_pos = pos.detach().cpu().numpy()
        md17_z = np.asarray(atomic_number, dtype=int).reshape(-1)

        n_atoms = int(md17_pos.shape[0])
        assert md17_z.shape[0] == n_atoms, "MD17 atomic number count != coordinate atom count"

        # 1) Build atom-only molecule (atom order matches MD17)
        rw_mol = Chem.RWMol()
        for z_val in md17_z.tolist():
            rw_mol.AddAtom(Chem.Atom(int(z_val)))
        mol = rw_mol.GetMol()

        # 2) Write MD17 conformer
        conf = rdchem.Conformer(n_atoms)
        for i in range(n_atoms):
            x, y, zc = float(md17_pos[i, 0]), float(md17_pos[i, 1]), float(md17_pos[i, 2])
            conf.SetAtomPosition(i, (x, y, zc))
        mol.AddConformer(conf, assignId=True)

        # 3) Auto-infer bonds and bond orders from coordinates
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)

        # RDKit order matches MD17 order
        rdkit_to_md17 = np.arange(n_atoms, dtype=int)
        md17_to_rdkit = np.arange(n_atoms, dtype=int)
        pos = torch.from_numpy(md17_pos).float()

        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # Use the coordinates already set above
        # conf = mol.GetConformer()
        # pos = conf.GetPositions()
        # pos = torch.tensor(pos, dtype=torch.float)

        type_idx = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        # Get atomic numbers from the actual molecule to ensure consistent ordering
        actual_atomic_numbers = []
        for atom in mol.GetAtoms():
            actual_atomic_numbers.append(atom.GetAtomicNum())
            type_idx.append(types[atom.GetSymbol()])
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        # Use actual molecule's atomic numbers instead of MD17's
        z = torch.tensor(actual_atomic_numbers, dtype=torch.long)

        # Build edge information
        rows, cols, edge_types = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_types += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_attr = one_hot(edge_type, num_classes=len(bonds))

        if len(rows) > 0:
            perm = (edge_index[0] * n_atoms + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

        row, col = edge_index if len(rows) > 0 else (torch.tensor([]), torch.tensor([]))
        hs = (z == 1).to(torch.float)
        num_hs = (
            scatter(hs[row], col, dim_size=n_atoms, reduce="sum").tolist()
            if len(rows) > 0
            else [0] * n_atoms
        )

        x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = (
            torch.tensor([actual_atomic_numbers, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float)
            .t()
            .contiguous()
        )
        x = torch.cat([x1, x2], dim=-1)

        # Unique atom IDs and index mappings:
        # - atom_uid: unique ID in RDKit atom order (0..n_atoms-1)
        # - rdkit_to_md17 / md17_to_rdkit: bidirectional index mappings
        atom_uid = torch.arange(n_atoms, dtype=torch.long)
        rdkit_to_md17_t = torch.from_numpy(rdkit_to_md17).long()
        md17_to_rdkit_t = torch.from_numpy(md17_to_rdkit).long()

        # Export current RDKit mol as MolBlock text for stable reconstruction later
        mol_block = Chem.MolToMolBlock(mol)

        data = Data(
            x=x,
            z=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=self.aspirin_smiles,
            name="aspirin",
            atom_uid=atom_uid,
            rdkit_to_md17=rdkit_to_md17_t,
            md17_to_rdkit=md17_to_rdkit_t,
            mol_block=mol_block,
        )

        torch.save([data] * 4096, raw_path)
        print(f"Initialized molecule from MD17 z/pos with auto-inferred bonds, saved to: {raw_path}")

    def process(self) -> None:
        try:
            raw_data = torch.load(self.raw_paths[0], weights_only=False)

            data_list = []
            if isinstance(raw_data, list):
                for data in raw_data:
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
            else:
                data = raw_data
                if self.pre_filter is None or self.pre_filter(data):
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)

            self.save(data_list, self.processed_paths[0])

        except Exception as e:
            print(f"Error processing data: {e}")
            print("Retrying with fresh raw data...")
            for raw_path in self.raw_paths:
                if osp.exists(raw_path):
                    os.unlink(raw_path)
            self.download()
            raw_data = torch.load(self.raw_paths[0])
            data_list = []
            if isinstance(raw_data, list):
                for data in raw_data:
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
            else:
                data = raw_data
                if self.pre_filter is None or self.pre_filter(data):
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)

            self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"
