import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D
from scipy.ndimage import gaussian_filter


def set_plot_style():
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


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


def calc_phi1_phi2(mol):
    # Phi1: Acetoxy group rotation relative to the ring
    phi1_atoms = [1, 6, 12, 11]
    # Phi2: Carboxyl group rotation relative to the ring
    phi2_atoms = [0, 5, 10, 7]

    conf = mol.GetConformer()
    phi1 = rdMolTransforms.GetDihedralDeg(conf, *phi1_atoms)
    phi2 = rdMolTransforms.GetDihedralDeg(conf, *phi2_atoms)
    return phi1, phi2


def plot_heatmap(phi1_val, phi2_val, plot_path, sigma):
    mask = ~np.isnan(phi1_val) & ~np.isnan(phi2_val)
    phi1_filtered = phi1_val[mask]
    phi2_filtered = phi2_val[mask]

    assert len(phi1_filtered) != 0, "No valid data points to plot"
    assert (phi1_filtered.max() <= 180 and phi1_filtered.min() >= -180) and (
        phi2_filtered.max() <= 180 and phi2_filtered.min() >= -180
    ), "Dihedral angles must be within [-180, 180]"

    hist, xedges, yedges = np.histogram2d(
        phi1_filtered,
        phi2_filtered,
        bins=100,
        range=[[-180, 180], [-180, 180]],
    )
    total = hist.sum()
    hist_freq = hist / total if total > 0 else hist
    hist_smooth = gaussian_filter(hist_freq, sigma=sigma)

    _fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.set_facecolor("black")
    im = ax.imshow(
        hist_smooth.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
        vmin=0,
        aspect="auto",
    )
    ax.set_xlabel(r"Acetoxy rotation $\phi_1$")
    ax.set_ylabel(r"Carboxyl rotation $\phi_2$")
    # ax.set_title("Dihedral Angles Heatmap")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)

    ticks = np.arange(-180, 181, 60)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    cbar = plt.colorbar(im, ax=ax, label="Frequency")
    vmax = float(np.nanmax(hist_smooth))
    cbar_ticks = np.linspace(0.0, vmax, 5) if np.isfinite(vmax) and vmax > 0 else np.array([0.0])
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.4f}" for tick in cbar_ticks])
    plt.tight_layout()
    plt.savefig(plot_path, dpi=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="./plots/psm/dihedral.png")
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    set_plot_style()

    atoms_samples = read(args.data_path, index=":")
    atomic_numbers = atoms_samples[0].numbers
    pos = np.stack([atoms.positions for atoms in atoms_samples], axis=0)

    phi1_list, phi2_list = [], []
    for i in range(pos.shape[0]):
        mol_pos = pos[i]
        mol_atomic_numbers = atomic_numbers
        try:
            mol = create_mol_from_pos(mol_pos, mol_atomic_numbers)
            phi1, phi2 = calc_phi1_phi2(mol)
            phi1_list.append(phi1)
            phi2_list.append(phi2)
        except Exception as e:
            print(f"Error processing mol {i}: {e}")
            phi1_list.append(np.nan)
            phi2_list.append(np.nan)

    phi1_arr = np.array(phi1_list)
    phi2_arr = np.array(phi2_list)

    plot_heatmap(phi1_arr, phi2_arr, args.out_path, args.sigma)
