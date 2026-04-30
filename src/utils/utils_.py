import numpy as np
import torch


def rewrite_ckpt(ckpt_path, sample_params=None, rewrite_path=None):
    if sample_params is not None:
        ori_params = torch.load(ckpt_path, map_location="cuda:0")
        # ori_params = torch.load(ckpt_path)
        ori_params["hyper_parameters"]["diffusion_params"]["sample_params"] = sample_params
        torch.save(ori_params, rewrite_path)


def get_hr(traj, xlim=10, n_bins=200):
    """Compute h(r) for MD17 simulations.

    traj: T x N_atoms x 3
    """
    bins = np.linspace(1e-3, xlim, n_bins + 1)
    pdist = torch.cdist(traj, traj).flatten()
    hist, _ = np.histogram(pdist[:].flatten().numpy(), bins, density=True)
    return hist


def batch_jacobian(g, x, create_graph=True):
    jac = []
    for d in range(g.shape[1]):
        jac.append(
            torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=create_graph)[0].view(
                x.shape[0], 1, x.shape[1]
            )
        )
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, :: M.shape[1] + 1].sum(1)


def norm_2(x):
    return torch.sum(torch.square(x).reshape((x.shape[0], -1)), -1, keepdim=True)


def inner_prod(x, y):
    return torch.sum((x * y).reshape((x.shape[0], -1)), -1, keepdim=True)
