import math

import lightning as L  # noqa: N812
import torch
from diffusers.utils.torch_utils import randn_tensor
from mattersim.forcefield.potential import Potential
from tqdm import tqdm

from src.ani_calculator import ANICalculator
from src.uff_calculator import UFFCalculator
from src.uma_calculator import UMABatchCalculator

from .equiformer_v2.equiformer_v2 import EquiformerV2
from .guide_utils import obtain_guidance
from .utils import Batch2Atoms


class Lit_EquiformerV2(L.LightningModule):  # noqa: N801
    def __init__(self, model_params, diffusion_params, train_params=None, data_params=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = EquiformerV2(**model_params)
        self.diffusion_params = diffusion_params
        self.model_params = model_params

        self.sde_type = self.model_params["sde_type"]
        self.model_type = self.model_params["model_type"]
        self.timesteps = torch.linspace(1e-5, 1, self.diffusion_params["num_steps"])
        self.timesteps_neg_sequence = torch.linspace(1, 1e-5, self.diffusion_params["num_steps"])
        self.init_noise_sigma = self.diffusion_params["sigma_max"]

        self.discrete_sigmas = torch.exp(
            torch.linspace(
                math.log(self.diffusion_params["sigma_min"]),
                math.log(self.diffusion_params["sigma_max"]),
                self.diffusion_params["num_steps"],
            ),
        )

        self.with_bond_info = self.model_params.get("with_bondinfo", False)

        if self.sde_type == "ve":
            self.sigmas = torch.tensor(
                [
                    self.diffusion_params["sigma_min"]
                    * (self.diffusion_params["sigma_max"] / self.diffusion_params["sigma_min"]) ** t
                    for t in self.timesteps
                ]
            )
            self.sigmas_sample = torch.tensor(
                [
                    self.diffusion_params["sigma_min"]
                    * (self.diffusion_params["sigma_max"] / self.diffusion_params["sigma_min"]) ** t
                    for t in self.timesteps_neg_sequence
                ]
            )

            self.g = torch.tensor(
                [
                    self.diffusion_params["sigma_min"]
                    * (self.diffusion_params["sigma_max"] / self.diffusion_params["sigma_min"]) ** t
                    * torch.sqrt(
                        2
                        * torch.log(
                            torch.tensor(
                                self.diffusion_params["sigma_max"]
                                / self.diffusion_params["sigma_min"]
                            )
                        )
                    )
                    for t in self.timesteps
                ]
            )

    def forward(
        self,
        repeat,
        timesteps,
        atomic_numbers,
        pos,
        edge_index,
        num_graphs,
        batch,
        bond_info=None,
        cell=None,
        cell_offsets=None,
        neighbors=None,
    ):
        return self.model(
            repeat,
            timesteps,
            atomic_numbers,
            pos,
            edge_index,
            num_graphs,
            batch,
            bond_info,
            cell,
            cell_offsets,
            neighbors,
        )

    def get_adjacent_sigma(self, timesteps, t):
        return torch.where(
            timesteps == 0,
            torch.zeros_like(t.to(timesteps.device)),
            self.discrete_sigmas[timesteps - 1].to(timesteps.device),
        )

    def step_pred(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: torch.Generator | None = None,
    ):
        if self.timesteps_neg_sequence is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' "
                "after creating the scheduler"
            )

        timestep = timestep * torch.ones(sample.shape[0], device=sample.device)  # (N, )
        timesteps = (timestep * (len(self.timesteps_neg_sequence) - 1)).long()

        timesteps = timesteps.to(self.discrete_sigmas.device)

        if self.sde_type == "ve":
            sigma = self.discrete_sigmas[timesteps].to(sample.device)
            adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep).to(sample.device)
            drift = torch.zeros_like(sample)
            diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5  # (N, )
            diffusion = diffusion.flatten()
            while len(diffusion.shape) < len(sample.shape):
                diffusion = diffusion.unsqueeze(-1)
            drift = drift - diffusion**2 * model_output  # (N, 3)

            noise = randn_tensor(
                sample.shape,
                layout=sample.layout,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample_mean = sample - drift
            prev_sample = prev_sample_mean + diffusion * noise

        elif self.sde_type == "ddpm" or self.sde_type == "vp":
            beta_t = self.diffusion_params["beta_min"] + timestep * (
                self.diffusion_params["beta_max"] - self.diffusion_params["beta_min"]
            )
            beta_t = beta_t.to(sample.device)
            drift = -0.5 * beta_t[:, None] * sample - beta_t[:, None] * model_output
            diffusion = torch.sqrt(beta_t).unsqueeze(-1)

            step_size = 1.0 / (self.diffusion_params["num_steps"])
            noise = randn_tensor(
                sample.shape,
                layout=sample.layout,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample_mean = sample - drift * step_size
            prev_sample = (
                prev_sample_mean + diffusion * torch.sqrt(torch.tensor(step_size)) * noise
            )

        return prev_sample, prev_sample_mean

    def step_correct(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        timesteps: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        if self.timesteps_neg_sequence is None:
            raise ValueError(
                "`self.timesteps_neg` is not set, you need to run 'set_timesteps' "
                "after creating the scheduler"
            )

        noise = randn_tensor(sample.shape, layout=sample.layout, generator=generator).to(
            sample.device
        )

        grad_norm = torch.norm(model_output.reshape(model_output.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (
            self.diffusion_params["sample_params"]["snr"] * noise_norm / grad_norm
        ) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)  # (N, )

        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        return prev_sample

    @torch.no_grad()
    def sample(
        self,
        example,
        calculator: Potential
        | UFFCalculator
        | ANICalculator
        | UMABatchCalculator,
        uff: UFFCalculator,
        sampling: str = "mn",
        guide_scale: float = 0.1,
        t_mid: float = 100,
        t_scale: float = 150,
        step_num: int = 750,
    ):
        atomic_numbers, edge_index, num_graphs, batch = (
            example.atomic_numbers,
            example.edge_index,
            example.num_graphs,
            example.batch,
        )
        if "cell" in example:
            cell, cell_offsets, neighbors = example.cell, example.cell_offsets, example.neighbors
        else:
            cell, cell_offsets, neighbors = None, None, None

        num_nodes = atomic_numbers.size(0)
        shape = [num_nodes, 3]
        repeat = torch.tensor([example.get_example(i).num_nodes for i in range(num_graphs)]).to(
            atomic_numbers.device
        )

        if self.sde_type == "ve":
            sample = randn_tensor(shape) * self.init_noise_sigma
        elif self.sde_type == "ddpm" or self.sde_type == "vp":
            sample = randn_tensor(shape)
        sample = sample.to(self.device)

        sample_len = self.timesteps_neg_sequence
        bond_info = None if not self.with_bond_info else example.bond_attr
        for i, t in enumerate(tqdm(sample_len, desc="Sampling frames", leave=False)):
            sigma_t = self.sigmas_sample[i] * torch.ones(shape[0], device=self.device)  # (N, )
            timesteps = torch.tensor(
                [self.diffusion_params["num_steps"] - i - 1],
                device=self.device,
            ) * torch.ones(num_graphs, device=self.device)

            # prediction step
            model_output = self(
                repeat,
                timesteps,
                atomic_numbers,
                sample,
                edge_index,
                num_graphs,
                batch,
                bond_info,
                cell,
                cell_offsets,
                neighbors,
            )

            # Guidance schedule: α_i = sigmoid((i - t_mid) / t_scale), active for last step_num steps

            sample_prev = sample.clone()
            if self.model_type == "score":
                sample, sample_mean = self.step_pred(model_output, t, sample)
            elif self.model_type == "epsilon":
                sample, sample_mean = self.step_pred(-model_output / sigma_t[..., None], t, sample)
            pos_delta = sample - sample_prev

            guide_start = self.diffusion_params["num_steps"] - step_num
            if i > guide_start and sampling.lower() == "mi":
                # Mi-guided: apply forces on current noisy state
                guidance = obtain_guidance(atomic_numbers, sample, batch, calculator) * guide_scale
                guidance = torch.where(torch.isnan(guidance), torch.zeros_like(guidance), guidance)

                uff_force = obtain_guidance(atomic_numbers, sample, batch, uff) * guide_scale * 1e-1
                guidance = uff_force + guidance

                sigmoid_logit = torch.full([1], (i - t_mid) / t_scale)
                relative_weight = torch.sigmoid(sigmoid_logit).to(pos_delta)
                pos_delta = (1 - relative_weight) * pos_delta + relative_weight * guidance

                sample = sample_prev + pos_delta
                sample_mean = sample
            elif i > guide_start and sampling.lower() == "mn":
                # MN-guided: apply forces on predicted clean state E[X0|Xt]
                assert self.model_type == "epsilon", "MN sampling only supports epsilon"
                assert self.sde_type == "ve", "MN sampling only supports ve"

                eps_t = self(
                    repeat,
                    timesteps,
                    atomic_numbers,
                    sample_prev,
                    edge_index,
                    num_graphs,
                    batch,
                    bond_info,
                    cell,
                    cell_offsets,
                    neighbors,
                )
                x_0_pred = sample_prev - sigma_t[..., None] * eps_t
                x_0_pred = torch.where(torch.isnan(x_0_pred), sample_prev, x_0_pred)

                guidance = (
                    obtain_guidance(atomic_numbers, x_0_pred, batch, calculator) * guide_scale
                )
                guidance = torch.where(torch.isnan(guidance), torch.zeros_like(guidance), guidance)

                uff_force = (
                    obtain_guidance(atomic_numbers, x_0_pred, batch, uff) * guide_scale * 1e-1
                )
                guidance = uff_force + guidance

                sample = sample + guidance
                sample_mean = sample

        example.pos = sample_mean
        example = example.detach().cpu()
        atoms = Batch2Atoms(example)

        return atoms, None
