import sys
from typing import ClassVar

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import Filter
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from mattersim.datasets.utils.build import build_dataloader as build_mattersim_dataloader
from mattersim.forcefield.potential import Potential
from tqdm import tqdm

from src.ani_calculator import ANICalculator
from src.uff_calculator import UFFCalculator
from src.uff_calculator import build_dataloader as build_uff_dataloader
from src.uma_calculator import UMABatchCalculator
from src.uma_calculator import build_dataloader as build_uma_dataloader


class DummyBatchCalculator(Calculator):
    def __init__(self):
        super().__init__()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass

    def get_potential_energy(self, atoms=None):
        return atoms.info["total_energy"]

    def get_forces(self, atoms=None):
        return atoms.arrays["forces"]

    def get_stress(self, atoms=None):
        return units.GPa * atoms.info["stress"]


class BatchRelaxer:
    """BatchRelaxer is a class for batch structural relaxation.
    It is more efficient than Relaxer when relaxing a large number of structures.

    The relax method returns two dictionaries:
    - converged_dict: structures that converged, grouped by chemical formula
    - unconverged_dict: structures that stopped due to max_relaxation_step
    """

    SUPPORTED_OPTIMIZERS: ClassVar[dict[str, type[Optimizer]]] = {
        "LBFGS": LBFGS,
        "BFGS": BFGS,
        "FIRE": FIRE,
    }
    SUPPORTED_FILTERS: ClassVar[dict[str, type[Filter]]] = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        potential: Potential | ANICalculator | UFFCalculator,
        optimizer: str | type[Optimizer] = "FIRE",
        filter: type[Filter] | str | None = None,
        fmax: float = 0.05,
        max_natoms_per_batch: int = 512,
        max_relaxation_step: int | None = None,
    ):
        self.potential = potential
        self.is_mattersim = isinstance(potential, Potential)
        self.is_uma = isinstance(potential, UMABatchCalculator)
        self.device = potential.device

        self.max_opt_step = 0.1

        if isinstance(optimizer, str):
            if optimizer.upper() not in self.SUPPORTED_OPTIMIZERS:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
        elif issubclass(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        if isinstance(filter, str):
            if filter.upper() not in self.SUPPORTED_FILTERS:
                raise ValueError(f"Unsupported filter: {filter}")
            self.filter = self.SUPPORTED_FILTERS[filter.upper()]
        elif filter is None or issubclass(filter, Filter):
            self.filter = filter
        else:
            raise ValueError(f"Unsupported filter: {filter}")
        self.fmax = fmax
        self.max_natoms_per_batch = max_natoms_per_batch
        self.max_relaxation_step = max_relaxation_step
        self.optimizer_instances: list[Optimizer] = []
        self.is_active_instance: list[bool] = []
        self.step_count: list[int] = []
        self.finished = False
        self.total_converged = 0
        self.trajectories: dict[int, list[Atoms]] = {}
        self.converged_status: dict[int, bool] = {}

    def insert(self, atoms: Atoms):
        # atoms.set_calculator(DummyBatchCalculator())
        atoms.calc = DummyBatchCalculator()
        optimizer_instance = self.optimizer(
            self.filter(atoms) if self.filter else atoms, maxstep=self.max_opt_step
        )
        optimizer_instance.fmax = self.fmax
        self.optimizer_instances.append(optimizer_instance)
        self.is_active_instance.append(True)
        self.step_count.append(0)

    def step_batch(self):
        atoms_list = []
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                atoms_list.append(opt.atoms)

        # Note: we use a batch size of len(atoms_list)
        # because we only want to run one batch at a time
        if self.is_mattersim:
            dataloader = build_mattersim_dataloader(
                atoms_list, batch_size=len(atoms_list), only_inference=True
            )
        elif isinstance(self.potential, UFFCalculator):
            dataloader = build_uff_dataloader(
                atoms_list, batch_size=len(atoms_list), only_inference=True
            )
        elif self.is_uma:
            dataloader = build_uma_dataloader(
                atoms_list, batch_size=len(atoms_list), only_inference=True
            )
        elif isinstance(self.potential, ANICalculator):
            from src.ani_calculator import build_dataloader as build_ani_dataloader

            dataloader = build_ani_dataloader(
                atoms_list, batch_size=len(atoms_list), only_inference=True
            )
        else:
            raise ValueError(f"Unsupported calculator type: {type(self.potential)}")
        energy_batch, forces_batch, stress_batch = self.potential.predict_properties(
            dataloader, include_forces=True, include_stresses=True
        )

        counter = 0
        self.finished = True
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                # Set the properties so the dummy calculator can
                # return them within the optimizer step
                opt.atoms.info["total_energy"] = energy_batch[counter]
                opt.atoms.arrays["forces"] = forces_batch[counter]
                opt.atoms.info["stress"] = stress_batch[counter]
                try:
                    self.trajectories[opt.atoms.info["structure_index"]].append(opt.atoms.copy())
                except KeyError:
                    self.trajectories[opt.atoms.info["structure_index"]] = [opt.atoms.copy()]

                opt.step()
                self.step_count[idx] += 1
                if opt.converged():
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    self.converged_status[opt.atoms.info["structure_index"]] = True
                    self.tqdmcounter.update(1)
                    # if self.total_converged % 100 == 0:
                    # logger.info(f"Relaxed {self.total_converged} structures.")
                elif (
                    self.max_relaxation_step is not None
                    and self.step_count[idx] >= self.max_relaxation_step
                ):
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    self.converged_status[opt.atoms.info["structure_index"]] = False
                    self.tqdmcounter.update(1)
                else:
                    self.finished = False
                counter += 1

        # remove inactive instances
        self.optimizer_instances = [
            opt
            for opt, active in zip(self.optimizer_instances, self.is_active_instance, strict=False)
            if active
        ]
        self.step_count = [
            count
            for count, active in zip(self.step_count, self.is_active_instance, strict=False)
            if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)

    def relax(
        self,
        atoms_list: list[Atoms],
    ) -> tuple[dict[str, list[Atoms]], dict[str, list[Atoms]]]:
        self.trajectories = {}
        self.converged_status = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout, desc="Relaxing structures")
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while (
            pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                sum([len(opt.atoms) for opt in self.optimizer_instances]) + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):
                # While there are enough n_atoms slots in the
                # batch and we have not reached the end of the list.
                self.insert(atoms_list_[pointer])  # Insert new structure to fire instances
                pointer += 1
            self.step_batch()
        self.tqdmcounter.close()

        # converged_dict: dict[str, list[Atoms]] = {}
        # unconverged_dict: dict[str, list[Atoms]] = {}
        converged_list: list[Atoms] = []
        unconverged_list: list[Atoms] = []

        for structure_idx, trajectory in self.trajectories.items():
            final_atoms = trajectory[-1]
            if self.converged_status.get(structure_idx, False):
                converged_list.append(final_atoms)
            else:
                unconverged_list.append(final_atoms)

        return converged_list, unconverged_list
