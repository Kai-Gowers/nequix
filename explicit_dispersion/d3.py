"""
Explicit DFT-D3(BJ) dispersion for Nequix delta-learning ("bare PBE" fine-tuning).

The reference DFT data (workflow/FINAL_RESULTS_HEALTHY) was computed with VASP PBE + DFT-D3(BJ),
`IVDW = 12` with VASP defaults. This module evaluates the *same* D3 term with the simple-dftd3
library so it can be (a) subtracted from the training targets and (b) added back to model
predictions. Parameters and cutoffs below are copied from a VASP OUTCAR of this dataset:

    VDW_S6 = 1.0000  VDW_S8 = 0.7875  VDW_A1 = 0.4289  VDW_A2 = 4.4407
    VDW_RADIUS = 50.2022 A  (two-body real-space cutoff)
    VDW_CNRADIUS = 21.1671 A (coordination-number cutoff)
    no three-body (ATM) term  ->  s9 = 0

Checked on MoS2 disp-001 (48 atoms): E_D3 here = -14.99917 eV vs VASP Edisp = -14.99931 eV.

Units everywhere in this module: eV, Angstrom (ASE conventions). simple-dftd3 works in
Hartree/Bohr internally; conversions happen at the boundary.

Two Hessian layouts are used in this project and must never be mixed by hand:
  * "phonopy" layout  : (n, n, 3, 3)      — what NequixCalculator.get_hessian returns and phonopy consumes
  * "training" layout : (3n, 3n) from fc.swapaxes(1, 2).reshape(3n, 3n) — what convert_to_nequix.py stores
Use `to_training_layout` / `from_training_layout`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree
from phonopy.harmonic.force_constants import compact_fc_to_full_fc

import dftd3.interface as d3i

# --------------------------------------------------------------------------------------
# VASP IVDW=12 (PBE-D3(BJ)) parametrisation, as printed in this dataset's OUTCARs
# --------------------------------------------------------------------------------------
VASP_PBE_D3BJ = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407, "s9": 0.0}
VASP_CUTOFF_DISP2_A = 50.2022  # VDW_RADIUS
VASP_CUTOFF_CN_A = 21.1671  # VDW_CNRADIUS
_CUTOFF_DISP3_A = 0.0  # three-body term is off (s9 = 0); cutoff irrelevant but set to zero


def make_param() -> d3i.RationalDampingParam:
    """Rational (Becke-Johnson) damping with VASP's PBE defaults and no three-body term."""
    return d3i.RationalDampingParam(**VASP_PBE_D3BJ)


def _model(atoms: Atoms) -> d3i.DispersionModel:
    if not atoms.pbc.all():
        raise ValueError("D3 term is set up for fully periodic cells (2D slabs use a vacuum along z)")
    model = d3i.DispersionModel(
        np.asarray(atoms.numbers),
        np.asarray(atoms.positions) / Bohr,
        np.asarray(atoms.cell.array) / Bohr,
        np.array([True, True, True]),
    )
    model.set_realspace_cutoff(VASP_CUTOFF_DISP2_A / Bohr, _CUTOFF_DISP3_A / Bohr, VASP_CUTOFF_CN_A / Bohr)
    return model


def d3_efs(atoms: Atoms) -> dict:
    """D3 energy [eV], forces [(n,3) eV/A] and stress [(3,3) eV/A^3, ASE sign convention]."""
    res = _model(atoms).get_dispersion(make_param(), grad=True)
    energy = float(res["energy"]) * Hartree
    forces = -np.asarray(res["gradient"]) * (Hartree / Bohr)
    stress = np.asarray(res["virial"]) * Hartree / atoms.get_volume()  # same as dftd3.ase.DFTD3
    return {"energy": energy, "forces": forces, "stress": stress}


def d3_hessian_analytic(atoms: Atoms) -> np.ndarray:
    """Analytic D3 Hessian in phonopy layout (n, n, 3, 3), eV/A^2."""
    n = len(atoms)
    h = np.asarray(_model(atoms).get_hessian(make_param())["hessian"]) * (Hartree / Bohr**2)  # (3n,3n), index 3i+c
    return h.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)


def d3_force_constants_phonopy(ph) -> np.ndarray:
    """
    D3 force constants built *exactly* the way the DFT reference FCs were built: evaluate D3
    forces on phonopy's displaced supercells (the same displacements VASP ran, from FORCE_SETS),
    then `produce_force_constants` + `symmetrize_force_constants`. Because FC construction is
    linear in the forces, FC_ref - FC_D3 is precisely the FC phonopy would give for PBE-only forces.

    `ph` must have been loaded with `force_sets_filename=...` so `ph.dataset` holds the displacements.
    Returns full (n_sc, n_sc, 3, 3) in eV/A^2.
    """
    if ph.dataset is None or "first_atoms" not in ph.dataset:
        raise ValueError("phonopy object has no displacement dataset; load it with force_sets_filename=")
    ph_d3 = copy.deepcopy(ph)
    dataset = copy.deepcopy(ph.dataset)
    for disp, supercell in zip(dataset["first_atoms"], ph_d3.supercells_with_displacements):
        atoms = Atoms(numbers=supercell.numbers, positions=supercell.positions, cell=supercell.cell, pbc=True)
        disp["forces"] = d3_efs(atoms)["forces"]
    ph_d3.dataset = dataset
    ph_d3.produce_force_constants()
    ph_d3.symmetrize_force_constants()
    fc = ph_d3.force_constants
    if fc.shape[0] != fc.shape[1]:
        fc = compact_fc_to_full_fc(ph_d3.primitive, fc)
    return np.asarray(fc)


# --------------------------------------------------------------------------------------
# Hessian layout helpers
# --------------------------------------------------------------------------------------
def to_training_layout(fc_full: np.ndarray) -> np.ndarray:
    n = fc_full.shape[0]
    return np.asarray(fc_full).swapaxes(1, 2).reshape(3 * n, 3 * n)


def from_training_layout(h: np.ndarray) -> np.ndarray:
    n = h.shape[0] // 3
    return np.asarray(h).reshape(n, 3, n, 3).swapaxes(1, 2)


def asr_residual(fc_full: np.ndarray) -> float:
    """Max |sum_j FC_ij| — zero for a translationally invariant force-constant matrix."""
    return float(np.abs(fc_full.sum(axis=1)).max())


# --------------------------------------------------------------------------------------
# Cache of per-material D3 terms (written by strip_d3_dataset.py, read at inference)
# --------------------------------------------------------------------------------------
def save_d3_cache(path: Path, *, fc_d3: np.ndarray, energy: float, forces: np.ndarray, stress: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, fc_d3=fc_d3.astype(np.float64), energy=energy, forces=forces, stress=stress,
        s6=VASP_PBE_D3BJ["s6"], s8=VASP_PBE_D3BJ["s8"], a1=VASP_PBE_D3BJ["a1"], a2=VASP_PBE_D3BJ["a2"],
        cutoff_disp2_A=VASP_CUTOFF_DISP2_A, cutoff_cn_A=VASP_CUTOFF_CN_A,
    )


def load_d3_cache(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------
# ASE calculators: explicit D3 alone, and Nequix + D3
# --------------------------------------------------------------------------------------
class D3Calculator(Calculator):
    """Stand-alone ASE calculator for the VASP-matched D3(BJ) term."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        res = d3_efs(self.atoms)
        self.results = {
            "energy": res["energy"],
            "free_energy": res["energy"],
            "forces": res["forces"],
            "stress": res["stress"].flat[[0, 4, 8, 5, 2, 1]],  # voigt, like every ASE calculator
        }


class NequixD3Calculator(Calculator):
    """
    Nequix model trained on bare-PBE targets + explicit D3 added back on the fly:
        E = E_nequix + E_D3,  F = F_nequix + F_D3,  sigma = sigma_nequix + sigma_D3,
    and `get_hessian(atoms, fc_d3=None)` = H_nequix + H_D3 in phonopy layout (n, n, 3, 3).
    Pass `fc_d3` (e.g. from the per-material cache) to reuse exactly the D3 FCs that were
    subtracted at training time; otherwise the analytic D3 Hessian is computed on the fly.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, nequix_calc, **kwargs):
        super().__init__(**kwargs)
        self.nequix = nequix_calc
        self.d3 = D3Calculator()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms.copy()
        a.calc = self.nequix
        e_ml, f_ml, s_ml = a.get_potential_energy(), a.get_forces(), a.get_stress()
        b = self.atoms.copy()
        b.calc = self.d3
        e_d3, f_d3, s_d3 = b.get_potential_energy(), b.get_forces(), b.get_stress()
        self.results = {"energy": e_ml + e_d3, "free_energy": e_ml + e_d3, "forces": f_ml + f_d3, "stress": s_ml + s_d3}

    def get_hessian(self, atoms: Atoms, fc_d3: np.ndarray | None = None) -> np.ndarray:
        h_ml = np.asarray(self.nequix.get_hessian(atoms))
        h_d3 = d3_hessian_analytic(atoms) if fc_d3 is None else np.asarray(fc_d3)
        if h_d3.shape != h_ml.shape:
            raise ValueError(f"D3 Hessian shape {h_d3.shape} != model Hessian shape {h_ml.shape}")
        return h_ml + h_d3
