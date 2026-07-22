from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tigris_tools.refine_restart.param_block import ParameterBlock

HYDROGEN_MASS_CGS = 1.6738234e-24
K_BOLTZMANN_CGS = 1.38065e-16
LIGHT_SPEED_CGS = 2.99792458e10
E_CHARGE_CGS = 4.80320427e-10
ELECTRON_VOLT_CGS = 1.60218e-12

XE_TABLE = np.array(
    [
        0.0023,
        0.0142,
        0.0704,
        0.2534,
        0.5678,
        0.8177,
        0.9324,
        0.978,
        1.004,
        1.035,
        1.067,
        1.086,
        1.094,
        1.098,
        1.1,
        1.102,
        1.11,
        1.128,
        1.154,
        1.176,
        1.188,
        1.195,
        1.198,
        1.199,
        1.2,
        1.201,
        1.201,
        1.201,
        1.202,
        1.203,
        1.203,
        1.204,
        1.204,
        1.204,
        1.204,
        1.204,
        1.204,
        1.205,
        1.205,
        1.205,
        1.205,
        1.205,
        1.206,
        1.206,
        1.206,
        1.206,
        1.207,
        1.207,
        1.207,
        1.208,
        1.208,
        1.208,
        1.208,
        1.208,
        1.208,
        1.208,
        1.208,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
        1.209,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CodeUnits:
    mass_cgs: float
    length_cgs: float
    time_cgs: float
    mean_mass_per_hydrogen: float

    @property
    def density_cgs(self) -> float:
        return self.mass_cgs / self.length_cgs**3

    @property
    def velocity_cgs(self) -> float:
        return self.length_cgs / self.time_cgs

    @property
    def energy_density_cgs(self) -> float:
        return self.density_cgs * self.velocity_cgs**2

    @property
    def temperature_mu_cgs(self) -> float:
        return self.velocity_cgs**2 * HYDROGEN_MASS_CGS / K_BOLTZMANN_CGS

    @property
    def density_to_nh(self) -> float:
        return self.density_cgs / self.mean_mass_per_hydrogen

    @property
    def cm(self) -> float:
        return 1.0 / self.length_cgs

    @property
    def second(self) -> float:
        return 1.0 / self.time_cgs

    @property
    def gram(self) -> float:
        return 1.0 / self.mass_cgs

    @property
    def speed_of_light(self) -> float:
        return LIGHT_SPEED_CGS / self.velocity_cgs

    @property
    def hydrogen_mass(self) -> float:
        return HYDROGEN_MASS_CGS / self.mass_cgs

    @property
    def erg(self) -> float:
        return self.time_cgs**2 / (self.mass_cgs * self.length_cgs**2)

    @property
    def echarge(self) -> float:
        dyne = self.gram * self.cm / self.second**2
        return E_CHARGE_CGS * np.sqrt(dyne * 4.0 * np.pi) * self.cm


class CRReconstructor:
    """Vectorized port of the single-group TIGRESS++ DefaultOpacity path."""

    def __init__(self, params: ParameterBlock, restart_dir: Path) -> None:
        units = params.values.get("units", {})
        self.units = CodeUnits(
            mass_cgs=float(units["mass_cgs"]),
            length_cgs=float(units["length_cgs"]),
            time_cgs=float(units["time_cgs"]),
            mean_mass_per_hydrogen=float(units["mean_mass_per_hydrogen"]),
        )
        cr = params.values.get("cr", {})
        self.stream = int(cr.get("vs_flag", 1)) != 0
        self.self_consistent = int(cr.get("self_consistent_flag", 0)) != 0
        self.ion_alfven = self.self_consistent or int(cr.get("valfven_flag", 0)) != 0
        self.collisional_ionization = int(cr.get("coll_ion_flag", 0)) != 0
        self.perpendicular_diffusion = int(cr.get("perp_diff_flag", 0)) != 0
        self.sigma_eff = int(cr.get("sigma_eff_flag", 0)) != 0
        self.vmax = float(cr.get("vmax", 1.0e9)) / self.units.velocity_cgs
        self.ecfloor = (
            float(cr.get("ecfloor", np.finfo(float).tiny)) / self.units.energy_density_cgs
        )
        sigma = float(cr.get("sigma", 1.0e-28))
        self.sigma = sigma * self.vmax * self.units.second / self.units.cm**2
        self.max_opacity = float(cr.get("max_opacity", 1.0e10))
        self.perp_to_parallel = float(cr.get("perp_to_par_diff", 10.0))
        self.ion_rate_norm = 1.0e-4
        self.cooling_log_t1, self.cooling_mu = _read_cooling_table(params, restart_dir)
        self.ekin, self.particle_speed = _single_group_momentum(self.units)

    def reconstruct(
        self,
        density: np.ndarray,
        pressure: np.ndarray,
        magnetic: np.ndarray,
        ecr: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        dx, dy, dz = spacing
        grad_pc = np.stack(
            [
                (np.roll(ecr, -1, axis=2) - np.roll(ecr, 1, axis=2)) / (6.0 * dx),
                (np.roll(ecr, -1, axis=1) - np.roll(ecr, 1, axis=1)) / (6.0 * dy),
                (np.roll(ecr, -1, axis=0) - np.roll(ecr, 1, axis=0)) / (6.0 * dz),
            ]
        )
        bmag = np.sqrt(np.sum(magnetic * magnetic, axis=0))
        b_dot_grad = np.sum(magnetic * grad_pc, axis=0)
        grad_parallel = np.divide(
            np.abs(b_dot_grad),
            bmag,
            out=np.zeros_like(bmag),
            where=bmag > np.finfo(float).tiny,
        )
        ion_density = self._ion_density(density, pressure, ecr)
        sigma_parallel = self._sigma_parallel(density, pressure, ecr, ion_density, grad_parallel)
        sigma_perpendicular = (
            sigma_parallel * self.perp_to_parallel
            if self.perpendicular_diffusion
            else np.full_like(sigma_parallel, self.max_opacity)
        )

        sign = np.sign(b_dot_grad)
        alfven_density = ion_density if self.ion_alfven else density
        streaming = -magnetic / np.sqrt(alfven_density) * sign
        if not self.stream:
            streaming[:] = 0.0

        va = bmag / np.sqrt(alfven_density)
        sigma_adv = np.full_like(sigma_parallel, self.max_opacity)
        valid_va = va > np.finfo(float).tiny
        sigma_adv[valid_va] = np.abs(b_dot_grad[valid_va]) / (
            bmag[valid_va] * va[valid_va] * (4.0 / 3.0) * (1.0 / self.vmax) * ecr[valid_va]
        )
        if not self.stream:
            sigma_adv[:] = self.max_opacity

        bhat = np.divide(
            magnetic,
            bmag,
            out=np.zeros_like(magnetic),
            where=bmag[None, ...] > np.finfo(float).tiny,
        )
        gradient_along_b = bhat * np.sum(bhat * grad_pc, axis=0)
        gradient_across_b = grad_pc - gradient_along_b
        factor = -self.vmax / ((4.0 / 3.0) * ecr)
        with np.errstate(divide="ignore", invalid="ignore"):
            diffusion_velocity = factor * (
                gradient_along_b / sigma_parallel + gradient_across_b / sigma_perpendicular
            )
        return {
            "Sigma_diff1": sigma_parallel,
            "Sigma_adv1": sigma_adv,
            "Vs1": streaming[0],
            "Vs2": streaming[1],
            "Vs3": streaming[2],
            "Vd1": diffusion_velocity[0],
            "Vd2": diffusion_velocity[1],
            "Vd3": diffusion_velocity[2],
        }

    def _ion_density(self, density, pressure, ecr):
        temperature, mu = self._temperature_mu(density, pressure)
        nh = density * self.units.density_to_nh
        ecr_cgs = np.maximum(ecr, self.ecfloor) * self.units.energy_density_cgs
        xion = self._ion_fraction(temperature, nh, ecr_cgs)
        mui = _ion_mean_mass(temperature, mu, xion)
        rho_ion = xion * nh * mui * HYDROGEN_MASS_CGS / self.units.density_cgs
        return np.minimum(rho_ion, 0.9999 * density)

    def _sigma_parallel(self, density, pressure, ecr, ion_density, grad_parallel):
        if not self.self_consistent:
            return np.full_like(ecr, self.sigma)
        temperature, mu = self._temperature_mu(density, pressure)
        xion = self._ion_fraction(
            temperature,
            density * self.units.density_to_nh,
            np.maximum(ecr, self.ecfloor) * self.units.energy_density_cgs,
        )
        mui = _ion_mean_mass(temperature, mu, xion)
        neutral_density = _neutral_number_density(
            temperature, density, ion_density, mui, self.units.density_cgs
        )
        ion_thermal_speed = np.sqrt(1.67 * pressure / density)
        n1 = 0.1 * np.maximum(ecr, self.ecfloor) / self.ekin
        nu = 3.0e-9 / self.units.second
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_in = (
                3.0
                / 8.0
                * np.pi
                * grad_parallel
                / ecr
                * n1
                * self.units.echarge
                / self.units.speed_of_light
                / np.sqrt(ion_density)
                / (nu * neutral_density)
            )
            sigma_nll = np.sqrt(
                3.0
                / 16.0
                * np.pi
                * grad_parallel
                / ecr
                * n1
                * self.units.echarge
                / self.particle_speed**2
                / np.sqrt(ion_density)
                / (0.3 * ion_thermal_speed)
            )
        if self.sigma_eff:
            sigma_nll *= 0.4796471
            sigma_in *= 0.3206159
        sigma_parallel = np.minimum(sigma_nll, sigma_in) * self.vmax
        return np.minimum(sigma_parallel, self.sigma)

    def _temperature_mu(self, density, pressure):
        t1 = pressure / density * self.units.temperature_mu_cgs
        mu = _linear_uniform_extrapolate(np.log10(t1), self.cooling_log_t1, self.cooling_mu)
        return mu * t1, mu

    def _ion_fraction(self, temperature, nh, ecr_cgs):
        result = np.empty_like(temperature)
        low = temperature < 2.0e4
        xi_cr = ecr_cgs[low] * self.ion_rate_norm
        if self.collisional_ionization:
            temp = temperature[low]
            alpha_rr = 2.59e-13 * (temp * 1.0e-4) ** -0.7
            beta_coll = 5.84e-11 * np.sqrt(temp) * np.exp(-157821.44645308115 / temp) / alpha_rr
            beta_cr = 1.5 * xi_cr / (alpha_rr * nh[low])
            beta_gr = 2.83e-14 / alpha_rr
            xm = 1.68e-4
            aa = 1.0 + beta_coll
            bb = -2.0 - xm * (1.0 + beta_coll) - (beta_gr + beta_cr + beta_coll)
            cc = 1.0 + xm + beta_gr
            xhi = (-bb - np.sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa)
            result[low] = 1.0 - xhi + xm
        else:
            alpha_rr = 1.42e-12
            beta = 1.5 * xi_cr / nh[low] / alpha_rr
            gamma_cr = 2.83e-14 / alpha_rr
            xm = 1.68e-4
            xe = (
                0.5 * (np.sqrt((beta + gamma_cr + xm) ** 2 + 4.0 * beta) - (beta + gamma_cr + xm))
                + xm
            )
            result[low] = np.minimum(np.minimum(xe, 1.209), 1.099)
        high_temperature = temperature[~low]
        log_temperature = np.log10(high_temperature)
        table_log_t = 4.0 + 0.05 * np.arange(XE_TABLE.size)
        table_index = ((log_temperature - table_log_t[0]) / 0.05).astype(np.int64)
        table_index = np.maximum(table_index, 0)
        table_index = np.where(log_temperature >= table_log_t[-1], XE_TABLE.size - 2, table_index)
        lower_temperature = 10.0 ** table_log_t[table_index]
        upper_temperature = 10.0 ** table_log_t[table_index + 1]
        fraction = (high_temperature - lower_temperature) / (upper_temperature - lower_temperature)
        high = XE_TABLE[table_index] * (1.0 - fraction) + XE_TABLE[table_index + 1] * fraction
        result[~low] = np.minimum(high, 1.099)
        return result


def _single_group_momentum(units: CodeUnits) -> tuple[float, float]:
    kinetic_energy = 1.0e9 * ELECTRON_VOLT_CGS * units.erg
    momentum = np.sqrt(
        (kinetic_energy / units.speed_of_light) ** 2 + 2.0 * kinetic_energy * units.hydrogen_mass
    )
    total_energy = np.sqrt(
        (momentum * units.speed_of_light) ** 2
        + (units.hydrogen_mass * units.speed_of_light**2) ** 2
    )
    particle_speed = units.speed_of_light * np.sqrt(
        1.0 - (units.hydrogen_mass * units.speed_of_light**2 / total_energy) ** 2
    )
    return total_energy - units.hydrogen_mass * units.speed_of_light**2, particle_speed


def _read_cooling_table(params: ParameterBlock, restart_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    filename = params.get("cooling", "coolftn_file")
    if filename is None:
        raise ValueError("self-consistent CR reconstruction requires <cooling>/coolftn_file")
    path = Path(filename)
    if not path.is_absolute():
        path = restart_dir / path
    rows = []
    with path.open() as stream:
        lines = (line for line in stream if not line.startswith("#"))
        shape = next(lines).split()
        nvar, count = int(shape[0]), int(shape[1])
        next(lines)
        for _ in range(count):
            rows.append([float(value) for value in next(lines).split()[:nvar]])
    table = np.asarray(rows)
    return table[:, 0], table[:, 1]


def _linear_uniform_extrapolate(values, coordinates, table):
    scale = (len(coordinates) - 1) / (coordinates[-1] - coordinates[0])
    position = (values - coordinates[0]) * scale
    lower = position.astype(np.int64)
    lower = np.where(values < coordinates[0], 0, lower)
    lower = np.where(lower >= len(coordinates) - 1, len(coordinates) - 2, lower)
    residual = 1.0 + lower - position
    return residual * table[lower] + (1.0 - residual) * table[lower + 1]


def _ion_mean_mass(temperature, mu, xion):
    del temperature, mu  # Retained in the signature to match TIGRESS++ Get_mui.
    xm = 1.68e-4
    xh = xion - xm
    xhe = np.maximum(xh - 1.0, 0.0)
    xh = np.minimum(xh, 1.0)
    return (xh + 4.0 * xhe + 12.0 * xm) / xion


def _neutral_number_density(temperature, density, ion_density, mui, density_cgs):
    mun = np.where(temperature <= 100.0, 2.0, 1.0)
    return (density - ion_density) * density_cgs / ((mun + mui) * HYDROGEN_MASS_CGS)
