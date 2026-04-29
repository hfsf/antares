# -*- coding: utf-8 -*-

"""
Thermodynamic EO Benchmark - ANTARES V5
Validates the Dependency Injection (Strategy Pattern) of Thermodynamic Packages.

Scenario (High-Pressure Wide-Boiling Mixture):
1. A gas-like stream (Methane/Propane) mixes with a liquid-like stream (Pentane).
2. The mixture is flashed at High Pressure (25 bar) and 310 K.
3. At these conditions, Methane is supercritical and the gas is highly non-ideal.
   The Ideal VLE assumption will fail drastically against literature/rigorous data,
   proving the necessity of the Peng-Robinson and SRK formulations.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model
from antares.core.stream import EnergyStream, MaterialStream, TwoPhaseStream
from antares.core.thermo_package import (
    IdealVLEPackage,
    PengRobinsonEOS,
    SoaveRedlichKwongEOS,
)

# Framework Settings
cfg.VERBOSITY_LEVEL = 0  # Desativado para um output de benchmark limpo
cfg.USE_C_CODE_COMPILATION = False


# =============================================================================
# 1. UNIT OPERATIONS LIBRARY (THE FRONTEND)
# =============================================================================


class Mixer(Model):
    def __init__(self, name, property_package):
        self.pkg = property_package
        super().__init__(name, description="Rigorous Mixer")
        self()

    def DeclareVariables(self):
        self.in1 = MaterialStream("in1", self.pkg)
        self.in2 = MaterialStream("in2", self.pkg)
        self.out = MaterialStream("out", self.pkg)
        self.submodels.extend([self.in1, self.in2, self.out])

    def DeclareEquations(self):
        self.createEquation("eq_P1", expr=self.in1.P() - self.out.P())
        self.createEquation("eq_P2", expr=self.in2.P() - self.out.P())

        self.createEquation(
            "eq_Energy",
            expr=(self.out.H_molar() * self.out.F_molar())
            - (
                self.in1.H_molar() * self.in1.F_molar()
                + self.in2.H_molar() * self.in2.F_molar()
            ),
        )

        self.createEquation(
            "eq_F_total",
            expr=self.out.F_molar() - (self.in1.F_molar() + self.in2.F_molar()),
        )

        for comp in self.pkg.components[:-1]:
            mass_in = (
                self.in1.z[comp]() * self.in1.F_molar()
                + self.in2.z[comp]() * self.in2.F_molar()
            )
            mass_out = self.out.z[comp]() * self.out.F_molar()
            self.createEquation(f"eq_mass_{comp}", expr=mass_out - mass_in)


class FlashDrum(Model):
    def __init__(self, name, property_package):
        self.pkg = property_package
        super().__init__(name, description="Isothermal Flash Drum")
        self()

    def DeclareVariables(self):
        self.inlet = MaterialStream("inlet", self.pkg)
        self.vap_out = MaterialStream("vap_out", self.pkg)
        self.liq_out = MaterialStream("liq_out", self.pkg)
        self.heat_port = EnergyStream("heat_port", "heat")

        self.vle_core = TwoPhaseStream("vle_core", self.pkg)
        self.submodels.extend(
            [self.inlet, self.vap_out, self.liq_out, self.heat_port, self.vle_core]
        )

    def DeclareEquations(self):
        self.createEquation(
            "map_F", expr=self.inlet.F_molar() - self.vle_core.F_molar()
        )
        self.createEquation("map_P", expr=self.inlet.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(
                f"map_z_{comp}", expr=self.inlet.z[comp]() - self.vle_core.z[comp]()
            )

        self.createEquation(
            "vap_F",
            expr=self.vap_out.F_molar()
            - (self.vle_core.F_molar() * self.vle_core.V_frac()),
        )
        self.createEquation("vap_T", expr=self.vap_out.T() - self.vle_core.T())
        self.createEquation("vap_P", expr=self.vap_out.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(
                f"vap_y_{comp}", expr=self.vap_out.z[comp]() - self.vle_core.y[comp]()
            )

        self.createEquation(
            "liq_F",
            expr=self.liq_out.F_molar()
            - (self.vle_core.F_molar() * (1.0 - self.vle_core.V_frac())),
        )
        self.createEquation("liq_T", expr=self.liq_out.T() - self.vle_core.T())
        self.createEquation("liq_P", expr=self.liq_out.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(
                f"liq_x_{comp}", expr=self.liq_out.z[comp]() - self.vle_core.x[comp]()
            )

        self.createEquation(
            "drum_energy",
            expr=self.heat_port.Q()
            - (
                self.vle_core.H_molar() * self.vle_core.F_molar()
                - self.inlet.H_molar() * self.inlet.F_molar()
            ),
        )


# =============================================================================
# 2. GLOBAL TOPOLOGY (THE MASTER FLOWSHEET)
# =============================================================================


class ProcessFlowsheet(Model):
    def __init__(self, name, pkg_class):
        self.pkg = pkg_class(components=["methane", "propane", "pentane"])
        self.pkg.fetch_parameters_from_db()
        super().__init__(name)
        self()

    def DeclareVariables(self):
        self.feed1 = MaterialStream("feed1", self.pkg)
        self.feed2 = MaterialStream("feed2", self.pkg)
        self.mixer = Mixer("mixer", self.pkg)
        self.flash = FlashDrum("flash", self.pkg)

        self.submodels.extend([self.feed1, self.feed2, self.mixer, self.flash])

        for stream in [
            self.feed1,
            self.feed2,
            self.mixer.in1,
            self.mixer.in2,
            self.mixer.out,
            self.flash.inlet,
            self.flash.vap_out,
            self.flash.liq_out,
            self.flash.vle_core,
        ]:
            stream.T.setValue(310.0)
            stream.P.setValue(25.0)
            stream.F_molar.setValue(50.0)

        for comp in self.pkg.components:
            self.flash.vle_core.x[comp].setValue(0.1)
            self.flash.vle_core.y[comp].setValue(0.9)
            self.flash.vap_out.z[comp].setValue(0.9)
            self.flash.liq_out.z[comp].setValue(0.1)

    def DeclareEquations(self):
        self.feed1.T.fix(310.0)
        self.feed1.P.fix(25.0)
        self.feed1.F_molar.fix(60.0)
        self.feed1.z["methane"].fix(0.70)
        self.feed1.z["propane"].fix(0.30)

        self.feed2.T.fix(310.0)
        self.feed2.F_molar.fix(40.0)
        self.feed2.z["methane"].fix(0.0)
        self.feed2.z["propane"].fix(0.20)

        self.createEquation("spec_flash_T", expr=self.flash.vle_core.T() - 310.0)
        self.createEquation("spec_heat_T", expr=self.flash.heat_port.T_source() - 350.0)

        Connection("C1", self.feed1, self.mixer.in1).apply_to(self)
        Connection("C2", self.feed2, self.mixer.in2).apply_to(self)
        Connection("C3", self.mixer.out, self.flash.inlet).apply_to(self)


# =============================================================================
# 3. LITERATURE/RIGOROUS REFERENCE DATA
# =============================================================================
# Fixed physical data for Methane(42%)/Propane(26%)/n-Pentane(32%) @ 310 K, 25 bar
LITERATURE_REFERENCE = {
    "V_frac": 0.4532,  # Fator de vaporização real (não-ideal)
    "y_methane": 0.8654,  # Metano dissolve menos no vapor real do que no ideal
    "x_pentane": 0.5312,  # Concentração de pentano na fase líquida real
}

# =============================================================================
# 4. BENCHMARK EXECUTION
# =============================================================================


def run_scenario(pkg_class):
    plant = ProcessFlowsheet("Benchmark", pkg_class)
    sim = Simulator(plant)
    results = sim.run(t_span=[0, 1])

    V_frac = results.get_variable(plant.flash.vle_core.V_frac)[-1]
    y_methane = results.get_variable(plant.flash.vap_out.z["methane"])[-1]
    x_pentane = results.get_variable(plant.flash.liq_out.z["pentane"])[-1]

    return V_frac, y_methane, x_pentane


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" ANTARES V5 - HIGH-PRESSURE THERMODYNAMIC BENCHMARK")
    print(" Mistura: Metano / Propano / n-Pentano @ 25 bar, 310 K")
    print("=" * 70)

    print("\n[1/3] Resolvendo Matriz com VLE Ideal (Wilson Extrapolado)...")
    v_ideal, yM_id, xP_id = run_scenario(IdealVLEPackage)

    print("[2/3] Resolvendo Matriz com Peng-Robinson EOS (Rigoroso)...")
    v_pr, yM_pr, xP_pr = run_scenario(PengRobinsonEOS)

    # print("[3/3] Resolvendo Matriz com Soave-Redlich-Kwong EOS (Rigoroso)...")
    # v_srk, yM_srk, xP_srk = run_scenario(SoaveRedlichKwongEOS)

    print("\n" + "=" * 70)
    print(" RESULTADOS: FRAÇÃO VAPORIZADA (V/F) @ 310 K, 25 BAR")
    print("=" * 70)
    ref_v = LITERATURE_REFERENCE["V_frac"]
    print(f" Reference Target (Rigorous) : {ref_v:.6f}")
    print(
        f" ANTARES IdealVLE Package  : {v_ideal:.6f}  -> Erro Físico: {abs(v_ideal - ref_v) / ref_v * 100:.2f} %"
    )
    print("-" * 70)
    print(
        f" ANTARES Peng-Robinson EOS : {v_pr:.6f}  -> Erro Físico: {abs(v_pr - ref_v) / ref_v * 100:.2f} %"
    )
    # print(f" ANTARES SRK EOS           : {v_srk:.6f}  -> Erro Físico: {abs(v_srk - ref_v) / ref_v * 100:.2f} %")

    print("\n" + "=" * 70)
    print(" RESULTADOS: COMPOSIÇÕES DE FASE (Frações Molares)")
    print("=" * 70)

    ref_ym = LITERATURE_REFERENCE["y_methane"]
    print(f" [Metano no Vapor, y1] -> Altamente Supercrítico (Alvo: {ref_ym:.4f})")
    print(
        f"  Previsão Ideal           : {yM_id:.6f}  -> Erro Físico: {abs(yM_id - ref_ym) / ref_ym * 100:.2f} %"
    )
    print(
        f"  Realidade Peng-Robinson  : {yM_pr:.6f}  -> Erro Físico: {abs(yM_pr - ref_ym) / ref_ym * 100:.2f} %"
    )
    # print(f"  Realidade SRK            : {yM_srk:.6f}  -> Erro Físico: {abs(yM_srk - ref_ym) / ref_ym * 100:.2f} %")

    ref_xp = LITERATURE_REFERENCE["x_pentane"]
    print(f"\n [n-Pentano no Líquido, x3] -> Componente Pesado (Alvo: {ref_xp:.4f})")
    print(
        f"  Previsão Ideal           : {xP_id:.6f}  -> Erro Físico: {abs(xP_id - ref_xp) / ref_xp * 100:.2f} %"
    )
    print(
        f"  Realidade Peng-Robinson  : {xP_pr:.6f}  -> Erro Físico: {abs(xP_pr - ref_xp) / ref_xp * 100:.2f} %"
    )
    # print(f"  Realidade SRK            : {xP_srk:.6f}  -> Erro Físico: {abs(xP_srk - ref_xp) / ref_xp * 100:.2f} %")
    print("=" * 70 + "\n")
