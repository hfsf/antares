# -*- coding: utf-8 -*-

"""
Thermodynamic EO Benchmark - ANTARES V5
Classic Literature Benchmark: Isothermal Steady-State Flash (Algebraic).

Objetivo: Comprovar a solidez dos 3 pacotes termodinâmicos (Ideal, PR, SRK)
utilizando a matemática isolada que sabidamente converge no KINSOL.
"""

import casadi as ca
import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model
from antares.core.stream import MaterialStream, TwoPhaseStream
from antares.library.thermo_package import (
    IdealVLEPackage,
    PengRobinsonEOS,
    SoaveRedlichKwongEOS,
)

# =============================================================================
# BENCHMARK CONTROL PANEL
# =============================================================================
TEST_IDEAL = True
TEST_PR = True
TEST_SRK = True

cfg.VERBOSITY_LEVEL = 1
cfg.USE_C_CODE_COMPILATION = False


class IsothermalFlashDrum(Model):
    r"""Tambor de Flash Puramente Mássico e Algébrico."""
    def __init__(self, name, property_package):
        self.pkg = property_package
        super().__init__(name, description="Pure Mass Isothermal Flash Drum")
        self()

    def DeclareVariables(self):
        self.inlet = MaterialStream("inlet", self.pkg)
        self.vap_out = MaterialStream("vap_out", self.pkg)
        self.liq_out = MaterialStream("liq_out", self.pkg)
        
        # O Core de Equilíbrio Termodinâmico
        self.vle_core = TwoPhaseStream("vle_core", self.pkg)
        self.submodels.extend([self.inlet, self.vap_out, self.liq_out, self.vle_core])

    def DeclareEquations(self):
        # Mapeamento da Entrada
        self.createEquation("map_F", expr=self.inlet.F_molar() - self.vle_core.F_molar())
        self.createEquation("map_P", expr=self.inlet.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(f"map_z_{comp}", expr=self.inlet.z[comp]() - self.vle_core.z[comp]())

        # Mapeamento do Vapor
        self.createEquation("vap_F", expr=self.vap_out.F_molar() - (self.vle_core.F_molar() * self.vle_core.V_frac()))
        self.createEquation("vap_T", expr=self.vap_out.T() - self.vle_core.T())
        self.createEquation("vap_P", expr=self.vap_out.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(f"vap_y_{comp}", expr=self.vap_out.z[comp]() - self.vle_core.y[comp]())

        # Mapeamento do Líquido
        self.createEquation("liq_F", expr=self.liq_out.F_molar() - (self.vle_core.F_molar() * (1.0 - self.vle_core.V_frac())))
        self.createEquation("liq_T", expr=self.liq_out.T() - self.vle_core.T())
        self.createEquation("liq_P", expr=self.liq_out.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(f"liq_x_{comp}", expr=self.liq_out.z[comp]() - self.vle_core.x[comp]())


class MassFlashFlowsheet(Model):
    r"""Flowsheet do Benchmark (Sem Dinâmica, Estritamente Algébrico)."""
    def __init__(self, name, pkg_class):
        self.pkg = pkg_class(components=["methane", "propane", "pentane"])
        self.pkg.fetch_parameters_from_db()
        
        # Interações binárias do Benchmark original
        #self.pkg.set_binary_interactions([
        #    [0.0,   0.014, 0.150], 
        #    [0.014, 0.0,   0.010], 
        #    [0.150, 0.010, 0.0  ]
        #])
        
        super().__init__(name)
        self()

    def DeclareVariables(self):
        self.feed = MaterialStream("feed", self.pkg)
        self.flash = IsothermalFlashDrum("flash", self.pkg)
        self.submodels.extend([self.feed, self.flash])

        # Inicialização Universal
        for stream in [self.feed, self.flash.inlet, self.flash.vap_out, self.flash.liq_out, self.flash.vle_core]:
            stream.T.setValue(310.0)
            stream.P.setValue(25.0)
            stream.F_molar.setValue(100.0)

        # Chutes Iniciais Assimétricos (Cruciais para as raízes cúbicas)
        chute_x = {"methane": 0.10, "propane": 0.25, "pentane": 0.65}
        chute_y = {"methane": 0.75, "propane": 0.20, "pentane": 0.05}

        for comp in self.pkg.components:
            self.flash.vle_core.x[comp].setValue(chute_x[comp])
            self.flash.vle_core.y[comp].setValue(chute_y[comp])
            self.flash.liq_out.z[comp].setValue(chute_x[comp])
            self.flash.vap_out.z[comp].setValue(chute_y[comp])

            z_feed = 0.42 if comp == "methane" else 0.26 if comp == "propane" else 0.32
            self.feed.z[comp].setValue(z_feed)
            self.flash.inlet.z[comp].setValue(z_feed)
            self.flash.vle_core.z[comp].setValue(z_feed)

        self.flash.vle_core.V_frac.setValue(0.5)

    def DeclareEquations(self):
        # Condições Fixas da Alimentação (Mistura de Benchmark)
        self.feed.T.fix(310.0)
        self.feed.P.fix(25.0)
        self.feed.F_molar.fix(100.0)
        self.feed.z["methane"].fix(0.42)
        self.feed.z["propane"].fix(0.26)

        # Flash Isotérmico
        self.createEquation("spec_flash_T", expr=self.flash.vle_core.T() - 310.0)

        Connection("C1", self.feed, self.flash.inlet).apply_to(self)


# =============================================================================
# REFERENCE DATA: LITERATURA RIGOROSA (C1/C3/n-C5 @ 25 bar, 310 K)
# =============================================================================
LITERATURE_REFERENCE = {
    "V_frac": 0.4532, 
    "y_methane": 0.8654, 
    "x_pentane": 0.5312
}
# =============================================================================

def run_scenario(pkg_class):
    plant = MassFlashFlowsheet("Benchmark", pkg_class)
    sim = Simulator(plant)

    try:
        # Como o modelo não tem EDOs (nx = 0), o sim.run envia automaticamente
        # para o KINSOL (Smart Dispatcher) e retorna como se fosse temporal [0, 1].
        results = sim.run(t_span=[0, 1])
        
        V_frac = results.get_variable(plant.flash.vle_core.V_frac)[-1]
        y_m = results.get_variable(plant.flash.vap_out.z["methane"])[-1]
        x_p = results.get_variable(plant.flash.liq_out.z["pentane"])[-1]
        return V_frac, y_m, x_p
    

    except Exception as e:
        print(f"[{pkg_class.__name__} FAILED: {str(e)}]")
        return None, None, None



if __name__ == "__main__":
    
    cfg.VERBOSITY_LEVEL = 1
    print("\n" + "=" * 70)
    print(" ANTARES V5 - THERMODYNAMIC LITERATURE BENCHMARK")
    print(" Mixture: C1/C3/n-C5 @ 25 bar, 310 K | Reference: Rigorous Literature")
    print(" Simulation: Algebraic Steady-State Flash (KINSOL Sandbox)")
    print("=" * 70)

    v_id, yM_id, xP_id = None, None, None
    v_pr, yM_pr, xP_pr = None, None, None
    v_srk, yM_srk, xP_srk = None, None, None

    if TEST_IDEAL:
        print("\n[1/3] Solving Algebraic System with Ideal VLE Package...")
        v_id, yM_id, xP_id = run_scenario(IdealVLEPackage)

    if TEST_PR:
        print("\n[2/3] Solving Algebraic System with Peng-Robinson EOS...")
        v_pr, yM_pr, xP_pr = run_scenario(PengRobinsonEOS)

    if TEST_SRK:
        print("\n[3/3] Solving Algebraic System with Soave-Redlich-Kwong EOS...")
        v_srk, yM_srk, xP_srk = run_scenario(SoaveRedlichKwongEOS)

    print("\n" + "=" * 70)
    print(" LITERATURE COMPARISON: VAPORIZED FRACTION (V/F)")
    print("=" * 70)
    ref_v = LITERATURE_REFERENCE["V_frac"]
    print(f" Literature (Rigorous Data)  : {ref_v:.4f}")
    if v_id is not None: print(f" ANTARES IdealVLE Package    : {v_id:.4f}  -> Error: {abs(v_id - ref_v) / ref_v * 100:>6.2f} %")
    if v_pr is not None: print(f" ANTARES Peng-Robinson EOS   : {v_pr:.4f}  -> Error: {abs(v_pr - ref_v) / ref_v * 100:>6.2f} %")
    if v_srk is not None: print(f" ANTARES SRK EOS             : {v_srk:.4f}  -> Error: {abs(v_srk - ref_v) / ref_v * 100:>6.2f} %")

    print("\n" + "=" * 70)
    print(" LITERATURE COMPARISON: PHASE COMPOSITIONS (Molar Fractions)")
    print("=" * 70)

    ref_ym = LITERATURE_REFERENCE["y_methane"]
    print(f" [Methane in Vapor, y1] -> Highly Supercritical (Target: {ref_ym:.4f})")
    if yM_id is not None: print(f"  Ideal Prediction           : {yM_id:.4f}  -> Error: {abs(yM_id - ref_ym) / ref_ym * 100:>6.2f} %")
    if yM_pr is not None: print(f"  ANTARES Peng-Robinson      : {yM_pr:.4f}  -> Error: {abs(yM_pr - ref_ym) / ref_ym * 100:>6.2f} %")
    if yM_srk is not None: print(f"  ANTARES SRK                : {yM_srk:.4f}  -> Error: {abs(yM_srk - ref_ym) / ref_ym * 100:>6.2f} %")

    ref_xp = LITERATURE_REFERENCE["x_pentane"]
    print(f"\n [n-Pentane in Liquid, x3] -> Heavy Component (Target: {ref_xp:.4f})")
    if xP_id is not None: print(f"  Ideal Prediction           : {xP_id:.4f}  -> Error: {abs(xP_id - ref_xp) / ref_xp * 100:>6.2f} %")
    if xP_pr is not None: print(f"  ANTARES Peng-Robinson      : {xP_pr:.4f}  -> Error: {abs(xP_pr - ref_xp) / ref_xp * 100:>6.2f} %")
    if xP_srk is not None: print(f"  ANTARES SRK                : {xP_srk:.4f}  -> Error: {abs(xP_srk - ref_xp) / ref_xp * 100:>6.2f} %")
    print("=" * 70 + "\n")