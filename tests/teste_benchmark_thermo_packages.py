# -*- coding: utf-8 -*-

"""
Thermodynamic EO Benchmark - ANTARES V5
Validates the Dependency Injection of Thermodynamic Packages progressively.
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

# =============================================================================
# [⚙️] PAINEL DE CONTROLO DO BENCHMARK
# =============================================================================

# COMPLEXITY_LEVEL (1, 2, ou 3)
# Nível 1: Apenas Flash Mássico Isotérmico (Graus de liberdade fechados na alimentação)
# Nível 2: Flash Isotérmico + Balanço de Entalpia (Testa a Escala Numérica O(1e5))
# Nível 3: O Fluxograma Completo (Misturador + Flash + Entalpia)
COMPLEXITY_LEVEL = 3

# SELEÇÃO DE MODELOS TERMODINÂMICOS
TEST_IDEAL = True
TEST_PR = True
TEST_SRK = True

# =============================================================================

# Variáveis derivadas baseadas no Nível de Complexidade
ENABLE_MIXER = COMPLEXITY_LEVEL == 3
ENABLE_ENERGY_BALANCE = COMPLEXITY_LEVEL >= 2

cfg.VERBOSITY_LEVEL = 0
cfg.USE_C_CODE_COMPILATION = False


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

        if ENABLE_ENERGY_BALANCE:
            # SCALING DIAGNÓSTICO (O Leaky Abstraction temporário)
            eq_energy_expr = (
                (self.out.H_molar() * self.out.F_molar())
                - (
                    self.in1.H_molar() * self.in1.F_molar()
                    + self.in2.H_molar() * self.in2.F_molar()
                )
            ) / 1e5
            self.createEquation("eq_Energy", expr=eq_energy_expr)
        else:
            self.createEquation("eq_T_iso", expr=self.out.T() - self.in1.T())


class FlashDrum(Model):
    def __init__(self, name, property_package):
        self.pkg = property_package
        super().__init__(name, description="Isothermal Flash Drum")
        self()

    def DeclareVariables(self):
        self.inlet = MaterialStream("inlet", self.pkg)
        self.vap_out = MaterialStream("vap_out", self.pkg)
        self.liq_out = MaterialStream("liq_out", self.pkg)
        self.vle_core = TwoPhaseStream("vle_core", self.pkg)
        self.submodels.extend([self.inlet, self.vap_out, self.liq_out, self.vle_core])

        if ENABLE_ENERGY_BALANCE:
            self.heat_port = EnergyStream("heat_port", "heat")
            self.submodels.append(self.heat_port)

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

        if ENABLE_ENERGY_BALANCE:
            drum_energy_expr = (
                self.heat_port.Q()
                - (
                    self.vle_core.H_molar() * self.vle_core.F_molar()
                    - self.inlet.H_molar() * self.inlet.F_molar()
                )
            ) / 1e5
            self.createEquation("drum_energy", expr=drum_energy_expr)


class ProcessFlowsheet(Model):
    def __init__(self, name, pkg_class):
        self.pkg = pkg_class(components=["methane", "propane", "pentane"])
        self.pkg.fetch_parameters_from_db()
        super().__init__(name)
        self()

    def _map_streams(self, name_prefix, src, dest):
        """
        Mapeamento Topológico Estrito (Bypass da classe Connection).
        Mapeia apenas as variáveis independentes, prevenindo a duplicação
        de equações de fecho (Enthalpy_Closure e Fractions_Sum) no Jacobiano.
        """
        self.createEquation(f"{name_prefix}_T", expr=dest.T() - src.T())
        self.createEquation(f"{name_prefix}_P", expr=dest.P() - src.P())
        self.createEquation(f"{name_prefix}_F", expr=dest.F_molar() - src.F_molar())

        # Mapeia apenas N-1 componentes!
        for comp in self.pkg.components[:-1]:
            self.createEquation(
                f"{name_prefix}_z_{comp}", expr=dest.z[comp]() - src.z[comp]()
            )

    def DeclareVariables(self):
        self.flash = FlashDrum("flash", self.pkg)
        self.submodels.append(self.flash)

        if ENABLE_MIXER:
            self.feed1 = MaterialStream("feed1", self.pkg)
            self.feed2 = MaterialStream("feed2", self.pkg)
            self.mixer = Mixer("mixer", self.pkg)
            self.submodels.extend([self.feed1, self.feed2, self.mixer])
            all_streams = [
                self.feed1,
                self.feed2,
                self.mixer.in1,
                self.mixer.in2,
                self.mixer.out,
                self.flash.inlet,
                self.flash.vap_out,
                self.flash.liq_out,
                self.flash.vle_core,
            ]
        else:
            self.feed = MaterialStream("feed", self.pkg)
            self.submodels.append(self.feed)
            all_streams = [
                self.feed,
                self.flash.inlet,
                self.flash.vap_out,
                self.flash.liq_out,
                self.flash.vle_core,
            ]

        # Reset Térmico e de Pressão base
        for stream in all_streams:
            stream.T.setValue(310.0)
            stream.P.setValue(25.0)

        # =====================================================================
        # INICIALIZAÇÃO TOPOLÓGICA CONSISTENTE (Evita a Armadilha Bilinear)
        # =====================================================================
        if ENABLE_MIXER:
            # Fluxos
            self.feed1.F_molar.setValue(60.0)
            self.mixer.in1.F_molar.setValue(60.0)
            self.feed2.F_molar.setValue(40.0)
            self.mixer.in2.F_molar.setValue(40.0)
            self.mixer.out.F_molar.setValue(100.0)
            self.flash.inlet.F_molar.setValue(100.0)

            # Composições
            z1 = {"methane": 0.70, "propane": 0.30, "pentane": 0.00}
            z2 = {"methane": 0.00, "propane": 0.20, "pentane": 0.80}
            zm = {"methane": 0.42, "propane": 0.26, "pentane": 0.32}

            for comp in self.pkg.components:
                self.feed1.z[comp].setValue(z1[comp])
                self.mixer.in1.z[comp].setValue(z1[comp])
                self.feed2.z[comp].setValue(z2[comp])
                self.mixer.in2.z[comp].setValue(z2[comp])
                self.mixer.out.z[comp].setValue(zm[comp])
                self.flash.inlet.z[comp].setValue(zm[comp])
                self.flash.vle_core.z[comp].setValue(zm[comp])
        else:
            self.feed.F_molar.setValue(100.0)
            self.flash.inlet.F_molar.setValue(100.0)

            zm = {"methane": 0.42, "propane": 0.26, "pentane": 0.32}
            for comp in self.pkg.components:
                self.feed.z[comp].setValue(zm[comp])
                self.flash.inlet.z[comp].setValue(zm[comp])
                self.flash.vle_core.z[comp].setValue(zm[comp])

        # Chute Assimétrico para o Flash Drum
        chute_x = {"methane": 0.10, "propane": 0.25, "pentane": 0.65}
        chute_y = {"methane": 0.75, "propane": 0.20, "pentane": 0.05}

        for comp in self.pkg.components:
            self.flash.vle_core.x[comp].setValue(chute_x[comp])
            self.flash.vle_core.y[comp].setValue(chute_y[comp])
            self.flash.vap_out.z[comp].setValue(chute_y[comp])
            self.flash.liq_out.z[comp].setValue(chute_x[comp])

    def DeclareEquations(self):
        if ENABLE_MIXER:
            self.feed1.T.fix(310.0)
            self.feed1.P.fix(25.0)
            self.feed1.F_molar.fix(60.0)
            self.feed1.z["methane"].fix(0.70)
            self.feed1.z["propane"].fix(0.30)

            self.feed2.T.fix(310.0)
            self.feed2.F_molar.fix(40.0)
            self.feed2.z["methane"].fix(0.0)
            self.feed2.z["propane"].fix(0.20)

            # Uso do Mapeamento Estrito no lugar das Connections!
            self._map_streams("C1", self.feed1, self.mixer.in1)
            self._map_streams("C2", self.feed2, self.mixer.in2)
            self._map_streams("C3", self.mixer.out, self.flash.inlet)
        else:
            self.feed.T.fix(310.0)
            self.feed.P.fix(25.0)
            self.feed.F_molar.fix(100.0)
            self.feed.z["methane"].fix(0.42)
            self.feed.z["propane"].fix(0.26)

            self._map_streams("C1", self.feed, self.flash.inlet)

        self.createEquation("spec_flash_T", expr=self.flash.vle_core.T() - 310.0)

        if ENABLE_ENERGY_BALANCE:
            self.createEquation(
                "spec_heat_T", expr=self.flash.heat_port.T_source() - 350.0
            )


# =============================================================================
# 4. TABELA DE DADOS RIGOROSOS DA LITERATURA
# =============================================================================
LITERATURE_REFERENCE = {
    "V_frac": 0.4532,  # Fator K do Metano puxa o vapor, mas C5 retém o líquido
    "y_methane": 0.8654,  # Metano altamente volátil
    "x_pentane": 0.5312,  # Pentano domina a fase líquida
}


# =============================================================================
# 5. EXECUÇÃO DO BENCHMARK
# =============================================================================


def run_scenario(pkg_class):
    plant = ProcessFlowsheet("Benchmark", pkg_class)
    sim = Simulator(plant)

    try:
        results = sim.run(t_span=[0, 1])
        V_frac = results.get_variable(plant.flash.vle_core.V_frac)[-1]
        y_methane = results.get_variable(plant.flash.vap_out.z["methane"])[-1]
        x_pentane = results.get_variable(plant.flash.liq_out.z["pentane"])[-1]
        return V_frac, y_methane, x_pentane
    except Exception as e:
        print(f"[{pkg_class.__name__} FALHOU: {str(e)[:60]}...]")
        return None, None, None


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" ANTARES V5 - HIGH-PRESSURE THERMODYNAMIC BENCHMARK")
    print(f" C1/C3/n-C5 @ 25 bar, 310 K | Nível de Complexidade: {COMPLEXITY_LEVEL}")
    print("=" * 70)

    v_ideal, yM_id, xP_id = None, None, None
    v_pr, yM_pr, xP_pr = None, None, None
    v_srk, yM_srk, xP_srk = None, None, None

    if TEST_IDEAL:
        print("\n[1/3] Resolvendo Matriz com VLE Ideal (Wilson Extrapolado)...")
        v_ideal, yM_id, xP_id = run_scenario(IdealVLEPackage)

    if TEST_PR:
        print("[2/3] Resolvendo Matriz com Peng-Robinson EOS (Rigoroso)...")
        v_pr, yM_pr, xP_pr = run_scenario(PengRobinsonEOS)

    if TEST_SRK:
        print("[3/3] Resolvendo Matriz com Soave-Redlich-Kwong EOS (Rigoroso)...")
        v_srk, yM_srk, xP_srk = run_scenario(SoaveRedlichKwongEOS)

    print("\n" + "=" * 70)
    print(" COMPARAÇÃO COM LITERATURA: FRAÇÃO VAPORIZADA (V/F)")
    print("=" * 70)
    ref_v = LITERATURE_REFERENCE["V_frac"]
    print(f" Literatura (Rigorous Data)  : {ref_v:.4f}")
    if v_ideal is not None:
        print(
            f" ANTARES IdealVLE Package    : {v_ideal:.4f}  -> Erro: {abs(v_ideal - ref_v) / ref_v * 100:>6.2f} %"
        )
    if v_pr is not None:
        print(
            f" ANTARES Peng-Robinson EOS   : {v_pr:.4f}  -> Erro: {abs(v_pr - ref_v) / ref_v * 100:>6.2f} %"
        )
    if v_srk is not None:
        print(
            f" ANTARES SRK EOS             : {v_srk:.4f}  -> Erro: {abs(v_srk - ref_v) / ref_v * 100:>6.2f} %"
        )

    print("\n" + "=" * 70)
    print(" COMPARAÇÃO COM LITERATURA: COMPOSIÇÕES DE FASE (Molares)")
    print("=" * 70)

    ref_ym = LITERATURE_REFERENCE["y_methane"]
    print(f" [Metano no Vapor, y1] -> Altamente Supercrítico (Alvo: {ref_ym:.4f})")
    if yM_id is not None:
        print(
            f"  Previsão Ideal             : {yM_id:.4f}  -> Erro: {abs(yM_id - ref_ym) / ref_ym * 100:>6.2f} %"
        )
    if yM_pr is not None:
        print(
            f"  ANTARES Peng-Robinson      : {yM_pr:.4f}  -> Erro: {abs(yM_pr - ref_ym) / ref_ym * 100:>6.2f} %"
        )
    if yM_srk is not None:
        print(
            f"  ANTARES SRK                : {yM_srk:.4f}  -> Erro: {abs(yM_srk - ref_ym) / ref_ym * 100:>6.2f} %"
        )

    ref_xp = LITERATURE_REFERENCE["x_pentane"]
    print(f"\n [n-Pentano no Líquido, x3] -> Componente Pesado (Alvo: {ref_xp:.4f})")
    if xP_id is not None:
        print(
            f"  Previsão Ideal             : {xP_id:.4f}  -> Erro: {abs(xP_id - ref_xp) / ref_xp * 100:>6.2f} %"
        )
    if xP_pr is not None:
        print(
            f"  ANTARES Peng-Robinson      : {xP_pr:.4f}  -> Erro: {abs(xP_pr - ref_xp) / ref_xp * 100:>6.2f} %"
        )
    if xP_srk is not None:
        print(
            f"  ANTARES SRK                : {xP_srk:.4f}  -> Erro: {abs(xP_srk - ref_xp) / ref_xp * 100:>6.2f} %"
        )
    print("=" * 70 + "\n")
