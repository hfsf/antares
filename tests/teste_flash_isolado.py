# -*- coding: utf-8 -*-

"""
Teste Cirúrgico - Flash Drum Puramente Mássico (Isothermal Mass Flash)
Objetivo: Isolar a Desproporção de Escala da Entalpia.
Se este teste convergir, o Peng-Robinson está correto e o erro é de Scaling Energético.
"""

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model
from antares.core.stream import MaterialStream, TwoPhaseStream
from antares.core.thermo_package import PengRobinsonEOS

cfg.VERBOSITY_LEVEL = 0
cfg.USE_C_CODE_COMPILATION = False


class IsothermalFlashDrum(Model):
    def __init__(self, name, property_package):
        self.pkg = property_package
        super().__init__(name, description="Pure Mass Isothermal Flash Drum")
        self()

    def DeclareVariables(self):
        self.inlet = MaterialStream("inlet", self.pkg)
        self.vap_out = MaterialStream("vap_out", self.pkg)
        self.liq_out = MaterialStream("liq_out", self.pkg)

        # O Core do Equilíbrio
        self.vle_core = TwoPhaseStream("vle_core", self.pkg)

        self.submodels.extend([self.inlet, self.vap_out, self.liq_out, self.vle_core])

    def DeclareEquations(self):
        # Mapeamento de Entrada
        self.createEquation(
            "map_F", expr=self.inlet.F_molar() - self.vle_core.F_molar()
        )
        self.createEquation("map_P", expr=self.inlet.P() - self.vle_core.P())
        for comp in self.pkg.components[:-1]:
            self.createEquation(
                f"map_z_{comp}", expr=self.inlet.z[comp]() - self.vle_core.z[comp]()
            )

        # Mapeamento do Vapor
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

        # Mapeamento do Líquido
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

        # NOTA: O Balanço de Energia (drum_energy) e o Heat Port foram REMOVIDOS.
        # Assim, o Jacobiano lidará apenas com valores na escala O(1).


class MassFlashFlowsheet(Model):
    def __init__(self, name):
        self.pkg = PengRobinsonEOS(components=["methane", "propane", "pentane"])
        self.pkg.fetch_parameters_from_db()
        super().__init__(name)
        self()

    def DeclareVariables(self):
        self.feed = MaterialStream("feed", self.pkg)
        self.flash = IsothermalFlashDrum("flash", self.pkg)
        self.submodels.extend([self.feed, self.flash])

        # Inicialização das Condições de Contorno
        for stream in [
            self.feed,
            self.flash.inlet,
            self.flash.vap_out,
            self.flash.liq_out,
            self.flash.vle_core,
        ]:
            stream.T.setValue(310.0)
            stream.P.setValue(25.0)
            stream.F_molar.setValue(100.0)

        # =====================================================================
        # CHUTE INICIAL ASSIMÉTRICO (A vacina contra o Jacobiano Singular)
        # =====================================================================
        # O Metano é supercrítico (vai para o vapor). O Pentano é pesado (vai para o líquido).
        chute_x = {"methane": 0.10, "propane": 0.25, "pentane": 0.65}
        chute_y = {"methane": 0.75, "propane": 0.20, "pentane": 0.05}

        for comp in self.pkg.components:
            # Núcleo do Flash
            self.flash.vle_core.x[comp].setValue(chute_x[comp])
            self.flash.vle_core.y[comp].setValue(chute_y[comp])

            # Propagar para as correntes de saída (para zerar os resíduos de Mapeamento Topológico)
            self.flash.liq_out.z[comp].setValue(chute_x[comp])
            self.flash.vap_out.z[comp].setValue(chute_y[comp])

            # Alimentação
            z_feed = 0.42 if comp == "methane" else 0.26 if comp == "propane" else 0.32
            self.feed.z[comp].setValue(z_feed)
            self.flash.inlet.z[comp].setValue(z_feed)
            self.flash.vle_core.z[comp].setValue(z_feed)

        self.flash.vle_core.V_frac.setValue(0.5)

    def DeclareEquations(self):
        # Condições Fixas da Alimentação (A mistura exata de benchmark)
        self.feed.T.fix(310.0)
        self.feed.P.fix(25.0)
        self.feed.F_molar.fix(100.0)
        self.feed.z["methane"].fix(0.42)
        self.feed.z["propane"].fix(0.26)

        # Condição de Operação do Flash (Temperatura Fixa)
        self.createEquation("spec_flash_T", expr=self.flash.vle_core.T() - 310.0)

        Connection("C1", self.feed, self.flash.inlet).apply_to(self)


if __name__ == "__main__":
    print("=" * 70)
    print(" TESTE ISOLADO: MASS FLASH DRUM + PENG-ROBINSON")
    print("=" * 70)

    plant = MassFlashFlowsheet("MassFlash")
    sim = Simulator(plant)

    print("[1/1] Resolvendo Matriz Mássica do Flash Drum (Sem Entalpia)...")
    try:
        results = sim.run(t_span=[0, 1])
        V_frac = results.get_variable(plant.flash.vle_core.V_frac)[-1]
        y_m = results.get_variable(plant.flash.vap_out.z["methane"])[-1]
        x_p = results.get_variable(plant.flash.liq_out.z["pentane"])[-1]

        print("\nSUCESSO ESTRONDOSO! O Flash convergiu com Peng-Robinson.")
        print(f" V/F: {V_frac:.4f}")
        print(f" y_Metano: {y_m:.4f}")
        print(f" x_Pentano: {x_p:.4f}")
        print(
            "\nDIAGNÓSTICO: O EOS está perfeito. O que estava a matar o KINSOL era a desproporção de escala da Equação de Entalpia!"
        )

    except Exception as e:
        print("\nFALHA. O Erro persiste mesmo sem o Balanço de Energia.")
        print(
            "DIAGNÓSTICO: A falha está na interação das restrições de fase internas (TwoPhaseStream) com as fugacidades."
        )
        print(e)
