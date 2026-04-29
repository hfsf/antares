# -*- coding: utf-8 -*-

"""
Teste Cirúrgico - Flash Drum com Peng-Robinson
Isola a Operação Unitária de Flash para verificar se o erro KIN_MXNEWT_5X_EXCEEDED
está no Balanço do Tambor ou se estava no Misturador.
"""

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.connection import Connection
from antares.core.model import Model
from antares.core.stream import EnergyStream, MaterialStream, TwoPhaseStream
from antares.core.thermo_package import PengRobinsonEOS

cfg.VERBOSITY_LEVEL = 0
cfg.USE_C_CODE_COMPILATION = False


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
        for comp in self.pkg.components:
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
        for comp in self.pkg.components:
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
        for comp in self.pkg.components:
            self.createEquation(
                f"liq_x_{comp}", expr=self.liq_out.z[comp]() - self.vle_core.x[comp]()
            )

        # Balanço de energia simplificado
        self.createEquation(
            "drum_energy",
            expr=self.heat_port.Q()
            - (
                self.vle_core.H_molar() * self.vle_core.F_molar()
                - self.inlet.H_molar() * self.inlet.F_molar()
            ),
        )


class SingleFlashFlowsheet(Model):
    def __init__(self, name):
        self.pkg = PengRobinsonEOS(components=["methane", "propane", "pentane"])
        self.pkg.fetch_parameters_from_db()
        super().__init__(name)
        self()

    def DeclareVariables(self):
        self.feed = MaterialStream("feed", self.pkg)
        self.flash = FlashDrum("flash", self.pkg)
        self.submodels.extend([self.feed, self.flash])

        # Inicialização Segura
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

        for comp in self.pkg.components:
            self.flash.vle_core.x[comp].setValue(0.1)
            self.flash.vle_core.y[comp].setValue(0.9)

    def DeclareEquations(self):
        # Condições Fixas da Alimentação (A mistura exata das feeds anteriores)
        self.feed.T.fix(310.0)
        self.feed.P.fix(25.0)
        self.feed.F_molar.fix(100.0)
        self.feed.z["methane"].fix(0.42)
        self.feed.z["propane"].fix(0.26)
        self.feed.z["pentane"].fix(0.32)

        # Condição de Operação do Flash
        self.createEquation("spec_flash_T", expr=self.flash.vle_core.T() - 310.0)
        self.createEquation("spec_heat_T", expr=self.flash.heat_port.T_source() - 350.0)

        Connection("C1", self.feed, self.flash.inlet).apply_to(self)


if __name__ == "__main__":
    print("=" * 70)
    print(" TESTE ISOLADO: FLASH DRUM + PENG-ROBINSON")
    print("=" * 70)

    plant = SingleFlashFlowsheet("SingleFlash")
    sim = Simulator(plant)

    print("[1/1] Resolvendo Matriz do Flash Drum...")
    try:
        results = sim.run(t_span=[0, 1])
        V_frac = results.get_variable(plant.flash.vle_core.V_frac)[-1]
        y_m = results.get_variable(plant.flash.vap_out.z["methane"])[-1]
        x_p = results.get_variable(plant.flash.liq_out.z["pentane"])[-1]

        print("\nSUCESSO! O Flash convergiu com Peng-Robinson.")
        print(f" V/F: {V_frac:.4f}")
        print(f" y_Metano: {y_m:.4f}")
        print(f" x_Pentano: {x_p:.4f}")
        print(
            "\nCONCLUSÃO: O erro não está no PR. O 'Mixer' do benchmark original está a envenenar a simulação!"
        )

    except Exception as e:
        print("\nFALHA. O Erro persiste APENAS com o Flash Drum.")
        print(e)
