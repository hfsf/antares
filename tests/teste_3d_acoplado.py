# -*- coding: utf-8 -*-

"""
Teste de Escala Extrema do Framework ANTARES:
Acoplamento de Topologias 0D (Tanque) com 3D (Dispersão Volumétrica).
Malha 25x25x25 (15.625 graus de liberdade espaciais).
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.domain import Domain3D
from antares.core.model import Model
from antares.core.template_units import m, s, mol
from antares.plotter import Plotter

from teste_tanques import Tanque

cfg.VERBOSITY_LEVEL = 2

# =============================================================================
# 1. MODELO 3D (Cubo de Dispersão)
# =============================================================================
class CuboDispersao(Model):
    def __init__(self, name):
        super().__init__(name, description="Volume 3D de Dispersão")
        self()

    def DeclareVariables(self):
        N = 25
        L = 2.0 
        self.eixo_x = self.createDomain("X", unit=m, length=L, n_points=N)
        self.eixo_y = self.createDomain("Y", unit=m, length=L, n_points=N)
        self.eixo_z = self.createDomain("Z", unit=m, length=L, n_points=N)
        
        self.dominio_3d = Domain3D("Matriz3D", self.eixo_x, self.eixo_y, self.eixo_z)

        self.C = self.createVariable("C", "mol/m^3", "Concentração", exposure_type="differential")
        self.C.distributeOn(self.dominio_3d)

        self.C_in = self.createVariable("C_in", "mol/m^3", "Conc. na Face Frontal", exposure_type="algebraic")

        self.setInitialCondition(self.C, value=0.0, location="all")

    def DeclareParameters(self):
        self.D = self.createParameter("D", "m^2/s", "Difusividade", value=0.08)

    def DeclareEquations(self):
        pde_dispersao = self.C.Diff() - (self.D() * self.C.Div())
        self.addBulkEquation("eq_dispersao_miolo", expression=pde_dispersao, domain=self.dominio_3d)

        self.setBoundaryCondition(self.C, self.dominio_3d, "front", "dirichlet", self.C_in())
        self.setBoundaryCondition(self.C, self.dominio_3d, "back",   "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.dominio_3d, "left",   "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.dominio_3d, "right",  "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.dominio_3d, "top",    "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.dominio_3d, "bottom", "neumann", 0.0)

# =============================================================================
# 2. MASTER FLOWSHEET (A Planta Acoplada)
# =============================================================================
class PlantaMista(Model):
    def __init__(self, name):
        self.tanque = Tanque("T1")
        self.cubo = CuboDispersao("Cubo3D")
        super().__init__(name, submodels=[self.tanque, self.cubo])
        self()

    def DeclareVariables(self):
        self.setInitialCondition(self.tanque.h, value=5.0)

    def DeclareParameters(self):
        # O Parâmetro de acoplamento com a unidade física exata para a conversão
        self.k_couple = self.createParameter(
            "k_couple", 
            "mol * s / m^6", 
            "Constante de Acoplamento Físico", 
            value=100.0
        )

    def DeclareEquations(self):
        eq_alim_tanque = self.tanque.F_in() - 0.0
        self.createEquation("fechar_torneira_tanque", expr=eq_alim_tanque)

        # AGORA A FÍSICA ESTÁ COERENTE: [mol/m³] = [mol*s/m⁶] * [m³/s]
        eq_acoplamento = self.cubo.C_in() - (self.k_couple() * self.tanque.F_out())
        self.createEquation("acoplamento_tanque_cubo", expr=eq_acoplamento)
# =============================================================================
# 3. EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    planta = PlantaMista("Planta_Global")
    t_span = np.linspace(0, 15, 60)

    simulador = Simulator(model=planta)
    resultados = simulador.run(t_span, use_c_code=False)

    # =============================================================================
    # 4. VISUALIZAÇÃO PONTUAL (TOTALMENTE ABSTRAÍDA)
    # =============================================================================
    plotador = Plotter(resultados)

    # A MÁGICA: O Plotter aceita as variáveis 0D juntamente com coordenadas 3D!
    # Vamos extrair pontos exatamente no meio geométrico da placa (X=1.0, Y=1.0)
    # Variando a profundidade: Z=0.0 (Entrada), Z=1.0 (Meio) e Z=2.0 (Fundo)
    
    plotador.plot(
        variables=[planta.tanque.h.name],            # Variáveis Lumped
        variable=planta.cubo.C,                      # Variável Distribuída
        domain=planta.cubo.dominio_3d,
        coordinates=[
            (1.0, 1.0, 0.0), 
            (1.0, 1.0, 1.0), 
            (1.0, 1.0, 2.0)
        ],
        title="Acoplamento 0D-3D: Tanque Drenando para Cubo de Dispersão",
        xlabel="Tempo de Simulação (s)",
        ylabel="Altura (m) / Concentração (mol/m³)",
        legend_labels={planta.tanque.h.name: "Altura do Tanque (m)"},
        show=True
    )