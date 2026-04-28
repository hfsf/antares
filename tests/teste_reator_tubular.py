# -*- coding: utf-8 -*-

"""
Teste End-to-End do Framework ANTARES V5:
Simulação de um Reator Tubular 1D (Advecção-Difusão) usando o Método das Linhas.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.model import Model
from antares.core.template_units import m
from antares.plotter import Plotter

# Verbosidade normal para ver o relatório de DOF e a inicialização limpa da V5
cfg.VERBOSITY_LEVEL = 2


class ReatorTubular(Model):
    def __init__(self, name):
        super().__init__(name)
        self()

    def DeclareVariables(self):
        self.z = self.createDomain(
            "Eixo_Z",
            unit=m,
            length=10.0,
            n_points=50,
            method="mol",
            diff_scheme="backward",
        )
        self.T = self.createVariable(
            "T", "K", "Temperatura", exposure_type="differential"
        )

        self.T.distributeOn(self.z)

        # Reator inicia inteiro a 300K
        self.setInitialCondition(self.T, value=300.0, location="all")

    def DeclareParameters(self):
        self.v = self.createParameter("v", "m/s", "Velocidade", value=0.5)
        self.alpha = self.createParameter("alpha", "m^2/s", "Difusividade", value=0.05)

    def DeclareEquations(self):
        # 1. PDE Governante (Resolvida nativamente em C++ CasADi via V5)
        pde_energia = self.T.Diff() - (
            -(self.v() * self.T.Grad()) + (self.alpha() * self.T.Div())
        )
        self.addBulkEquation("eq_energia_bulk", expression=pde_energia, domain=self.z)

        # 2. CC Entrada (Z=0) - Degrau de Aquecimento (Dirichlet)
        self.setBoundaryCondition(self.T, self.z, "start", "dirichlet", 350.0)

        # 3. CC Saída (Z=L) - Condição de Danckwerts (Neumann = 0)
        self.setBoundaryCondition(self.T, self.z, "end", "neumann", 0.0)


if __name__ == "__main__":
    reator = ReatorTubular("PFR_Aquecimento")
    t_span = np.linspace(0, 30, 150)

    # Execução limpa: O Motor DAE cuida das CIs encapsuladas e do cast algébrico.
    # Sem SymPy Unrolling, a compilação agora é instantânea.
    simulador = Simulator(model=reator)
    resultados = simulador.run(
        t_span, use_c_code=False
    )  # Roda instantaneamente sem JIT para 1D

    # =============================================================================
    # 4. GERAÇÃO DE GRÁFICOS (V5 TOPOLOGY READY)
    # =============================================================================
    plotador = Plotter(resultados)

    # Gráfico 1: Evolução no Tempo (Extração Inteligente por Coordenada)
    plotador.plot(
        variable=reator.T,
        domain=reator.z,
        coordinates=[0.0, 2.5, 5.0, 10.0],
        title="Dinâmica de Aquecimento do Reator",
        xlabel="Tempo de Simulação (s)",
        ylabel="Temperatura (K)",
        show=True,
    )

    # Gráfico 2: Perfis Espaciais Sobrepostos (Frente de Onda Térmica)
    plotador.plot_spatial(
        variables=reator.T,
        domain=reator.z,
        time=[0.0, 5.0, 10.0, 20.0, 30.0],
        title="Propagação da Frente Térmica no Reator",
        show=True,
    )
