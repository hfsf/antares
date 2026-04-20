# -*- coding: utf-8 -*-

"""
Teste End-to-End do Framework ANTARES:
Simulação de um Reator Tubular 1D (Advecção-Difusão) usando o Método das Linhas.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.model import Model
from antares.core.template_units import m
from antares.plotter import Plotter

# Verbosidade normal para ver o relatório de DOF que criamos
cfg.VERBOSITY_LEVEL = 2


class ReatorTubular(Model):
    def __init__(self, name):
        super().__init__(name)
        self()

    def DeclareVariables(self):
        self.z = self.createDomain("Eixo_Z", unit=m, length=10.0, n_points=50, method="mol", diff_scheme="backward")
        self.T = self.createVariable("T", "K", "Temperatura", exposure_type="differential")
        
        self.T.distributeOn(self.z)
        
        # Reator inicia inteiro a 300K
        self.setInitialCondition(self.T, value=300.0, location="all")

    def DeclareParameters(self):
        self.v = self.createParameter("v", "m/s", "Velocidade", value=0.5)
        self.alpha = self.createParameter("alpha", "m^2/s", "Difusividade", value=0.05)

    def DeclareEquations(self):
        # 1. PDE Governante
        pde_energia = self.T.Diff() - (-(self.v() * self.T.Grad()) + (self.alpha() * self.T.Div()))
        self.addBulkEquation("eq_energia_bulk", expression=pde_energia, domain=self.z)

        # 2. CC Entrada (Z=0) - Degrau de Aquecimento (Dirichlet)
        self.setBoundaryCondition(self.T, self.z, "start", "dirichlet", 350.0)

        # 3. CC Saída (Z=L) - Condição de Danckwerts (Neumann)
        self.setBoundaryCondition(self.T, self.z, "end", "neumann", 0.0)


if __name__ == "__main__":
    reator = ReatorTubular("PFR_Aquecimento")
    t_span = np.linspace(0, 30, 150)

    # Execução limpa. O Motor DAE cuida das CIs encapsuladas e do cast algébrico.
    simulador = Simulator(model=reator)
    resultados = simulador.run(t_span)

    # =============================================================================
    # 4. GERAÇÃO DE GRÁFICOS (DINÂMICO, ESPACIAL E SUPERFÍCIE 3D)
    # =============================================================================
    plotador = Plotter(resultados)

    # Gráfico 1: Evolução no Tempo (Extração Inteligente por Coordenada)
    plotador.plot(
        variable=reator.T,
        domain=reator.z,
        coordinates=[0.0, 2.5, 5.0, 10.0], # A física extrai os nós automaticamente!
        title="Dinâmica de Aquecimento do Reator",
        xlabel="Tempo de Simulação (s)",
        ylabel="Temperatura (K)",
        show=True,
    )

    # Gráfico 2: Perfis Espaciais Sobrepostos (Frente de Onda Térmica)
    plotador.plot_spatial(
        variable=reator.T,
        domain=reator.z,
        time=[0.0, 5.0, 10.0, 20.0, 30.0], # Múltiplos tempos numa única chamada!
        title="Propagação da Frente Térmica no Reator",
        show=True
    )

    # Gráfico 3: A Obra-Prima (Superfície Espaço-Temporal 3D)
    plotador.plot_surface(
        variable=reator.T,
        domain=reator.z,
        title="Superfície de Temperatura (1D + Tempo)",
        cmap="inferno", # O mapa de calor "inferno" é excelente para transferência de calor
        show=True
    )