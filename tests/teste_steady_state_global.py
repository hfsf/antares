# -*- coding: utf-8 -*-

"""
Teste Definitivo do Framework ANTARES:
Resolução de Estado Estacionário (Steady-State) Multiescala (0D, 2D e 3D).

Associa três modelos fenomenológicos num Master Flowsheet resolvido 
simultaneamente pelo motor KINSOL via CasADi, utilizando compilação JIT em C.
Demonstra a renderização nativa de Mapas de Calor 2D e Fatiamentos 3D.
-----------------------------------------------------------------------------
SOBRE ESTE EXEMPLO:
Este script simula o equilíbrio termodinâmico de uma planta de tratamento
composta por três sistemas acoplados em dimensões distintas:
1. Uma restrição de processo (0D) que fixa o nível de um tanque de contenção.
2. Um dissipador térmico (2D) ilustrando a condução de calor (Lei de Fourier)
   entre uma fonte quente (500 K) e um sorvedouro frio (300 K).
3. Um filtro catalítico (3D) resolvendo um problema de Difusão-Reação, onde
   um reagente entra por uma face e é consumido ao longo do volume por uma
   cinética de primeira ordem.

O objetivo é demonstrar a capacidade do transpilador e do motor KINSOL 
(backend em C) de encontrar instantaneamente as raízes de milhares de equações
algébricas não-lineares resultantes da discretização espacial de PDEs, sem 
necessidade de integração no tempo. Demonstra também a renderização nativa 
de Mapas de Calor 2D e Fatiamentos 3D.
"""


import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.domain import Domain2D, Domain3D
from antares.core.model import Model
from antares.core.template_units import m
from antares.plotter import Plotter

cfg.VERBOSITY_LEVEL = 1


# =============================================================================
# 1. MODELO 2D: DISSIPADOR TÉRMICO (ESTADO ESTACIONÁRIO)
# =============================================================================
class Dissipador2D(Model):
    def __init__(self, name):
        super().__init__(name, description="Placa condutora de calor em SS")
        self()

    def DeclareVariables(self):
        self.x = self.createDomain("X", unit=m, length=1.0, n_points=20, method="mol")
        self.y = self.createDomain("Y", unit=m, length=1.0, n_points=20, method="mol")
        self.malha_2d = Domain2D("Superficie", self.x, self.y)

        # A MÁGICA ESTÁ AQUI: exposure_type="algebraic" silencia a derivada temporal
        self.T = self.createVariable("T", "K", "Temperatura", exposure_type="algebraic")
        self.T.distributeOn(self.malha_2d)

    def DeclareParameters(self):
        self.alpha = self.createParameter("alpha", "m^2/s", value=1.0)
        self.T_fonte = self.createParameter("T_fonte", "K", value=500.0)

    def DeclareEquations(self):
        # PDE Estacionária (Fourier): Laplaciano = 0
        pde_conducao = self.alpha() * self.T.Div()
        self.addBulkEquation("eq_conducao_2d", expression=pde_conducao, domain=self.malha_2d)

        # Gradiente forçado (500K na esquerda dissipando para 300K na direita)
        self.setBoundaryCondition(self.T, self.malha_2d, "left", "dirichlet", self.T_fonte())
        self.setBoundaryCondition(self.T, self.malha_2d, "right", "dirichlet", 300.0)
        self.setBoundaryCondition(self.T, self.malha_2d, "top", "neumann", 0.0)
        self.setBoundaryCondition(self.T, self.malha_2d, "bottom", "neumann", 0.0)


# =============================================================================
# 2. MODELO 3D: FILTRO DE DIFUSÃO-REAÇÃO (ESTADO ESTACIONÁRIO)
# =============================================================================
class FiltroReativo3D(Model):
    def __init__(self, name):
        super().__init__(name, description="Bloco reativo 3D em equilíbrio")
        self()

    def DeclareVariables(self):
        self.x = self.createDomain("X", unit=m, length=1.0, n_points=10, method="mol")
        self.y = self.createDomain("Y", unit=m, length=1.0, n_points=10, method="mol")
        self.z = self.createDomain("Z", unit=m, length=1.0, n_points=10, method="mol")
        self.malha_3d = Domain3D("Volume", self.x, self.y, self.z)

        # Variável puramente algébrica para a malha 3D
        self.C = self.createVariable("C", "mol/m^3", "Concentração", exposure_type="algebraic")
        self.C.distributeOn(self.malha_3d)

    def DeclareParameters(self):
        self.D = self.createParameter("D", "m^2/s", value=0.05)
        self.k = self.createParameter("k", "s^-1", value=2.0)
        self.C_entrada = self.createParameter("C_in", "mol/m^3", value=100.0)

    def DeclareEquations(self):
        # PDE Estacionária (Fick + Cinética): D * Laplaciano - k * C = 0
        pde_difusao = (self.D() * self.C.Div()) - (self.k() * self.C())
        self.addBulkEquation("eq_difusao_reacao_3d", expression=pde_difusao, domain=self.malha_3d)

        # Massa entra pela frente e é consumida no interior
        self.setBoundaryCondition(self.C, self.malha_3d, "front", "dirichlet", self.C_entrada())
        self.setBoundaryCondition(self.C, self.malha_3d, "back", "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.malha_3d, "left", "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.malha_3d, "right", "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.malha_3d, "top", "neumann", 0.0)
        self.setBoundaryCondition(self.C, self.malha_3d, "bottom", "neumann", 0.0)


# =============================================================================
# 3. MASTER FLOWSHEET (A Planta Acoplada)
# =============================================================================
class PlantaMista(Model):
    def __init__(self, name):
        self.dissipador = Dissipador2D("Dissipador")
        self.filtro = FiltroReativo3D("Filtro")
        super().__init__(name, submodels=[self.dissipador, self.filtro])
        self()

    def DeclareVariables(self):
        # Uma variável 0D que representa o sistema estabilizado globalmente
        self.Nivel_Tanque = self.createVariable("h", "m", exposure_type="algebraic")

    def DeclareEquations(self):
        # Restrição 0D: Nível estabilizado em 5.0m
        self.createEquation("eq_nivel_estacionario", expr=self.Nivel_Tanque() - 5.0)


# =============================================================================
# 4. RESOLUÇÃO NUMÉRICA E VISUALIZAÇÃO
# =============================================================================
if __name__ == "__main__":
    planta = PlantaMista("Complexo_Industrial")
    
    simulador = Simulator(model=planta)
    
    # Podemos passar "chutes" iniciais (Initial Guesses) para ajudar o Newton-Raphson.
    # O motor preenche automaticamente as outras milhares de variáveis com 0.0
    chute_inicial = {
        planta.Nivel_Tanque.name: 3.0
    }
    
    # Executa o Rootfinder KINSOL (Transpila e resolve os gradientes de uma vez)
    resultados = simulador.run_steady_state(initial_guesses=chute_inicial, use_c_code=True)
    
    plotador = Plotter(resultados)
    
    # -------------------------------------------------------------------------
    # Visualização 1: Variável 0D (Deteção Automática de Steady-State)
    # O plotter vai desenhar um ponto em vez de tentar desenhar uma linha.
    # -------------------------------------------------------------------------
    plotador.plot(
        variables=[planta.Nivel_Tanque.name],
        title="Variável 0D: Nível Estabilizado do Tanque",
        ylabel="Altura (m)",
        show=True
    )
    
    # -------------------------------------------------------------------------
    # Visualização 2: Mapa de Calor 2D NATIVO
    # Perfeito para visualizar a dissipação térmica na placa condutora.
    # -------------------------------------------------------------------------
    plotador.plot_heatmap_2d(
        variable=planta.dissipador.T,
        domain=planta.dissipador.malha_2d,
        title="Estado Estacionário: Distribuição Térmica no Dissipador 2D",
        show=True
    )
    
    # -------------------------------------------------------------------------
    # Visualização 3: Fatiamento 3D NATIVO
    # Realiza um corte transversal no meio do reator (eixo Z) para revelar 
    # a concentração interior consumida pela cinética de reação.
    # -------------------------------------------------------------------------
    plotador.plot_slice_3d(
        variable=planta.filtro.C,
        domain=planta.filtro.malha_3d,
        slice_axis='z',
        title="Fatiamento 3D (Eixo Z): Perfil de Concentração Interna",
        show=True
    )

    plotador.plot_slice_3d(
        variable=planta.filtro.C,
        domain=planta.filtro.malha_3d,
        slice_axis='y', # <-- Corta o reator de lado!
        title="Fatiamento 3D (Eixo Y): Decaimento Longitudinal",
        show=True
    )