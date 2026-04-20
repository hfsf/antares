# -*- coding: utf-8 -*-

"""
Teste de Alta Performance do Framework ANTARES:
Simulação de Transferência de Calor 2D Transiente.
Testa a abstração tensorial do Domain2D e a compilação JIT para C.
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.domain import Domain2D
from antares.core.model import Model
from antares.core.template_units import m
from antares.plotter import Plotter

# Verbosidade no nível 1 para vermos o Relatório de Graus de Liberdade e a Compilação C
cfg.VERBOSITY_LEVEL = 1


# =============================================================================
# 1. DECLARAÇÃO DO MODELO FENOMENOLÓGICO 2D
# =============================================================================
class PlacaAquecida(Model):
    def __init__(self, name):
        super().__init__(name)
        self()

    def DeclareVariables(self):
        # 1. Criação dos eixos 1D individuais (Malha 40x40 = 1600 nós)
        self.eixo_x = self.createDomain(
            "X", unit=m, length=1.0, n_points=40, method="mol", diff_scheme="central"
        )
        self.eixo_y = self.createDomain(
            "Y", unit=m, length=1.0, n_points=40, method="mol", diff_scheme="central"
        )
        
        # 2. Produto Tensorial: Criação do Domínio 2D
        self.placa = Domain2D("Placa_Matriz", self.eixo_x, self.eixo_y)

        # 3. Variável de Estado Distribuída
        self.T = self.createVariable("T", "K", "Temperatura", exposure_type="differential")
        self.T.distributeOn(self.placa)

        # 4. Condição Inicial Universal (Placa fria a 300 K)
        self.setInitialCondition(self.T, value=300.0, location="all")

    def DeclareParameters(self):
        # Difusividade térmica (Ex: Aço carbono)
        self.alpha = self.createParameter("alpha", "m^2/s", "Difusividade Térmica", value=1.2e-5)

    def DeclareEquations(self):
        # 1. EQUAÇÃO GOVERNANTE (PDE 2D)
        # O método Div() automaticamente deteta o Domain2D e aplica (d2T/dx2 + d2T/dy2)
        pde_calor = self.T.Diff() - (self.alpha() * self.T.Div())
        
        self.addBulkEquation("eq_calor_miolo", expression=pde_calor, domain=self.placa)

        # 2. CONDIÇÕES DE CONTORNO (Bordas da Placa)
        # Base quente (Dirichlet)
        self.setBoundaryCondition(self.T, self.placa, "bottom", "dirichlet", 500.0)
        
        # Parede esquerda resfriada (Dirichlet)
        self.setBoundaryCondition(self.T, self.placa, "left", "dirichlet", 300.0)
        
        # Topo e Parede direita perfeitamente isolados (Neumann = 0 de fluxo térmico)
        self.setBoundaryCondition(self.T, self.placa, "top", "neumann", 0.0)
        self.setBoundaryCondition(self.T, self.placa, "right", "neumann", 0.0)


# =============================================================================
# 2. INSTANCIAÇÃO E SETUP
# =============================================================================
if __name__ == "__main__":
    modelo = PlacaAquecida("Placa_Assimetrica")

    # Vamos simular 5 horas de aquecimento (18000 segundos) para ver o calor propagar
    t_span = np.linspace(0, 18000, 200)

    # =============================================================================
    # 3. EXECUÇÃO DA SIMULAÇÃO (COMPILAÇÃO NATIVA EM C)
    # =============================================================================
    # Aqui ativamos o use_c_code=True. O ANTARES vai gerar código C para as 1600 equações!
    simulador = Simulator(model=modelo)
    resultados = simulador.run(t_span, use_c_code=True)

    # =============================================================================
    # 4. VISUALIZAÇÃO DINÂMICA
    # =============================================================================
    plotador = Plotter(resultados)

    # Como o nosso Plotter atual ainda não extrai coordenadas (x,y) de matrizes 2D
    # de forma automática, vamos buscar os nós cruciais pelo seu índice exato [i, j].
    # Malha 40x40 -> Índices vão de 0 a 39.
    
    no_centro = "T_Placa_Assimetrica_Placa_Matriz_20_20"
    no_canto_isolado = "T_Placa_Assimetrica_Placa_Matriz_39_39" # Topo-Direita
    no_perto_base = "T_Placa_Assimetrica_Placa_Matriz_20_2"     # Perto do calor
    
    variaveis = [no_perto_base, no_centro, no_canto_isolado]
    legendas = {
        no_perto_base: "Perto da Base Quente (X=0.5, Y=0.05)",
        no_centro: "Centro da Placa (X=0.5, Y=0.5)",
        no_canto_isolado: "Canto Isolado (X=1.0, Y=1.0)"
    }

    plotador.plot(
        variables=variaveis,
        title="Dinâmica de Aquecimento (Condução 2D)",
        xlabel="Tempo de Simulação (s)",
        ylabel="Temperatura (K)",
        legend_labels=legendas,
        show=True
    )