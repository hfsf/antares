# -*- coding: utf-8 -*-

"""
Teste de Benchmark: Equações Acopladas de Luikov (Secagem 3D).

SOBRE ESTE EXEMPLO:
Este é um benchmark clássico de Fenômenos de Transporte focado na 
termodinâmica do não-equilíbrio. Simula a secagem de um meio poroso
(ex: um cubo de madeira de 10cm x 10cm x 10cm) onde os gradientes de 
temperatura e umidade estão intimamente acoplados (Efeitos de Soret e Dufour).

Física:
  dT/dt = k11 * Laplaciano(T) + k12 * Laplaciano(M)
  dM/dt = k21 * Laplaciano(T) + k22 * Laplaciano(M)

O objetivo deste teste é validar a estabilidade e a performance do 
motor de Delegação Matricial Pura (Sparse Matrices via CasADi) ao lidar 
com matrizes Jacobianas massivas cruzadas em geometria 3D (resultando 
em 2.000 EDOs acopladas resolvidas simultaneamente).
"""

import numpy as np

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.domain import Domain1D, Domain3D
from antares.core.model import Model
from antares.core.template_units import m, K, s
from antares.plotter import Plotter

# Forçamos a compilação máxima para provar a escalabilidade do motor
cfg.VERBOSITY_LEVEL = 1
cfg.USE_C_CODE_COMPILATION = True
cfg.KEEP_TEMPORARY_COMPILATION_FILES = False


class BlocoMadeiraLuikov(Model):
    def __init__(self, name):
        super().__init__(name, description="Secagem 3D - Equações de Luikov")
        self()

    def DeclareVariables(self):
        # Malha 3D: 10 x 10 x 10 = 1.000 nós espaciais
        self.x = self.createDomain("X", unit=m, length=0.1, n_points=10, method="mol")
        self.y = self.createDomain("Y", unit=m, length=0.1, n_points=10, method="mol")
        self.z = self.createDomain("Z", unit=m, length=0.1, n_points=10, method="mol")
        self.malha_3d = Domain3D("Volume", self.x, self.y, self.z)

        # 1.000 EDOs de Temperatura
        self.T = self.createVariable("T", "K", "Temperatura", exposure_type="differential")
        self.T.distributeOn(self.malha_3d)

        # 1.000 EDOs de Umidade (Concentração)
        self.M = self.createVariable("M", "mol/m^3", "Umidade", exposure_type="differential")
        self.M.distributeOn(self.malha_3d)

    def DeclareParameters(self):
        # Coeficientes fenomenológicos cruzados
        self.k11 = self.createParameter("k11", "m^2/s", value=1.0e-5) # Difusividade Térmica Principal
        self.k12 = self.createParameter("k12", "K*m^5/(mol*s)", value=2.0e-6) # Efeito Dufour
        self.k21 = self.createParameter("k21", "mol/(m*K*s)", value=1.5e-6) # Efeito Soret (Termodifusão)
        self.k22 = self.createParameter("k22", "m^2/s", value=0.5e-5) # Difusividade de Massa Principal

        # Condições de Contorno (Estufa de Secagem)
        self.T_amb = self.createParameter("T_amb", "K", value=350.0)
        self.M_amb = self.createParameter("M_amb", "mol/m^3", value=5.0)

    def DeclareEquations(self):
        # A BELEZA DO CÓDIGO: Transcrição 1:1 da matemática. 
        # O Transpilador converterá estes '.Div()' diretamente para matrizes Kronecker.
        eq_calor = self.T.Diff() - (self.k11() * self.T.Div() + self.k12() * self.M.Div())
        eq_massa = self.M.Diff() - (self.k21() * self.T.Div() + self.k22() * self.M.Div())

        self.addBulkEquation("Eq_Termica", expression=eq_calor, domain=self.malha_3d)
        self.addBulkEquation("Eq_Massica", expression=eq_massa, domain=self.malha_3d)

        # Condições de Contorno (Faces expostas à estufa)
        faces = ["front", "back", "left", "right", "top", "bottom"]
        for face in faces:
            self.setBoundaryCondition(self.T, self.malha_3d, face, "dirichlet", self.T_amb())
            self.setBoundaryCondition(self.M, self.malha_3d, face, "dirichlet", self.M_amb())

        # Condições Iniciais (Bloco de madeira inicialmente frio e muito úmido)
        self.setInitialCondition(self.T, 300.0)
        self.setInitialCondition(self.M, 50.0)


if __name__ == "__main__":
    modelo_luikov = BlocoMadeiraLuikov("Benchmark_Secagem_Luikov")
    
    simulador = Simulator(model=modelo_luikov)
    
    # Vamos observar 5000 segundos de processo de secagem
    t_span = np.linspace(0, 5000, 50)
    
    # Compilação JIT e Integração das 2.000 EDOs Acopladas
    resultados = simulador.run(t_span, use_c_code=True)
    
    plotador = Plotter(resultados)
    
    # -------------------------------------------------------------------------
    # Visualização 1: Inércia Térmica no Centro do Cubo
    # -------------------------------------------------------------------------
    plotador.plot(
        variable=modelo_luikov.T,
        domain=modelo_luikov.malha_3d,
        coordinates=[(0.05, 0.05, 0.05)], # Exatamente no centro geométrico (Length = 0.1m)
        title="Dinâmica Transiente: Aquecimento no Centro do Bloco (T)",
        xlabel="Tempo de Secagem (s)",
        ylabel="Temperatura (K)",
        show=True
    )
    
    # -------------------------------------------------------------------------
    # Visualização 2: Fatiamento 3D Espacial da Frente de Secagem
    # Escolhemos visualizar o tempo médio (t=1000s) para ver os gradientes antes
    # de o sistema atingir o equilíbrio total com o ambiente.
    # -------------------------------------------------------------------------
    plotador.plot_slice_3d(
        variable=modelo_luikov.M,
        domain=modelo_luikov.malha_3d,
        slice_axis='z',
        time=1020.4, # Tempo aproximado dentro do t_span gerado
        title="Corte Transversal 3D: Gradiente de Umidade Central (t ≈ 1000s)",
        cmap="viridis",
        show=True
    )