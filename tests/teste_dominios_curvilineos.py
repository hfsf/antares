# -*- coding: utf-8 -*-

"""
Curvilinear Domains Benchmark - ANTARES V5
Numerical Verification against Exact Analytical Solutions.
Problem: Intraparticle Diffusion with Zero-Order Reaction in Solid and Hollow geometries.
"""

import numpy as np
import matplotlib.pyplot as plt

import antares.core.GLOBAL_CFG as cfg
from antares.backend.simulator import Simulator
from antares.core.model import Model

# Framework Settings
cfg.VERBOSITY_LEVEL = 0  # Silenced to focus on the final report
cfg.USE_C_CODE_COMPILATION = False


class PastilhaCatalitica(Model):
    """
    Model representing a catalytic particle undergoing a zero-order reaction.
    Dynamically supports both solid geometries (using L'Hôpital at the core) 
    and hollow/annular geometries (opening the inner boundary).
    """

    def __init__(self, name, geometria="cilindro", inner_radius=0.0):
        """
        Instantiates the Catalytic Particle model.

        :param str name: The instance name.
        :param str geometria: Geometry type ('cilindro', 'esfera', 'cilindro_oco', 'esfera_oca').
        :param float inner_radius: The inner radius. If > 0.0, the domain is hollow/annular.
        """
        self.geometria = geometria
        self.inner_radius = float(inner_radius)
        super().__init__(name, description=f"Catalyst Particle ({geometria})")
        self()

    def DeclareVariables(self):
        """Declares spatial domains and state variables."""
        base_geo = "cilindro" if "cilindro" in self.geometria else "esfera"
        
        # 1. Domain Creation (Injecting the inner_radius dynamically)
        if base_geo == "cilindro":
            self.r = self.createRadialDomain(
                "Raio", unit="m", radius=0.1, n_points=500, inner_radius=self.inner_radius
            )
        else:
            self.r = self.createSphericalDomain(
                "Raio", unit="m", radius=0.1, n_points=500, inner_radius=self.inner_radius
            )

        # 2. State Variable (Concentration of Reactant A)
        self.C = self.createVariable("C", "mol/m^3", exposure_type="differential")
        self.distributeVariable(self.C, self.r)
        
        # Initial Condition: Particle starts filled uniformly
        self.setInitialCondition(self.C, 100.0)

    def DeclareParameters(self):
        """Declares the kinetic and transport parameters."""
        self.Deff = self.createParameter("Deff", "m^2/s", value=0.01)
        self.k0 = self.createParameter("k0", "mol/(m^3 * s)", value=50.0)
        self.Cs = self.createParameter("Cs", "mol/m^3", value=100.0)  # Surface Concentration

    def DeclareEquations(self):
        """Declares the governing PDEs and boundary conditions."""
        # 1. Governing Bulk Equation: dC/dt = Deff * Laplacian(C) - k0
        eq_difusao = self.C.Diff() - (self.Deff() * self.r.apply_laplacian(self.C) - self.k0())
        self.addBulkEquation("Bal_Massa_A", eq_difusao, self.r)

        # 2. Outer Boundary Condition (Always Dirichlet at r = R_out)
        self.setBoundaryCondition(self.C, self.r, boundary_locator="end", bc_type="dirichlet", value=self.Cs())
        
        # 3. Inner Boundary Condition (Only for Hollow/Annular Domains)
        # If inner_radius > 0.0, the "start" boundary is physically exposed (e.g., a cooling pipe).
        # We enforce the same surface concentration Cs on the inside wall.
        if self.inner_radius > 0.0:
            self.setBoundaryCondition(self.C, self.r, boundary_locator="start", bc_type="dirichlet", value=self.Cs())


# =============================================================================
# EXACT ANALYTICAL SOLUTIONS
# =============================================================================
def solucao_analitica(r_grid, R_out, R_in, Cs, k0, Deff, geometria):
    """
    Computes the exact analytical steady-state solution for zero-order diffusion-reaction.

    :param ndarray r_grid: Spatial radial grid.
    :param float R_out: Outer radius.
    :param float R_in: Inner radius.
    :param float Cs: Surface concentration.
    :param float k0: Kinetic constant.
    :param float Deff: Effective diffusivity.
    :param str geometria: The chosen geometry configuration.
    :return: Exact concentration profile.
    :rtype: ndarray
    """
    if geometria == "cilindro":
        return Cs - (k0 * R_out**2 / (4 * Deff)) * (1 - (r_grid / R_out)**2)
        
    elif geometria == "esfera":
        return Cs - (k0 * R_out**2 / (6 * Deff)) * (1 - (r_grid / R_out)**2)
        
    elif geometria == "cilindro_oco":
        # Annular Cylinder Solution
        C1 = - (k0 * (R_out**2 - R_in**2)) / (4 * Deff * np.log(R_out / R_in))
        C2 = Cs - (k0 / (4 * Deff)) * R_out**2 - C1 * np.log(R_out)
        return (k0 / (4 * Deff)) * r_grid**2 + C1 * np.log(r_grid) + C2
        
    elif geometria == "esfera_oca":
        # Hollow Sphere Solution
        C1 = (k0 * (R_out**2 - R_in**2)) / (6 * Deff * (1/R_out - 1/R_in))
        C2 = Cs - (k0 / (6 * Deff)) * R_out**2 + C1 / R_out
        return (k0 / (6 * Deff)) * r_grid**2 - C1 / r_grid + C2


# =============================================================================
# EXECUTION & AUDITING
# =============================================================================
if __name__ == "__main__":
    
    # Define test cases: Solid vs Hollow
    casos = [
        {"nome": "Cilindro Maciço", "geo": "cilindro", "R_in": 0.0, "cor": "blue", "marker": "o"},
        {"nome": "Esfera Maciça", "geo": "esfera", "R_in": 0.0, "cor": "red", "marker": "s"},
        {"nome": "Cilindro Anular", "geo": "cilindro_oco", "R_in": 0.02, "cor": "cyan", "marker": "^"},
        {"nome": "Esfera Oca", "geo": "esfera_oca", "R_in": 0.02, "cor": "orange", "marker": "D"},
    ]
    
    plt.figure(figsize=(12, 8))

    for caso in casos:
        geo = caso["geo"]
        R_in = caso["R_in"]
        print(f"\n[{caso['nome'].upper()}] Compilando e Resolvendo...")
        
        modelo = PastilhaCatalitica(f"Modelo_{geo}", geometria=geo, inner_radius=R_in)
        
        # Simulate for 15 seconds to reach steady-state
        simulador = Simulator(model=modelo)
        resultados = simulador.run(t_span=np.linspace(0, 15, 100))
        
        # Extract Spatial Data
        r_grid = modelo.r.grid
        R_max = modelo.r.outer_radius
        
        # Extract the stitched tensor from ANTARES V5 Results (last time step)
        C_antares = resultados.get_variable(modelo.C)[-1]
        
        # Compute exact literature solution
        C_exato = solucao_analitica(r_grid, R_max, R_in, modelo.Cs.value, modelo.k0.value, modelo.Deff.value, geo)
        
        # Auditing: Maximum Absolute and Relative Errors
        erro_abs_max = np.max(np.abs(C_antares - C_exato))
        # Use a reference point in the middle of the domain to avoid division by zero issues
        C_ref = C_exato[len(C_exato)//2] 
        erro_relativo = (erro_abs_max / C_ref) * 100
        
        print(f" -> Erro Máximo Absoluto: {erro_abs_max:.2e} mol/m³")
        print(f" -> Erro Relativo Máximo: {erro_relativo:.4f} %")
        
        # Visual Plotting
        plt.plot(r_grid, C_antares, caso["marker"], color=caso["cor"], label=f"ANTARES ({caso['nome']})", alpha=0.7)
        plt.plot(r_grid, C_exato, '-', color="black", linewidth=1.5, label="Analítico" if geo == "cilindro" else "")

    plt.title("Validação do Motor Geométrico V5: Simetria (L'Hôpital) vs. Domínios Anulares", fontsize=15)
    plt.xlabel("Raio $r$ (m)", fontsize=13)
    plt.ylabel("Concentração $C(r)$ (mol/m³)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Highlight the boundaries
    plt.axvline(0, color="gray", linestyle="-", linewidth=2)
    plt.axvline(0.02, color="green", linestyle="--", alpha=0.6)
    plt.text(0.002, 85, "Eixo Central\n(L'Hôpital / Simetria)", color="gray", fontsize=10)
    plt.text(0.022, 85, "Parede Interna (Anular)\n(R_in = 0.02 m)", color="green", fontsize=10)
    
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()