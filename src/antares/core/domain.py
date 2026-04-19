# -*- coding: utf-8 -*-

"""
Define the spatial Domain class for PDE discretization.
Automates the Method of Lines (MoL) and matrix operations for the AST Expansion.
"""

import numpy as np

from .error_definitions import UnexpectedValueError


def _ast_matmul(matrix, vector, original_unit, domain_unit, deriv_order=1):
    """
    Realiza o produto matricial seguro para árvores AST (EquationNodes).
    Garante que a coerência dimensional é mantida e protege o CasadiTranspiler.
    """
    N = len(vector)
    result = []

    target_unit = original_unit / (domain_unit**deriv_order)

    for i in range(N):
        row_sum = None  # CORREÇÃO 1: Usamos None para não acionar o nosso novo __eq__
        for j in range(N):
            # CORREÇÃO 2: Forçamos o cast para float nativo do Python!
            # Isto impede que o SymPy crie 'sympy.Floats' residuais que enlouquecem o CasADi.
            val = float(matrix[i, j])

            if val != 0.0:
                term = vector[j] * val
                if row_sum is None:
                    row_sum = term
                else:
                    row_sum = row_sum + term

        # Fallback de segurança se a linha for inteiramente zero
        if row_sum is None:
            row_sum = vector[0] * 0.0

        if hasattr(row_sum, "unit_object"):
            row_sum.unit_object = target_unit
            # row_sum.name = f"d{deriv_order}_{vector[i].name}_d{domain_unit.name}"

        result.append(row_sum)

    return np.array(result)


class Domain1D:
    def __init__(
        self, name, length, n_points, unit, description="", method="mol", diff_scheme="central"
    ):
        """
        Instancia um domínio espacial 1D.

        :param str name: Nome do eixo (ex: 'z').
        :param float length: Comprimento físico total.
        :param int n_points: Número de nós na malha.
        :param str description: Domain description
        :param Unit unit: Unidade de medida do eixo (ex: _m_).
        :param str method: 'mol' (Diferenças Finitas) ou 'collocation'.
        """
        self.name = name
        self.length = float(length)
        self.n_points = int(n_points)
        self.unit = unit
        self.description = description
        self.method = method.lower()
        self.diff_scheme = diff_scheme.lower()

        self.grid = None
        self.dz = None
        self.A_matrix = None  # Matriz da 1ª derivada (Gradiente)
        self.B_matrix = None  # Matriz da 2ª derivada (Laplaciano/Divergente)

        self._owner_model_instance = None

        self._build_mesh()

    def _build_mesh(self):
        if self.method == "mol":
            self._build_finite_difference_matrices()
        elif self.method == "collocation":
            raise NotImplementedError(
                "Colocação Ortogonal será ativada na próxima atualização."
            )
        else:
            raise UnexpectedValueError("Method must be 'mol' or 'collocation'.")

    def _build_finite_difference_matrices(self):
        """Gera as matrizes banda para o Método das Linhas (MoL)."""
        self.dz = self.length / (self.n_points - 1)
        self.grid = np.linspace(0, self.length, self.n_points)

        N = self.n_points
        dz = self.dz

        self.A_matrix = np.zeros((N, N))
        self.B_matrix = np.zeros((N, N))

        # --- Matriz B (2ª Derivada - Diferença Central O(h^2)) ---
        for i in range(1, N - 1):
            self.B_matrix[i, i - 1] = 1.0 / (dz**2)
            self.B_matrix[i, i] = -2.0 / (dz**2)
            self.B_matrix[i, i + 1] = 1.0 / (dz**2)

        # Contornos da Matriz B (Forward/Backward diff O(h^2))
        self.B_matrix[0, 0:3] = [1.0 / (dz**2), -2.0 / (dz**2), 1.0 / (dz**2)]
        self.B_matrix[-1, -3:] = [1.0 / (dz**2), -2.0 / (dz**2), 1.0 / (dz**2)]

        # --- Matriz A (1ª Derivada) ---
        if self.diff_scheme == "backward":  # Ideal para advecção (escoamento pistonado)
            for i in range(1, N):
                self.A_matrix[i, i] = 1.0 / dz
                self.A_matrix[i, i - 1] = -1.0 / dz
            self.A_matrix[0, 0:2] = [-1.0 / dz, 1.0 / dz]

        elif self.diff_scheme == "central":  # Ideal para difusão pura
            for i in range(1, N - 1):
                self.A_matrix[i, i + 1] = 1.0 / (2 * dz)
                self.A_matrix[i, i - 1] = -1.0 / (2 * dz)
            self.A_matrix[0, 0:3] = [-3.0 / (2 * dz), 4.0 / (2 * dz), -1.0 / (2 * dz)]
            self.A_matrix[-1, -3:] = [1.0 / (2 * dz), -4.0 / (2 * dz), 3.0 / (2 * dz)]
