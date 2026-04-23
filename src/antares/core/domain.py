# -*- coding: utf-8 -*-

"""
Domain Module.

Defines the spatial domain classes (1D, 2D, 3D) for PDE discretization.
V4 UPDATE: Generates declarative symbolic Tokens for N-Dimensional spatial
operations, offloading the heavy Kronecker linear algebra directly to the
CasADi Transpiler to prevent SymPy graph explosions.
"""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from .error_definitions import UnexpectedValueError
from .expression_evaluation import EquationNode


class Domain(ABC):
    """Abstract Base Class enforcing the architectural contract for all domains."""

    def __init__(self, name, description="", method="mol"):
        self.name = name
        self.description = description
        self.method = method.lower()
        self._owner_model_instance = None

    @abstractmethod
    def get_bulk_slice(self):
        pass

    @abstractmethod
    def get_boundary(self, locator):
        pass

    @abstractmethod
    def get_mesh_indices(self):
        pass

    def _get_length_unit(self):
        """Helper para extrair a unidade de comprimento do domínio (1D, 2D ou 3D)."""
        if hasattr(self, "unit"):
            return self.unit
        elif hasattr(self, "x"):
            return self.x.unit
        raise ValueError(
            f"O domínio {self.name} não possui uma unidade de comprimento definida."
        )

    def apply_laplacian(self, variable):
        """Generates the Laplacian symbolic token for the transpiler."""
        sym = sp.Function(f"Laplacian_{self.name}")(variable().symbolic_object)
        len_unit = self._get_length_unit()

        return EquationNode(
            name=f"Laplacian({variable.name})",
            symbolic_object=sym,
            repr_symbolic=sym,
            unit_object=variable.units / (len_unit**2),  # [Var] / [L]^2
        )

    def get_normal_gradient(self, variable, locator):
        """Generates the Normal Gradient symbolic token for the transpiler."""
        sym = sp.Function(f"NormalGradient_{self.name}_{locator}")(
            variable().symbolic_object
        )
        len_unit = self._get_length_unit()

        return EquationNode(
            name=f"NormalGrad({variable.name})",
            symbolic_object=sym,
            repr_symbolic=sym,
            unit_object=variable.units / len_unit,  # [Var] / [L]
        )

    def apply_gradient(self, variable):
        """Generates the full spatial gradient symbolic token (1D primarily)."""
        sym = sp.Function(f"Gradient_{self.name}")(variable().symbolic_object)
        len_unit = self._get_length_unit()

        return EquationNode(
            name=f"Grad({variable.name})",
            symbolic_object=sym,
            repr_symbolic=sym,
            unit_object=variable.units / len_unit,  # [Var] / [L]
        )


class Domain1D(Domain):
    """1-Dimensional spatial domain using Method of Lines."""

    def __init__(
        self,
        name,
        length,
        n_points,
        unit,
        description="",
        method="mol",
        diff_scheme="central",
    ):
        super().__init__(name, description, method)
        self.length = float(length)
        self.n_points = int(n_points)
        self.shape = (self.n_points,)
        self.unit = unit
        self.diff_scheme = diff_scheme.lower()
        self.grid = None
        self.dz = None
        self.A_matrix = None
        self.B_matrix = None
        self._build_mesh()

    def get_bulk_slice(self):
        return slice(1, -1)

    def get_boundary(self, locator):
        pos = str(locator).lower()
        if pos in ["start", "inlet", "left", "bottom", "front"]:
            return 0, "start"
        if pos in ["end", "outlet", "right", "top", "back"]:
            return -1, "end"
        return locator, f"idx_{str(locator).replace(' ', '')}"

    def get_mesh_indices(self):
        return np.arange(self.n_points).reshape(self.shape)

    def _build_mesh(self):
        if self.method != "mol":
            raise UnexpectedValueError("Only 'mol' is supported currently.")

        self.dz = self.length / (self.n_points - 1)
        self.grid = np.linspace(0, self.length, self.n_points)
        N, dz = self.n_points, self.dz
        self.A_matrix = np.zeros((N, N))
        self.B_matrix = np.zeros((N, N))

        for i in range(1, N - 1):
            self.B_matrix[i, i - 1] = 1.0 / (dz**2)
            self.B_matrix[i, i] = -2.0 / (dz**2)
            self.B_matrix[i, i + 1] = 1.0 / (dz**2)
        self.B_matrix[0, 0:3] = [1.0 / (dz**2), -2.0 / (dz**2), 1.0 / (dz**2)]
        self.B_matrix[-1, -3:] = [1.0 / (dz**2), -2.0 / (dz**2), 1.0 / (dz**2)]

        if self.diff_scheme == "backward":
            for i in range(1, N):
                self.A_matrix[i, i] = 1.0 / dz
                self.A_matrix[i, i - 1] = -1.0 / dz
            self.A_matrix[0, 0:2] = [-1.0 / dz, 1.0 / dz]
        elif self.diff_scheme == "central":
            for i in range(1, N - 1):
                self.A_matrix[i, i + 1] = 1.0 / (2 * dz)
                self.A_matrix[i, i - 1] = -1.0 / (2 * dz)
            self.A_matrix[0, 0:3] = [-3.0 / (2 * dz), 4.0 / (2 * dz), -1.0 / (2 * dz)]
            self.A_matrix[-1, -3:] = [1.0 / (2 * dz), -4.0 / (2 * dz), 3.0 / (2 * dz)]


class Domain2D(Domain):
    """2-Dimensional spatial domain constructed as a Tensor Product of two 1D Domains."""

    def __init__(self, name, x_domain, y_domain, description=""):
        super().__init__(name, description, method=x_domain.method)
        if not isinstance(x_domain, Domain1D) or not isinstance(y_domain, Domain1D):
            raise TypeError("Domain2D requires two Domain1D instances as axes.")

        self.x = x_domain
        self.y = y_domain
        self.shape = (self.x.n_points, self.y.n_points)
        self.n_points = self.shape[0] * self.shape[1]
        self.X_grid, self.Y_grid = np.meshgrid(self.x.grid, self.y.grid, indexing="ij")

    def get_bulk_slice(self):
        return (slice(1, -1), slice(1, -1))

    def get_boundary(self, locator):
        pos = str(locator).lower()
        if pos in ["left", "west", "x_start"]:
            return (0, slice(None)), "left"
        elif pos in ["right", "east", "x_end"]:
            return (-1, slice(None)), "right"
        elif pos in ["bottom", "south", "y_start"]:
            return (slice(None), 0), "bottom"
        elif pos in ["top", "north", "y_end"]:
            return (slice(None), -1), "top"
        else:
            raise ValueError(f"Unknown 2D boundary locator '{locator}'.")

    def get_mesh_indices(self):
        return np.arange(self.n_points).reshape(self.shape)


class Domain3D(Domain):
    """3-Dimensional spatial domain constructed as a Tensor Product of three 1D Domains."""

    def __init__(self, name, x_domain, y_domain, z_domain, description=""):
        super().__init__(name, description, method=x_domain.method)
        if (
            not isinstance(x_domain, Domain1D)
            or not isinstance(y_domain, Domain1D)
            or not isinstance(z_domain, Domain1D)
        ):
            raise TypeError("Domain3D requires three Domain1D instances as axes.")

        self.x = x_domain
        self.y = y_domain
        self.z = z_domain
        self.shape = (self.x.n_points, self.y.n_points, self.z.n_points)
        self.n_points = self.shape[0] * self.shape[1] * self.shape[2]

        self.X_grid, self.Y_grid, self.Z_grid = np.meshgrid(
            self.x.grid, self.y.grid, self.z.grid, indexing="ij"
        )

    def get_bulk_slice(self):
        return (slice(1, -1), slice(1, -1), slice(1, -1))

    def get_boundary(self, locator):
        pos = str(locator).lower()
        if pos in ["left", "west", "x_start"]:
            return (0, slice(None), slice(None)), "left"
        elif pos in ["right", "east", "x_end"]:
            return (-1, slice(None), slice(None)), "right"
        elif pos in ["bottom", "south", "y_start"]:
            return (slice(None), 0, slice(None)), "bottom"
        elif pos in ["top", "north", "y_end"]:
            return (slice(None), -1, slice(None)), "top"
        elif pos in ["front", "z_start"]:
            return (slice(None), slice(None), 0), "front"
        elif pos in ["back", "z_end"]:
            return (slice(None), slice(None), -1), "back"
        else:
            raise ValueError(f"Unknown 3D boundary locator '{locator}'.")

    def get_mesh_indices(self):
        return np.arange(self.n_points).reshape(self.shape)
