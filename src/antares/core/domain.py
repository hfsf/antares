# -*- coding: utf-8 -*-

"""
Domain Module (V5 Native CasADi Architecture).

Defines the spatial domain classes (1D, 2D, 3D) for PDE discretization.
In the V5 Architecture, the Domain actively constructs and caches its own
sparse finite difference matrices (CasADi DM). It completely bypasses SymPy
and applies spatial operators (Laplacian, Gradient) via native C++ matrix
multiplication (ca.mtimes) directly to the MX symbolic vectors.
"""

from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
import scipy.sparse as sps

from .error_definitions import UnexpectedValueError
from .expression_evaluation import EquationNode


class Domain(ABC):
    """
    Abstract Base Class enforcing the architectural contract for all domains.
    Provides the standard interface for boundary topological mapping and
    native CasADi sparse matrix generation.
    """

    def __init__(self, name, description="", method="mol"):
        self.name = name
        self.description = description
        self.method = method.lower()
        self._owner_model_instance = None

        # Caches for CasADi Data Matrices (DM) to ensure O(1) performance
        # on repeated spatial operator calls within the model.
        self._B_ca = None
        self._A_ca = {}

    @abstractmethod
    def get_bulk_slice(self):
        """Returns the slicing tuple representing the interior nodes."""
        pass

    @abstractmethod
    def get_boundary(self, locator):
        """Returns the slicing tuple and topological suffix for a specified boundary."""
        pass

    @abstractmethod
    def get_mesh_indices(self):
        """Returns an N-Dimensional numpy array containing the flat indices of the mesh."""
        pass

    @abstractmethod
    def _build_laplacian_matrix(self):
        """Builds the N-Dimensional Laplacian sparse matrix (scipy.sparse)."""
        pass

    @abstractmethod
    def _build_gradient_matrix(self, axis=None):
        """Builds the N-Dimensional Gradient sparse matrix (scipy.sparse) for a specific axis."""
        pass

    def _get_length_unit(self):
        """
        Helper to extract the fundamental length unit of the domain.

        :return: Unit object representing the spatial length.
        :rtype: Unit
        :raises ValueError: If no length unit is defined.
        """
        if hasattr(self, "unit"):
            return self.unit
        elif hasattr(self, "x"):
            return self.x.unit
        raise ValueError(f"Domain '{self.name}' lacks a defined length unit.")

    def _scipy_to_casadi(self, mat):
        """
        Converts a scipy.sparse matrix to a native CasADi Data Matrix (DM).

        :param scipy.sparse.spmatrix mat: The sparse matrix.
        :return: CasADi Sparse DM.
        :rtype: casadi.DM
        """
        mat = mat.tocsc()
        colind = mat.indptr.astype(int).tolist()
        row = mat.indices.astype(int).tolist()
        data = mat.data.tolist()
        sparsity = ca.Sparsity(mat.shape[0], mat.shape[1], colind, row)
        return ca.DM(sparsity, data)

    def _get_casadi_laplacian(self):
        """Fetches or builds the cached CasADi Laplacian matrix."""
        if self._B_ca is None:
            self._B_ca = self._scipy_to_casadi(self._build_laplacian_matrix())
        return self._B_ca

    def _get_casadi_gradient(self, axis=None):
        """Fetches or builds the cached CasADi Gradient matrix for the given axis."""
        if axis not in self._A_ca:
            self._A_ca[axis] = self._scipy_to_casadi(self._build_gradient_matrix(axis))
        return self._A_ca[axis]

    def apply_laplacian(self, variable):
        """
        Applies the spatial Laplacian operator to the distributed variable.
        Performs native CasADi matrix-vector multiplication (B * x).

        :param Variable variable: The target distributed state variable.
        :return: An EquationNode containing the resulting C++ graph.
        :rtype: EquationNode
        """
        B_ca = self._get_casadi_laplacian()
        sym_res = ca.mtimes(B_ca, variable.symbolic_object)
        len_unit = self._get_length_unit()

        return EquationNode(
            name=f"Laplacian({variable.name})",
            symbolic_object=sym_res,
            unit_object=variable.units / (len_unit**2),
        )

    def get_normal_gradient(self, variable, locator):
        """
        Computes the spatial gradient normal to the specified boundary.

        :param Variable variable: The target distributed state variable.
        :param str locator: The semantic boundary locator (e.g., 'left', 'top').
        :return: An EquationNode containing the resulting C++ graph.
        :rtype: EquationNode
        """
        pos = str(locator).lower()
        axis = None

        if pos in [
            "start",
            "inlet",
            "left",
            "west",
            "x_start",
            "end",
            "outlet",
            "right",
            "east",
            "x_end",
        ]:
            axis = self.x.name if hasattr(self, "x") else None
        elif pos in ["bottom", "south", "y_start", "top", "north", "y_end"]:
            axis = self.y.name if hasattr(self, "y") else None
        elif pos in ["front", "z_start", "back", "z_end"]:
            axis = self.z.name if hasattr(self, "z") else None

        A_ca = self._get_casadi_gradient(axis)
        sym_res = ca.mtimes(A_ca, variable.symbolic_object)
        len_unit = self._get_length_unit()

        return EquationNode(
            name=f"NormalGrad_{locator}({variable.name})",
            symbolic_object=sym_res,
            unit_object=variable.units / len_unit,
        )

    def apply_gradient(self, variable):
        """
        Applies the full spatial gradient operator to the distributed variable.
        Primarily designed for 1D systems.

        :param Variable variable: The target distributed state variable.
        :return: An EquationNode containing the resulting C++ graph.
        :rtype: EquationNode
        """
        A_ca = self._get_casadi_gradient()
        sym_res = ca.mtimes(A_ca, variable.symbolic_object)
        len_unit = self._get_length_unit()

        return EquationNode(
            name=f"Grad({variable.name})",
            symbolic_object=sym_res,
            unit_object=variable.units / len_unit,
        )


class Domain1D(Domain):
    """1-Dimensional spatial domain utilizing the Method of Lines (MoL)."""

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

    def _build_laplacian_matrix(self):
        return sps.csr_matrix(self.B_matrix)

    def _build_gradient_matrix(self, axis=None):
        return sps.csr_matrix(self.A_matrix)

    def _build_mesh(self):
        """Generates the 1D finite difference core matrices (A and B)."""
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

    def _build_laplacian_matrix(self):
        Lx, Ly = sps.csr_matrix(self.x.B_matrix), sps.csr_matrix(self.y.B_matrix)
        Ix, Iy = sps.eye(self.x.n_points), sps.eye(self.y.n_points)
        return sps.kron(Lx, Iy) + sps.kron(Ix, Ly)

    def _build_gradient_matrix(self, axis=None):
        Ix, Iy = sps.eye(self.x.n_points), sps.eye(self.y.n_points)
        if axis == self.x.name:
            return sps.kron(sps.csr_matrix(self.x.A_matrix), Iy)
        elif axis == self.y.name:
            return sps.kron(Ix, sps.csr_matrix(self.y.A_matrix))
        return sps.csr_matrix(self.x.A_matrix)  # Fallback


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

    def _build_laplacian_matrix(self):
        Lx, Ly, Lz = (
            sps.csr_matrix(self.x.B_matrix),
            sps.csr_matrix(self.y.B_matrix),
            sps.csr_matrix(self.z.B_matrix),
        )
        Ix, Iy, Iz = (
            sps.eye(self.x.n_points),
            sps.eye(self.y.n_points),
            sps.eye(self.z.n_points),
        )

        return (
            sps.kron(sps.kron(Lx, Iy), Iz)
            + sps.kron(sps.kron(Ix, Ly), Iz)
            + sps.kron(sps.kron(Ix, Iy), Lz)
        )

    def _build_gradient_matrix(self, axis=None):
        Ix, Iy, Iz = (
            sps.eye(self.x.n_points),
            sps.eye(self.y.n_points),
            sps.eye(self.z.n_points),
        )

        if axis == self.x.name:
            return sps.kron(sps.kron(sps.csr_matrix(self.x.A_matrix), Iy), Iz)
        elif axis == self.y.name:
            return sps.kron(sps.kron(Ix, sps.csr_matrix(self.y.A_matrix)), Iz)
        elif axis == self.z.name:
            return sps.kron(sps.kron(Ix, Iy), sps.csr_matrix(self.z.A_matrix))

        return sps.csr_matrix(self.x.A_matrix)  # Fallback
