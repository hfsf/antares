# -*- coding: utf-8 -*-

"""
Domain Module.

Defines the spatial domain abstract base class and its dimension-specific implementations
(e.g., Domain1D). Automates the Method of Lines (MoL) matrix operations and prepares 
the infrastructure for N-dimensional tensor expansions.
"""

from abc import ABC, abstractmethod

import numpy as np

from .error_definitions import UnexpectedValueError


def _ast_matmul(matrix, vector, original_unit, domain_unit, deriv_order=1):
    """
    Performs a safe matrix multiplication for Abstract Syntax Tree (AST) EquationNodes.
    Ensures dimensional coherence is maintained and protects the CasadiTranspiler
    from residual SymPy objects.

    :param np.ndarray matrix: The finite difference coefficient matrix.
    :param list/np.ndarray vector: The vector of symbolic EquationNodes.
    :param Unit original_unit: The physical unit of the state variable.
    :param Unit domain_unit: The physical unit of the spatial domain.
    :param int deriv_order: The derivative order (1 for Gradient, 2 for Laplacian).
    :return: An array of symbolic nodes representing the spatial derivative.
    :rtype: np.ndarray
    """
    N = len(vector)
    result = []

    target_unit = original_unit / (domain_unit**deriv_order)

    for i in range(N):
        row_sum = None  # Use None to prevent triggering overloaded __add__ prematurely
        for j in range(N):
            # Strict cast to native Python float.
            # Prevents SymPy from generating 'sympy.Float' residuals that break CasADi JIT.
            val = float(matrix[i, j])

            if val != 0.0:
                term = vector[j] * val
                if row_sum is None:
                    row_sum = term
                else:
                    row_sum = row_sum + term

        # Safety fallback if the entire row is zero (e.g., boundary nodes)
        if row_sum is None:
            row_sum = vector[0] * 0.0

        if hasattr(row_sum, "unit_object"):
            row_sum.unit_object = target_unit

        result.append(row_sum)

    return np.array(result)


class Domain(ABC):
    """
    Abstract Base Class for all spatial domains (1D, 2D, 3D).
    Enforces a strict architectural contract for dimension-agnostic modeling,
    ensuring that the Model class never needs to know the underlying geometry.
    """

    def __init__(self, name, description="", method="mol"):
        """
        Initializes the base Domain properties.

        :param str name: The identifier for the domain configuration.
        :param str description: Optional domain description.
        :param str method: Discretization method. 'mol' (Method of Lines) or 'collocation'.
        """
        self.name = name
        self.description = description
        self.method = method.lower()
        self._owner_model_instance = None

    @abstractmethod
    def get_bulk_slice(self):
        """
        Dimension-agnostic bulk locator.
        Must return the slice (or tuple of slices) required to extract the 
        interior nodes of the domain, excluding the boundary elements.
        """
        pass

    @abstractmethod
    def get_boundary(self, locator):
        """
        Dimension-agnostic boundary translator.
        Must translate a semantic location (e.g., 'start', 'top') or a raw index 
        into the appropriate NumPy slicing objects for the specific dimension.

        :param str/slice/tuple locator: The user-defined boundary location.
        :return: A tuple containing (numpy_index, string_suffix).
        :rtype: tuple
        """
        pass

    @abstractmethod
    def _build_mesh(self):
        """Internal method to construct numerical grids and operator matrices."""
        pass


class Domain1D(Domain):
    """
    Represents a 1-Dimensional spatial domain.
    Acts as the foundational building block for multi-dimensional spatial systems.
    """

    def __init__(
        self, name, length, n_points, unit, description="", method="mol", diff_scheme="central"
    ):
        """
        Instantiates a 1D spatial domain.

        :param str name: The identifier for the axis (e.g., 'z', 'r').
        :param float length: The total physical length of the domain.
        :param int n_points: The number of discrete nodes in the mesh.
        :param Unit unit: The physical unit of the axis.
        :param str description: Optional domain description.
        :param str method: Discretization method. 'mol' (Finite Differences) or 'collocation'.
        :param str diff_scheme: The finite difference scheme ('central', 'backward', 'forward').
        """
        super().__init__(name, description, method)
        
        self.length = float(length)
        self.n_points = int(n_points)
        self.unit = unit
        self.diff_scheme = diff_scheme.lower()

        self.grid = None
        self.dz = None
        
        # Spatial operators
        self.A_matrix = None  # 1st Derivative Matrix (Gradient)
        self.B_matrix = None  # 2nd Derivative Matrix (Laplacian)

        self._build_mesh()

    def get_bulk_slice(self):
        """
        Returns the indexing slice required to extract the 1D interior nodes.
        
        :return: A slice representing the bulk interior.
        :rtype: slice
        """
        if self.method == "mol":
            # For finite differences, the bulk strictly excludes node 0 and node -1.
            return slice(1, -1)
        elif self.method == "collocation":
            return slice(1, -1)

    def get_boundary(self, locator):
        """
        Translates a semantic 1D boundary string into an exact index.

        :param str/int/slice locator: 'start', 'end', 'inlet', 'outlet', or an explicit index.
        :return: Tuple containing the numeric index and a naming suffix for the solver.
        :rtype: tuple
        :raises ValueError: If an incompatible 2D/3D term (e.g., 'top') is used.
        """
        if isinstance(locator, str):
            pos_lower = locator.lower()
            if pos_lower in ["start", "inlet", "left", "bottom"]:
                return 0, "start"
            elif pos_lower in ["end", "outlet", "right", "top"]:
                return -1, "end"
            else:
                raise ValueError(
                    f"Invalid boundary locator '{locator}' for a 1D Domain. "
                    f"Accepted terms are: 'start', 'end', 'inlet', 'outlet', 'left', 'right'."
                )
        else:
            # Fallback for explicit slices or advanced tuple logic provided by the user
            return locator, f"idx_{str(locator).replace(' ', '')}"

    def _build_mesh(self):
        """
        Routes the mesh generation to the appropriate numerical method.
        """
        if self.method == "mol":
            self._build_finite_difference_matrices()
        elif self.method == "collocation":
            raise NotImplementedError(
                "[ANTARES ARCHITECTURE] Orthogonal Collocation on Finite Elements (OCFE) "
                "requires Jacobi/Radau polynomial root finding algorithms to establish "
                "non-uniform grids. This feature is scheduled for a future release."
            )
        else:
            raise UnexpectedValueError("Method must be 'mol' or 'collocation'.")

    def _build_finite_difference_matrices(self):
        """
        Generates the banded coefficient matrices for the Method of Lines (MoL).
        """
        self.dz = self.length / (self.n_points - 1)
        self.grid = np.linspace(0, self.length, self.n_points)

        N = self.n_points
        dz = self.dz

        self.A_matrix = np.zeros((N, N))
        self.B_matrix = np.zeros((N, N))

        # --- Matrix B (2nd Derivative - Central Difference O(h^2)) ---
        for i in range(1, N - 1):
            self.B_matrix[i, i - 1] = 1.0 / (dz**2)
            self.B_matrix[i, i] = -2.0 / (dz**2)
            self.B_matrix[i, i + 1] = 1.0 / (dz**2)

        # Matrix B Boundaries (Forward/Backward diff O(h^2))
        self.B_matrix[0, 0:3] = [1.0 / (dz**2), -2.0 / (dz**2), 1.0 / (dz**2)]
        self.B_matrix[-1, -3:] = [1.0 / (dz**2), -2.0 / (dz**2), 1.0 / (dz**2)]

        # --- Matrix A (1st Derivative) ---
        if self.diff_scheme == "backward":
            # Ideal for highly advective flows (plug-flow regime)
            for i in range(1, N):
                self.A_matrix[i, i] = 1.0 / dz
                self.A_matrix[i, i - 1] = -1.0 / dz
            
            # Forward diff fallback for the first node
            self.A_matrix[0, 0:2] = [-1.0 / dz, 1.0 / dz]

        elif self.diff_scheme == "central":
            # Ideal for diffusion-dominated regimes
            for i in range(1, N - 1):
                self.A_matrix[i, i + 1] = 1.0 / (2 * dz)
                self.A_matrix[i, i - 1] = -1.0 / (2 * dz)
            
            # Forward diff for start, Backward diff for end
            self.A_matrix[0, 0:3] = [-3.0 / (2 * dz), 4.0 / (2 * dz), -1.0 / (2 * dz)]
            self.A_matrix[-1, -3:] = [1.0 / (2 * dz), -4.0 / (2 * dz), 3.0 / (2 * dz)]