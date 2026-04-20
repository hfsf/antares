# -*- coding: utf-8 -*-

"""
Domain Module.

Defines the spatial domain classes (1D, 2D) for PDE discretization.
Automates the Method of Lines (MoL) matrix operations and applies N-dimensional 
tensorial derivatives safely to SymPy Abstract Syntax Trees (AST).
"""

from abc import ABC, abstractmethod
import numpy as np

from .error_definitions import UnexpectedValueError


def _ast_matmul_nd(matrix, tensor_sym, axis, target_unit):
    """
    Performs a safe Tensorial Matrix Multiplication for AST EquationNodes.
    Prevents SymPy from generating 'sympy.Float' residuals that break CasADi JIT.
    
    :param np.ndarray matrix: The 1D finite difference matrix (N, N).
    :param np.ndarray tensor_sym: The N-Dimensional array of symbolic nodes.
    :param int axis: The axis along which to apply the derivative (0 for X, 1 for Y).
    :param Unit target_unit: The resulting physical unit.
    :return: An array of symbolic nodes of the same shape as tensor_sym.
    :rtype: np.ndarray
    """
    shape = tensor_sym.shape
    result = np.empty(shape, dtype=object)
    
    if len(shape) == 1:
        N = shape[0]
        for i in range(N):
            row_sum = None
            for j in range(N):
                val = float(matrix[i, j])
                if val != 0.0:
                    term = tensor_sym[j] * val
                    row_sum = term if row_sum is None else row_sum + term
            
            if row_sum is None:
                row_sum = tensor_sym[0] * 0.0
                
            row_sum.unit_object = target_unit
            result[i] = row_sum
        return result
    
    elif len(shape) == 2:
        Nx, Ny = shape
        if axis == 0:  # Apply derivative along X-axis (Rows) -> M @ T
            for i in range(Nx):
                for j in range(Ny):
                    row_sum = None
                    for k in range(Nx):
                        val = float(matrix[i, k])
                        if val != 0.0:
                            term = tensor_sym[k, j] * val
                            row_sum = term if row_sum is None else row_sum + term
                    if row_sum is None:
                        row_sum = tensor_sym[0, j] * 0.0
                    row_sum.unit_object = target_unit
                    result[i, j] = row_sum
            return result
            
        elif axis == 1:  # Apply derivative along Y-axis (Cols) -> T @ M.T
            for i in range(Nx):
                for j in range(Ny):
                    row_sum = None
                    for k in range(Ny):
                        val = float(matrix[j, k])
                        if val != 0.0:
                            term = tensor_sym[i, k] * val
                            row_sum = term if row_sum is None else row_sum + term
                    if row_sum is None:
                        row_sum = tensor_sym[i, 0] * 0.0
                    row_sum.unit_object = target_unit
                    result[i, j] = row_sum
            return result
    else:
        raise NotImplementedError("Derivatives for 3D tensors are planned for future releases.")


class Domain(ABC):
    """Abstract Base Class enforcing the architectural contract for all domains."""
    def __init__(self, name, description="", method="mol"):
        self.name = name
        self.description = description
        self.method = method.lower()
        self._owner_model_instance = None

    @abstractmethod
    def get_bulk_slice(self): pass

    @abstractmethod
    def get_boundary(self, locator): pass

    @abstractmethod
    def apply_gradient(self, variable): pass

    @abstractmethod
    def apply_laplacian(self, variable): pass

    @abstractmethod
    def get_normal_gradient(self, variable, locator): pass


class Domain1D(Domain):
    """1-Dimensional spatial domain using Method of Lines."""
    def __init__(self, name, length, n_points, unit, description="", method="mol", diff_scheme="central"):
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
        if pos in ["start", "inlet", "left", "bottom"]: return 0, "start"
        if pos in ["end", "outlet", "right", "top"]: return -1, "end"
        return locator, f"idx_{str(locator).replace(' ', '')}"

    def apply_gradient(self, variable):
        base_unit = variable.discrete_nodes[0]().unit_object
        target_unit = base_unit / self.unit
        return _ast_matmul_nd(self.A_matrix, variable(), axis=0, target_unit=target_unit)

    def apply_laplacian(self, variable):
        base_unit = variable.discrete_nodes[0]().unit_object
        target_unit = base_unit / (self.unit**2)
        return _ast_matmul_nd(self.B_matrix, variable(), axis=0, target_unit=target_unit)

    def get_normal_gradient(self, variable, locator):
        """
        Computes the spatial gradient normal to the specified boundary.
        In 1D, there is only one axis, so it returns the standard gradient.

        :param Variable variable: The distributed state variable.
        :param str/int/slice locator: The boundary locator.
        :return: A NumPy array containing the symbolic normal gradient.
        :rtype: np.ndarray
        """
        return self.apply_gradient(variable)

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
    """
    2-Dimensional spatial domain constructed as a Tensor Product of two 1D Domains.
    """
    def __init__(self, name, x_domain, y_domain, description=""):
        super().__init__(name, description, method=x_domain.method)
        
        if not isinstance(x_domain, Domain1D) or not isinstance(y_domain, Domain1D):
            raise TypeError("Domain2D requires two Domain1D instances as axes.")
        
        self.x = x_domain
        self.y = y_domain
        self.shape = (self.x.n_points, self.y.n_points)
        self.n_points = self.shape[0] * self.shape[1]
        self.X_grid, self.Y_grid = np.meshgrid(self.x.grid, self.y.grid, indexing='ij')

    def get_bulk_slice(self):
        return (slice(1, -1), slice(1, -1))

    def get_boundary(self, locator):
        pos = str(locator).lower()
        if pos in ["left", "west", "x_start"]: return (0, slice(None)), "left"
        elif pos in ["right", "east", "x_end"]: return (-1, slice(None)), "right"
        elif pos in ["bottom", "south", "y_start"]: return (slice(None), 0), "bottom"
        elif pos in ["top", "north", "y_end"]: return (slice(None), -1), "top"
        else: raise ValueError(f"Unknown 2D boundary locator '{locator}'.")

    def apply_gradient(self, variable):
        raise NotImplementedError(
            "In 2D, the gradient is a vector field (dT/dx, dT/dy). "
            "Please apply gradients directly via individual axes if needed."
        )

    def apply_laplacian(self, variable):
        sym_tensor = variable()
        base_unit = variable.discrete_nodes[0, 0]().unit_object
        
        d2_dx2 = _ast_matmul_nd(self.x.B_matrix, sym_tensor, axis=0, target_unit=base_unit/(self.x.unit**2))
        d2_dy2 = _ast_matmul_nd(self.y.B_matrix, sym_tensor, axis=1, target_unit=base_unit/(self.y.unit**2))
        
        return d2_dx2 + d2_dy2

    def get_normal_gradient(self, variable, locator):
        """
        Computes the spatial gradient normal to the specified 2D boundary.
        Automatically selects the orthogonal derivative (X or Y) based on the boundary.

        :param Variable variable: The distributed state variable.
        :param str locator: The semantic boundary locator (e.g., 'top', 'left').
        :return: A NumPy array containing the symbolic normal gradient.
        :rtype: np.ndarray
        """
        pos = str(locator).lower()
        sym_tensor = variable()
        base_unit = variable.discrete_nodes[0, 0]().unit_object
        
        if pos in ["left", "west", "x_start", "right", "east", "x_end"]:
            # Normal to Left/Right is the X-axis (axis 0)
            return _ast_matmul_nd(self.x.A_matrix, sym_tensor, axis=0, target_unit=base_unit/self.x.unit)
            
        elif pos in ["bottom", "south", "y_start", "top", "north", "y_end"]:
            # Normal to Bottom/Top is the Y-axis (axis 1)
            return _ast_matmul_nd(self.y.A_matrix, sym_tensor, axis=1, target_unit=base_unit/self.y.unit)
            
        else:
            raise ValueError(f"Unknown 2D boundary locator '{locator}' for normal gradient.")