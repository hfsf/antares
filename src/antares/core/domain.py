# -*- coding: utf-8 -*-

"""
Domain Module.

Defines the spatial domain classes (1D, 2D, 3D) for PDE discretization.
Automates the Method of Lines (MoL) matrix operations and applies N-dimensional 
tensorial derivatives safely to SymPy Abstract Syntax Trees (AST).
"""

from abc import ABC, abstractmethod
import numpy as np

from .error_definitions import UnexpectedValueError


def _ast_matmul_nd(matrix, tensor_sym, axis, target_unit):
    """
    Optimized N-Dimensional Tensorial Matrix Multiplication for AST EquationNodes.
    
    Uses Sparsity Mapping and dynamic axis transposition to avoid O(N^2) inner loops 
    over structural zeros, accelerating 3D PDE AST generation by over 90%.
    It is agnostic to the dimensionality of the tensor.

    :param np.ndarray matrix: The 1D finite difference sparse matrix (N, N).
    :param np.ndarray tensor_sym: The N-Dimensional array of symbolic nodes.
    :param int axis: The specific axis to apply the spatial derivative.
    :param Unit target_unit: The physical unit resulting from the derivation.
    :return: An array of symbolic nodes of the same shape as tensor_sym.
    :rtype: np.ndarray
    """
    shape = tensor_sym.shape
    N = shape[axis]
    
    # 1. SPARSE MAPPING: Pre-computes non-zero coordinates
    # For a tridiagonal matrix, this reduces the inner loop from N to 3.
    nonzero_elements = [[] for _ in range(N)]
    rows, cols = np.nonzero(matrix)
    for r, c in zip(rows, cols):
        nonzero_elements[r].append((c, float(matrix[r, c])))
        
    # 2. DYNAMIC AXIS SHIFTING
    # Moves the target derivation axis to position 0. Works for 1D, 2D, 3D, etc.
    tensor_moved = np.moveaxis(tensor_sym, axis, 0)
    orig_moved_shape = tensor_moved.shape
    
    # Flattens all other spatial dimensions into a single linear dimension
    rest_dim = np.prod(orig_moved_shape[1:], dtype=int)
    tensor_flat = tensor_moved.reshape(N, rest_dim)
    
    result_flat = np.empty((N, rest_dim), dtype=object)
    
    # 3. HIGH-PERFORMANCE SYMBOLIC CONTRACTION
    for i in range(N):
        nz_row = nonzero_elements[i]  # Usually just 2 or 3 elements!
        for j in range(rest_dim):
            row_sum = None
            
            for c, val in nz_row:
                term = tensor_flat[c, j] * val
                row_sum = term if row_sum is None else row_sum + term
            
            if row_sum is None:
                row_sum = tensor_flat[0, j] * 0.0
                
            row_sum.unit_object = target_unit
            result_flat[i, j] = row_sum
            
    # 4. RECONSTRUCTION
    # Reshapes back to the moved format, then returns the axis to its original physical spot
    result_moved = result_flat.reshape(orig_moved_shape)
    return np.moveaxis(result_moved, 0, axis)


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
        if pos in ["start", "inlet", "left", "bottom", "front"]: return 0, "start"
        if pos in ["end", "outlet", "right", "top", "back"]: return -1, "end"
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
            "In 2D, the gradient is a vector field. Apply gradients via normal axes."
        )

    def apply_laplacian(self, variable):
        sym_tensor = variable()
        base_unit = variable.discrete_nodes[0, 0]().unit_object
        
        d2_dx2 = _ast_matmul_nd(self.x.B_matrix, sym_tensor, axis=0, target_unit=base_unit/(self.x.unit**2))
        d2_dy2 = _ast_matmul_nd(self.y.B_matrix, sym_tensor, axis=1, target_unit=base_unit/(self.y.unit**2))
        
        return d2_dx2 + d2_dy2

    def get_normal_gradient(self, variable, locator):
        pos = str(locator).lower()
        sym_tensor = variable()
        base_unit = variable.discrete_nodes[0, 0]().unit_object
        
        if pos in ["left", "west", "x_start", "right", "east", "x_end"]:
            return _ast_matmul_nd(self.x.A_matrix, sym_tensor, axis=0, target_unit=base_unit/self.x.unit)
        elif pos in ["bottom", "south", "y_start", "top", "north", "y_end"]:
            return _ast_matmul_nd(self.y.A_matrix, sym_tensor, axis=1, target_unit=base_unit/self.y.unit)
        else:
            raise ValueError(f"Unknown 2D boundary locator '{locator}' for normal gradient.")


class Domain3D(Domain):
    """
    3-Dimensional spatial domain constructed as a Tensor Product of three 1D Domains.
    """
    def __init__(self, name, x_domain, y_domain, z_domain, description=""):
        super().__init__(name, description, method=x_domain.method)
        
        if not isinstance(x_domain, Domain1D) or not isinstance(y_domain, Domain1D) or not isinstance(z_domain, Domain1D):
            raise TypeError("Domain3D requires three Domain1D instances as axes.")
        
        self.x = x_domain
        self.y = y_domain
        self.z = z_domain
        self.shape = (self.x.n_points, self.y.n_points, self.z.n_points)
        self.n_points = self.shape[0] * self.shape[1] * self.shape[2]
        
        self.X_grid, self.Y_grid, self.Z_grid = np.meshgrid(
            self.x.grid, self.y.grid, self.z.grid, indexing='ij'
        )

    def get_bulk_slice(self):
        """Returns the internal volume slice, excluding all 6 faces."""
        return (slice(1, -1), slice(1, -1), slice(1, -1))

    def get_boundary(self, locator):
        """Translates semantic 3D face locators into precise NumPy 3D tuples."""
        pos = str(locator).lower()
        if pos in ["left", "west", "x_start"]: return (0, slice(None), slice(None)), "left"
        elif pos in ["right", "east", "x_end"]: return (-1, slice(None), slice(None)), "right"
        elif pos in ["bottom", "south", "y_start"]: return (slice(None), 0, slice(None)), "bottom"
        elif pos in ["top", "north", "y_end"]: return (slice(None), -1, slice(None)), "top"
        elif pos in ["front", "z_start"]: return (slice(None), slice(None), 0), "front"
        elif pos in ["back", "z_end"]: return (slice(None), slice(None), -1), "back"
        else: raise ValueError(f"Unknown 3D boundary locator '{locator}'.")

    def apply_gradient(self, variable):
        raise NotImplementedError(
            "In 3D, the gradient is a vector field. Apply gradients via normal axes."
        )

    def apply_laplacian(self, variable):
        """Calculates the 3D Laplacian: (d2T/dx2) + (d2T/dy2) + (d2T/dz2)."""
        sym_tensor = variable()
        base_unit = variable.discrete_nodes[0, 0, 0]().unit_object
        
        d2_dx2 = _ast_matmul_nd(self.x.B_matrix, sym_tensor, axis=0, target_unit=base_unit/(self.x.unit**2))
        d2_dy2 = _ast_matmul_nd(self.y.B_matrix, sym_tensor, axis=1, target_unit=base_unit/(self.y.unit**2))
        d2_dz2 = _ast_matmul_nd(self.z.B_matrix, sym_tensor, axis=2, target_unit=base_unit/(self.z.unit**2))
        
        return d2_dx2 + d2_dy2 + d2_dz2

    def get_normal_gradient(self, variable, locator):
        """Computes the spatial gradient normal to the specified 3D face."""
        pos = str(locator).lower()
        sym_tensor = variable()
        base_unit = variable.discrete_nodes[0, 0, 0]().unit_object
        
        if pos in ["left", "west", "x_start", "right", "east", "x_end"]:
            return _ast_matmul_nd(self.x.A_matrix, sym_tensor, axis=0, target_unit=base_unit/self.x.unit)
        elif pos in ["bottom", "south", "y_start", "top", "north", "y_end"]:
            return _ast_matmul_nd(self.y.A_matrix, sym_tensor, axis=1, target_unit=base_unit/self.y.unit)
        elif pos in ["front", "z_start", "back", "z_end"]:
            return _ast_matmul_nd(self.z.A_matrix, sym_tensor, axis=2, target_unit=base_unit/self.z.unit)
        else:
            raise ValueError(f"Unknown 3D boundary locator '{locator}' for normal gradient.")