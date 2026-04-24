# -*- coding: utf-8 -*-

"""
Variable Module (V5 Native CasADi Architecture).

This module defines the Variable class for the ANTARES framework.
In V5, SymPy is completely eradicated. Variables directly instantiate
CasADi MX symbolic objects, whether lumped (scalar) or distributed (N-Dimensional tensors).
This guarantees zero-overhead compilation and immediate matrix generation for PDEs.
"""

import casadi as ca
import numpy as np

from .equation_operators import _Diff
from .expression_evaluation import EquationNode
from .quantity import Quantity


class Variable(Quantity):
    """
    Represents a mathematical state variable within the ANTARES flowsheet.
    Inherits from Quantity to maintain strict dimensional coherence.
    In V5, it acts as a direct wrapper for CasADi MX symbolic vectors.
    """

    def __init__(
        self,
        name,
        units,
        description="",
        is_lower_bounded=False,
        is_upper_bounded=False,
        lower_bound=None,
        upper_bound=None,
        value=0.0,
        is_exposed=False,
        exposure_type="",
        latex_text="",
        owner_model_name="",
        domain=None,
    ):
        """
        Instantiates a physical Variable.

        :param str name: Unique symbolic identifier.
        :param Unit units: Physical unit object for dimensional coherence.
        :param str description: Physical description of the variable.
        :param bool is_lower_bounded: Flag for active lower bounds in solvers.
        :param bool is_upper_bounded: Flag for active upper bounds in solvers.
        :param float lower_bound: Numerical value for the minimum limit.
        :param float upper_bound: Numerical value for the maximum limit.
        :param float value: Nominal or initial scalar value.
        :param bool is_exposed: True if the variable acts as a flowsheet port.
        :param str exposure_type: DAE classification (e.g., 'differential', 'algebraic').
        :param str latex_text: LaTeX formatting string for reports.
        :param str owner_model_name: Name of the parent model.
        :param Domain domain: Optional spatial domain for immediate distribution.
        """
        self._raw_units = units
        super().__init__(name, units, description, value, latex_text, owner_model_name)

        self.is_lower_bounded = lower_bound is not None
        self.is_upper_bounded = upper_bound is not None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_exposed = is_exposed
        self.type = exposure_type

        # Spatial Topology Attributes
        self.is_distributed = False
        self.domain = None
        self.tensor_shape = None
        self.n_points = 1

        # V5 NATIVE CASADI ALLOCATION (Starts as scalar by default)
        self.symbolic_object = ca.MX.sym(self.name)

        # Arrays for DAE partitioning and Initial Conditions
        self.mesh_indices = None
        self.node_types = np.array([self.type], dtype=object)
        self.initial_condition_array = np.array([value], dtype=float)

    def distributeOnDomain(self, domain):
        """
        Maps the variable to a spatial domain as a unified tensorial block.
        Re-allocates the CasADi symbolic object natively as an N-dimensional vector.

        :param Domain domain: The target discretization domain.
        """
        self.domain = domain
        self.is_distributed = True
        self.tensor_shape = getattr(domain, "shape", (domain.n_points,))
        self.n_points = domain.n_points
        self.mesh_indices = domain.get_mesh_indices()

        # V5 VECTORIAL RE-ALLOCATION: The core of the new architecture
        # Allocates a 1D column vector in C++ representing the entire spatial mesh
        self.symbolic_object = ca.MX.sym(self.name, self.n_points)

        # Reset DAE topological maps
        self.node_types = np.full(self.n_points, self.type, dtype=object)
        self.initial_condition_array = np.zeros(self.n_points)

    def distributeOn(self, domain):
        """
        Syntactic sugar to delegate discretization to the Master Model.

        :param Domain domain: The target discretization domain.
        :raises RuntimeError: If the variable is not bound to a Model instance.
        """
        if (
            hasattr(self, "_owner_model_instance")
            and self._owner_model_instance is not None
        ):
            self._owner_model_instance.distributeVariable(self, domain)
        else:
            raise RuntimeError(
                f"Variable '{self.name}' is orphaned! It must be created via "
                f"model.createVariable() to be bound to a parent Model instance."
            )

    def setNodeType(self, indices, new_type):
        """
        Overrides the DAE classification for a specific subset of the variable.
        Vital for establishing Algebraic boundaries in Differential fields.

        :param array-like indices: Flat indices of the targeted nodes.
        :param str new_type: The new DAE type (e.g., 'algebraic').
        """
        self.node_types[indices] = new_type

    def setVectorialInitialCondition(self, value, location=None):
        """
        Applies initial values to specific topological slices of the block tensor.

        :param float or ndarray value: The initial value to inject.
        :param location: The topological slice where the value is applied.
        """
        if location is None or location == "all":
            self.initial_condition_array[:] = value
        else:
            self.initial_condition_array[location] = value

    def __call__(self):
        """
        Evaluates the variable, returning its EquationNode wrapper.

        :return: A node encapsulating the native CasADi symbol and its units.
        :rtype: EquationNode
        """
        return EquationNode(
            name=self.name, symbolic_object=self.symbolic_object, unit_object=self.units
        )

    def Diff(self, ind_var=None):
        """
        Applies the temporal derivative operator.
        In V5, directly instantiates the companion `_dot` CasADi symbolic vector.

        :param ind_var: Optional independent variable (typically time).
        :return: An EquationNode representing the temporal derivative vector.
        :rtype: EquationNode
        """
        dim = self.n_points if self.is_distributed else 1
        sym_dot = ca.MX.sym(self.name + "_dot", dim)

        # Delegates physical unit resolution to the native operator
        dummy_diff_node = _Diff(self, ind_var)
        time_derivative_unit = dummy_diff_node.unit_object

        enode = EquationNode(
            name=f"d({self.name})",
            symbolic_object=sym_dot,
            unit_object=time_derivative_unit,
        )
        enode.equation_type["is_differential"] = True
        return enode

    def Grad(self, domain=None):
        """
        Delegates spatial gradient matrix multiplication to the Domain.

        :param Domain domain: Optional override for the target domain.
        :return: An EquationNode representing the spatial gradient.
        :rtype: EquationNode
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed.")
        return dom.apply_gradient(self)

    def Div(self, domain=None):
        """
        Delegates spatial Laplacian matrix multiplication to the Domain.

        :param Domain domain: Optional override for the target domain.
        :return: An EquationNode representing the Laplacian operator.
        :rtype: EquationNode
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed.")
        return dom.apply_laplacian(self)
