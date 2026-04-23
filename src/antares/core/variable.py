# -*- coding: utf-8 -*-

"""
Variable Module.

This module defines the Variable class for the ANTARES framework, handling
both lumped (scalar) and distributed (vector/tensor) states within the symbolic
computational graph.

V4 UPDATE: Block Vectorization Support.
Variables natively represent N-Dimensional fields as unified mathematical blocks,
preventing graph explosion during AD (Algorithmic Differentiation) in solvers.
"""

import numpy as np
import sympy as sp

from .equation_operators import _Diff
from .expression_evaluation import EquationNode
from .quantity import Quantity


class Variable(Quantity):
    """
    Represents a mathematical variable within the ANTARES flowsheet.
    Maintains dimensional coherence and tracks the spatial topology for PDEs.
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
        self._raw_units = units
        super().__init__(name, units, description, value, latex_text, owner_model_name)

        # Base Symbol for the Block Vectorization
        self.symbolic_object = sp.Symbol(self.name)

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

        # Arrays for DAE partitioning and Initial Conditions
        self.mesh_indices = None
        self.node_types = None
        self.initial_condition_array = None

    def distributeOnDomain(self, domain):
        """
        Maps the variable to a spatial domain as a unified tensorial block.
        Prepares the DAE classification array to resolve Boundary Conditions.

        :param Domain domain: The target discretization domain.
        """
        self.domain = domain
        self.is_distributed = True
        self.tensor_shape = getattr(domain, "shape", (domain.n_points,))
        self.n_points = domain.n_points
        self.mesh_indices = domain.get_mesh_indices()

        # All nodes start with the base exposure type (usually 'differential')
        self.node_types = np.full(self.n_points, self.type, dtype=object)
        self.initial_condition_array = np.zeros(self.n_points)

    def distributeOn(self, domain):
        """Syntactic sugar to delegate discretization to the Master Model."""
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
        Applies initial values to the block tensor.

        :param float/ndarray value: The initial value to inject.
        :param location: The topological slice where the value is applied.
        """
        if location is None or location == "all":
            self.initial_condition_array[:] = value
        else:
            self.initial_condition_array[location] = value

    def __call__(self):
        """
        Evaluates the variable, returning its symbolic representation.
        In V4, returns a single EquationNode masking the entire spatial block.
        """
        if self.is_distributed:
            return EquationNode(
                name=self.name,
                symbolic_object=self.symbolic_object,
                repr_symbolic=self.symbolic_object,
                unit_object=self.units,
            )
        return super().__call__()

    def Diff(self, ind_var=None):
        """Applies the temporal derivative operator."""
        if self.is_distributed:
            sym_dot = sp.Symbol(self.name + "_dot")

            # Delega a resolução rigorosa da unidade temporal para o operador nativo
            # Isso garante que a unidade seja dividida pelo tempo da simulação (ex: /s)
            dummy_diff_node = _Diff(self, ind_var)
            time_derivative_unit = dummy_diff_node.unit_object

            enode = EquationNode(
                name=f"d({self.name})",
                symbolic_object=sym_dot,
                repr_symbolic=sym_dot,
                unit_object=time_derivative_unit,
            )
            enode.equation_type["is_differential"] = True
            return enode
        return _Diff(self, ind_var)

    def Grad(self, domain=None):
        """Delegates spatial gradient tagging to the Domain."""
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed.")
        return dom.apply_gradient(self)

    def Div(self, domain=None):
        """Delegates spatial Laplacian tagging to the Domain."""
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed.")
        return dom.apply_laplacian(self)
