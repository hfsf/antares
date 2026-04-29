# -*- coding: utf-8 -*-

"""
Variable Module (V5 Native CasADi Architecture).

This module defines the Variable class for the ANTARES framework.
V5 leverages direct CasADi MX symbolic objects for both lumped and distributed 
parameters, ensuring high-performance Jacobian evaluation. It includes 
full support for topological tensor reshaping required by 1D, 2D, and 3D PDEs.
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
        :param Unit units: Physical unit object.
        :param str description: Physical description.
        :param float value: Nominal initial value.
        """
        self._raw_units = units
        super().__init__(name, units, description, value, latex_text, owner_model_name)

        self.is_lower_bounded = lower_bound is not None
        self.is_upper_bounded = upper_bound is not None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_exposed = is_exposed
        self.type = exposure_type

        # Topology and Allocation
        self.is_distributed = False
        self.domain = None
        self.tensor_shape = None
        self.mesh_indices = None
        self.n_points = 1
        
        # Native CasADi allocation
        self.symbolic_object = ca.MX.sym(self.name)
        self.initial_condition_array = np.array([value], dtype=float)
        self.node_types = np.array([self.type], dtype=object)

    def fix(self, value):
        """
        Fixes the variable to a constant numerical value by injecting a 
        structural residual equation into the model hierarchy.
        
        This method resolves the abstraction leak by automating the 
        Degrees of Freedom (DOF) closure. It ensures the specification 
        is propagated to the root flowsheet regardless of submodel depth.

        :param float value: The numerical value to fix the variable to.
        :raises RuntimeError: If the variable is not bound to a model instance.
        """
        if not hasattr(self, "_owner_model_instance") or self._owner_model_instance is None:
            raise RuntimeError(
                f"Variable '{self.name}' is orphaned. It must be created via "
                "model.createVariable() to support the .fix() method."
            )

        # 1. Update the Initial Guess (Numerical Stability)
        if not self.is_distributed:
            if hasattr(self, "setValue"):
                self.setValue(value)
            else:
                self.value = value
                self.initial_condition_array[0] = value
        else:
            self.setVectorialInitialCondition(value)

        # 2. Generate the residual expression (Var - Value = 0)
        residual_expr = self() - value

        # 3. Extract pure local name to prevent dictionary overwrite collisions
        owner_name = self._owner_model_instance.name
        local_name = self.name
        suffix = f"_{owner_name}"
        if local_name.endswith(suffix):
            local_name = local_name[:-len(suffix)]

        spec_name = f"Spec_{local_name}"
        desc = f"Fixed specification for {local_name}"
        
        # 4. Request the owner model to inject the equation (triggers upward propagation)
        self._owner_model_instance.createEquation(
            name=spec_name, 
            description=desc, 
            expr=residual_expr
        )

    def distributeOnDomain(self, domain):
        """
        Maps the variable to a spatial domain as a unified vector.
        Preserves the tensor shape to allow multidimensional unrolling (2D/3D).

        :param Domain domain: The spatial domain to distribute upon.
        """
        self.domain = domain
        self.is_distributed = True
        
        # Restored Topological Trackers required by the Simulator and Plotter
        self.n_points = domain.n_points
        self.tensor_shape = getattr(domain, "shape", (domain.n_points,))
        if hasattr(domain, "get_mesh_indices"):
            self.mesh_indices = domain.get_mesh_indices()
            
        # Reallocate the CasADi symbol as a flattened N-dimensional vector
        self.symbolic_object = ca.MX.sym(self.name, self.n_points)
        self.node_types = np.full(self.n_points, self.type, dtype=object)
        self.initial_condition_array = np.zeros(self.n_points)

    def distributeOn(self, domain):
        """
        Syntactic sugar to delegate discretization to the Master Model.
        
        :param Domain domain: The spatial domain.
        """
        if hasattr(self, "_owner_model_instance") and self._owner_model_instance:
            self._owner_model_instance.distributeVariable(self, domain)
        else:
            raise RuntimeError(f"Variable '{self.name}' is not bound to a Model.")

    def setNodeType(self, indices, new_type):
        """
        Overrides the DAE classification for a specific subset of nodes.
        
        :param list indices: Spatial indices to modify.
        :param str new_type: New exposure type (e.g., 'algebraic').
        """
        self.node_types[indices] = new_type

    def setVectorialInitialCondition(self, value, location=None):
        """
        Applies initial values to specific topological slices.
        
        :param float|ndarray value: Value to apply.
        :param slice|list location: Topological location.
        """
        if location is None or location == "all":
            self.initial_condition_array[:] = value
        else:
            self.initial_condition_array[location] = value

    def __call__(self):
        """
        Evaluates the variable, returning its EquationNode wrapper.
        
        :return: EquationNode containing the CasADi symbol.
        :rtype: EquationNode
        """
        return EquationNode(
            name=self.name, symbolic_object=self.symbolic_object, unit_object=self.units
        )

    def Diff(self, ind_var=None):
        """
        Instantiates the temporal derivative CasADi symbolic vector.
        
        :param Variable ind_var: The independent variable (e.g., time).
        :return: EquationNode representing the temporal derivative.
        :rtype: EquationNode
        """
        dim = self.n_points if self.is_distributed else 1
        sym_dot = ca.MX.sym(self.name + "_dot", dim)
        dummy_diff_node = _Diff(self, ind_var)
        enode = EquationNode(
            name=f"d({self.name})",
            symbolic_object=sym_dot,
            unit_object=dummy_diff_node.unit_object,
        )
        enode.equation_type["is_differential"] = True
        return enode

    def Grad(self, domain=None):
        """
        Delegates spatial gradient to the Domain.
        
        :param Domain domain: Optional domain override.
        :return: EquationNode representing the gradient.
        :rtype: EquationNode
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed.")
        return dom.apply_gradient(self)

    def Div(self, domain=None):
        """
        Delegates spatial Laplacian to the Domain.
        
        :param Domain domain: Optional domain override.
        :return: EquationNode representing the Laplacian.
        :rtype: EquationNode
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed.")
        return dom.apply_laplacian(self)