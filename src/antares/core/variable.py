# -*- coding: utf-8 -*-

"""
Variable Module.

This module defines the Variable class for the ANTARES framework, handling 
both lumped (scalar) and distributed (vector) states within the symbolic 
computational graph.
"""

import numpy as np

from .domain import _ast_matmul
from .equation_operators import _Diff
from .quantity import Quantity


class Variable(Quantity):
    """
    Represents a mathematical variable within the ANTARES flowsheet.
    Inherits from Quantity to maintain dimensional and physical consistency.
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
        Initializes a Variable instance.

        :param str name: The unique identifier for the variable.
        :param str units: The physical unit of the variable (e.g., 'K', 'm/s').
        :param str description: A brief description of the variable's physical meaning.
        :param bool is_lower_bounded: Flag indicating if a lower bound exists.
        :param bool is_upper_bounded: Flag indicating if an upper bound exists.
        :param float lower_bound: The lower boundary value for the solver.
        :param float upper_bound: The upper boundary value for the solver.
        :param float value: The initial or nominal value of the variable.
        :param bool is_exposed: Flag indicating if the variable is exposed to external ports.
        :param str exposure_type: Defines the mathematical nature ('algebraic' or 'differential').
        :param str latex_text: LaTeX representation for report generation.
        :param str owner_model_name: The name of the parent model containing this variable.
        :param Domain1D domain: The spatial or temporal domain if distributed.
        """
        # Store the raw string unit to preserve it for discrete node generation
        self._raw_units = units

        super().__init__(name, units, description, value, latex_text, owner_model_name)

        self.is_lower_bounded = lower_bound is not None
        self.is_upper_bounded = upper_bound is not None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_exposed = is_exposed
        self.type = exposure_type

        self.is_distributed = False
        self.domain = domain
        self.discrete_nodes = []

    def distributeOnDomain(self, domain):
        """
        Internal method to generate discrete nodes along a specified domain.
        This does not register the nodes in the solver; it only builds the internal array.

        :param Domain1D domain: The target discretization domain.
        """
        self.domain = domain
        self.is_distributed = True
        self.discrete_nodes = []

        for i in range(domain.n_points):
            node_name = f"{self.name}_{domain.name}_{i}"
            node_var = Variable(
                name=node_name,
                units=self._raw_units,
                description=f"{self.description} at node {i}",
                exposure_type=self.type,
                owner_model_name=self.owner_model_name,
            )
            self.discrete_nodes.append(node_var)

    def distributeOn(self, domain):
        """
        Syntactic sugar to delegate discretization to the Master Model.
        """
        # Verificamos se o vínculo injetado pelo Factory existe
        if hasattr(self, '_owner_model_instance') and self._owner_model_instance is not None:
            self._owner_model_instance.distributeVariable(self, domain)
        else:
            raise RuntimeError(
                f"Variable '{self.name}' is orphaned! It must be created via "
                f"model.createVariable() to be bound to a parent Model instance."
            )
        
    def __call__(self):
        """
        Evaluates the variable, returning its symbolic representation.
        If distributed, returns an array of symbolic nodes.

        :return: A single EquationNode or a NumPy array of EquationNodes.
        :rtype: EquationNode or np.ndarray
        """
        if self.is_distributed:
            return np.array([node() for node in self.discrete_nodes])
        return super().__call__()

    def Diff(self, ind_var=None):
        """
        Applies the temporal derivative operator.

        :param Variable ind_var: The independent variable (usually time). Defaults to None.
        :return: The symbolic representation of the temporal derivative.
        :rtype: EquationNode or np.ndarray
        """
        if self.is_distributed:
            return np.array([node.Diff(ind_var) for node in self.discrete_nodes])
        return _Diff(self, ind_var)

    def Grad(self, domain=None):
        """
        Applies the first spatial derivative operator ($ \\frac{\\partial}{\\partial z} $) 
        using finite differences over the distributed domain.

        :param Domain1D domain: The target domain. Defaults to the variable's primary domain.
        :return: A NumPy array containing the symbolic gradient representation.
        :rtype: np.ndarray
        :raises Exception: If the variable is lumped (not distributed).
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed on a domain.")

        base_unit_obj = self.discrete_nodes[0]().unit_object
        return _ast_matmul(
            dom.A_matrix, self.__call__(), base_unit_obj, dom.unit, deriv_order=1
        )

    def Div(self, domain=None):
        """
        Applies the second spatial derivative (Laplacian) operator ($ \\frac{\\partial^2}{\\partial z^2} $) 
        using finite differences over the distributed domain.

        :param Domain1D domain: The target domain. Defaults to the variable's primary domain.
        :return: A NumPy array containing the symbolic Laplacian representation.
        :rtype: np.ndarray
        :raises Exception: If the variable is lumped (not distributed).
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed on a domain.")

        base_unit_obj = self.discrete_nodes[0]().unit_object
        return _ast_matmul(
            dom.B_matrix, self.__call__(), base_unit_obj, dom.unit, deriv_order=2
        )