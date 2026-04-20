# -*- coding: utf-8 -*-

"""
Variable Module.

This module defines the Variable class for the ANTARES framework, handling 
both lumped (scalar) and distributed (vector/tensor) states within the symbolic 
computational graph.
"""

import numpy as np

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
        self._raw_units = units
        super().__init__(name, units, description, value, latex_text, owner_model_name)

        self.is_lower_bounded = lower_bound is not None
        self.is_upper_bounded = upper_bound is not None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_exposed = is_exposed
        
        # O atributo interno obrigatório para o Transpilador CasADi
        self.type = exposure_type

        self.is_distributed = False
        self.domain = None
        self.discrete_nodes = []

    def distributeOnDomain(self, domain):
        """
        Internal method to generate discrete nodes along a specified domain.
        Creates N-Dimensional numpy arrays of discrete Variable nodes to support 
        1D, 2D, or 3D spatial representations.

        :param Domain domain: The target discretization domain (Domain1D, Domain2D, etc.).
        """
        self.domain = domain
        self.is_distributed = True
        
        # Discover the shape of the domain (e.g., (50,) for 1D or (40, 40) for 2D)
        shape = getattr(domain, "shape", (domain.n_points,))
        
        # Initialize an empty object array of the correct dimensions
        nodes_array = np.empty(shape, dtype=object)
        
        # Safe N-Dimensional iteration using ndindex (avoids nditer reference issues)
        for idx in np.ndindex(*shape):
            idx_str = "_".join(map(str, idx))
            node_name = f"{self.name}_{domain.name}_{idx_str}"
            
            node_var = Variable(
                name=node_name,
                units=self._raw_units,
                description=f"{self.description} at idx {idx}",
                exposure_type=self.type, 
                owner_model_name=self.owner_model_name,
            )
            nodes_array[idx] = node_var
            
        self.discrete_nodes = nodes_array
        
    def distributeOn(self, domain):
        """Syntactic sugar to delegate discretization to the Master Model."""
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
        If distributed, returns an N-Dimensional array of symbolic nodes.
        """
        if self.is_distributed:
            # Vectorize the evaluation to work on N-Dimensional arrays natively
            func = np.vectorize(lambda n: n())
            return func(self.discrete_nodes)
        return super().__call__()

    def Diff(self, ind_var=None):
        """Applies the temporal derivative operator."""
        if self.is_distributed:
            func = np.vectorize(lambda n: n.Diff(ind_var))
            return func(self.discrete_nodes)
        return _Diff(self, ind_var)

    def Grad(self, domain=None):
        """
        Delegates the computation of the spatial gradient to the attached Domain.
        This enables agnostic 1D, 2D, or 3D differential operators.
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed on a domain.")
        return dom.apply_gradient(self)

    def Div(self, domain=None):
        """
        Delegates the computation of the spatial Laplacian to the attached Domain.
        """
        dom = domain if domain else self.domain
        if not self.is_distributed:
            raise Exception(f"Variable '{self.name}' is not distributed on a domain.")
        return dom.apply_laplacian(self)