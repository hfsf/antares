# -*- coding: utf-8 -*-

"""
Model Module.

Defines the Model class for the ANTARES framework.
This class acts as a mathematical container, storing equations, variables,
parameters, and constants.
V4 UPDATE: Eliminates Scalar Unrolling. Uses topological slicing to store
Bulk and Boundary PDEs as unified block vectors.
"""

import numpy as np
import sympy as sp

from . import GLOBAL_CFG as cfg
from .constant import Constant
from .equation import Equation
from .error_definitions import UnexpectedObjectDeclarationError
from .parameter import Parameter
from .print_headings import print_heading
from .variable import Variable

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable


class Model:
    """
    Base Model class definition. Users should inherit from this class and
    implement the declarative methods to construct phenomenological systems.
    """

    def __init__(self, name, description="", submodels=None):
        print_heading()
        self.name = name
        self.description = description
        self.submodels = submodels if submodels is not None else []

        self.variables = {}
        self.parameters = {}
        self.constants = {}
        self.equations = {}

        self.exposed_vars = {"input": [], "output": []}

    def __call__(self):
        """Resolves the configuration of the model and its hierarchical submodels."""
        self.DeclareConstants()
        self.DeclareParameters()
        self.DeclareVariables()
        self.DeclareEquations()

        for child in self.submodels:
            self.incorporateFromExternalModel(child)

        if len(self.variables) == 0 and getattr(cfg, "STRICT_MODE", False):
            raise UnexpectedObjectDeclarationError(
                f"No variables declared in '{self.name}'."
            )
        if len(self.equations) == 0 and getattr(cfg, "STRICT_MODE", False):
            raise UnexpectedObjectDeclarationError(
                f"No equations declared in '{self.name}'."
            )

    def incorporateFromExternalModel(self, child_model):
        """Absorbs objects from a submodel into this master Flowsheet."""
        self.variables.update(child_model.variables)
        self.parameters.update(child_model.parameters)
        self.constants.update(child_model.constants)
        self.equations.update(child_model.equations)

    # =========================================================================
    # FACTORY METHODS (V4 SYNC FIX)
    # =========================================================================

    def createVariable(
        self,
        name,
        units,
        description="",
        is_lower_bounded=False,
        is_upper_bounded=False,
        lower_bound=None,
        upper_bound=None,
        is_exposed=False,
        exposure_type="",
        latex_text="",
        value=0.0,
    ):
        if not latex_text:
            latex_text = name
        var = Variable(
            name,
            units,
            description,
            is_lower_bounded,
            is_upper_bounded,
            lower_bound,
            upper_bound,
            value,
            is_exposed,
            exposure_type,
            latex_text,
            owner_model_name=self.name,
        )

        var.name = f"{var.name}_{self.name}"
        var.symbolic_object = sp.Symbol(var.name)  # Sincroniza o Símbolo SymPy
        var._owner_model_instance = self

        self.variables[var.name] = var
        setattr(self, name, var)
        if is_exposed:
            self.exposed_vars[exposure_type].append(var)
        return var

    def createParameter(self, name, units, description="", value=0.0, latex_text=""):
        if not latex_text:
            latex_text = name
        par = Parameter(
            name, units, description, value, latex_text, owner_model_name=self.name
        )
        par.name = f"{par.name}_{self.name}"
        par.symbolic_object = sp.Symbol(par.name)  # Sincroniza o Símbolo SymPy
        par._owner_model_instance = self
        self.parameters[par.name] = par
        setattr(self, name, par)
        return par

    def createConstant(self, name, units, description="", value=0.0, latex_text=""):
        if not latex_text:
            latex_text = name
        con = Constant(
            name, units, description, value, latex_text, owner_model_name=self.name
        )
        con.name = f"{con.name}_{self.name}"
        con.symbolic_object = sp.Symbol(con.name)  # Sincroniza o Símbolo SymPy
        con._owner_model_instance = self
        self.constants[con.name] = con
        setattr(self, name, con)
        return con

    def createEquation(self, name, description="", expr=None):
        eq = Equation(name, description, expr, owner_model_name=self.name)
        eq.name = f"{eq.name}_{self.name}"
        self.equations[eq.name] = eq
        setattr(self, name, eq)
        return eq

    def createDomain(
        self,
        name,
        unit,
        description="",
        length=1.0,
        n_points=10,
        method="mol",
        diff_scheme="central",
    ):
        from .domain import Domain1D

        domain_obj = Domain1D(
            name, length, n_points, unit, description, method, diff_scheme
        )
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)
        return domain_obj

    def distributeVariable(self, variable, domain):
        """Discretizes the variable along the domain as a Unified Block Tensor."""
        variable.distributeOnDomain(domain)
        variable._owner_model_instance = self
        if variable.name not in self.variables:
            self.variables[variable.name] = variable

    # =========================================================================
    # INITIAL & BOUNDARY CONDITIONS (V4 Vector-Safe)
    # =========================================================================

    def setInitialCondition(self, variable, value, location=None):
        """Sets Initial Conditions using vectorized numpy assignments."""
        if not variable.is_distributed:
            variable.setValue(value)
        else:
            variable.setVectorialInitialCondition(value, location)

    def setBoundaryCondition(
        self, variable, domain, boundary_locator, bc_type, value=0.0
    ):
        """
        Generates Boundary Conditions safely mapping the flat topological slices
        directly into the Unified Equation Block. Includes a Node-Claiming system
        to prevent edge/corner overlapping in 2D/3D domains.
        """
        slice_idx, node_suffix = domain.get_boundary(boundary_locator)
        flat_idx = domain.get_mesh_indices()[slice_idx].flatten().tolist()

        if not hasattr(variable, "_assigned_boundary_indices"):
            variable._assigned_boundary_indices = set()

        unique_flat_idx = [
            i for i in flat_idx if i not in variable._assigned_boundary_indices
        ]
        if not unique_flat_idx:
            return

        variable._assigned_boundary_indices.update(unique_flat_idx)
        variable.setNodeType(unique_flat_idx, "algebraic")

        bc_lower = str(bc_type).lower()
        if bc_lower == "dirichlet":
            target_expr = variable()
        elif bc_lower == "neumann":
            target_expr = domain.get_normal_gradient(variable, boundary_locator)
        else:
            raise ValueError(f"Unsupported BC type '{bc_type}'.")

        residual = target_expr - value
        unique_name = f"bc_eq_{variable.name}_{boundary_locator}"

        eq = self.createEquation(
            unique_name, description=f"{bc_type} at {node_suffix}", expr=residual
        )
        eq.flat_indices = unique_flat_idx
        eq.is_distributed = True
        eq.type = "algebraic"

    def addBulkEquation(self, name, expression, domain, description=""):
        """Registers a governing ODE directly to the interior bulk using topological flat mapping."""
        bulk_slice = (
            domain.get_bulk_slice()
            if hasattr(domain, "get_bulk_slice")
            else slice(1, -1)
        )
        flat_idx = domain.get_mesh_indices()[bulk_slice].flatten().tolist()

        eq = self.createEquation(name, description=description, expr=expression)
        eq.flat_indices = flat_idx
        eq.is_distributed = True
        eq.type = "differential"

    # =========================================================================
    # DECLARATIVE INTERFACES
    # =========================================================================
    def DeclareVariables(self):
        pass

    def DeclareParameters(self):
        pass

    def DeclareConstants(self):
        pass

    def DeclareEquations(self):
        pass

    def print_dof_report(self):
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"\n--- Model Abstract Report: {self.name} ---")
            print(f"Variables blocks:  {len(self.variables)}")
            print(f"Equations blocks:  {len(self.equations)}")
