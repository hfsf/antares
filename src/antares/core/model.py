# -*- coding: utf-8 -*-

"""
Model Module (V5 Native CasADi Architecture).

Defines the Model class for the ANTARES framework.
This class acts as the central mathematical container, orchestrating equations,
variables, parameters, and constants.
In the V5 Architecture, it acts as a pure declarative interface, delegating all
CasADi C++ graph generation to its underlying objects and topology assignments
to the Domains, completely free of SymPy scalar unrolling or symbolic tracking.
"""

import numpy as np

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
    Base Model class definition.
    Users must inherit from this class and implement the declarative methods
    (DeclareVariables, DeclareEquations, etc.) to construct phenomenological systems.
    """

    def __init__(self, name, description="", submodels=None):
        """
        Initializes the Model container.

        :param str name: Unique identifier for the model (used for namespace scoping).
        :param str description: Physical description of the modeled system.
        :param list submodels: List of child Model instances to be incorporated.
        """
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
        """
        Resolves the configuration of the model and its hierarchical submodels.
        Executes the declarative lifecycle methods.

        :raises UnexpectedObjectDeclarationError: If STRICT_MODE is enabled and the model is empty.
        """
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
        """
        Absorbs objects from a submodel into this master Flowsheet.
        Due to V5 Native CasADi memory referencing, namespace conflicts are
        handled intrinsically without requiring expression string sweeps.

        :param Model child_model: The instantiated submodel to absorb.
        """
        self.variables.update(child_model.variables)
        self.parameters.update(child_model.parameters)
        self.constants.update(child_model.constants)
        self.equations.update(child_model.equations)

    # =========================================================================
    # FACTORY METHODS (V5 NATIVE ALLOCATION)
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
        """
        Factory method to instantiate a Variable and bind it to the model.
        Automatically handles namespace scoping for the CasADi C++ symbol.

        :param str name: Local name of the variable.
        :param Unit units: Dimensional unit object.
        :param str description: Physical description.
        :param float value: Nominal initial value.
        :return: The generated Variable object.
        :rtype: Variable
        """
        scoped_name = f"{name}_{self.name}"
        latex_text = latex_text if latex_text else name

        var = Variable(
            name=scoped_name,
            units=units,
            description=description,
            is_lower_bounded=is_lower_bounded,
            is_upper_bounded=is_upper_bounded,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            value=value,
            is_exposed=is_exposed,
            exposure_type=exposure_type,
            latex_text=latex_text,
            owner_model_name=self.name,
        )

        var._owner_model_instance = self

        self.variables[scoped_name] = var
        setattr(self, name, var)

        if is_exposed:
            self.exposed_vars[exposure_type].append(var)

        return var

    def createParameter(self, name, units, description="", value=0.0, latex_text=""):
        """
        Factory method to instantiate a Parameter and bind it to the model.

        :param str name: Local name of the parameter.
        :param Unit units: Dimensional unit object.
        :return: The generated Parameter object.
        :rtype: Parameter
        """
        scoped_name = f"{name}_{self.name}"
        latex_text = latex_text if latex_text else name

        par = Parameter(
            name=scoped_name,
            units=units,
            description=description,
            value=value,
            latex_text=latex_text,
            owner_model_name=self.name,
        )

        par._owner_model_instance = self
        self.parameters[scoped_name] = par
        setattr(self, name, par)

        return par

    def createConstant(self, name, units, description="", value=0.0, latex_text=""):
        """
        Factory method to instantiate a Constant and bind it to the model.

        :param str name: Local name of the constant.
        :param Unit units: Dimensional unit object.
        :return: The generated Constant object.
        :rtype: Constant
        """
        scoped_name = f"{name}_{self.name}"
        latex_text = latex_text if latex_text else name

        con = Constant(
            name=scoped_name,
            units=units,
            description=description,
            value=value,
            latex_text=latex_text,
            owner_model_name=self.name,
        )

        con._owner_model_instance = self
        self.constants[scoped_name] = con
        setattr(self, name, con)

        return con

    def createEquation(self, name, description="", expr=None):
        """
        Factory method to instantiate an Equation and bind it to the model.

        :param str name: Local name of the equation.
        :param str description: Physical description.
        :param EquationNode|tuple expr: The residual expression or equality tuple.
        :return: The generated Equation object.
        :rtype: Equation
        """
        scoped_name = f"{name}_{self.name}"

        eq = Equation(
            name=scoped_name,
            description=description,
            fast_expr=expr,
            owner_model_name=self.name,
        )

        self.equations[scoped_name] = eq
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
        """
        Factory method to instantiate a 1D Cartesian spatial domain.

        :param str name: Local name of the domain.
        :param Unit unit: Dimensional length unit.
        :param float length: Total physical length.
        :param int n_points: Number of discretization nodes.
        :param str method: Method of discretization. Defaults to 'mol'.
        :param str diff_scheme: Differentiation scheme. Defaults to 'central'.
        :return: The generated Domain1D object.
        :rtype: Domain1D
        """
        from .domain import Domain1D

        domain_obj = Domain1D(
            name, length, n_points, unit, description, method, diff_scheme
        )
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)

        return domain_obj

    def createRadialDomain(
        self,
        name,
        unit,
        description="",
        radius=1.0,
        n_points=10,
        method="mol",
        diff_scheme="central",
        inner_radius=0.0,
    ):
        """
        Factory method to instantiate a 1D Radial domain (Cylindrical coordinates).
        Automatically incorporates 1/r terms into the gradient and laplacian operators.
        
        ANNULAR DOMAINS VS SOLID CYLINDERS:
        If `inner_radius == 0.0` (Solid), the framework applies L'Hôpital's rule at the core 
        to handle the mathematical singularity, enforcing continuous symmetry.
        If `inner_radius > 0.0` (Annular/Hollow), the domain opens up, exposing the inner 
        wall boundary for user-defined Dirichlet or Neumann conditions.

        IMPORTANT NOTE: This creates only the radial symmetric axis. To simulate 
        multi-dimensional engineering problems (e.g., a tubular reactor), couple this 
        domain with a Cartesian Domain1D (longitudinal axis) using the Domain2D tensor product.

        :param str name: Local name of the domain.
        :param Unit unit: Dimensional length unit.
        :param float radius: Total physical outer radius (from inner_radius to R).
        :param int n_points: Number of discretization nodes.
        :param str method: Method of discretization. Defaults to 'mol'.
        :param str diff_scheme: Differentiation scheme. Defaults to 'central'.
        :param float inner_radius: Starting radius for annular configurations. Defaults to 0.0.
        :return: The generated RadialDomain object.
        :rtype: RadialDomain
        """
        from .domain import RadialDomain

        domain_obj = RadialDomain(
            name, radius, n_points, unit, description, method, diff_scheme, inner_radius
        )
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)

        return domain_obj

    def createSphericalDomain(
        self,
        name,
        unit,
        description="",
        radius=1.0,
        n_points=10,
        method="mol",
        diff_scheme="central",
        inner_radius=0.0,
    ):
        """
        Factory method to instantiate a 1D Spherical domain (Spherical coordinates).
        Automatically incorporates 2/r terms into the gradient and laplacian operators.
        
        ANNULAR DOMAINS VS SOLID SPHERES:
        If `inner_radius == 0.0` (Solid), the framework applies L'Hôpital's rule at the core 
        to handle the mathematical singularity, enforcing continuous symmetry.
        If `inner_radius > 0.0` (Hollow), the domain opens up, exposing the inner 
        wall boundary for user-defined Dirichlet or Neumann conditions.

        IMPORTANT NOTE: This creates only the radial symmetric axis. In process engineering, 
        angular symmetries are universally assumed for spherical catalytic particles 
        and droplets, making this 1D symmetry strictly sufficient for most phenomenological cases.

        :param str name: Local name of the domain.
        :param Unit unit: Dimensional length unit.
        :param float radius: Total physical outer radius (from inner_radius to R).
        :param int n_points: Number of discretization nodes.
        :param str method: Method of discretization. Defaults to 'mol'.
        :param str diff_scheme: Differentiation scheme. Defaults to 'central'.
        :param float inner_radius: Starting radius for hollow configurations. Defaults to 0.0.
        :return: The generated SphericalDomain object.
        :rtype: SphericalDomain
        """
        from .domain import SphericalDomain

        domain_obj = SphericalDomain(
            name, radius, n_points, unit, description, method, diff_scheme, inner_radius
        )
        domain_obj._owner_model_instance = self
        setattr(self, name, domain_obj)

        return domain_obj

    def distributeVariable(self, variable, domain):
        """
        Discretizes the variable along the domain as a Unified Block Tensor.
        Automatically re-allocates the underlying CasADi vector.

        :param Variable variable: The variable to be distributed.
        :param Domain domain: The target discretization domain.
        """
        variable.distributeOnDomain(domain)
        variable._owner_model_instance = self
        if variable.name not in self.variables:
            self.variables[variable.name] = variable

    # =========================================================================
    # INITIAL & BOUNDARY CONDITIONS (V5 Vector-Safe)
    # =========================================================================

    def setInitialCondition(self, variable, value, location=None):
        """
        Sets Initial Conditions safely.

        :param Variable variable: Target variable.
        :param float|ndarray value: Value to apply.
        :param tuple|slice location: Optional specific node location.
        """
        if not getattr(variable, "is_distributed", False):
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

        :param Variable variable: Target distributed variable.
        :param Domain domain: The domain defining the geometry.
        :param str boundary_locator: Semantic location (e.g., 'left', 'top').
        :param str bc_type: 'dirichlet' or 'neumann'.
        :param float value: Boundary fixed value or flux.
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

        # The overloaded operators in EquationNode automatically yield residual format
        residual = target_expr - value
        unique_name = f"bc_eq_{variable.name}_{boundary_locator}"

        eq = self.createEquation(
            unique_name, description=f"{bc_type} at {node_suffix}", expr=residual
        )
        eq.flat_indices = unique_flat_idx
        eq.is_distributed = True
        eq.type = "algebraic"

    def addBulkEquation(self, name, expression, domain, description=""):
        """
        Registers a governing PDE directly to the interior bulk using
        topological flat mapping.

        :param str name: Local name of the equation.
        :param EquationNode|tuple expression: The governing PDE expression.
        :param Domain domain: The domain of the variable.
        :param str description: Physical description.
        """
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
    # DECLARATIVE INTERFACES (To be overridden by user)
    # =========================================================================

    def DeclareVariables(self):
        """User implementation block for declaring variables."""
        pass

    def DeclareParameters(self):
        """User implementation block for declaring parameters."""
        pass

    def DeclareConstants(self):
        """User implementation block for declaring constants."""
        pass

    def DeclareEquations(self):
        """User implementation block for declaring equations."""
        pass

    def print_dof_report(self):
        """Outputs an abstract mapping report of the current formulation."""
        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 1:
            print(f"\n--- Model Abstract Report: {self.name} ---")
            print(f"Variables blocks:  {len(self.variables)}")
            print(f"Equations blocks:  {len(self.equations)}")