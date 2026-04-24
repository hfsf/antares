# -*- coding: utf-8 -*-

"""
Parameter Module (V5 Native CasADi Architecture).

Defines the Parameter class for the ANTARES framework.
In the V5 Architecture, Parameters instantiate their own native CasADi MX
symbolic objects. They represent fixed or tunable scalars in the equations
that are isolated into the 'p' vector of the DAE system.
"""

import casadi as ca

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import DimensionalCoherenceError, UnexpectedValueError
from .quantity import Quantity


class Parameter(Quantity):
    """
    Parameter class definition.
    Acts as a physical parameter in the mathematical model, enforcing
    dimensional coherence. Natively holds a CasADi MX symbolic scalar.
    """

    def __init__(
        self,
        name,
        units,
        description="",
        value=0.0,
        latex_text="",
        is_specified=False,
        owner_model_name="",
    ):
        """
        Initializes the Parameter object.

        :param str name: Internal and symbolic name of the parameter.
        :param Unit units: Dimensional unit definition of the current parameter.
        :param str description: Short description for the parameter. Defaults to "".
        :param float value: Numerical value of the parameter. Defaults to 0.0.
        :param str latex_text: String representing the parameter in LaTeX format.
        :param bool is_specified: Flag indicating if the parameter's value has been explicitly set.
        :param str owner_model_name: Name of the model that owns this parameter.
        """
        # The base class Quantity handles the assignment of physical metadata
        super().__init__(name, units, description, value, latex_text, owner_model_name)

        self.is_specified = is_specified

        # V5 NATIVE CASADI ALLOCATION
        # Instantiates the underlying C++ symbolic scalar for the solver
        self.symbolic_object = ca.MX.sym(self.name)

    def setValue(self, quantity_value, quantity_unit=None):
        """
        Sets the numerical value for the Parameter object.
        Performs dimensional checks based on the GLOBAL_CFG settings.

        :param quantity_value: Numerical value or another Quantity object.
        :type quantity_value: float, int, or Quantity
        :param Unit quantity_unit: Optional unit object for the parameter. Defaults to None.
        :raises DimensionalCoherenceError: If units are incompatible and global checks are enabled.
        :raises UnexpectedValueError: If an unsupported type is provided.
        """

        # Helper function to respect the global dimensional check configuration
        def is_dimensionally_coherent(unit1, unit2):
            if not cfg.DIMENSIONAL_COHERENCE_CHECK:
                return True
            return unit1._check_dimensional_coherence(unit2)

        # Case 1: Value is passed as another Parameter/Quantity object
        if isinstance(quantity_value, self.__class__):
            if quantity_unit is None and is_dimensionally_coherent(
                quantity_value.units, self.units
            ):
                self.value = float(quantity_value.value)
                self.is_specified = True
            else:
                raise DimensionalCoherenceError(self.units, quantity_value.units)

        # Case 2: Value is a pure number and no specific unit is provided
        elif isinstance(quantity_value, (float, int)) and quantity_unit is None:
            self.value = float(quantity_value)
            self.is_specified = True

        # Case 3: Value is a number but a specific unit is provided alongside it
        elif isinstance(quantity_value, (float, int)) and quantity_unit is not None:
            if is_dimensionally_coherent(quantity_unit, self.units):
                self.value = float(quantity_value)
                self.is_specified = True
            else:
                raise DimensionalCoherenceError(self.units, quantity_unit)

        # Invalid Input
        else:
            raise UnexpectedValueError("(Quantity, float, int)")
