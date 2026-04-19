# -*- coding: utf-8 -*-

"""
Define the Parameter class for the ANTARES framework.
"""

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import DimensionalCoherenceError, UnexpectedValueError
from .quantity import Quantity


class Parameter(Quantity):
    """
    Parameter class definition, holding capabilities for:
    - Parameter definition, including its units for dimensional coherence analysis.
    - Parameter operations using overloaded mathematical operators, enabling an
      almost-writing-syntax (e.g., a() + b()).
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

        :param str name: Name of the current parameter.
        :param Unit units: Dimensional unit definition of the current parameter.
        :param str description: Short description for the parameter. Defaults to "".
        :param float value: Numerical value of the parameter. Defaults to 0.0.
        :param str latex_text: String representing the parameter in LaTeX format.
        :param bool is_specified: Flag indicating if the parameter's value has been explicitly set.
        :param str owner_model_name: Name of the model that owns this parameter.
        """
        # The base class Quantity handles the assignment of name, units, description, value, etc.
        super().__init__(name, units, description, value, latex_text, owner_model_name)

        self.is_specified = is_specified

    def setValue(self, quantity_value, quantity_unit=None):
        """
        Method for value specification of the Parameter object.
        Performs dimensional checks based on the GLOBAL_CFG settings.

        :param [float, int, Quantity] quantity_value: Numerical value or another Quantity object.
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
