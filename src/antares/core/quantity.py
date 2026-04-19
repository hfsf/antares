# -*- coding: utf-8 -*-

"""
Define the Quantity (QTY) class.
Acts as the base class for all unit-containing objects (Variables, Parameters, Constants)
in the ANTARES framework.
"""

import sympy as sp

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import DimensionalCoherenceError, UnexpectedValueError
from .expression_evaluation import EquationNode
from .unit import Unit

# Null dimension dictionary
null_dimension = {
    "m": 0.0,
    "kg": 0.0,
    "s": 0.0,
    "A": 0.0,
    "K": 0.0,
    "mol": 0.0,
    "cd": 0.0,
}


class Quantity:
    """
    Base class for quantities holding a name, description, value, and dimensional units.
    It enables dimensional coherence analysis and overloads mathematical operators
    to provide a natural equation-writing syntax (e.g., a() + b()).
    """

    def __init__(
        self, name, units, description="", value=0.0, latex_text="", owner_model_name=""
    ):
        """
        Instantiate a Quantity object.

        :param str name: Name of the current Quantity.
        :param Unit/str units: Dimensional unit definition (Unit object or string).
        :param str description: Short description for the Quantity. Defaults to "".
        :param float value: Numerical value of the Quantity. Defaults to 0.0.
        :param str latex_text: String representing the Quantity in LaTeX format.
        :param str owner_model_name: Name of the model that owns this object.
        """
        self.name = name

        # If the defined unit is a string, cast a dummy Unit object to process it
        if isinstance(units, str):
            self.units = Unit("", units)
        else:
            self.units = units

        self.description = description
        self.value = float(value)
        self.latex_text = latex_text
        self.is_specified = False
        self.owner_model_name = owner_model_name
        self._owner_model_instance = None

    def setValue(self, quantity_value, quantity_unit=None):
        """
        Sets the numerical value for the Quantity object, with optional dimensional checks.
        """

        def is_dimensionally_coherent(unit1, unit2):
            if not cfg.DIMENSIONAL_COHERENCE_CHECK:
                return True
            return unit1._check_dimensional_coherence(unit2)

        if isinstance(quantity_value, self.__class__):
            if quantity_unit is None and is_dimensionally_coherent(
                quantity_value.units, self.units
            ):
                self.value = float(quantity_value.value)
                self.is_specified = True
            else:
                raise DimensionalCoherenceError(self.units, quantity_value.units)

        elif isinstance(quantity_value, (float, int)) and quantity_unit is None:
            self.value = float(quantity_value)
            self.is_specified = True

        elif isinstance(quantity_value, (float, int)) and quantity_unit is not None:
            if is_dimensionally_coherent(quantity_unit, self.units):
                self.value = float(quantity_value)
                self.is_specified = True
            else:
                raise DimensionalCoherenceError(self.units, quantity_unit)
        else:
            raise UnexpectedValueError("(Quantity, float, int)")

    def __call__(self):
        """
        Overloaded function for calling the Quantity as a function (e.g., self.h()).
        Returns an EquationNode object containing the symbolic representation.

        :return: An EquationNode object corresponding to the current Quantity.
        :rtype EquationNode:
        """
        # If the object is not explicitly specified (e.g., a standard variable)
        if not self.is_specified:
            return EquationNode(
                name=self.name,
                symbolic_object=sp.Symbol(self.name),
                symbolic_map={self.name: self},
                variable_map={self.name: self},
                unit_object=self.units,
                latex_text=self.latex_text,
                repr_symbolic=sp.Symbol(self.name),
            )

        # If the object is specified (e.g., a Parameter with a fixed value)
        else:
            return EquationNode(
                name=self.name,
                symbolic_object=self.value,
                symbolic_map={self.name: self},
                variable_map={},
                unit_object=self.units,
                latex_text=self.latex_text,
                repr_symbolic=sp.Symbol(self.name),
            )

    # =========================================================================
    # OVERLOADED MATHEMATICAL OPERATORS
    # =========================================================================

    def __add__(self, other_obj):
        """
        Overloaded operator for summation (+). Checks dimensional coherence based on GLOBAL_CFG.
        """
        if (
            not cfg.DIMENSIONAL_COHERENCE_CHECK
            or self.units._check_dimensional_coherence(other_obj.units)
        ):
            new_obj = self.__class__(
                name=f"({self.name}_plus_obj)",
                units=self.units,
                value=self.value + other_obj.value,
            )
            return new_obj
        else:
            raise DimensionalCoherenceError(self.units, other_obj.units)

    def __sub__(self, other_obj):
        """
        Overloaded operator for subtraction (-). Checks dimensional coherence based on GLOBAL_CFG.
        """
        if (
            not cfg.DIMENSIONAL_COHERENCE_CHECK
            or self.units._check_dimensional_coherence(other_obj.units)
        ):
            new_obj = self.__class__(
                name=f"({self.name}_minus_obj)",
                units=self.units,
                value=self.value - other_obj.value,
            )
            return new_obj
        else:
            raise DimensionalCoherenceError(self.units, other_obj.units)

    def __mul__(self, other_obj):
        """
        Overloaded operator for multiplication (*). Adjusts resulting units.
        """
        new_obj = self.__class__(
            name=f"({self.name}_mul_obj)",
            units=self.units * other_obj.units,
            value=self.value * other_obj.value,
        )
        return new_obj

    def __truediv__(self, other_obj):
        """
        Overloaded operator for true division (/) in Python 3. Adjusts resulting units.
        """
        new_obj = self.__class__(
            name=f"({self.name}_div_obj)",
            units=self.units / other_obj.units,
            value=self.value / other_obj.value,
        )
        return new_obj

    def __pow__(self, power):
        """
        Overloaded operator for exponentiation (**).
        Accepts both pure numerical types (float/int) and Quantity objects.
        """
        # Safely extract the numerical value of the exponent
        power_val = power.value if hasattr(power, "value") else float(power)

        new_obj = self.__class__(
            name=f"({self.name}_pow)",
            units=self.units**power_val,
            value=self.value**power_val,
        )
        return new_obj
