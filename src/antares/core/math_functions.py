# -*- coding: utf-8 -*-

"""
Numerical Math Functions Module (V5 Native CasADi Architecture).

Defines mathematical functions designed to work with the numerical values
of unit-containing objects (e.g., Variable, Parameter, Constant) prior to
or outside of the CasADi symbolic graph evaluation.

Unlike `equation_operators.py` (which builds CasADi C++ AST nodes), these
functions eagerly evaluate the `.value` attribute using NumPy and return
a cloned numerical Quantity object.
"""

import copy

import numpy as np

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError


def _numerical_wrapper(func_name, obj, np_func, ignore_dimensions=False):
    """
    Internal wrapper to execute NumPy mathematical functions on Quantity objects.
    Validates dimensional coherence by ensuring the object is dimensionless.

    :param str func_name: The name of the mathematical operation.
    :param Quantity obj: The physical quantity object to be evaluated.
    :param callable np_func: The corresponding NumPy function to apply.
    :param bool ignore_dimensions: If True, bypasses the dimensional check.
    :return: A cloned object with the updated numerical value.
    :rtype: Quantity
    :raises TypeError: If the object is not dimensionless and checks are enabled.
    :raises UnexpectedValueError: If the object lacks a '.value' attribute.
    """
    # Dimensional coherence check linked to GLOBAL_CFG
    if not ignore_dimensions and getattr(cfg, "DIMENSIONAL_COHERENCE_CHECK", True):
        if hasattr(obj, "units") and hasattr(obj.units, "_is_dimensionless"):
            if not obj.units._is_dimensionless():
                raise TypeError(
                    f"A dimensionless argument was expected for the '{func_name}' function, "
                    f"but received an object with dimensions: {obj.units.dimension}. "
                    f"Set ignore_dimensions=True or disable DIMENSIONAL_COHERENCE_CHECK."
                )

    try:
        # Clone the object to prevent in-place mutation of the original Quantity
        res = copy.copy(obj)
        res.name = f"{func_name}_{obj.name}"
        res.value = float(np_func(obj.value))
        return res
    except AttributeError:
        raise UnexpectedValueError(
            f"Function '{func_name}' expected a Quantity-like object with a '.value' attribute."
        )


def Log(obj, ignore_dimensions=False):
    """
    Computes the natural logarithm (base e) of the numerical value of a Quantity.

    :param Quantity obj: The input Quantity object.
    :param bool ignore_dimensions: Flag to bypass dimensional coherence check.
    :return: Cloned object with the evaluated logarithm.
    :rtype: Quantity
    """
    return _numerical_wrapper("Log", obj, np.log, ignore_dimensions)


def Log10(obj, ignore_dimensions=False):
    """
    Computes the base-10 logarithm of the numerical value of a Quantity.

    :param Quantity obj: The input Quantity object.
    :param bool ignore_dimensions: Flag to bypass dimensional coherence check.
    :return: Cloned object with the evaluated base-10 logarithm.
    :rtype: Quantity
    """
    return _numerical_wrapper("Log10", obj, np.log10, ignore_dimensions)


def Exp(obj, ignore_dimensions=False):
    """
    Computes the exponential (e^x) of the numerical value of a Quantity.

    :param Quantity obj: The input Quantity object.
    :param bool ignore_dimensions: Flag to bypass dimensional coherence check.
    :return: Cloned object with the evaluated exponential.
    :rtype: Quantity
    """
    return _numerical_wrapper("Exp", obj, np.exp, ignore_dimensions)


def Abs(obj, ignore_dimensions=False):
    """
    Computes the absolute value of the numerical value of a Quantity.

    :param Quantity obj: The input Quantity object.
    :param bool ignore_dimensions: Flag to bypass dimensional coherence check.
    :return: Cloned object with the evaluated absolute value.
    :rtype: Quantity
    """
    return _numerical_wrapper("Abs", obj, np.abs, ignore_dimensions)


def Sin(obj, ignore_dimensions=False):
    """
    Computes the sine of the numerical value of a Quantity.

    :param Quantity obj: The input Quantity object.
    :param bool ignore_dimensions: Flag to bypass dimensional coherence check.
    :return: Cloned object with the evaluated sine.
    :rtype: Quantity
    """
    return _numerical_wrapper("Sin", obj, np.sin, ignore_dimensions)


def Cos(obj, ignore_dimensions=False):
    """
    Computes the cosine of the numerical value of a Quantity.

    :param Quantity obj: The input Quantity object.
    :param bool ignore_dimensions: Flag to bypass dimensional coherence check.
    :return: Cloned object with the evaluated cosine.
    :rtype: Quantity
    """
    return _numerical_wrapper("Cos", obj, np.cos, ignore_dimensions)


def Tan(obj, ignore_dimensions=False):
    """
    Computes the tangent of the numerical value of a Quantity.

    :param Quantity obj: The input Quantity object.
    :param bool ignore_dimensions: Flag to bypass dimensional coherence check.
    :return: Cloned object with the evaluated tangent.
    :rtype: Quantity
    """
    return _numerical_wrapper("Tan", obj, np.tan, ignore_dimensions)
