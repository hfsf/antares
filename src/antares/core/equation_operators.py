# -*- coding: utf-8 -*-

"""
Equation Operators Module (V5 Native CasADi Architecture).

Defines wrapper functions to safely inject mathematical operators (trigonometric,
transcendental, bounds, and differentials) into the ANTARES mathematical tree.
In V5, these operators directly bind to CasADi MX C++ functions and strictly
enforce dimensional coherence before the computational graph is augmented.
"""

import functools

import casadi as ca

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError
from .expression_evaluation import EquationNode
from .template_units import dimless

# =============================================================================
# TRANSCENDENTAL FUNCTIONS WRAPPER
# =============================================================================


def wrapper(
    own_func,
    obj,
    base_func,
    latex_func_name=None,
    equation_type=None,
    dim_check=True,
):
    """
    Wrapper function to encapsulate CasADi mathematical functions into ANTARES EquationNodes.
    It rigorously checks for dimensional coherence (transcendental functions require
    dimensionless arguments) based on the GLOBAL_CFG settings.

    :param callable own_func: The ANTARES function calling this wrapper (e.g., Log).
    :param obj: The argument being evaluated.
    :type obj: int, float, or EquationNode
    :param callable base_func: The native CasADi MX mathematical function (e.g., ca.log).
    :param str latex_func_name: Optional custom LaTeX string. Defaults to function name.
    :param dict equation_type: Custom equation topological classification mapping.
    :param bool dim_check: Flag indicating if the argument must be dimensionless.
    :return: An EquationNode containing the new CasADi graph branch.
    :rtype: EquationNode
    :raises TypeError: If dimensional coherence fails or an unsupported type is provided.
    """
    if equation_type is None:
        equation_type_ = {
            "is_linear": False,
            "is_nonlinear": True,
            "is_differential": False,
        }
    else:
        equation_type_ = {
            "is_linear": False,
            "is_nonlinear": False,
            "is_differential": False,
        }
        equation_type_.update(equation_type)

    if latex_func_name is None:
        latex_func_name = own_func.__name__.lower()

    def f_name(func_name, obj_name):
        return f"{func_name}({obj_name})"

    # Handle pure numerical inputs
    if isinstance(obj, (float, int)):
        enode_ = EquationNode(
            name=f_name(own_func.__name__, str(obj)),
            symbolic_object=base_func(float(obj)),
            unit_object=dimless,
            latex_text=f"\\{latex_func_name}({obj})",
        )
        enode_.equation_type = equation_type_
        return enode_

    # Handle ANTARES EquationNodes
    elif isinstance(obj, EquationNode):
        if (
            not cfg.DIMENSIONAL_COHERENCE_CHECK
            or obj.unit_object._is_dimensionless()
            or not dim_check
        ):
            enode_ = EquationNode(
                name=f_name(own_func.__name__, obj.name),
                symbolic_object=base_func(obj.symbolic_object),
                unit_object=dimless,
                latex_text=f"\\{latex_func_name}({obj.latex_text})",
            )
            enode_.equation_type = equation_type_
            return enode_
        else:
            raise TypeError(
                f"A dimensionless argument was expected for the '{own_func.__name__}' function, "
                f"but received an object with dimensions: {obj.unit_object.dimension}. "
                f"Disable DIMENSIONAL_COHERENCE_CHECK in GLOBAL_CFG to bypass this physical constraint."
            )
    else:
        raise TypeError(
            f"Unexpected value error. An (int, float, EquationNode) was expected, "
            f"but {type(obj)} was supplied."
        )


def Log(obj):
    """Natural logarithm."""
    return wrapper(Log, obj, ca.log)


def Log10(obj):
    """Base-10 logarithm."""
    return wrapper(Log10, obj, ca.log10)


def Sqrt(obj):
    """Square root."""
    return wrapper(Sqrt, obj, ca.sqrt)


def Abs(obj):
    """Absolute value."""
    return wrapper(Abs, obj, ca.fabs)


def Exp(obj):
    """Exponential function (e^x)."""
    return wrapper(Exp, obj, ca.exp)


def Sin(obj):
    """Sine trigonometric function."""
    return wrapper(Sin, obj, ca.sin)


def Cos(obj):
    """Cosine trigonometric function."""
    return wrapper(Cos, obj, ca.cos)


def Tan(obj):
    """Tangent trigonometric function."""
    return wrapper(Tan, obj, ca.tan)


# =============================================================================
# MIN / MAX OPERATORS (Vector-Safe)
# =============================================================================


def Min(*args):
    """
    Evaluates the minimum across multiple arguments, propagating dimensions.
    Safely utilizes CasADi's element-wise 'fmin' through reduction, making it
    fully compatible with N-dimensional block tensors.

    :param args: Elements to be compared.
    :type args: int, float, EquationNode, Quantity
    :return: An EquationNode containing the minimum threshold logic.
    :rtype: EquationNode
    :raises UnexpectedValueError: If the arguments do not share coherent physical units.
    """
    args_list = list(args)
    names = []
    syms = []
    units = []

    for obj in args_list:
        obj_node = obj.__call__() if hasattr(obj, "__call__") else obj
        if isinstance(obj_node, EquationNode):
            names.append(obj_node.name)
            syms.append(obj_node.symbolic_object)
            units.append(obj_node.unit_object)
        elif isinstance(obj_node, (float, int)):
            names.append(str(obj_node))
            syms.append(ca.MX(float(obj_node)))
            units.append(dimless)

    f_name = f"Min({','.join(names)})"
    latex_text = f"\\min({','.join(names)})"

    # Coherence Check
    target_unit = units[0] if units else dimless
    if cfg.DIMENSIONAL_COHERENCE_CHECK:
        for u in units[1:]:
            if not target_unit._check_dimensional_coherence(u):
                raise UnexpectedValueError(
                    f"A set of objects with coherent dimensions is required for Min()."
                )

    # Native CasADi Element-wise Reduction
    sym_res = functools.reduce(ca.fmin, syms)

    enode_ = EquationNode(
        name=f_name,
        symbolic_object=sym_res,
        unit_object=target_unit,
        latex_text=latex_text,
    )
    enode_.equation_type = {
        "is_linear": False,
        "is_nonlinear": True,
        "is_differential": False,
    }
    return enode_


def Max(*args):
    """
    Evaluates the maximum across multiple arguments, propagating dimensions.
    Safely utilizes CasADi's element-wise 'fmax' through reduction, making it
    fully compatible with N-dimensional block tensors.

    :param args: Elements to be compared.
    :type args: int, float, EquationNode, Quantity
    :return: An EquationNode containing the maximum threshold logic.
    :rtype: EquationNode
    :raises UnexpectedValueError: If the arguments do not share coherent physical units.
    """
    args_list = list(args)
    names = []
    syms = []
    units = []

    for obj in args_list:
        obj_node = obj.__call__() if hasattr(obj, "__call__") else obj
        if isinstance(obj_node, EquationNode):
            names.append(obj_node.name)
            syms.append(obj_node.symbolic_object)
            units.append(obj_node.unit_object)
        elif isinstance(obj_node, (float, int)):
            names.append(str(obj_node))
            syms.append(ca.MX(float(obj_node)))
            units.append(dimless)

    f_name = f"Max({','.join(names)})"
    latex_text = f"\\max({','.join(names)})"

    # Coherence Check
    target_unit = units[0] if units else dimless
    if cfg.DIMENSIONAL_COHERENCE_CHECK:
        for u in units[1:]:
            if not target_unit._check_dimensional_coherence(u):
                raise UnexpectedValueError(
                    f"A set of objects with coherent dimensions is required for Max()."
                )

    # Native CasADi Element-wise Reduction
    sym_res = functools.reduce(ca.fmax, syms)

    enode_ = EquationNode(
        name=f_name,
        symbolic_object=sym_res,
        unit_object=target_unit,
        latex_text=latex_text,
    )
    enode_.equation_type = {
        "is_linear": False,
        "is_nonlinear": True,
        "is_differential": False,
    }
    return enode_


# =============================================================================
# DIFFERENTIAL OPERATOR
# =============================================================================


def _Diff(obj, ind_var_=None):
    """
    Instantiates the temporal derivative token for the ANTARES mathematical tree.
    In V5, this safely allocates the CasADi '_dot' MX vector mapping the exact
    dimension of the distributed physical state.

    :param obj: The state variable to be differentiated.
    :type obj: EquationNode or Variable
    :param ind_var_: The independent variable (e.g., time). Defaults to None.
    :return: An EquationNode acting as the Differential marker.
    :rtype: EquationNode
    """
    from .unit import _s_  # SI base unit for implicit time

    # Guarantee access to the evaluated EquationNode
    obj_ = obj.__call__() if hasattr(obj, "__call__") else obj

    # Ensure valid shape allocation for N-Dimensional PDEs
    sym_shape = (
        obj_.symbolic_object.size1() if hasattr(obj_.symbolic_object, "size1") else 1
    )
    dot_name = f"{obj_.name}_dot"

    # Native CasADi C++ Vector Allocation
    dot_symbol = ca.MX.sym(dot_name, sym_shape)

    # Dimensional coherence tracking (dx/dt)
    if ind_var_ is None:
        unit_object_ = obj_.unit_object / _s_
    else:
        unit_object_ = obj_.unit_object / ind_var_.__call__().unit_object

    enode_ = EquationNode(
        name=dot_name,
        symbolic_object=dot_symbol,
        unit_object=unit_object_,
        latex_text=f"\\frac{{d {obj_.latex_text} }}{{d t}}",
    )

    # Architectural flag vital for the DaeAssembler topological routing
    enode_.equation_type = {
        "is_linear": False,
        "is_nonlinear": False,
        "is_differential": True,
    }

    return enode_
