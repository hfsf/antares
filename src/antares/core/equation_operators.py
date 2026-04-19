# -*- coding: utf-8 -*-

"""
Define functions for utilization in the equation writing.
All these functions return EquationNode objects and map native SymPy
operations to the ANTARES mathematical tree.
"""

import sympy as sp

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError
from .expression_evaluation import EquationNode
from .template_units import dimless

# =============================================================================
# TRANSCENDENTAL FUNCTIONS
# =============================================================================


def _Log10(sp_obj, evaluate=True):
    """
    Helper function to evaluate base-10 logarithms using SymPy.
    """
    return sp.log(sp_obj, 10, evaluate=evaluate)


def wrapper(
    own_func,
    obj,
    base_func,
    latex_func_name=None,
    equation_type=None,
    dim_check=True,
    ind_var=None,
):
    """
    Wrapper function to encapsulate SymPy mathematical functions into ANTARES EquationNodes.
    It checks for dimensional coherence based on the GLOBAL_CFG settings.
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
        latex_func_name = own_func.__name__

    def f_name(func_name, obj_name):
        return f"{func_name}({obj_name})"

    if isinstance(obj, (float, int)):
        # obj is a pure number
        enode_ = EquationNode(
            name=f_name(own_func.__name__, str(obj)),
            symbolic_object=base_func(obj, evaluate=False),
            symbolic_map={},
            variable_map={},
            unit_object=dimless,
            latex_text=f_name(latex_func_name, str(obj)),
            repr_symbolic=base_func(obj, evaluate=False),
        )
        return enode_

    elif isinstance(obj, EquationNode):
        # Dimensional coherence check linked to GLOBAL_CFG
        if (
            not cfg.DIMENSIONAL_COHERENCE_CHECK
            or obj.unit_object._is_dimensionless()
            or not dim_check
        ):
            # obj is an EquationNode and passes the dimensional check (or check is bypassed)
            enode_ = EquationNode(
                name=f_name(own_func.__name__, obj.name),
                symbolic_object=base_func(obj.symbolic_object, evaluate=False),
                symbolic_map={**obj.symbolic_map},
                variable_map={**obj.variable_map},
                unit_object=obj.unit_object,
                latex_text=f_name(latex_func_name, obj.latex_text),
                repr_symbolic=base_func(obj.repr_symbolic, evaluate=False),
            )
            enode_.equation_type = equation_type_
            return enode_
        else:
            raise TypeError(
                f"A dimensionless argument was expected for the '{own_func.__name__}' function, "
                f"but received an object with dimensions: {obj.unit_object.dimension}. "
                f"Disable DIMENSIONAL_COHERENCE_CHECK in GLOBAL_CFG to bypass this."
            )
    else:
        # Defined directly to avoid circular dependency error while importing expression_evaluation
        raise TypeError(
            "Unexpected value error. An (int, float, EquationNode) was expected, "
            "but a divergent type was supplied."
        )


def Log(obj):
    return wrapper(Log, obj, sp.log)


def Log10(obj):
    return wrapper(Log10, obj, _Log10)


def Sqrt(obj):
    return wrapper(Sqrt, obj, sp.sqrt)


def Abs(obj):
    return wrapper(Abs, obj, sp.Abs)


def Exp(obj):
    return wrapper(Exp, obj, sp.exp)


def Sin(obj):
    return wrapper(Sin, obj, sp.sin)


def Cos(obj):
    return wrapper(Cos, obj, sp.cos)


def Tan(obj):
    return wrapper(Tan, obj, sp.tan)


# =============================================================================
# MIN / MAX OPERATORS
# =============================================================================


def Min(*obj):
    obj = list(obj)
    latex_func_name = "min"
    f_name = f"{latex_func_name}\\right ("

    obj_symb_map = {}
    obj_var_map = {}
    obj_symb_objcts = []

    for obj_i in obj:
        if hasattr(obj_i, "obj_latex_name"):
            obj_latex_name = obj_i.obj_latex_name
        else:
            try:
                obj_latex_name = obj_i.name
            except AttributeError:
                obj_latex_name = str(obj_i)

        if isinstance(obj_i, EquationNode):
            obj_name = obj_i.name
        else:
            obj_name = str(obj_i)
            obj_latex_name = str(obj_i)

        f_name += obj_name

        if hasattr(obj_i, "symbolic_object"):
            obj_symb_objcts.append(obj_i.symbolic_object)
        else:
            obj_symb_objcts.append(obj_i)

        # Gather all the symbolic and variable maps from the object
        try:
            obj_symb_map = {**obj_symb_map, **obj_i.symbolic_map}
            obj_var_map = {**obj_var_map, **obj_i.variable_map}
        except AttributeError:
            pass

    f_name += ")"
    latex_func_name += "\\right )"

    if all(isinstance(obj_i, (float, int)) for obj_i in obj):
        obj_dims = dimless
    elif all(isinstance(obj_i, EquationNode) for obj_i in obj):
        if all(the_unit.unit_object == obj[0].unit_object for the_unit in obj):
            obj_dims = obj[0].unit_object
        else:
            if cfg.DIMENSIONAL_COHERENCE_CHECK:
                raise UnexpectedValueError(
                    "A set of objects with equivalent dimensions is required for Min()."
                )
            else:
                obj_dims = obj[0].unit_object
    else:
        obj_dims_list = [
            obj_i.unit_object for obj_i in obj if hasattr(obj_i, "unit_object")
        ]
        obj_dims = obj_dims_list[0] if obj_dims_list else dimless

    enode_ = EquationNode(
        name=f_name,
        symbolic_object=sp.Min(*obj_symb_objcts, evaluate=False),
        symbolic_map=obj_symb_map,
        variable_map=obj_var_map,
        unit_object=obj_dims,
        latex_text=latex_func_name,
        repr_symbolic=sp.Min(*obj_symb_objcts, evaluate=False),
    )
    return enode_


def Max(*obj):
    obj = list(obj)
    latex_func_name = "max"
    f_name = f"{latex_func_name}\\right ("

    obj_symb_map = {}
    obj_var_map = {}
    obj_symb_objcts = []

    for obj_i in obj:
        if hasattr(obj_i, "obj_latex_name"):
            obj_latex_name = obj_i.obj_latex_name
        else:
            try:
                obj_latex_name = obj_i.name
            except AttributeError:
                obj_latex_name = str(obj_i)

        if isinstance(obj_i, EquationNode):
            obj_name = obj_i.name
        else:
            obj_name = str(obj_i)
            obj_latex_name = str(obj_i)

        f_name += obj_name

        if hasattr(obj_i, "symbolic_object"):
            obj_symb_objcts.append(obj_i.symbolic_object)
        else:
            obj_symb_objcts.append(obj_i)

        # Gather all the symbolic and variable maps from the object
        try:
            obj_symb_map = {**obj_symb_map, **obj_i.symbolic_map}
            obj_var_map = {**obj_var_map, **obj_i.variable_map}
        except AttributeError:
            pass

    f_name += ")"
    latex_func_name += "\\right )"

    if all(isinstance(obj_i, (float, int)) for obj_i in obj):
        obj_dims = dimless
    elif all(isinstance(obj_i, EquationNode) for obj_i in obj):
        if all(the_unit.unit_object == obj[0].unit_object for the_unit in obj):
            obj_dims = obj[0].unit_object
        else:
            if cfg.DIMENSIONAL_COHERENCE_CHECK:
                raise UnexpectedValueError(
                    "A set of objects with equivalent dimensions is required for Max()."
                )
            else:
                obj_dims = obj[0].unit_object
    else:
        obj_dims_list = [
            obj_i.unit_object for obj_i in obj if hasattr(obj_i, "unit_object")
        ]
        obj_dims = obj_dims_list[0] if obj_dims_list else dimless

    enode_ = EquationNode(
        name=f_name,
        symbolic_object=sp.Max(*obj_symb_objcts, evaluate=False),
        symbolic_map=obj_symb_map,
        variable_map=obj_var_map,
        unit_object=obj_dims,
        latex_text=latex_func_name,
        repr_symbolic=sp.Max(*obj_symb_objcts, evaluate=False),
    )
    return enode_


# =============================================================================
# DIFFERENTIAL OPERATOR
# =============================================================================


def _Diff(obj, ind_var_=None):
    """
    No ANTARES, o operador de derivada gera um NOVO nó simbólico (com o sufixo '_dot')
    que atuará como um marcador na árvore de equações.
    """
    import sympy as sp

    from .expression_evaluation import EquationNode
    from .unit import _s_  # <-- Importamos a unidade de tempo padrão do SI

    # Garante que estamos lidando com o nó da variável
    obj_ = obj.__call__() if hasattr(obj, "__call__") else obj

    # 1. Criamos um NOME único para o símbolo da derivada
    dot_name = f"{obj_.name}_dot"

    # 2. Criamos um NOVO símbolo SymPy
    dot_symbol = sp.Symbol(dot_name)

    # 3. Tratamento das unidades (dx/dt)
    if ind_var_ is None:
        # Derivada implícita no tempo (divide pela unidade de segundo)
        unit_object_ = obj_.unit_object / _s_
    else:
        unit_object_ = obj_.unit_object / ind_var_.__call__().unit_object

    # 4. Criamos o nó de equação
    enode_ = EquationNode(
        name=dot_name,
        symbolic_object=dot_symbol,
        symbolic_map={**obj_.symbolic_map, dot_name: obj},
        variable_map={**obj_.variable_map},
        unit_object=unit_object_,
        latex_text=f"\\frac{{d {obj_.latex_text} }}{{d t}}",
        repr_symbolic=dot_symbol,
    )

    # 5. FLAG VITAL PARA O TRANSPILADOR
    enode_.equation_type = {
        "is_linear": False,
        "is_nonlinear": False,
        "is_differential": True,
    }
    enode_.is_derivative_node = True
    enode_.parent_variable_name = obj_.name

    return enode_
