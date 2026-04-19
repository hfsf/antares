# -*- coding: utf-8 -*-

"""
Define the Equation class for the ANTARES framework.
Acts as a passive container that stores the symbolic expression tree (SymPy)
for subsequent transpilation to the CasADi backend.
"""

import uuid

import sympy as sp

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError
from .expression_evaluation import EquationNode


def _generate_unique_id():
    """
    Generates a unique short string identifier for unnamed equations or constants.
    """
    return uuid.uuid4().hex[:8]


class Equation:
    """
    Definition of the Equation class.
    Maintains the mathematical expression in residual form (f(x) = 0).
    """

    def __init__(self, name, description="", fast_expr=None, owner_model_name=""):
        """
        Initializes the Equation object.

        :param str name: Name of the equation. If empty, a unique ID is assigned.
        :param str description: Short description of the equation.
        :param fast_expr: The mathematical expression (EquationNode or Tuple).
        :param str owner_model_name: The name of the model that owns this equation.
        """
        self.name = name if name != "" else f"eq_{_generate_unique_id()}"
        self.description = description
        self.owner_model_name = owner_model_name

        # Antares strictly requires the residual format: (LHS - RHS = 0)
        self.equation_expression = None

        # Mathematical classification: 'algebraic' or 'differential'
        self.type = None

        # Dictionary storing all symbolic objects present in this equation
        self.objects_declared = {}

        if fast_expr is not None:
            self.setResidual(fast_expr)

    def _getTypeFromExpression(self):
        """
        Determines whether the equation is Differential or Algebraic based on
        the flags stored in the underlying expression tree.
        CasADi DAE solvers do not require linear/nonlinear distinctions.
        """
        if self.equation_expression.equation_type.get("is_differential", False):
            self.type = "differential"
        else:
            self.type = "algebraic"

    def _sweepObjects(self):
        """
        Maps all symbolic objects declared in the equation.
        This is an essential mapping step for the Transpiler.
        """
        self.objects_declared = {**self.equation_expression.symbolic_map}

    def setResidual(self, equation_expression):
        """
        Ensures the mathematical expression is stored in the residual
        format (f(x) = 0), resolving tuples and constant integers/floats.

        :param equation_expression: Can be an EquationNode directly or a
                                    tuple (LHS, RHS) representing equality.
        """
        if isinstance(equation_expression, tuple):
            # The equation was passed in elemental format: (LHS, RHS)
            lhs, rhs = equation_expression

            if isinstance(rhs, (float, int)):
                # If RHS is a pure number, convert it to an EquationNode
                # It inherently inherits the LHS unit for dimensional safety
                rhs_node = EquationNode(
                    name=f"constant_{_generate_unique_id()}",
                    symbolic_object=rhs,
                    unit_object=lhs.unit_object,
                    repr_symbolic=rhs,
                )
                self.equation_expression = lhs - rhs_node

            elif isinstance(rhs, EquationNode):
                # If both are nodes, subtract to form the residual.
                # The overloaded '-' operator handles dimensional coherence checks.
                self.equation_expression = lhs - rhs
            else:
                raise UnexpectedValueError("RHS must be a float, int, or EquationNode.")

        elif isinstance(equation_expression, EquationNode):
            # The equation was already passed in residual format
            self.equation_expression = equation_expression
        else:
            raise UnexpectedValueError(
                "Expected an EquationNode or tuple(EquationNode, float|int|EquationNode)."
            )

        # Update metadata
        self._sweepObjects()
        self._getTypeFromExpression()

        if cfg.VERBOSITY_LEVEL >= 2:
            print(
                f"[DEBUG] Equation '{self.name}' successfully parsed as a {self.type} equation."
            )

    def _convertEquationSymbolicExpression(self, names_map, whole_obj_map):
        """
        Translates internal SymPy symbols during model composition/incorporation.
        This renames variables to prevent namespace collisions when multiple
        sub-models are flattened into a Master Flowsheet.

        :param dict names_map: Mapping of {old_symbol: new_symbol}.
        :param dict whole_obj_map: Mapping of {symbol_name: Quantity_object}.
        """
        self.equation_expression.repr_symbolic = (
            self.equation_expression.repr_symbolic.subs(names_map)
        )
        self.equation_expression.symbolic_object = (
            self.equation_expression.symbolic_object.subs(names_map)
        )

        # Re-map the active symbols based on the new expression tree
        symbols_used = [
            str(i) for i in list(self.equation_expression.repr_symbolic.free_symbols)
        ]
        new_symbolic_map = {
            k: whole_obj_map[k] for k in symbols_used if k in whole_obj_map
        }

        self.equation_expression.symbolic_map = new_symbolic_map
        self._sweepObjects()
