# -*- coding: utf-8 -*-

"""
Equation Module (V5 Native CasADi Architecture).

Defines the Equation class for the ANTARES framework.
In V5, this class acts as a lightweight, passive container that stores
the pre-validated CasADi MX expression tree (wrapped in an EquationNode).
It eliminates legacy SymPy overhead, AST sweeping, and string-based substitutions.
"""

import uuid

import antares.core.GLOBAL_CFG as cfg

from .error_definitions import UnexpectedValueError
from .expression_evaluation import EquationNode


def _generate_unique_id():
    """
    Generates a unique short string identifier for unnamed equations.

    :return: A random 8-character hexadecimal string.
    :rtype: str
    """
    return uuid.uuid4().hex[:8]


class Equation:
    """
    Definition of the Equation class.
    Maintains the mathematical expression strictly in residual form (f(x) = 0),
    holding the native CasADi computational graph.
    """

    def __init__(self, name, description="", fast_expr=None, owner_model_name=""):
        """
        Initializes the Equation object.

        :param str name: Name of the equation. If empty, a unique ID is assigned.
        :param str description: Short physical description of the equation.
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

        # Attributes dynamically populated by the Master Model for PDE vectorization
        self.is_distributed = False
        self.flat_indices = None

        if fast_expr is not None:
            self.setResidual(fast_expr)

    def _getTypeFromExpression(self):
        """
        Determines whether the equation is Differential or Algebraic based on
        the pre-calculated flags stored in the underlying EquationNode.
        CasADi DAE solvers route equations based on this classification.
        """
        if self.equation_expression.equation_type.get("is_differential", False):
            self.type = "differential"
        else:
            self.type = "algebraic"

    def setResidual(self, equation_expression):
        """
        Ensures the mathematical expression is stored in the residual
        format (f(x) = 0). It dynamically handles native residual EquationNodes
        or elemental equality tuples, delegating physical unit checks to the nodes.

        :param equation_expression: Can be an EquationNode directly or a
                                    tuple (LHS, RHS) representing equality.
        :raises UnexpectedValueError: If the expression format is incompatible.
        """
        if isinstance(equation_expression, tuple):
            # The equation was passed in elemental format: (LHS, RHS)
            lhs, rhs = equation_expression

            if isinstance(rhs, (float, int, EquationNode)):
                # The overloaded '-' operator in EquationNode safely handles
                # unit coherence checks and native CasADi MX subtractions.
                self.equation_expression = lhs - rhs
            else:
                raise UnexpectedValueError(
                    "Equation RHS must be a float, int, or EquationNode."
                )

        elif isinstance(equation_expression, EquationNode):
            # The equation was naturally passed in residual format
            # (e.g., via the overloaded '==' operator yielding self - other)
            self.equation_expression = equation_expression

        else:
            raise UnexpectedValueError(
                "Expected an EquationNode or tuple(EquationNode, float|int|EquationNode)."
            )

        # Update core mathematical classification
        self._getTypeFromExpression()

        if getattr(cfg, "VERBOSITY_LEVEL", 1) >= 2:
            print(
                f"[DEBUG] Equation '{self.name}' successfully mounted as a {self.type} block."
            )
