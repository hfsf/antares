# -*- coding: utf-8 -*-

"""
Connection Module (V5 Native CasADi Architecture).

Defines the generic Connection class for the ANTARES framework.
This class acts as a topological link between two models or ports,
generating equality constraints to represent physical or logical streams.
It leverages the master Model's creation methods to ensure full tensor-safety
across 1D, 2D, and 3D domain couplings, feeding directly into the CasADi graph.

Now features topological overloading for MaterialStreams and EnergyStreams,
ensuring thermodynamic coherence and preventing domain-mixing leaks.
"""

from . import GLOBAL_CFG as cfg
from .error_definitions import AbsentRequiredObjectError, UnexpectedValueError
from .variable import Variable


class Connection:
    """
    Generic Connection class.

    Dynamically inspects ports for auto-connection or explicitly connects
    user-defined variables between two phenomenological models.
    Provides overloaded topological routing specifically for MaterialStream
    and EnergyStream objects to enforce thermodynamic domain isolation.
    """

    def __init__(
        self, name, source_port, sink_port, source_var_name=None, sink_var_name=None
    ):
        """
        Initializes the topological connection and maps the equality expressions.

        :param str name: Unique string identifier for the connection.
        :param Model source_port: The origin object (source model/port).
        :param Model sink_port: The destination object (sink model/port).
        :param str source_var_name: Specific variable name in the source port.
        :param str sink_var_name: Specific variable name in the sink port.
        :raises TypeError: If the user attempts to cross-link Material and Energy streams.
        """
        self.name = name
        self.source_port = source_port
        self.sink_port = sink_port

        # Stores the generated mathematical constraints before applying them
        self._pending_links = []

        # Local imports to prevent circular dependencies in the framework core
        from .stream import EnergyStream, MaterialStream

        # Type reflection for overloaded routing
        is_source_mat = isinstance(source_port, MaterialStream)
        is_sink_mat = isinstance(sink_port, MaterialStream)
        is_source_eng = isinstance(source_port, EnergyStream)
        is_sink_eng = isinstance(sink_port, EnergyStream)

        # 1. Safety Guard: Prevent physical domain leaks
        if (is_source_mat and is_sink_eng) or (is_source_eng and is_sink_mat):
            raise TypeError(
                f"Connection '{self.name}': Cannot connect a MaterialStream to an EnergyStream. "
                "Thermodynamic domains are strictly incompatible."
            )

        # 2. Decision Logic Tree
        if source_var_name and sink_var_name:
            self._generate_specific_equation(source_var_name, sink_var_name)
        elif is_source_mat and is_sink_mat:
            self._generate_material_stream_connection()
        elif is_source_eng and is_sink_eng:
            self._generate_energy_stream_connection()
        else:
            # Fallback for Equipment-to-Equipment or generic logical connections
            self._generate_auto_connection_equations()

    def _generate_material_stream_connection(self):
        """
        Generates equality constraints tailored for MaterialStreams.

        Equates fundamental states (F_molar, T, P) and the composition vector (z).
        Enthalpy (H_molar) is purposely omitted from explicit equality to prevent
        redundant constraints (singular Jacobians).

        CRITICAL ARCHITECTURE NOTE:
        To prevent structurally singular Jacobians (DOF = -1), only N-1 components
        are equated. Both streams independently possess a sum(z) = 1.0 constraint.
        Connecting all N components would result in linear dependence.
        """
        # Equate Intensive and Extensive base variables
        for attr in ["F_molar", "T", "P"]:
            var_source = getattr(self.source_port, attr)
            var_sink = getattr(self.sink_port, attr)
            expr = var_source() - var_sink()
            eq_name = f"{self.name}_{attr}_equality"
            desc = f"Material stream link {attr}: {self.source_port.name} -> {self.sink_port.name}"
            self._pending_links.append((eq_name, desc, expr))

        # Equate Composition arrays (z dictionary) using the N-1 rule
        source_comps = set(self.source_port.components)
        sink_comps = set(self.sink_port.components)

        if source_comps != sink_comps:
            raise ValueError(
                f"Connection '{self.name}': Mismatched chemical components between "
                f"'{self.source_port.name}' ({source_comps}) and "
                f"'{self.sink_port.name}' ({sink_comps})."
            )

        # Slicing [:-1] guarantees exactly N-1 equations are generated
        for comp in self.source_port.components[:-1]:
            var_source = self.source_port.z[comp]
            var_sink = self.sink_port.z[comp]
            expr = var_source() - var_sink()
            eq_name = f"{self.name}_z_{comp}_equality"
            desc = f"Material stream link z_{comp}: {self.source_port.name} -> {self.sink_port.name}"
            self._pending_links.append((eq_name, desc, expr))

    def _generate_energy_stream_connection(self):
        """
        Generates equality constraints tailored for EnergyStreams.
        Matches the thermodynamic energy rates and their respective driving potentials.
        """
        if self.source_port.stream_type != self.sink_port.stream_type:
            raise TypeError(
                f"Connection '{self.name}': Incompatible EnergyStream types. "
                f"Source is '{self.source_port.stream_type}', sink is '{self.sink_port.stream_type}'."
            )

        attrs = (
            ["Q", "T_source"]
            if self.source_port.stream_type == "heat"
            else ["W", "RPM"]
        )

        for attr in attrs:
            var_source = getattr(self.source_port, attr)
            var_sink = getattr(self.sink_port, attr)
            expr = var_source() - var_sink()
            eq_name = f"{self.name}_{attr}_equality"
            desc = f"Energy stream link {attr}: {self.source_port.name} -> {self.sink_port.name}"
            self._pending_links.append((eq_name, desc, expr))

    def _generate_specific_equation(self, source_var_name, sink_var_name):
        """Generates a single explicit equality expression."""
        if not hasattr(self.source_port, source_var_name) or not hasattr(
            self.sink_port, sink_var_name
        ):
            raise AbsentRequiredObjectError(
                "Variables not found for explicit connection."
            )

        var_source = getattr(self.source_port, source_var_name)
        var_sink = getattr(self.sink_port, sink_var_name)

        expr = var_source() - var_sink()
        eq_name = f"{self.name}_{source_var_name}_to_{sink_var_name}"
        desc = f"Explicit link: {self.source_port.name}.{source_var_name} -> {self.sink_port.name}.{sink_var_name}"

        self._pending_links.append((eq_name, desc, expr))

    def _generate_auto_connection_equations(self):
        """Fallback dynamic inspection for Equipment-to-Equipment routing."""
        for attr_name in dir(self.source_port):
            if attr_name.startswith("_"):
                continue

            var_source = getattr(self.source_port, attr_name)
            if isinstance(var_source, Variable) and hasattr(self.sink_port, attr_name):
                var_sink = getattr(self.sink_port, attr_name)
                if isinstance(var_sink, Variable):
                    expr = var_source() - var_sink()
                    eq_name = f"{self.name}_{attr_name}_equality"
                    desc = f"Auto link for {attr_name}: {self.source_port.name} -> {self.sink_port.name}"
                    self._pending_links.append((eq_name, desc, expr))

    def apply_to(self, target_model):
        """Injects the generated equality constraints into a target model."""
        for eq_name, desc, expr in self._pending_links:
            target_model.createEquation(name=eq_name, description=desc, expr=expr)
