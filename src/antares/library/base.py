# -*- coding: utf-8 -*-

r"""
Base Equipment Module (V5 Native CasADi Architecture).

Provides the abstract base classes for constructing phenomenological
unit operations within the ANTARES framework. It enforces a strict
topological separation between purely algebraic (`SteadyEquipment`)
and differential (`DynamicEquipment`) systems. 

These classes offer standardized port registration methods (inlets, outlets, 
energy streams) to facilitate generic, automated mass and energy balances, 
and ensure seamless coupling with `Connection` objects.
"""

from abc import ABC, abstractmethod
import casadi as ca

from antares.core.model import Model
from antares.library.streams import MaterialStream, EnergyStream


class EquipmentBase(Model, ABC):
    r"""
    Abstract Base Class for all phenomenological equipment models.

    Provides the structural dictionaries to hold registered inlet and
    outlet ports, and standardized methods to attach streams. This ensures
    that any subclass can generically iterate over its boundaries to
    formulate conservation laws.

    :param str name: The unique string identifier for the equipment.
    :param PropertyPackage property_package: The thermodynamic package governing the unit.
    :param str description: Optional descriptive string for documentation purposes.
    """

    def __init__(self, name, property_package, description=""):
        self.pkg = property_package
        
        # Topological Port Registries
        self.inlets = {}
        self.outlets = {}
        self.energy_inlets = {}
        self.energy_outlets = {}
        
        super().__init__(name, description)

    def add_material_inlet(self, port_name):
        r"""
        Creates and registers a MaterialStream as an inlet port.

        :param str port_name: The internal variable name for the port (e.g., 'feed').
        :return: The instantiated MaterialStream object.
        :rtype: MaterialStream
        """
        stream = MaterialStream(port_name, self.pkg)
        self.inlets[port_name] = stream
        self.submodels.append(stream)
        setattr(self, port_name, stream)
        return stream

    def add_material_outlet(self, port_name):
        r"""
        Creates and registers a MaterialStream as an outlet port.

        :param str port_name: The internal variable name for the port (e.g., 'vap_out').
        :return: The instantiated MaterialStream object.
        :rtype: MaterialStream
        """
        stream = MaterialStream(port_name, self.pkg)
        self.outlets[port_name] = stream
        self.submodels.append(stream)
        setattr(self, port_name, stream)
        return stream

    def add_energy_port(self, port_name, stream_type="heat", direction="in"):
        r"""
        Creates and registers an EnergyStream port.

        :param str port_name: The internal variable name for the port.
        :param str stream_type: The nature of the energy ('heat' or 'work').
        :param str direction: Topological flow direction ('in' or 'out').
        :return: The instantiated EnergyStream object.
        :rtype: EnergyStream
        :raises ValueError: If the direction is not 'in' or 'out'.
        """
        stream = EnergyStream(port_name, stream_type)
        if direction == "in":
            self.energy_inlets[port_name] = stream
        elif direction == "out":
            self.energy_outlets[port_name] = stream
        else:
            raise ValueError("Energy port direction must be 'in' or 'out'.")
            
        self.submodels.append(stream)
        setattr(self, port_name, stream)
        return stream


class SteadyEquipment(EquipmentBase):
    r"""
    Base Class for purely algebraic (Steady-State) equipment models.

    Assumes strictly zero accumulation ($\frac{dM}{dt} = 0$, $\frac{dU}{dt} = 0$).
    Provides helper methods to automatically generate robust, zero-DOF 
    algebraic material and energy balances across all registered ports.
    """

    def generate_steady_mass_balances(self):
        r"""
        Generates overarching algebraic component mass balances.
        
        Equation: $\sum (\dot{n}_{in} \cdot z_{in, i}) - \sum (\dot{n}_{out} \cdot z_{out, i}) = 0$
        
        Relies on the overloaded addition operators of `EquationNode` objects 
        to natively construct the Abstract Syntax Tree (AST).
        """
        for comp in self.pkg.components:
            in_mass = 0.0
            for port in self.inlets.values():
                in_mass += port.F_molar() * port.z[comp]()
                
            out_mass = 0.0
            for port in self.outlets.values():
                out_mass += port.F_molar() * port.z[comp]()

            self.createEquation(
                name=f"Steady_Mass_Bal_{comp}",
                description=f"Algebraic mass conservation for component {comp}",
                expr=in_mass - out_mass
            )


class DynamicEquipment(EquipmentBase):
    r"""
    Base Class for dynamic (Holdup-bearing) equipment models.

    Provides structural support for internal control volumes. Introduces
    methods to automatically sweep registered boundaries (inlets/outlets)
    and formulate Ordinary Differential Equations (ODEs) for mass and internal energy.
    """

    def generate_dynamic_mass_balances(self, holdup_mass_dict):
        r"""
        Generates Ordinary Differential Equations (ODEs) for component mass accumulation.

        Equation: $\frac{dM_i}{dt} = \sum (\dot{n}_{in} \cdot z_{in, i}) - \sum (\dot{n}_{out} \cdot z_{out, i})$

        :param dict holdup_mass_dict: Dictionary mapping component names to their respective `differential` mass Variables.
        """
        for comp in self.pkg.components:
            in_mass = 0.0
            for port in self.inlets.values():
                in_mass += port.F_molar() * port.z[comp]()
                
            out_mass = 0.0
            for port in self.outlets.values():
                out_mass += port.F_molar() * port.z[comp]()

            self.createEquation(
                name=f"dM_{comp}_dt",
                description=f"Dynamic mass accumulation for component {comp}",
                expr=holdup_mass_dict[comp].Diff() - (in_mass - out_mass)
            )

    def generate_dynamic_energy_balance(self, holdup_energy_var):
        r"""
        Generates the Ordinary Differential Equation (ODE) for internal energy accumulation.

        Equation: $\frac{dU}{dt} = \sum (\dot{H}_{in}) - \sum (\dot{H}_{out}) + \sum (\dot{Q}_{in} + \dot{W}_{in}) - \sum (\dot{Q}_{out} + \dot{W}_{out})$

        :param Variable holdup_energy_var: The `differential` Variable representing total internal energy ($U$).
        """
        in_enth = 0.0
        for port in self.inlets.values():
            in_enth += port.Energy_Flow()
            
        out_enth = 0.0
        for port in self.outlets.values():
            out_enth += port.Energy_Flow()

        in_energy_ports = 0.0
        for port in self.energy_inlets.values():
            in_energy_ports += port.Q() if port.stream_type == "heat" else port.W()

        out_energy_ports = 0.0
        for port in self.energy_outlets.values():
            out_energy_ports += port.Q() if port.stream_type == "heat" else port.W()

        self.createEquation(
            name="dU_dt",
            description="Dynamic internal energy accumulation",
            expr=holdup_energy_var.Diff() - (in_enth - out_enth + in_energy_ports - out_energy_ports)
        )