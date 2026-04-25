# Project ANTARES 

###### (Evolving towards its next iteration)

![antares_logo](https://github.com/hfsf/antares/blob/main/docs/antares_logo.png?raw=true)

![version](https://img.shields.io/badge/version-0.1.0a-orange?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)
[![License: GPL](https://img.shields.io/badge/License-GPL-blue.svg?style=for-the-badge)](https://opensource.org/licenses/GPL)

## What is Project ANTARES?

**Project ANTARES** is an experimental, Python-based computational framework designed for **Equation-Oriented (EO)** modeling and simulation of phenomenological systems. 

Its primary goal is to bridge the gap between human-readable, declarative engineering models and high-performance numerical solvers. By assembling complex Differential-Algebraic Equations (DAEs) into a single monolithic system, ANTARES solves plant topologies and recycle loops simultaneously, bypassing the convergence issues often found in traditional sequential-modular simulators.

### The V5 Native Architecture
In its recent V5 iteration, the framework underwent a structural paradigm shift. Intermediate symbolic translation layers (such as SymPy) were completely removed. ANTARES now operates as a direct assembler for **CasADi**, constructing native C++ computational graphs (`ca.MX`) and sparse matrices from the ground up. This eliminates AST (Abstract Syntax Tree) memory explosions and significantly accelerates the compilation of large-scale systems.

### Core Capabilities

- **Equation-Oriented Solving:** Translates object-oriented topologies into a global DAE matrix, integrated via the robust SUNDIALS suite (IDAS for dynamics, KINSOL for steady-state).
- **Strict Dimensional Guardian:** A rigid Python-layer unit manager that enforces physical coherence during equation formulation. It supports engineering derived units (e.g., `L/min`, `bar`, `cP`) and performs automatic SI-scaling normalization before feeding data to the C++ numerical engine.
- **Abstracted Spatial Discretization (PDEs):** Supports lumped (0D) models and extends to multi-dimensional geometries using the Method of Lines (MoL) and sparse tensor products. Features geometrically-aware curvilinear domains (`RadialDomain`, `SphericalDomain`) that seamlessly handle central singularities (at $r=0$) using L'Hôpital's rule. Support for Orthogonal Collocation Method will de implemented in the future.
- **Declarative OOP Frontend:** Encourages the creation of modular equipment libraries through clear class inheritance (`DeclareVariables`, `DeclareEquations`, `DeclareParameters`), paired with a robust `Connection` framework for flowsheet mapping.

### Roadmap & Future Horizons

The architecture is being laid out to support advanced numerical methodologies in the future:
- **Advanced Optimization Suite:** Parameter estimation, Nonlinear Model Predictive Control (NMPC), and Dynamic Real-Time Optimization (DRTO) using CasADi's exact Jacobians and Hessians.
- **Hybrid Modeling:** Exploring pathways to integrate data-driven approaches (Machine Learning) directly into the phenomenological DAE graphs.
- **Property Delegation:** Continued decoupling of thermodynamic property calculations to specialized external packages.
- **Hybrid Modeling (Physics + ML):** Integration with Machine Learning frameworks to solve PDEs and enhance phenomenological models using data-driven approaches.

- **Thermodynamic Delegation:** Seamlessly delegating property calculations to specialized external libraries (e.g., *Thermo*, *CoolProp*).


## Current Status

⚠️ **Alpha Version (v0.1.0a)**

Project ANTARES is currently in a highly fluid, architectural development phase. While the core mathematical engine is functional, the API is experimental and subject to profound changes without notice. It is primarily a general-purpose research tool and is **not yet mature enough for production environments or external contributions.**

## Contact

For architectural discussions, concepts, or academic inquiries, please contact: **[hanniel.freitas@ifrn.edu.br](mailto:hanniel.freitas@ifrn.edu.br)**.