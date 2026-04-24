# ANTARES

###### (Derived from SLOTH project)

![antares_logo](https://github.com/hfsf/antares/blob/main/docs/antares_logo.png?raw=true)

![version](https://img.shields.io/badge/version-0.1.5a-orange?style=for-the-badge)
![python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)
[![License: GPL](https://img.shields.io/badge/License-GPL-blue.svg?style=for-the-badge)](https://opensource.org/licenses/GPL)

<!-- Placeholder for CI/CD and Docs badges
[![Build Status](https://img.shields.io/travis/com/hfsf/antares/main?style=for-the-badge)](https://travis-ci.com/hfsf/antares)
[![Documentation Status](https://readthedocs.org/projects/antares/badge/?style=for-the-badge&version=latest)](https://antares.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/00000.svg)](https://zenodo.org/badge/latestdoi/00000)
-->

## What is ANTARES?

**ANTARES** is an evolving computational framework designed to solve complex dynamic models and Differential-Algebraic Equations (DAE). Its primary goal is to provide a flexible and highly expressive environment where researchers and engineers can translate phenomenological concepts into high-performance numerical simulations.

Rather than being a rigid, domain-specific simulator, ANTARES acts as a mathematical bridge between high-level, human-readable declarations (via Python/SymPy) and robust, low-level execution engines (like CasADi).

### Core Philosophy

- **Expressive yet Fast:** Combine intuitive, object-oriented modeling syntax with the speed of graph-based optimization and JIT C++ compilation.
- **Adaptable to Complexity:** Built to handle the specific challenges of advanced engineering—such as stiff systems, highly nonlinear kinetics, and complex boundary conditions.
- **Delegate Physical Properties:** Focus purely on mathematical resolution and optimization, seamlessly delegating thermodynamic and physical property calculations to specialized external libraries (e.g., _Cantera_, _Thermo_).

### Key Features (Current)

- **Equation-Oriented Modeling:** Declare variables, domains, and mass/energy balances naturally using Python objects.
- **Flexible Discretization:** Native support for translating PDEs into algebraic or differential systems using the **Method of Lines (MOL)** or **Orthogonal Collocation**.
- **Automatic Differentiation (AD):** Exact Jacobians and Hessians provided natively through the CasADi backend, ensuring solver convergence.
- **Symbolic Transpilation:** Automatically converts Python-based Abstract Syntax Trees (AST) into optimized computational graphs.

### Roadmap & Future Horizons

ANTARES is being architected from the ground up to support advanced methodologies. Upcoming capabilities include:

- **Hybrid Modeling (Physics + ML):** Native integration with Machine Learning frameworks (such as **Keras/TensorFlow**) to solve PDEs and enhance phenomenological models using data-driven approaches.
- **Advanced Optimization Suite:** Built-in wrappers and routines for Dynamic Real-Time Optimization (DRTO), Nonlinear Model Predictive Control (NMPC), and automated parameter estimation.
- **Multidimensional Domains:** Expanding spatial discretization and tensor operations to natively support 2D and 3D modeling scenarios.

## Current Status

⚠️ **Alpha Version (v0.1.5a)**

ANTARES is currently in its core architectural development phase. The codebase is fluid, and the API is subject to change as we lay the groundwork for the future features mentioned above. The project is not yet mature enough for external contributions.

## Contact

For architectural discussions, ideas, or collaborations, please contact: **[hanniel.freitas@ifrn.edu.br](mailto:hanniel.freitas@ifrn.edu.br)**.
