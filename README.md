Introduction
============

Quick and dirty SQP experiment with `qpmad` and `CppAD`, unsupported and
probably broken.


Dependencies
============

System requirements:
- C++ compiler with C++11 support
- cmake
- git
- Eigen


git submodule dependencies:
- googletest
- qpmad
- CppAD


Compilation
===========

Using `make`:
- clean
- build
- test


Architecture
============

All code is in headers located in `include` folder and contain the following
classes:

1. `sqp::Parameters` -- parameters of the SQP solver: tolerances, initial trust
   region, etc.

2. abstract objective classes:
    - `sqp::ObjectiveBase` -- interface of an objective.

    - `sqp::ObjectiveAnalyticBase` -- analytic objective, a derived class must
      implement functions for computation of the value of the objective
      function, Jacobian, and Hessian.

    - `sqp::ObjectiveAutoDiffBase` -- a base class which uses `CppAD` to
      automatically compute Jacobians and Hessians, user must implement only
      the objective function itself.

3. `sqp::Solver` -- solver class, which is configured using `sqp::Parameters`
   and provides a function which searches for a minimizer of a given
   `sqp::ObjectiveBase`

See tests in `test` folder for examples.
