---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Factors and Constraints](Factors and Constraints.md)

## Undirected factors

All factors listed on this page are **experimental** and currently are only supported when using the MaxProductBeliefPropagation algorithm. Support for undirected models will be extended in future releases of Infer.NET.

This page lists the built-in methods for creating common undirected factors, as used in many undirected graphical models. Strictly speaking these are stochastic constraints, but we follow convention and refer to them as undirected factors. In these methods, you can usually pass in random variables as arguments e.g. `Variable<double>` instead of `double`. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable.Constrain` and passing in the undirected factor method, as described [on this page](Attaching constraints to variables.md). 

| **Factor** | **Syntax** | **Description** |
|-------------------------------------------|
| _Potts (bool)_ | `Variable.Potts(bool a, bool b, double logCost)` | Adds a Potts factor between two **bool** random variables which evaluates to 1 if a==b and _exp(-logCost)_ otherwise. |
| _Potts (int)_ | `Variable.Potts(int a, int b, double logCost)` | Adds a Potts factor between two **int** random variables which evaluates to 1 if a==b and _exp(-logCost)_ otherwise. |
| _Linear (int)_ | `Variable.Linear(int a, int b, double logUnitCost)` | Adds a linear factor between two **int** random variables which evaluates to _exp(-abs(a-b)*logUnitCost)_. |
| _Truncated linear (int)_ | `Variable.LinearTrunc(int a, int b, double logUnitCost, double maxCost)` | Adds a truncated linear factor between two **int** random variables which evaluates to _exp(-min(abs(a-b)*logUnitCost, maxCost))_. |
