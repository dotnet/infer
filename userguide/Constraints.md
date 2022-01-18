---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Factors and Constraints](Factors and Constraints.md)

## Constraints

This page lists the built-in methods for applying constraints. In these methods, you can often pass in random variables as arguments e.g. `Variable<double>` instead of **double**. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable.Constrain` and passing in the constraint method, as described [on this page](Attaching constraints to variables.md). 

| **Constraint** | **Syntax** | **Description** |
|-----------------------------------------------|
| _True_ | `Variable.ConstrainTrue(bool v)` | Constrains the **bool** random variable _`v`_ to be true. |
| _False_ | `Variable.ConstrainFalse(bool v)` | Constrains the **bool** random variable _`v`_ to be false. |
| _Positivity_ | `Variable.ConstrainPositive(double v)` | Constrains the **double** random variable _`v`_ to be strictly positive (i.e. such that _`v>0`_).
| _Between limits_ | `Variable.ConstrainBetween(double x, double lowerBound, double upperBound)` | Constrains the **double** random variable _`x`_ to be between the specified limits, such that _`lowerBound`_ <= _`x`_ < _`upperBound`._  
| _Equality_ | `Variable.ConstrainEqual<T>(T a, T b)` | Constrains the random variables _`a`_ and _`b`_ to have equal values. The variables must have the same type **T**. |
| _Stochastic equality_ | `Variable.ConstrainEqualRandom<T,TDist>(T a, TDist b)` | Constrains the random variable _`a`_ to be equal to a sample from a distribution _`b`_. Equivalently, [increment the log-density](Increment log density.md) by `b.GetLogProb(a)`. If _a_ has type **T**, then _`b`_ must be a distribution with domain type **T**. |
