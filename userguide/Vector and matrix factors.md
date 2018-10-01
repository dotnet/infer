---
layout: default
---
[Factors and Constraints](Factors and Constraints.md)

## Vector and matrix factors

This page lists the built-in methods and operators for creating random variables of type **Vector** and **Positive Definite Matrix.** For both static methods and operators, you can often pass in random variables as arguments e.g. **Variable<Vector\>** instead of **Vector**. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using **Variable<T\>.Factor** and passing in the factor method, as described [on this page](Applying functions and operators to variables.md).

#### Distribution Factors

A distribution factor creates a random variable from a parameterised distribution.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Dirichlet_ | `Variable.Dirichlet(double[] u)` `Variable.Dirichlet(Vector v)` | Creates a **Vector** random variable with a Dirichlet distribution with the specified array or vector of pseudo-counts. |
| _Dirichlet_ | `Variable.DirichletSymmetric(int dimension, double pseudocount)` | Creates a **Vector** random variable with a Dirichlet distribution whose pseudo-counts are all set to pseudocount. |
| _Dirichlet_ | `Variable.DirichletUniform(int dimension)` | Creates a **Vector** random variable with a Dirichlet distribution whose pseudo-counts are all set to 1. |
| _Multivariate Gaussian_ | `Variable.VectorGaussianFromMeanAndPrecision(Vector mean, PositiveDefiniteMatrix precision)` | Creates a **Vector** random variable with a multivariate Gaussian distribution with the given mean and precision matrix. |
| _Multivariate Gaussian_ | `Variable.VectorGaussianFromMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)` | Creates a **Vector** random variable with a multivariate Gaussian distribution with the given mean and variance matrix. |
| _Wishart_ | `Variable.WishartFromShapeAndScale(double shape, PositiveDefiniteMatrix scale)` | Creates a **PositiveDefiniteMatrix** random variable with a Wishart distribution with the specified shape parameter and scale matrix. |
| _Wishart_ | `Variable.WishartFromShapeAndRate(double shape, PositiveDefiniteMatrix rate)` | Creates a **PositiveDefiniteMatrix** random variable with a Wishart distribution with the specified shape parameter and rate matrix. |

#### Mathematical operations

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _Softmax_ | `Variable.Softmax(Vector x)` | Creates a **Vector** random variable equal to the [Softmax](http://en.wikipedia.org/wiki/Softmax_activation_function) of another **Vector** variable. |
| _Softmax_ | `Variable.Softmax(IList x)` | Creates a **Vector** random variable equal to the [Softmax](http://en.wikipedia.org/wiki/Softmax_activation_function) of an `IList` variable. |
| _Softmax_ | `Variable.Softmax(ISparseList x)` | Creates a **Vector** random variable equal to the [Softmax](http://en.wikipedia.org/wiki/Softmax_activation_function) of an `ISparseList` variable. |
| _Rotate_ | `Variable.Rotate(double x, double y, double angle)` | Creates a 2-D **Vector** random variable from coordinates x and y, and an angle given in radians. |
| _Matrix times scalar_ | `Variable.MatrixTimesScalar(PositiveDefiniteMatrix a, double b)` | Creates a **PositiveDefiniteMatrix** random variable as the product of another `PositiveDefiniteMatrix` variable a and a scalar variable b. |
| _Matrix times vector_ | `Variable.MatrixTimesScalar(Matrix a, Vector b)` | Creates a **Vector** random variable as the product of a Matrix variable a and a Vector variable b. |
| _Matrix times matrix_ | `Variable.MatrixMultiply(double[,] a, double[,] b) | Creates a **double**[,] random variable as the product of two other double[,] variables. |

#### Element Operations

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _Concatenation_ | `Variable.Concat(Vector first, Vector second)` | Creates a **Vector** random variable equal to the concatenation of two other **Vector** variables. |
| _Vector_ | `Variable.Vector(double[] array)` | Creates a **Vector** random variable from an array. |
| _Subvector_ | `Variable.Subvector(Vector source, int startIndex, int count)` | Creates a **Vector** random variable as a sub-vector of another Vector variable. |
