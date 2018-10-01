---
layout: default
---
[Factors and Constraints](Factors and Constraints.md)

## Double factors

This page lists the built-in methods and operators for creating random variables of type **double.** For both static methods and operators, you can often pass in random variables as arguments e.g. `Variable<double>` instead of **double**. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable<T>.Factor` and passing in the factor method, as described [on this page](Applying functions and operators to variables.md).

#### Distribution Factors

A distribution factor creates a random variable from a parameterised distribution.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Gaussian_ | `Variable.GaussianFromMeanAndPrecision(double mean, double precision)` | Creates a **double** random variable with a _Gaussian_ distribution of the specified mean and precision (inverse variance). |
| _Gaussian_ | `Variable.GaussianFromMeanAndVariance(double mean, double variance)` | Creates a **double** random variable with a _Gaussian_ distribution of the specified mean and variance. |
| _Gamma_ | `Variable.GammaFromShapeAndScale(double shape, double scale)` | Creates a positive **double** random variable with a _Gamma_ distribution of the specified shape and scale. |
| _Gamma_ | `Variable.GammaFromShapeAndRate(double shape, double rate)` | Creates a positive **double** random variable with a _Gamma_ distribution of the specified shape and rate. |
| _Gamma_ | `Variable.GammaFromMeanAndVariance(double mean, double variance)` | Creates a positive **double** random variable with a _Gamma_ distribution of the specified mean and variance. |
| _Beta_ | `Variable.Beta(double trueCount, double falseCount)` | Creates a **double** random variable with a _Beta_ distribution with the specified counts. |
| _Beta_ | `Variable.BetaFromMeanAndVariance(double mean, double variance)` | Creates a **double** random variable with a _Beta_ distribution of the specified mean and variance. |
| _Truncated Gaussian_ | `Variable.TruncatedGaussian(double mean, double variance, double lowerBound, double upperBound)` | Creates a **double** random variable with a _TruncatedGaussian_ distribution of the specified mean, variance, lower bound, and upper bound. |
| _Truncated Gamma_ | `Variable.TruncatedGamma(double shape, double rate, double lowerBound, double upperBound)` | Creates a **double** random variable with a _TruncatedGamma_ distribution of the specified shape, rate, lower bound, and upper bound. |

#### Arithmetic and Mathematical Operations

Arithmetic operations are supported via operator overloads or static methods.

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _Plus_ | a + b | Creates a **double** random variable equal to the sum of _`a`_ and _`b`._ |
| _Minus_ | a - b | Creates a **double** random variable equal to the difference of _`a`_ and _`b`._ |
| _Negative_ | -a | Creates a **double** random variable equal to the negative of _`a`._ |
| _Times_ | a * b | Creates a **double** random variable equal to the product of _`a`_ and _`b`._ |
| _Divide_ | a / b | Creates a **double** random variable equal to the ratio of _`a`_ to _`b`._ |
| _Power_ | a ^ b | Creates a **double**random variable equal to _`a`_ raised to the power _`b`._ |
| _Sum of elements_ | `Variable.Sum(double[] array)` `Variable.Sum(IList<double> array)` | Creates a **double** random variable equal to the sum of the elements of the array. |
| _Exp_ | `Variable.Exp(double exponent)` | Creates a **double** random variable which takes e to the power of _exponent._ |
| _Log_ | `Variable.Exp(double x)` | Creates a **double** random variable equal to the natural log of _`x`._ |
| _Logistic_ | `Variable.Logistic(double x)` | Creates a **double** random variable equal to `1/(1+exp(-x))`. |
| _Max_ | `Variable.Max(double a, double b)` | Creates a **double** random variable equal to the maximum of _`a`_ and _`b`._ |
| _Min_ | `Variable.Min(double a, double b)` | Creates a **double** random variable equal to the minimum of _`a`_ and _`b`._ |
| _Conditional sum of elements_ | `Variable.SumWhere(bool[] a, Vector b)` | Creates a **double** random variable equal to the sum of the elements of _`b`_ where _`a`_ is true. |
| _Inner product_ | `Variable.InnerProduct(Vector a, Vector b)` `Variable.InnerProduct(double[] a, Vector b)` `Variable.InnerProduct(double[] a, double[] b)` | Creates a **double** random variable equal to the inner product of vectors _`a`_ and _`b`_. |

#### Miscellaneous Operations

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _FunctionEvaluate_ | `Variable.FunctionEvaluate(IFunction func, Vector x)` | Evaluate a random function at a point. Used to construct Gaussian Process models like [this one](Gaussian Process classifier.md). |
| _GetItem_ | `Variable.GetItem(Vector source, int index)` | Extract an element of a random vector. |
| _Double_ | `Variable.Double(int integer)` | Convert an integer to double. |

​
