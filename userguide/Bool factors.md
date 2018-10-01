---
layout: default
---
[Factors and Constraints](Factors and Constraints.md)

## Boolean factors

This page lists the built-in methods and operators for creating random variables of type `bool`. For both static methods and operators, you can often pass in random variables as arguments e.g. `Variable<bool>` instead of `bool`. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable<T>.Factor` and passing in the factor method, as described [on this page](Applying functions and operators to variables.md).

#### Distribution Factors

A distribution factor creates a random variable from a parameterised distribution.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| Bernoulli | `Variable.Bernoulli(double probTrue)` | Creates a **boolean** random variable from the probability of being true P(true). |
| Bernoulli | `Variable.BernoulliFromLogOdds(double logOdds)` | Creates a **boolean** random variable from the log odds i.e. log P(true)/P(false). Equivalent to `Variable.Bernoulli(Variable.Logistic(logOdds))`. |

#### Logical Operations

Logical operations are supported via operator overloads or static methods.

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _And_ | a & b | Creates a **boolean** random variable which is true if both _`a`_ and _`b`_ are true. |
| _Or_ | a `or` b | Creates a **boolean** random variable which is true if either _`a`_ or _`b`_ are true. |
| _Not_ | !a | Creates a **boolean** random variable which is true if _`a`_ is false. |
| _AllTrue_ | `Variable.AllTrue(bool[] array) Variable.AllTrue(IList<bool> array)` | Creates a **boolean** random variable which is true if all the elements of _array_ are true. In other words, this is a N-valued AND. Where the array has length two, & should be used instead. |

#### Comparison Operations

Comparison operations are supported via operator overloads or static methods.

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _Equals_ | a==b | Creates a **boolean** random variable which is true if _`a`_ and _`b`_ are equal. |
| _Not equals_ | a!=b | Creates a **boolean** random variable which is true if _`a`_ and _`b`_ are not equal. |
| _Greater than / less than_ | a>b, a<b, a>=b, a<=b Note: a and b must be both double or both int | Creates a **boolean** random variable which is true if _`a`_ is greater than/less than/greater than or equal to/less than or equal to _`b`_. |
| _IsPositive_ | `Variable.IsPositive(double x)` | Creates a **boolean** random variable which is true if _`x`_ is positive. |
| _IsBetween_ | `Variable.IsBetween(double x, double lowerBound, double upperBound)` | Creates a **boolean** random variable which is true if _lowerBound_ <= x < _upperBound_. |
