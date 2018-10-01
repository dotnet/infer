---
layout: default
---
[Factors and Constraints](Factors and Constraints.md)

## Integer and enum factors

This page lists the built-in methods and operators for creating random variables of type **int** and of type **enum**. For both static methods and operators, you can often pass in random variables as arguments e.g. `Variable<int>` instead of int. For compactness, this is not shown in the syntax below.

These methods provide a convenient short alternative to using `Variable<T>.Factor` and passing in the factor method, as described [on this page](Applying functions and operators to variables.md).

#### Distribution Factors

A distribution factor creates a random variable from a parameterised distribution.

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Discrete_ | `Variable.Discrete(double[] probs)` `Variable.Discrete(Vector probs)` | Creates an **int** random variable from supplied array or vector of probabilities (which must add up to 1). The variable can take values between 0 and N-1 where N is the length of the array/vector. |
| _Discrete_ | `Variable.DiscreteFromLogProbs(double[] logProbs)` | Creates an **int** random variable from supplied array or vector of log probabilities. Equivalent to `Variable.Discrete(Variable.Softmax(logProbs))`. |
| _Discrete_ | `Variable.EnumDiscrete<TEnum>(double[] probs)` `Variable.EnumDiscrete<TEnum>(Vector probs)` | Creates an **int** random variable which can take values from 0 to size-1 with equal probability. |
| _Discrete enum_ | `Variable.EnumDiscrete<TEnum>(double[] probs)` `Variable.EnumDiscrete<TEnum>(Vector probs)` | Creates an **enum** random variable where the probability of each enum value is given by the specified array or vector. _TEnum_ specifies the enum type. |
| _Discrete enum_ | `Variable.EnumUniform<TEnum>()` | Creates an **enum** random variable with equal probability of taking each possible value of the enumeration. _TEnum_ specifies the enum type. |
| _Binomial_ | `Variable.Binomial(int trialCount, double probSuccess)` | Creates an **int** random variable which has a Binomial distribution with the specified probability of success per trial and number of trials |
| _Poisson_ | `Variable.Poisson(double mean)` | Creates an **int** random variable which has a Poisson distribution with the specified mean. |

#### Arihmetic Operations

Arithmetic operations are supported via operator overloads or static methods.

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _Plus_ | a + b | Creates an **int** random variable equal to the sum of _`a`_ and _`b`_ |
| _Multiply_ | a * b | Creates an **int**random variable equal to the product of _`a`_ and _`b`_ |
| _CountTrue_ | `Variable.CountTrue(bool[] array)` | Creates an **int** random variable which counts up the number of elements in the array that are true. |

#### Conversion Operations

| **Operation** | **Syntax** | **Description** |
|----------------------------------------------|
| _EnumToInt_ | `Variable.EnumToInt(TEnum v)` | Creates an **int** random variable corresponding to an **enum** random variable. The returned variable can be used as the condition for a _Switch_ or _Case_ block. |

​
