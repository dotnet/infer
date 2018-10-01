---
layout: default 
--- 
[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](The Infer.NET modelling API.md)

## Applying functions and operators to variables

As well as creating random variables directly from prior distributions, you can also derive them from other variables by operations such as addition, multiplication, and comparison. For example, if a and b are random variables, then (a+b) returns a new random variable. All of the operators in C# have been overloaded to provide this functionality. Additionally, the library provides functions on random variables that are useful for statistical modelling, such as taking the inner product between random vectors. These can all be found as static methods of the Variable class. A wide variety of models can be constructed from these primitives. If you need new primitive operations on random variables, you can use the extension mechanism described in [How to add new factors and message operators](How to add a new factor and message operators.md).

#### Deriving a variable using overloaded operators

Here is an example of creating a random variable **y** which is the sum of the variables **a** and **b:**

```csharp
Variable<double> y = a + b;
```

This creates a boolean random variable **c** which indicates if **a** is greater than **b:**

```csharp
Variable<bool> c = a > b;
```

Notice that the results of operators are strongly typed as appropriate (in this case the result is a boolean random variable). Operators and functions can also be chained together to produce complex expressions, such as

```csharp
Variable<double> z = (a + b) * Variable.GaussianFromMeanAndPrecision(mean + x, precision);
```

Note that under the hood this expression creates all the necessary variables: (a+b), (mean+x) and the Gaussian variable with mean (mean+x) and precision "precision". This means that the expression a*(b+c) is not exactly equivalent to ab+ac. While the former makes the variables (b+c) and a*(b+c), the later makes ab, ac and ab+ac.

**! Caution:** **All C# operators are overridden, including both the == and != operators, which return Variable<bool>. This means you cannot use them to compare Variable objects themselves, for example, to see if a Variable object is null. Instead you must first cast the Variable to object and compare that to null. This is illustrated in the following code snippet,**

```csharp
Variable<double> var = null;  
if (var == null) Console.WriteLine("Null variable"); // will not compile  
if (((object)var) == null) Console.WriteLine("Null variable"); // will compile
```

**This design decision was made to ensure consistent behaviour with respect to all operators.**

Under the hood, each overloaded operator is converted into an equivalent call to `Variable<double>`. For example, the definition of **y** above is equivalent to:
```csharp
Variable<double> y = Variable<double>.Factor(Factor.Sum, a, b);
```
All overloaded operators have equivalent calls to static methods on Factor such as **Sum**, **Difference**, **Product**, **Ratio**, **And**, **Or**, and **GreaterThan**. This underlying mechanism for invoking functions is described below.

#### Deriving a variable using a function

Besides overloaded operators, the Variable class also provides static methods to create derived variables. For example:

```csharp
Variable<double> h = Variable.InnerProduct(randomVector1, randomVector2);
```

```csharp
Variable<double> x = Variable.GaussianFromMeanAndPrecision(randomMean, randomPrecision);
```

Like overloaded operators, these calls convert to an equivalent call to `Variable<double>.Factor`. For example, the definition of **h** and **x** above are equivalent to:

```csharp
Variable<double> h = Variable<double>.Factor(Factor.InnerProduct, randomVector1, randomVector2);
```

```csharp
Variable<double> x = Variable<double>.Factor(Factor.Gaussian, randomMean, randomPrecision);
```

The most general way of creating a derived random variable is:

```csharp
Variable<T> var = Variable<T>.Factor(MethodName, arg1, arg2, ...);
```

which creates a new random variable of type **T**, as the result of calling MethodName with the specified arguments (which may be random variables, constants or parameters). The term _factor_ comes from _factor graphs_. The Factor class contains most of the functions built into Infer.NET.

Built-in factor methods include: **And**, **Bernoulli**, **Difference**, **Gaussian**, **InnerProduct**, **IsBetween**, **IsPositive**, **MatrixMultiply**, **Not**, **Or**, **Product**, **Ratio**, **Sum** and **VectorGaussian** (see [the full list of factors](list of factors and constraints.md) for more). You can also [define your own factor methods](How to add a new factor and message operators.md). 

**! Caution:** **Some factors are limited in which arguments can be stochastic/deterministic, i.e. which can be stochastic random variables or deterministic constants/parameters (modelled as observed or constant variables). Most built-in factors support all combinations, but not all of them, depending on the inference algorithm chosen. If you use an invalid combination, you will get an error when the model is compiled for inference. You can see which factors support which argument combinations by looking at the [list of factors and constraints](list of factors and constraints.md).**
