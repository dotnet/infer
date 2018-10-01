---
layout: default
---
[Factors and Constraints](Factors and Constraints.md)

## Generic factors

This page lists the built-in methods for creating random variables of different types that depend on the type of the input variable; such factors refer to a generic C# type **T**. 
 

| **Distribution** | **Syntax** | **Description** |
|-------------------------------------------------|
| _Copy_ | `Variable.Copy<T>(Variable<T> x)` | Creates a random variable of type **T** that is a copy of the input variable. This is equivalent to creating a new variable and constraining it to be equal to the original variable. |
| _SequentialCopy_ | `Variable.SequentialCopy<T>(Variable<T> x, out Variable<T> second)` | Creates two copies of a random variable of type **T**. The copies will be updated in order during inference. Meant to be used as an inference hint. |
| _Cut_ | `Variable.Cut<T>(Variable<T> x)` | Same as the copy factor, but during inference, backward messages are cut off. |
| _Random_ | `Variable.Random<T, TDist>(TDist prior)` `Variable.Random<T, TDist>(Variable<TDist> prior)` | Creates a random variable of type T defined in terms of its prior distribution of type TDist. This is useful when you want to specify your prior at run-time, for example when doing online learning where you want the posteriors from one step to be the priors for the next step. All the distribution factors documented in the various factor sections could be rewritten in terms of the _Random_ factor. | 

​
