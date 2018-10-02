---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Running inference](Running inference.md)

## Working with different inference algorithms

Infer.NET supports the following inference algorithms:

*   [Expectation Propagation](Expectation Propagation.md) (EP), a generalized form of loopy belief propagation. This is the default algorithm used.
*   [Variational Message Passing](Variational Message Passing.md) (VMP), a generalized form of mean field approximation
*   [Gibbs sampling](Gibbs sampling.md) (including block Gibbs sampling)

Note: There is also experimental support for max product belief propagation, but this is supported only for a _very_ limited number of factors.

The algorithm is a property of the **InferenceEngine** object and can be set as follows:

```csharp
// Use Expectation Propagation  
InferenceEngine ie = new InferenceEngine();  
ie.Algorithm = new ExpectationPropagation();
```

```csharp
// Use Variational Message Passing  
InferenceEngine ie = new InferenceEngine();  
ie.Algorithm = new VariationalMessagePassing();
```

#### Comparing the algorithms

This table gives a brief summary of the properties of the different algorithms - for more information see the individual pages on each algorithm.

| | **[Expectation Propagation](Expectation Propagation.md)** | **[Variational Message Passing](Variational Message Passing.md)** | **[Gibbs sampling](Gibbs sampling.md)** |
|-------------------------------------------------------------------------------------------|
| **Deterministic?** Deterministic algorithms will give the same answer for the same problem (given the same initialization). | Yes | Yes | No |
| **Exact result?** Whether the algorithm will give the exact result. | Only in simple cases (if factor graph is discrete or linear-Gaussian and has no loops) | No | Ultimately (but this may involve running the algorithm for an unfeasibly long time) |
| **Guaranteed to converge?** Whether the algorithm is guaranteed to converge to a solution (exact or approximate). | No | Yes, VMP maximises a lower bound on the evidence. | Yes, but convergence may be very difficult to diagnose. |
| **Efficient?** Roughly how efficient the algorithm is. | Reasonably efficient | Often the most efficient | Generally less efficient than its deterministic counterparts. |

Although all three algorithms can handle a large range of factors, there are some factors which are more naturally and tractably handled by one or other of the algorithms. The [list of factors and constraints](list of factors and constraints.md) shows which factors each algorithm supports, and can also help guide your algorithm selection.

In the future, we plan to allow individual variables/factors to be tagged with algorithms, so that different algorithms can be used in different parts of the model. But in the meantime, you need to pick the one algorithm that best suits the variables in your model or divide your model into sections using [shared variables](Sharing variables between models.md); in particular, look see the section on [using shared variables to support hybrid algorithms](Shared variable hybrid.md).

EP and VMP are in fact both part of a more general class of algorithms referred to as Power EP which will be available in a later release, and which increases the class of problems for which there are tractable solutions.
