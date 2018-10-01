---
layout: default
---

[Infer.NET user guide](index.md)

## What is Infer.NET?

![Infer.NET logo](logo_with_text.png)

**Infer.NET** is a framework for running Bayesian inference in graphical models. If you don't know what that means, but would like to, take a look at the [Resources and References](Resources and References.md) page. Infer.NET provides the state-of-the-art message-passing algorithms and statistical routines needed to perform inference for a wide variety of applications. Infer.NET differs from existing inference software in a number of ways:

*   **Rich modelling language**  
    Support for univariate and multivariate variables, both continuous and discrete. Models can be constructed from a broad range of factors including arithmetic operations, linear algebra, range and positivity constraints, Boolean operators, Dirichlet-Discrete, Gaussian, and many others. Support for hierarchical mixtures with heterogeneous components.

*   **Multiple inference algorithms**  
    Built-in algorithms include Expectation Propagation, Belief Propagation (a special case of EP), Variational Message Passing and Gibbs sampling.

*   **Designed for large scale inference**  
    In most existing inference programs, inference is performed inside the program - the overhead of running the program slows down the inference. Instead, Infer.NET compiles models into inference source code which can be executed independently with no overhead. It can also be integrated directly into your application. In addition, the source code can be viewed, stepped through, profiled or modified as needed, using standard development tools.
    
*   **User-extendable**  
    Probability distributions, factors, message operations and inference algorithms can all be added by the user. Infer.NET uses a plug-in architecture which makes it open-ended and adaptable. Whilst the built-in libraries support a wide range of models and inference operations, there will always be special cases where a new factor or distribution type or algorithm is needed. In this case, custom code can be written and freely mixed with the built-in functionality, minimising the amount of extra work that is needed.

Take a look at [a simple example](a simple example.md) of using Infer.NET. The examples in this documentation are given in C#, but Infer.NET can also be used from any .NET language.