---
layout: default
---
[Infer.NET development](index.md)

## Comparison to other PP frameworks

*UNDER CONSTRUCTION*

|                                            | Infer.NET                 | Stan | Edward | PyMC3 | Pyro |
|---------------------------------------------------------------------------------------|
| **Inference**                              | EP, VMP, Gibbs            | NUTS, HMC, ADVI | | | |	
| **Language**                               | C#, F#, Csoft<sup>1</sup> | Stan (Mix of C++ and R) Interfaces with Python and others | Python | | |
| **License**                                | MIT                       | BSD (parts use GPL v3) | Apache v2 | | |		
| **Online inference latency**               | Tiny                      | | | | |				
| **Scalability**                            |                           | | | | |					
| **Discrete random variables**              | Yes                       | No | Algorithm dependent | | |		
| **Continuous random variables**            | Yes                       | Yes | Yes | | |		
| **String random variables**                | Yes                       | No | No | | |		
| **Stochastic gates (if / switch)**         | Yes                       | No | No | | |		
| **Mixture models**                         | Yes                       | Awkward | Yes | | |		
| **Repeat blocks / raising model to power** | Yes                       | No | No | | |		
| **Arrays of arrays (jagged arrays)**       | Yes                       | No | No | | |		
| **Increment log density**                  | Yes                       | Yes | Yes | | |		
| **Turing complete**                        | No                        | No | Yes? | | |		
| **Model evidence computation**             | Yes                       | No |	No | | |		
| **Non-parametric models**                  | GP only                   | No | No | | |		

1. Support for Csoft (Probabilistic C#) is still experimental.
2. Has an extensibility mechanism for adding new user-defined factors, message operators, distributions and constraints.

For a comparison to Bayesian network software, see [https://www.cs.ubc.ca/~murphyk/Software/bnsoft.html](https://www.cs.ubc.ca/~murphyk/Software/bnsoft.html).
