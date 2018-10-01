---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Running inference](Running inference.md) : [Working with different inference algorithms](Working with different inference algorithms.md)

## Gibbs sampling

Gibbs sampling is a special case of the Metropolis-Hastings algorithm. From a given state of the variables it randomly moves to another state, preserving the desired equilibrium distribution. The simplest version is single-site Gibbs sampling, where each variable is visited in turn and sampled from its conditional distribution given the current state of its neighbors. Variables may be visited in any order and some variables may be sampled more than others, as long as each variable gets sampled sufficiently often. 

A more efficient version of Gibbs sampling, especially in the case of deterministic constraints, is the block Gibbs sampler. Here the variables are grouped into acyclic blocks, each block is visited in turn, and the entire block is sampled from its conditional distribution given the state of its neighbors. This procedure is valid even if blocks overlap; variables in multiple blocks will simply be sampled more often. Typically you will not have to specify the blocking for Gibbs sampling as default blocking is automatically based on the deterministic factors and constraints in your graphical model. However, you do have the option of explicitly specifying blocks using the Group method on the [inference engine](inference engine settings.md).

The Gibbs sampling algorithm object has two configurable properties:

*   BurnIn: number of samples to discard at the beginning
*   Thin: reduction factor when constructing sample and conditional lists in order to avoid correlated samples

Gibbs sampling typically requires many more iterations than EP or VMP (each iteration is one sample).

```csharp
// Use Gibbs sampling
GibbsSampling gs = new GibbsSampling();  
gs.BurnIn = 100;  
gs.Thin = 10;  
InferenceEngine ie = new InferenceEngine(gs);  
ie.NumberOfIterations = 2000;
```

When choosing an inference algorithm, you may wish to consider the following properties of Gibbs sampling:

*   Gibbs sampling will eventually converge to the correct solution, but may take a long while to get there.
*   Speed of convergence may be helped by careful initialisation.
*   Gibbs sampling only supports conjugate factors - i.e. marginal posteriors and conditional distributions over any variable in the factor have the same parameteric form as the prior for that variable.
*   Non-conjugate factors require more sophisticated Monte Carlo methods which are not currently supported by Infer.NET.
*   Gibbs sampling allows you to query for (a) posterior marginals, (b) a list of conditional distributions, and/or (c) a list of samples. See QueryType in [Running inference](Running inference.md).
*   Gibbs sampling does not support variables defined within a gate, but does support mixture models which only have factors within a stochastic If, Case, or Switch statement.
*   Gibbs sampling is **stochastic** \- it will give different samples depending on the random seed for different initialisations. You can read about [how to change message initialisation](customising the algorithm initialisation.md)
