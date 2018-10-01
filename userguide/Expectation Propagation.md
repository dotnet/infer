---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Running inference](Running inference.md) : [Working with different inference algorithms](Working with different inference algorithms.md)

## Expectation Propagation

Expectation Propagation (EP) is a deterministic approximate inference method. It is a generalization of loopy belief propagation and assumed density filtering. Infer.NET only implements EP with a fully factorised approximating family. This version of EP is essentially an approximation of loopy belief propagation, in which the messages are not arbitrary distributions but are projected onto tractable families (such as Gaussians) by matching expectations. In the case where the BP messages are always tractable (such as linear-Gaussian or discrete networks) then it is precisely loopy belief propagation.

EP has the following properties:

*   Like loopy belief propagation, EP is **not guaranteed to converge** to an answer.
*   EP **does not try to find the mode** of the posterior like VMP or EM. Instead it tries to summarize the entire posterior. If the posterior is complex, with multiple conflicting solutions, this can lead to very broad approximations or non-convergence.
*   EP is zero-avoiding which means it tries to give adequate probability to all plausible outcomes, at the expense of sometimes giving high probability to outcomes that actually have zero probability (this is the opposite behavior of VMP).
*   The fixed point found by EP will **depend on the initialization** of certain messages, along with the message-passing schedule used. You can read about [how to change message initialization](customising the algorithm initialisation.md)
*   EP is **deterministic** \- it will give the same solution for the same initialization and schedule

Further reading on EP:

*   Minka's [Roadmap to research on EP](http://research.microsoft.com/~minka/papers/ep/roadmap.html).
*   The Wikipedia pages on [Expectation Propagation](http://en.wikipedia.org/wiki/Expectation_propagation) and [Belief Propagation](http://en.wikipedia.org/wiki/Belief_propagation).

EP and [VMP](Variational Message Passing.md) are in fact both part of a more general class of algorithms referred to as Power EP which will be available in a later release, and which greatly increases the class of problems for which there are tractable solutions.
