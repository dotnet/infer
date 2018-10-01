---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Running inference](Running inference.md) : [Working with different inference algorithms](Working with different inference algorithms.md)

## Variational Message Passing

Variational Message Passing (VMP) is a deterministic approximate inference method. It is similar to the [EM algorithm](http://en.wikipedia.org/wiki/Expectation-maximization_algorithm), except that it learns distributions over parameters, rather than maximum likelihood estimates - this makes it suitable for fully Bayesian inference. 
 

VMP is so called because it allows variational inference to be applied to a large class of graphical models, using only local message passing operations. It was developed by John Winn and Christopher Bishop during the former's Ph.D. at the University of Cambridge.

When choosing an inference algorithm, you may wish to consider the following properties of VMP:

*   Because VMP is maximising a lower bound on the model evidence, it is **guaranteed to converge** to some solution.
*   VMP is 'zero forcing' - it will narrow its distributions to avoid placing probability mass in regions where the posterior probability should be low. This means that VMP tends to select a single peak in the posterior and will be **over-confident** about the posterior probability of this peak.
*   The fixed point found by VMP will **depend on the initialization** of certain messages, along with the message-passing schedule used. You can read about [how to change message initialization](customising the algorithm initialisation.md)
*   VMP is **deterministic** \- it will give the same solution for the same initialization and schedule.
*   If the model is symmetric, it is necesary to **break symmetry** when using VMP.

Further reading on VMP:

*   Resources on VMP can be found at [http://www.johnwinn.org/Research/VMP.html](http://www.johnwinn.org/Research/VMP.html).
*   The [Variational Message Passing](http://en.wikipedia.org/wiki/Variational_message_passing) page on Wikipedia.
