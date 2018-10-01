---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## The Rats example from BUGS

You can run this example in the [Examples Browser](The examples browser.md). The [rats model](http://www.openbugs.net/Examples/Rats.html) is an example provided by the BUGS software for Gibbs sampling. Please follow the link for a description of the model. The Infer.NET code is a fairly straightforward conversion of the BUGS model. If you are familiar with BUGS, this example illustrates how you can convert a BUGS model into Infer.NET. Using the menu in the Examples Browser, you can run Variational Message Passing on this model and compare the results to Gibbs sampling. There is one subtle issue with inference in this model: because the priors are quite vague, it is important to tell Infer.NET how to initialise the Gibbs sampler. The solution taken here is to initialise with the mean of the prior. By default, Infer.NET initialises by drawing a sample from the prior. If the prior is vague, then this can be a poor initialisation. The results using Gibbs sampling are:

```
alpha0 = Gaussian(106.4, 13.99)[sd=3.74]  
betaC = Gaussian(6.186, 0.01315)[sd=0.1147]  
tauC = Gamma(42.26, 0.0006696)[mean=0.0283]
```

The results using Variational Message Passing are:

```
alpha0 = Gaussian(106.4, 11.04)[sd=3.323]  
betaC = Gaussian(6.186, 0.008808)[sd=0.09385]  
tauC = Gamma(75, 0.0003699)[mean=0.02774]
```

The posterior means are almost identical for both algorithms, but Variational Message Passing underestimates the posterior variance (as expected from the [general properties of that algorithm](Variational Message Passing.md)).​​
