---
layout: default
---

[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md) 
 
__Under construction__

## Factor analysis example

This tutorial shows how to implement Bayesian Factor Analysis in Infer.NET. Factor Analysis explains a high dimensional data matrix by linearly mapping it into a lower dimensional manifold. We will construct a probabilistic Factor Analysis model and illustrate how Automated Relevance Determination (ARD) can be used to automatically determine the most probable number of latent factors.

### Inference on Toy Data

The data in this example is purely fictitious. In fact the data matrix Y has been drawn from the generative model of a Factor Analysis model.
```
Y = WX + ε;
```
The purpose of the inference is to reconstruct the unknown mixing matrix W and the factor activations X. An important parameter for the inference is the number of factors which corresponds to the dimensionality of the lower dimensional manifold.

### Automatic Relevance Determination

An important component of the model definition is the prior on the mixing matrix `W`:

```
//ARD prior on mixing matrix
//Alpha[K] represents the shared precision for a row K in the mixing matrix
RandomVariableArray<double> alpha = Variable.Array<double>(K).Named("Alpha");
alpha[K]=Variable.GammaFromShapeAndScale(1E-3,1E3).ForEach(K);

// Mixing matrix:
RandomVariableArray2D<double> W = Variable.Array<double>(K,D).Named("W");
W[K, D] = Variable.GaussianFromMeanAndPrecision(0,alpha[K]).ForEach(D);
```
The entries of the RandomVariableArray W are Gaussian with mean 0 and precision Alpha[K], shared for each latent factor K. The prior parameters of the Gamma distribution Alpha[K] are chosen such that the prior expected mean of Alpha[K] is 1E6 which drives the variance of the entries in the mixing matrix to 0 and hence encourages factors to be switched off. In the light of data the posterior distribution of some of the Alpha[K] will decrease hence allowing for the factor to explain variance.

_**See also: ** [Computing model evidence for model selection.](Computing model evidence for model selection.md)_

This mechanism automatically chooses a suitable number of active factors. Alternatively we could have determined the most probable number of factors using the model evidence, rerunning the inference with different numbers of factors.

### Matrix Product

The remaining components of the model definition specify the latent factor activations x as well as the observation noise model tau.
Finally activations and mixing matrix are related to the observed data using the MatrixProduct factor:

```
// Noise
RandomVariableArray<double> tau = Variable.Array<double>(D).Named("tau");            tau[D] = Variable.GammaFromShapeAndScale(0.001, 1.0 / 0.001).ForEach(D);

// Factor activations
RandomVariableArray2D<double> x = Variable.Array<double>(N, K).Named("x");            x[N, K] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(N, K);

// Matrix Product X*W
RandomVariableArray2D<double> XtimesW = Variable.Array<double>(N,D).Named("xTimesW");
XtimesW.SetTo(Factor.MatrixMultiply, x, W);

// Set to data
Y[N, D] = Variable.GaussianFromMeanAndPrecision(XtimesW[N, D], tau[D]);

//Break Symmetry of W and X
W.InitialiseTo(tools.RandomGaussianArray(Nlatent, dimensions));
x.InitialiseTo(tools.RandomGaussianArray(observations, Nlatent));
```

Note in the last two lines InitialiseTo is used to set a user-defined, random initialization for W and x. This step is necessary to break the symmetry of the system. Using the standard uniform initialization W and x would not change.

### Running the Example

After performing inference, the example code first evaluates the reconstruction error of the data (Mean Squared Error), then the posterior mean values of Alpha[K] are printed for each K. From the output it should be apparent that 6 or 7 of the factors are used (low precision) while the remaining ones are switched off (high precision).