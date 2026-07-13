---
layout: default 
--- 

[Infer.NET user guide](index.md)

## Copulas and vines

This page describes an experimental feature that is likely to change in future releases

A **copula** describes the _dependence_ between several variables separately from their individual (marginal) distributions. A **vine** factorises a high-dimensional copula into a hierarchy of simple bivariate copulas, which makes it practical to model rich dependence in many dimensions. Infer.NET additionally implements **GPVINE**, in which the strength of each conditional copula is allowed to _vary with the variables it is conditioned on_, learned with a sparse Gaussian process and Expectation Propagation.

The implementation follows Lopez-Paz, Hernández-Lobato and Ghahramani, _Gaussian Process Vine Copulas for Multivariate Dependence_, ICML 2013.

### When to use this

Use copulas and vines when you want to model the joint distribution of several continuous variables and a multivariate Gaussian is too rigid — for example when the dependence is asymmetric in the tails (variables that crash together but do not boom together), or when the strength of dependence between two variables changes with the value of a third. Typical tasks are:

*   **Estimating a joint density** over continuous variables with flexible dependence.
*   **Discovering dependence structure** — which variables are directly coupled, and how strongly.
*   **Capturing context-dependent dependence** — "the correlation between X and Y changes with Z" — which a correlation matrix cannot express.
*   **Generating** synthetic joint data, **imputing** missing dimensions, or doing posterior-predictive inference given a partial observation.
*   **Comparing** dependence models by held-out likelihood.

A vine recovers statistical dependence, not causation: the trees are a conditional-dependence factorisation, not a causal graph. For causal questions see [Causal inference with Infer.NET](Causal inference with Infer.NET.md).

### Key concepts

**Probability integral transform (PIT).** Copulas operate on _pseudo-observations_: each variable is mapped to a uniform value in (0, 1) by its empirical CDF, using `rank / (n + 1)`. This strips out the marginals and leaves only the dependence. Infer.NET applies the PIT for you, so you pass raw data in its natural units.

**Kendall's tau.** A rank correlation in [-1, 1]. It plays two roles here: its absolute value is the weight used to _select_ the vine structure, and it is the parameter through which every copula family is expressed. Parameterising by tau rather than by each family's native parameter means a single latent function can drive any family.

**Vine trees.** The first tree `T1` connects the raw variables; each deeper tree `Ti` has the previous tree's _edges_ as its nodes and represents _conditional_ copulas. A full vine over `d` variables has `d - 1` trees and `d(d - 1)/2` edges. Truncating to fewer trees treats the omitted dependence as independence, and is the usual accuracy/cost knob.

**SVINE versus GPVINE.** A simplified vine (SVINE) fits each conditional copula with a single constant tau — the "simplifying assumption". GPVINE instead models tau as a function of the conditioning variables, `tau = 2 * Phi(f(z)) - 1`, where `f` is given a sparse Gaussian process prior and inferred by EP. When conditional dependence genuinely varies, GPVINE achieves a higher held-out likelihood; when it does not, the two agree.

### Where the types live

| Namespace | Types | Assembly |
|-----------|-------|----------|
| `Microsoft.ML.Probabilistic.Distributions.Copulas` | `IBivariateCopula`, `GaussianCopula`, `ClaytonCopula`, `GumbelCopula` | Runtime |
| `Microsoft.ML.Probabilistic.Distributions.Copulas.Vine` | `RegularVine`, `VineStructure`, `VineTree`, `VineEdge`, `Pit`, `KendallTau`, `MaxSpanningTree`, `IConditionalCopulaFitter`, `IConditionalCopulaPosterior`, `SparseGPCopulaPosterior` | Runtime |
| `Microsoft.ML.Probabilistic.Factors` | `CopulaFactor`, `BivariateCopulaOp` | Runtime |
| `Microsoft.ML.Probabilistic.Models` | `GaussianProcessCopulaFitter` | Compiler |

Everything except `GaussianProcessCopulaFitter` is pure numeric code with no dependency on the inference compiler. The GP fit needs the inference engine, and therefore lives in the modelling layer.

### Quick start

```csharp
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Distributions.Copulas.Vine;
using Microsoft.ML.Probabilistic.Models;

// Data: rows are observations, columns are variables, in raw units.
double[][] train = ...;   // [n][d]
double[][] test  = ...;   // [m][d]

// (a) SVINE baseline - closed form, no inference engine needed.
double llSvine = new RegularVine().Fit(train).LogLikelihood(test);

// (b) GPVINE - deeper-tree copulas fitted with a sparse GP and EP.
var fitter = new GaussianProcessCopulaFitter();
double llGpvine = new RegularVine().Fit(train, nTrees: 0, fitter).LogLikelihood(test);

// Higher held-out log-likelihood is better. On data whose conditional dependence
// varies with the conditioning variables, llGpvine > llSvine.
```

`nTrees: 0` means "build all `d - 1` trees". Passing a `fitter` is what turns an SVINE into a GPVINE.

### Read next

*   [Bivariate copula families](Bivariate copula families.md) — the `IBivariateCopula` interface and the Gaussian, Clayton and Gumbel families.
*   [Vine copulas](Vine copulas.md) — fitting, scoring, inspecting, sampling and conditional simulation with `RegularVine`.
*   [Gaussian Process vine copulas](Gaussian Process vine copulas.md) — the GPVINE model, the `GaussianProcessCopulaFitter`, and the copula EP factor.

### References

*   D. Lopez-Paz, J. M. Hernández-Lobato and Z. Ghahramani, [Gaussian Process Vine Copulas for Multivariate Dependence](http://proceedings.mlr.press/v28/lopez-paz13.html), ICML 2013.
*   K. Aas, C. Czado, A. Frigessi and H. Bakken, _Pair-copula constructions of multiple dependence_, Insurance: Mathematics and Economics, 2009. (The C-vine simulation algorithm used by `RegularVine.Sample`.)
*   T. Bedford and R. M. Cooke, _Vines - a new graphical model for dependent random variables_, Annals of Statistics, 2002.
