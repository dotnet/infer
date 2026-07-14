---
layout: default 
--- 

[Infer.NET user guide](index.md) : [Copulas and vines](Copulas and vines.md)

## Vine copulas

This page describes an experimental feature that is likely to change in future releases

A **regular vine** (R-vine) factorises a `d`-dimensional copula density into a product of bivariate copulas arranged in a nested sequence of trees. The first tree `T1` connects the raw variables; each deeper tree `Ti` has the edges of `T(i-1)` as its nodes, and joins two of them only if they share a node — the _proximity condition_. The pseudo-observations feeding a deeper edge are the h-functions (conditional CDFs) of the previous tree's fitted copulas, and the conditioning set grows by one variable per level.

The `RegularVine` class in `Microsoft.ML.Probabilistic.Distributions.Copulas.Vine` implements all of this. It is a plain numeric class: fitting an ordinary vine needs no inference engine at all. Only [GPVINE](Gaussian Process vine copulas.md), where the copula parameter is a learned function of the conditioning variables, brings in Infer.NET inference.

### The RegularVine API

| Member | Signature | Purpose |
|--------|-----------|---------|
| Constructor | `RegularVine(IBivariateCopula copula = null)` | The copula family for every edge. Defaults to `GaussianCopula`. |
| `Fit` | `RegularVine Fit(double[][] x, int nTrees = 0, IConditionalCopulaFitter fitter = null, VineStructure structure = VineStructure.Regular, int[] rootOrder = null)` | Fits the vine to raw data and returns `this`, so calls can be chained. |
| `LogLikelihood` | `double LogLikelihood(double[][] x)` | Total copula log-likelihood of `x` under the fitted vine. |
| `Sample` | `double[][] Sample(int n)` | `n` joint draws on the data scale. Requires a canonical fit. |
| `SampleConditional` | `double[][] SampleConditional(IDictionary<int, double> known, int n)` | Draws with some variables held at observed values. Requires a canonical fit. |
| `Trees` | `List<VineTree> { get; }` | The fitted trees, in order. |
| `Marginals` | `InnerQuantiles[] { get; }` | The empirical marginals captured at fit time, one per variable. |
| `Structure` | `VineStructure { get; }` | The tree-selection strategy used by the last `Fit`. |
| `Copula` | `IBivariateCopula { get; }` | The copula family this vine uses. |

### Input data

Data is `double[][]` with `x[i][j]` = variable `j` of observation `i`. All variables are treated as continuous.

**You do not transform the data yourself.** `Fit` applies the empirical PIT internally, so you pass raw values in their natural units. `LogLikelihood` re-applies the PIT to its own argument, so training and test data are transformed consistently and the result is a valid held-out score. The marginals are stored at fit time in `Marginals`, so generated samples come back on the original data scale.

### Fitting

```csharp
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Distributions.Copulas.Vine;

var vine = new RegularVine(new GaussianCopula()).Fit(train, nTrees: 2);
double ll = vine.LogLikelihood(test);
```

**`nTrees`** truncates the vine. A full vine has `d - 1` trees; passing `0` (or anything larger than `d - 1`) builds all of them. Deeper trees capture higher-order conditional dependence at increasing cost, and the omitted trees are treated as independence. This is the standard accuracy/cost knob: sweep `nTrees` and stop where held-out log-likelihood plateaus.

**`fitter`** selects the model class. With no fitter, every deeper-tree copula is fitted with a single Kendall's-tau MLE — the simplifying assumption, giving an SVINE. With a [`GaussianProcessCopulaFitter`](Gaussian Process vine copulas.md), deeper-tree copulas become _conditional_: tau varies with the conditioning variables. The first tree is always fitted unconditionally, so SVINE and GPVINE agree exactly at `nTrees: 1`.

**`structure`** selects the tree topology:

*   `VineStructure.Regular` (the default) picks a maximum spanning tree on the `|tau|` weights at each level, giving a general R-vine. This is the best choice for fitting and scoring.
*   `VineStructure.Canonical` builds a C-vine, in which every tree is a star centred on the variable with the strongest summed dependence. Fitting and scoring behave identically; the only difference is the topology. **Sampling and conditional simulation require a canonical fit** — see below.

**`rootOrder`** applies to canonical fits only. It forces the listed variables, in order, to be the leading roots of the vine, which is what makes `SampleConditional` able to condition on them exactly.

### Reading the fitted vine

After `Fit`, the vine is a list of trees, each a list of edges.

```csharp
foreach (VineTree tree in vine.Trees)
{
    Console.WriteLine($"Tree {tree.Level}:");
    foreach (VineEdge e in tree.Edges)
    {
        string strength = e.IsConditional
            ? $"tau varies with z (e.g. {e.Posterior.TauAt(new[] { 0.5 }):f2} at z = 0.5)"
            : $"tau = {e.Tau:f2}";
        Console.WriteLine($"  {e.Label,-10} weight |tau| = {e.Weight:f3}   {strength}");
    }
}
```

`VineEdge.Label` reads as _conditioned_ `|` _conditioning_: `0,2|1,3` means "the copula of variables 0 and 2 given variables 1 and 3". Tree `T1` therefore shows you the strongest direct pairwise dependencies, and deeper trees show what dependence remains after conditioning. Edge counts per tree are `d-1, d-2, ..., 1`.

The useful members of `VineEdge` are:

| Member | Type | Meaning |
|--------|------|---------|
| `Left`, `Right` | `int` | The two conditioned variables, aligned with the `U` and `V` series. |
| `Conditioning` | `int[]` | The conditioning set `D(e)`, sorted. Empty in `T1`. |
| `Tau` | `double` | The fitted unconditional Kendall's tau. Meaningful when `Posterior` is null. |
| `Posterior` | `IConditionalCopulaPosterior` | The fitted conditional posterior, or null for an unconditional edge. |
| `IsConditional` | `bool` | Shorthand for `Posterior != null`. |
| `Weight` | `double` | The <code>&#124;tau&#124;</code> used to select this edge during tree construction. |
| `Label` | `string` | The <code>"0,2&#124;1,3"</code> display form. |
| `U`, `V` | `double[]` | The pseudo-observation series feeding this edge. |
| `Z` | `double[][]` | The conditioning matrix, one row per observation, one column per conditioning variable. Empty in `T1`. |
| `HByVar` | `Dictionary<int, double[]>` | The h-function series computed at fit time, which seed the next tree. |
| `N()` | `int[]` | The complete set `{Left, Right} ∪ Conditioning`, sorted. |

For an unconditional edge, `Tau` is a single number: its magnitude is the strength of the dependence and its sign the direction. For a conditional (GPVINE) edge, tau is a _function_, and `Posterior.TauAt(z)` evaluates it. Sweeping `z` over a grid and plotting the result is the headline output of GPVINE: dependence as a learned curve rather than a single number.

### The log-likelihood, and how to read it

```csharp
double ll = vine.LogLikelihood(test);
```

This is the **copula** log-likelihood: the summed log copula densities over every edge of every tree. Marginal densities are deliberately _not_ included, which is exactly what you want when comparing dependence models on the same data — the marginal terms would be identical and would cancel anyway.

*   **Higher is better.** The value can be positive, because dependence concentrates probability mass relative to the independent case.
*   **Independence gives approximately zero.** A vine fitted to independent variables contributes roughly nothing.
*   **Always score on held-out data.** In-sample log-likelihood only ever increases as you add trees, so it cannot tell you when to stop.

The vine structure is replayed on the PIT of the argument, so `LogLikelihood(test)` is a genuine held-out score and not an in-sample artefact.

### Sampling

A fitted vine is a generative model. Sampling uses the inverse-Rosenblatt transform and **requires a canonical (C-vine) fit**; calling `Sample` on a `Regular` vine throws `NotSupportedException` telling you to refit.

```csharp
var vine = new RegularVine()
    .Fit(train, nTrees: 0, fitter, VineStructure.Canonical);

double[][] synthetic = vine.Sample(1000);   // [1000][d], in the original data units
```

Output is on the data scale: the stored empirical marginals (`InnerQuantiles.GetQuantile`) invert the PIT. For a GPVINE, each draw samples the latent function from the GP posterior rather than using its mean, so the synthetic data reflects both the learned context-varying dependence _and_ the posterior uncertainty about it — a genuine posterior-predictive sample rather than a plug-in.

A quick sanity check on a sample is that it should reproduce the training data's pairwise Kendall's tau:

```csharp
double tauData = KendallTau.Compute(Column(train, i), Column(train, j));
double tauSamp = KendallTau.Compute(Column(synthetic, i), Column(synthetic, j));
// these should agree to within sampling error
```

### Conditional simulation and imputation

`SampleConditional` fixes a chosen set of variables to observed values and draws the rest from their conditional distribution. This gives you imputation of missing dimensions, scenario analysis ("given this Z, what happens?"), and posterior-predictive inference given a partial observation.

The one rule is that **conditioning is exact when the conditioned set is the leading roots of the canonical vine**. So fit with `rootOrder` set to the variable ids you intend to condition on:

```csharp
using System.Collections.Generic;

int zId = 2;
var vine = new RegularVine()
    .Fit(train, nTrees: 0, fitter, VineStructure.Canonical, rootOrder: new[] { zId });

// "Given Z = 1.0, what are plausible (X, Y)?"
double[][] drawn = vine.SampleConditional(
    new Dictionary<int, double> { { zId, 1.0 } }, n: 1000);
```

Each returned row has `drawn[i][zId] == 1.0` exactly — conditioned variables are returned as given — and the remaining columns are drawn from their conditional distribution. Under a GPVINE those draws reflect the dependence _at that conditioning value_, so the sampled X-Y rank correlation matches the learned tau(z). If you condition on a set that is not the leading roots, or on a `Regular`-fit vine, you get a `NotSupportedException` that names the `rootOrder` you should refit with. Conditioning on an arbitrary subset without refitting is not supported.

### Helper classes

These are public and usable on their own.

**`Pit`** — the empirical probability integral transform.

```csharp
double[] u  = Pit.Transform(column);      // one column of raw values -> (0, 1)
double[][] U = Pit.Transform(x);          // [n][d] -> [n][d], column by column
double safe = Pit.ClampUnit(value);       // clamp into [eps, 1 - eps], eps = 1e-6 by default
```

`Transform` uses average ranks divided by `n + 1`, which keeps values strictly inside (0, 1) and so avoids the infinities that `Phi^-1(0)` and `Phi^-1(1)` would produce. Ties receive the mean of the ranks they span.

**`KendallTau`** — `double KendallTau.Compute(double[] a, double[] b)` returns Kendall's tau-b in [-1, 1], or 0 where it is undefined (constant input, or fewer than two points). The tau-b form corrects for ties; on tie-free continuous data it equals the ordinary (concordant - discordant) / (n(n-1)/2).

**`MaxSpanningTree`** — `List<(int I, int J)> MaxSpanningTree.Prim(double[,] weights)` returns the edges of the maximum spanning tree of a symmetric weight matrix. Mark a disallowed pair with `double.NegativeInfinity`; if the allowed edges do not connect every node, you get a spanning forest with fewer than `m - 1` edges. This is how each vine tree is selected.

**Marginals** are `InnerQuantiles`, so `vine.Marginals[j].GetProbLessThan(value)` maps a data value to (0, 1), and `vine.Marginals[j].GetQuantile(p)` maps back to the data scale.

### Next

*   [Gaussian Process vine copulas](Gaussian Process vine copulas.md) — modelling tau as a learned function of the conditioning variables.
*   [Bivariate copula families](Bivariate copula families.md) — the copula families a vine can be built from.
