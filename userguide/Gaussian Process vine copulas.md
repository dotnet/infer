---
layout: default 
--- 

[Infer.NET user guide](index.md) : [Copulas and vines](Copulas and vines.md)

## Gaussian Process vine copulas

This page describes an experimental feature that is likely to change in future releases

An ordinary vine makes the **simplifying assumption**: each conditional copula has a single, constant parameter, regardless of the values it is conditioned on. **GPVINE** drops that assumption. It models Kendall's tau of each conditional copula as a smooth function of the conditioning variables,

```
tau = g(f(z)) = 2 * Phi(f(z)) - 1
```

where `f` is a latent function given a sparse Gaussian process prior and inferred by Expectation Propagation. The link `g` maps the unbounded latent score into the valid tau range (-1, 1), and, being folded inside the factor, means EP only ever sees a smooth non-conjugate likelihood on a single Gaussian variable.

This implements Lopez-Paz, Hernández-Lobato and Ghahramani, _Gaussian Process Vine Copulas for Multivariate Dependence_, ICML 2013. You can run the bundled example from the [Examples Browser](The examples browser.md); the source is `src/Tutorials/GaussianProcessVine.cs`.

### The per-edge model

Each conditional edge of a vine is one small Infer.NET model, structurally identical to the [Gaussian Process classifier](Gaussian Process classifier.md): a sparse GP prior over `f`, a `score = f(z)` evaluation, and an observed likelihood on that score. The only new piece is the likelihood, which is the copula density of the observed pseudo-observation pair.

```csharp
Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
IfBlock block = Variable.If(evidence);

Variable<SparseGP> prior = Variable.New<SparseGP>().Named("copulaPrior");
Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

VariableArray<Vector> z = Variable.Observed(inputs).Named("z");   // the conditioning vectors
Range j = z.Range.Named("j");
Variable<double> score = Variable.FunctionEvaluate(f, z[j]).Named("score");

VariableArray<Vector> uv = Variable.Observed(pairs, j).Named("uv");   // the (u, v) pairs
uv[j] = Variable<Vector>.Factor(CopulaFactor.BivariateCopula, score, Variable.Observed(copula));

block.CloseBlock();

double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
SparseGP posterior = engine.Infer<SparseGP>(f);
```

The copula family is passed as **observed data** rather than baked into the factor, so a single factor and a single EP message operator serve every family. The `Variable.Bernoulli(0.5)` and `IfBlock` pattern gives you the [model evidence](Computing model evidence for model selection.md), which is what you use for hyperparameter and family comparison.

You rarely need to write this out: `GaussianProcessCopulaFitter` builds exactly this model for you.

### GaussianProcessCopulaFitter

`Microsoft.ML.Probabilistic.Models.GaussianProcessCopulaFitter` implements the Runtime interface `IConditionalCopulaFitter`. Hand one to `RegularVine.Fit` and the deeper trees become conditional.

| Member | Default | Purpose |
|--------|---------|---------|
| `GaussianProcessCopulaFitter(InferenceEngine engine = null)` | — | If no engine is supplied, a non-verbose Expectation Propagation engine is created. |
| `NumInducing` | `20` | Number of inducing inputs (pseudo-inputs) for the sparse GP. |
| `LogLengthScale` | `-1.5` | Log length-scale used to initialise every ARD kernel dimension. |
| `LogSignalSd` | `0.0` | Log signal standard deviation of the ARD kernel. |
| `LogNoiseSd` | `log(0.2)` | Log standard deviation of an added white-noise kernel component. |
| `NumberOfIterations` | `15` | EP iterations per edge fit. |
| `LastLogEvidence` | `NaN` | The EP log-evidence of the most recent `Fit`. |
| `Fit(double[] u, double[] v, double[][] z, IBivariateCopula copula)` | — | Fits one conditional copula, returning an `IConditionalCopulaPosterior`. |

```csharp
using Microsoft.ML.Probabilistic.Models;

var fitter = new GaussianProcessCopulaFitter
{
    NumInducing        = 20,
    NumberOfIterations = 15,
    LogLengthScale     = -1.5,
};

var gpvine = new RegularVine().Fit(train, nTrees: 0, fitter);
double ll = gpvine.LogLikelihood(test);
```

Reusing one fitter across every edge of a vine is fine and is what you should do — a fitter holds no per-edge state beyond `LastLogEvidence`, and sharing it shares the inference engine rather than constructing a new one per edge. Each `Fit` call does build a fresh model, so the settings above take effect on every edge.

Three points about the defaults are worth knowing.

**The length-scale must match the scale of your conditioning inputs.** Inside a vine, the conditioning inputs are pseudo-observations in (0, 1), and the default `LogLengthScale = -1.5` (a length of about 0.22) is tuned for that range. If you call `fitter.Fit(u, v, z, copula)` directly with **raw** conditioning values — say `z` in [-3, 3] — raise it to around `0.0` (a length of about 1), as the [GP classifier](Gaussian Process classifier.md) does.

**The white-noise term is doing real work.** It regularises the inducing-point covariance and acts as the GP nugget. Without it, the GP fit can fail with a positive-definiteness error on (0, 1) inputs. Leave `LogNoiseSd` alone unless you have a reason.

**The GP mean is initialised from the data.** It is set to `Phi^-1((tau_MLE + 1) / 2)`, where `tau_MLE` is the unconditional Kendall's tau of the edge, clamped away from +/-1. This happens automatically.

### The fitted posterior

`Fit` returns an `IConditionalCopulaPosterior`:

| Member | Signature | Purpose |
|--------|-----------|---------|
| `TauAt` | `double TauAt(double[] z)` | The posterior **mean** Kendall's tau at conditioning vector `z`. |
| `SampleTau` | `double SampleTau(double[] z)` | A Kendall's tau **drawn** from the posterior at `z`, reflecting the latent function's uncertainty. |

The concrete implementation is `SparseGPCopulaPosterior`, which wraps the fitted `SparseGP` and exposes it through its `Posterior` property if you want the raw GP. `RegularVine` uses `TauAt` when computing likelihoods and `SampleTau` when generating posterior-predictive samples — which is why GPVINE draws carry the GP's uncertainty rather than plugging in a point estimate.

You can use the fitter on its own, without a vine, to learn how the dependence between two variables varies with a covariate:

```csharp
// u, v: paired pseudo-observations in (0, 1). z: one conditioning vector per observation.
IConditionalCopulaPosterior post = fitter.Fit(u, v, z, new GaussianCopula());

for (double zStar = -3; zStar <= 3; zStar += 0.25)
    Console.WriteLine($"{zStar,5:f2}  tau = {post.TauAt(new[] { zStar }):f3}");

Console.WriteLine($"log evidence = {fitter.LastLogEvidence:f2}");
```

That sweep is the curve GPVINE exists to produce.

### The copula factor and its EP operator

`CopulaFactor.BivariateCopula(double score, IBivariateCopula copula)` returns a `Vector` pair `(u, v)`. It is a thin hook for the inference compiler: it names the factor that `BivariateCopulaOp` attaches its EP messages to. Its body samples from the copula at `tau = 2 * Phi(score) - 1`, but EP never calls it — the messages moment-match the copula likelihood against the Gaussian latent directly.

`BivariateCopulaOp` is the Expectation Propagation message operator, and carries the `Experimental` [quality band](Quality bands.md). For an observed pair `(u, v)` the factor contributes the copula likelihood as a non-conjugate term on the single Gaussian latent `score`. The message to `score` is obtained by moment matching: with cavity <code>N(s; m, v)</code> the tilted distribution is <code>N(s; m, v) c(u, v &#124; g(s))</code>, and its first two moments are computed by quadrature. The integrand is evaluated in log space and shifted by its value at the proposal mean before being exponentiated, which keeps it in range; the offset is folded back into the log normaliser analytically. The projected Gaussian divided by the cavity is the message, and the log normaliser of the tilted integral is the evidence contribution.

| Method | Purpose |
|--------|---------|
| `Gaussian ScoreAverageConditional(Vector pair, Gaussian score, IBivariateCopula copula, Gaussian result)` | The EP message to `score`. |
| `Gaussian ScoreAverageConditionalInit()` | Initialiser (returns a uniform Gaussian). |
| `double LogAverageFactor(Vector pair, Gaussian score, IBivariateCopula copula, Gaussian to_score)` | <code>log int N(s; m, v) c(u, v &#124; g(s)) ds</code> — the evidence contribution. |
| `double LogEvidenceRatio(Vector pair, Gaussian score, IBivariateCopula copula, Gaussian to_score)` | The evidence ratio. The pair is observed, so this equals `LogAverageFactor`. |

Three static fields tune the operator. The defaults are robust and you should only change them if you hit numerical trouble.

| Field | Default | Meaning |
|-------|---------|---------|
| `BivariateCopulaOp.QuadratureNodeCount` | `64` | Initial node count for the adaptive quadrature. |
| `BivariateCopulaOp.QuadratureRelTol` | `1e-10` | Relative tolerance for the adaptive quadrature. |
| `BivariateCopulaOp.ForceProper` | `true` | Force the outgoing message to have non-negative precision. |

Adaptive Clenshaw-Curtis is used rather than fixed Gauss-Hermite because the copula likelihood can blow up as the latent pushes tau towards +/-1, which makes the integrand heavy-tailed. Where the quadrature cannot produce a usable result — a collapsed variance, or a NaN — the operator returns a uniform message rather than an improper one, so a difficult edge degrades to uninformative rather than corrupting the fit.

### A worked comparison: does the simplifying assumption cost you anything?

This is the bundled example (`src/Tutorials/GaussianProcessVine.cs`), reduced to its essentials. The data is three variables where X and Y each depend on Z, and the copula of (X, Y) given Z has a Kendall's tau that _varies with Z_ — exactly the structure the simplifying assumption cannot represent.

```csharp
var engine = new InferenceEngine(new ExpectationPropagation()) { ShowProgress = false };
var fitter = new GaussianProcessCopulaFitter(engine) { NumInducing = 15, NumberOfIterations = 15 };

for (int nTrees = 1; nTrees <= 2; nTrees++)
{
    double svine  = new RegularVine(new GaussianCopula()).Fit(train, nTrees).LogLikelihood(test);
    double gpvine = new RegularVine(new GaussianCopula()).Fit(train, nTrees, fitter).LogLikelihood(test);
    Console.WriteLine($"  {nTrees}   |     {svine,10:f2}     |     {gpvine,10:f2}");
}
```

The data is generated from a fixed seed, so the example prints the same numbers every run:

```
trees | SVINE test log-lik | GPVINE test log-lik
------+--------------------+--------------------
  1   |         257.65     |         257.65
  2   |         257.57     |         304.26
```

The two agreeing exactly at one tree is a correctness check — the first tree is unconditional by construction, so there is nothing for the GP to do. At two trees SVINE gains nothing (it slips very slightly, to 257.57, because a constant-tau second-tree copula is the wrong model and costs a little on held-out data), while GPVINE gains about 47 nats by modelling how the (X, Y) dependence varies with Z. That gap is the value of dropping the simplifying assumption, and it is the entire claim of the method.

When you run this you will also see two compiler warnings, one for `ScoreAverageConditional` and one for `LogEvidenceRatio`, noting that `BivariateCopulaOp` has quality band `Experimental`. That is expected — see [Quality bands](Quality bands.md).

### Model comparison

Everything here is comparable by **held-out** copula log-likelihood, and that should be your primary criterion:

*   **GPVINE against SVINE** — fit both and compare `LogLikelihood(test)`. If they are close, your conditional dependence really is constant and the cheaper SVINE is the better model.
*   **Family selection** — fit Gaussian, Clayton and Gumbel and compare. See [Bivariate copula families](Bivariate copula families.md).
*   **Truncation** — sweep `nTrees` and stop where the held-out score plateaus.

For a Bayesian criterion at the level of a single edge, `fitter.LastLogEvidence` holds the EP log-evidence of the most recent fit, which you can compare across kernel hyperparameters or families. Automated hyperparameter and inducing-point optimisation is **not** built in: Infer.NET does not optimise kernel hyperparameters for you, so you compare a handful of settings manually, as the [GP classifier](Gaussian Process classifier.md) does.

### Cost

Fitting scales roughly linearly in the number of observations `n`, and quadratically in the number of variables `d` and in the number of inducing points. Each GPVINE conditional edge is a full EP run, taking on the order of seconds; SVINE edges are near-instant by comparison. The knobs that reduce cost, in the order you should reach for them, are: truncate `nTrees`; reduce `NumInducing`; reduce `NumberOfIterations`; and use an SVINE wherever conditional dependence is weak. Sampling from a fitted vine is cheap — the GP posterior is queried but never refitted.

### Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| A positive-definiteness error during a GP fit | The inducing covariance is ill-conditioned. Keep the white-noise nugget (`LogNoiseSd`), and make sure `LogLengthScale` matches the scale of the inputs: about `-1.5` for (0, 1) pseudo-observations, about `0.0` for raw inputs. |
| `NotSupportedException` from `Sample` or `SampleConditional` | Refit with `structure: VineStructure.Canonical`, and with `rootOrder` if you intend to condition. |
| `NotSupportedException: Conditioned set must be the leading roots...` | Refit with `rootOrder` set to the variable ids you want to condition on. The message names them for you. |
| A Clayton or Gumbel fit looks like independence | Those families model positive dependence only. Negative or sign-changing dependence needs `GaussianCopula`. |
| Held-out log-likelihood falls as you add trees | The deeper trees are fitting noise. Truncate `nTrees`. |
| The fit is slow | Reduce `NumInducing` and `NumberOfIterations`, truncate `nTrees`, or drop to an SVINE. |
| Recovered `tau(z)` is flat when it should vary | The length-scale is too long for the input range. Lower `LogLengthScale`, and check that `z` really is the variable the dependence varies with. |

### Limitations

Sampling and conditional simulation require a canonical (C-vine) fit; general R-vine sampling is not implemented. Conditioning is exact only for the leading roots of the vine, so conditioning on a different subset needs a refit with the appropriate `rootOrder`. The Frank family and rotated Clayton/Gumbel copulas (which would cover negative dependence in the Archimedean families) are not implemented. Kernel hyperparameters and inducing inputs are not optimised automatically.

### See also

*   [Copulas and vines](Copulas and vines.md) — the overview.
*   [Vine copulas](Vine copulas.md) — `RegularVine`, fitting, sampling, conditional simulation.
*   [Gaussian Process classifier](Gaussian Process classifier.md) — the sparse-GP modelling pattern this builds on.
*   [Computing model evidence for model selection](Computing model evidence for model selection.md).
