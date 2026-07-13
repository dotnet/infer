---
layout: default 
--- 

[Infer.NET user guide](index.md) : [Copulas and vines](Copulas and vines.md)

## Bivariate copula families

This page describes an experimental feature that is likely to change in future releases

A bivariate copula is a joint distribution on the unit square whose marginals are uniform. It is the building block of everything on these pages: a [vine](Vine copulas.md) is a product of bivariate copulas, and the [copula EP factor](Gaussian Process vine copulas.md) attaches a single bivariate copula likelihood to a latent score.

All families live in `Microsoft.ML.Probabilistic.Distributions.Copulas` and implement `IBivariateCopula`. They are plain numeric classes with no dependency on the inference compiler, so you can use them standalone.

### The IBivariateCopula interface

Every family is parameterised by **Kendall's tau**, not by its own native parameter. This is what lets one latent function and one EP message operator serve every family: `tau` is a common currency in [-1, 1], and each family converts it to its native `theta` internally.

| Member | Signature | Purpose |
|--------|-----------|---------|
| `Name` | `string { get; }` | Human-readable family name, e.g. `"Gaussian"`. |
| `TauRange` | `(double Min, double Max) { get; }` | The valid range of Kendall's tau for this family. |
| `TauToTheta` | `double TauToTheta(double tau)` | Maps Kendall's tau to the family's native parameter. |
| `ThetaToTau` | `double ThetaToTau(double theta)` | The inverse map. |
| `LogDensity` | `double LogDensity(double u, double v, double tau)` | Log copula density <code>log c(u, v &#124; tau)</code> for one observation. |
| `Cdf` | `double Cdf(double u, double v, double tau)` | The copula CDF `C(u, v) = P(U <= u, V <= v)`. |
| `ConditionalCdf` | `double ConditionalCdf(double u, double v, double tau, int given)` | The conditional CDF, known in the vine literature as the **h-function**. With `given = 1` it returns <code>P(U &lt;= u &#124; V = v)</code>; with `given = 0` it returns <code>P(V &lt;= v &#124; U = u)</code>. |
| `InverseConditionalCdf` | `double InverseConditionalCdf(double w, double x, double tau, int given)` | Inverts `ConditionalCdf` for the unknown variable, given the conditioning value `x` and the conditional-CDF level `w`. The building block of the inverse-Rosenblatt transform used to sample. |
| `Sample` | `Vector Sample(double tau)` | Draws one pair `(u, v)`, returned as a length-2 `Vector`. |

`u` and `v` are always pseudo-observations strictly inside (0, 1). The `given` argument selects _which_ variable is being conditioned on: `given = 1` conditions on `v` and returns a value for `u`; `given = 0` conditions on `u` and returns a value for `v`. The h-function is what generates the pseudo-observations that feed the deeper trees of a vine — see [Vine copulas](Vine copulas.md).

### The implemented families

| Family | Dependence captured | Tau range | theta from tau |
|--------|--------------------|-----------|----------------|
| `GaussianCopula` | Symmetric, no tail dependence | (-1, 1) | `theta = sin(pi/2 * tau)` |
| `ClaytonCopula` | **Lower**-tail dependence (variables crash together) | (0, 1) | `theta = 2 * tau / (1 - tau)` |
| `GumbelCopula` | **Upper**-tail dependence (variables boom together) | (0, 1) | `theta = 1 / (1 - tau)` |

`GaussianCopula` is the default and the most generally useful: it is the only one of the three that handles **negative** dependence.

**An important caveat for Clayton and Gumbel.** Both are Archimedean families that model positive dependence only, so their `TauRange` is (0, 1) and their `TauToTheta` clamps `tau` into that range. The GPVINE link `tau = 2 * Phi(f) - 1` spans (-1, 1), so any latent value implying negative dependence is mapped to (near-)independence rather than to negative dependence. If your data has negative or sign-changing dependence, use `GaussianCopula`. Rotated Archimedean copulas, which would cover negative dependence, are not implemented.

### Using a family directly

```csharp
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Math;

IBivariateCopula c = new ClaytonCopula();

double theta = c.TauToTheta(0.4);            // native parameter for tau = 0.4
double logDensity = c.LogDensity(0.3, 0.8, 0.4);
double joint = c.Cdf(0.3, 0.8, 0.4);         // P(U <= 0.3, V <= 0.8)
double h = c.ConditionalCdf(0.3, 0.8, 0.4, given: 1);   // P(U <= 0.3 | V = 0.8)
double back = c.InverseConditionalCdf(h, 0.8, 0.4, given: 1);   // recovers 0.3

Vector uv = c.Sample(0.4);                   // one draw, uv[0] and uv[1] in (0, 1)
```

`Sample` uses the shared random number generator, so call `Rand.Restart(seed)` if you need reproducible draws.

### Choosing a family

Fit each candidate and compare the **held-out** copula log-likelihood; the winner tells you the shape of the dependence:

```csharp
foreach (IBivariateCopula family in new IBivariateCopula[]
         { new GaussianCopula(), new ClaytonCopula(), new GumbelCopula() })
{
    double ll = new RegularVine(family).Fit(train).LogLikelihood(test);
    Console.WriteLine($"{family.Name}: {ll:f1}");
}
```

A Clayton win indicates lower-tail dependence, a Gumbel win upper-tail dependence, and a Gaussian win symmetric dependence with light tails. Because Clayton and Gumbel cannot represent negative dependence, a Gaussian win on data with mixed signs is expected and is not evidence about the tails.

### Numerical notes

You do not normally need to think about any of this, but it explains the behaviour you will see at extreme values.

*   `GaussianCopula` clamps `theta` to `(-1 + 1e-6, 1 - 1e-6)` (`GaussianCopula.ThetaEpsilon`) so that `1 - theta^2` stays strictly positive and the density stays finite as `tau` approaches +/-1.
*   `ClaytonCopula` clamps `tau` to `[1e-6, 1 - 1e-6]` (`ClaytonCopula.TauEpsilon`), and treats a sufficiently small `theta` as exact independence (density 1, CDF `u*v`).
*   `GumbelCopula` clamps `tau` to `[0, 1 - 1e-6]` (`GumbelCopula.TauEpsilon`) and evaluates its density entirely in log space, because the intermediate `x^theta` would overflow or underflow for large `theta` (tau near 1).
*   `GaussianCopula` and `ClaytonCopula` have closed-form inverse h-functions. `GumbelCopula` does not, so `InverseConditionalCdf` solves for the unknown by 60 steps of bisection on (0, 1) — correct, but noticeably slower if you call it in a tight loop.

### Adding a family

Implement `IBivariateCopula` and pass an instance wherever a copula is expected — `RegularVine(copula)`, `IConditionalCopulaFitter.Fit(u, v, z, copula)`, or the `CopulaFactor.BivariateCopula` factor. No changes to the vine layer or to the EP message operator are needed, because both are written against the interface and only ever speak in Kendall's tau. The one requirement is that `LogDensity`, `ConditionalCdf` and `InverseConditionalCdf` are mutually consistent, since the vine recursion relies on the h-function being the exact partial derivative of the CDF.
