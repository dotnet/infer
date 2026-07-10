# GPVINE.md — Extending Infer.NET with Gaussian Process Vine Copulas

A detailed implementation plan for adding the **GPVINE** method of

> Lopez‑Paz, Hernández‑Lobato & Ghahramani, *Gaussian Process Vine Copulas for
> Multivariate Dependence*, ICML 2013.

to the Infer.NET codebase. This document is a plan only — no code is implemented
yet. Equation numbers (eq. N) refer to the paper.

---

## 1. Why Infer.NET is a good host for this method

The method has two separable layers:

- **Layer A — inference core:** a sparse Gaussian‑process prior on a latent
  function `f`, an ARD kernel, **Expectation Propagation** for the posterior and
  the model evidence. This is the hard, risky part to build from scratch.
- **Layer B — copula/vine machinery:** marginal PIT, bivariate copula math,
  regular‑vine tree construction, h‑function recursion, sampling. Closed‑form,
  language‑independent application logic.

Infer.NET already provides essentially *all* of Layer A. The paper's authors
worked in the same lab that produced Infer.NET, and the approach (sparse GP + EP)
is squarely in its wheelhouse. The only genuinely new inference component is a
**custom copula likelihood factor with an EP message operator**; everything else
is composition of existing primitives plus a port of Layer B to C#.

### What already exists (verified in this repo)

| Paper requirement | Infer.NET asset |
|---|---|
| GP prior over functions, eq. 6 | `src/Runtime/Distributions/GaussianProcess/{GaussianProcess,SparseGP,SparseGPFixed}.cs` |
| ARD‑RBF kernel, eq. 7 | `src/Runtime/Distributions/Kernels/ARD.cs`, `SquaredExponential.cs` |
| FITC / inducing points, Sec. 3.1 | `SparseGPFixed` (the fixed `basis`); EP ops in `src/Runtime/Factors/SparseGPOp.cs` |
| Evaluate `f(z)` | `Factor.FunctionEvaluate(IFunction, Vector)` (`src/Runtime/Factors/Factor.cs:872`), op in `SparseGPOp.cs` |
| EP inference | `Microsoft.ML.Probabilistic.Algorithms.ExpectationPropagation` |
| Model evidence (Sec. 4 tuning/selection) | evidence blocks: `Variable.Bernoulli(0.5)` + `IfBlock` |
| 1‑D quadrature for moment matching | `src/Runtime/Core/Maths/Quadrature.cs` (`GaussianNodesAndWeights` = Gauss–Hermite, `AdaptiveClenshawCurtis`, `AdaptiveExpSinh`) |
| Φ, Φ⁻¹ for Gaussian copula + link | `MMath.NormalCdf`, `MMath.NormalCdfLn`, `MMath.NormalCdfInv` (`src/Runtime/Core/Maths/SpecialFunctions.cs`) |
| Predictive `τ(z*)` | `SparseGP.Marginal(Vector)` → `Gaussian` (`SparseGP.cs:507`) |
| Reference template | `src/Tutorials/GaussianProcessClassifier.cs` |

### What must be built

1. A bivariate **copula family** library (Gaussian first), with density,
   conditional CDFs (h‑functions), and Kendall‑τ ↔ θ maps (eqs. 11–14, Table 1).
2. A **custom EP factor** that injects the copula likelihood `c(u,v | τ=g(f(z)))`
   as a term on the latent score, with quadrature‑based moment matching and an
   evidence contribution.
3. The **regular‑vine** layer: tree construction (max spanning tree on |τ|),
   h‑function recursion to generate deeper‑tree pseudo‑observations (eqs. 4–5).
4. A driver/example **`GaussianProcessVine.cs`** mirroring the GP classifier.
5. An **outer loop** for kernel hyperparameters and inducing inputs via the EP
   evidence (Infer.NET does not optimize these automatically).

---

## 2. The modelling pattern (per conditional copula / per vine edge)

The classifier template:

```csharp
Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
Variable<double> score = Variable.FunctionEvaluate(f, x[j]).Named("score");
y[j] = (Variable.GaussianFromMeanAndVariance(score, 0.1) > 0);   // likelihood
```

GPVINE keeps the GP scaffolding identically and swaps the likelihood. For one
vine edge with conditioning vectors `z_i` and observed pseudo‑observation pairs
`(u_i, v_i)`:

```csharp
Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
Range j = z.Range;
Variable<double> score = Variable.FunctionEvaluate(f, z[j]).Named("score"); // f(z_i)

// Kendall's tau in (-1,1) via the paper's link g(f) = 2*Phi(f) - 1 (Sec. 3).
// Folded INSIDE the copula factor so EP only deals with `score`.

// Observed pair drives a copula likelihood term on `score`:
uv[j] = BivariateCopula(score, copula);   // NEW factor; uv observed
```

`τ_i = g(f(z_i)) = 2Φ(f(z_i)) − 1` is computed inside the factor, so to EP the
new factor is a non‑conjugate likelihood on a single Gaussian latent `score` —
structurally the same situation `LogisticOp` already handles.

**First tree `T_1` (unconditional copulas).** Conditioning set is empty, so τ is a
single scalar. Two options: (a) fit τ as the Kendall‑τ MLE directly (cheap, what
the paper effectively does for the GP mean init, Sec. 4: mean
`Φ⁻¹((τ̂_MLE+1)/2)`); or (b) reuse the same machinery with a constant input so the
GP collapses to a constant. Plan: use (a) for `T_1` and the full GP for trees ≥ 2.

---

## 3. New components, file by file

### 3.1 Copula family math — `src/Runtime/Distributions/Copulas/`

New folder, mirroring `Distributions/Kernels/` in style.

- `ICopula.cs` — interface:
  ```csharp
  public interface IBivariateCopula
  {
      string Name { get; }
      double TauToTheta(double tau);
      double ThetaToTau(double theta);
      double LogDensity(double u, double v, double tau);     // log c(u,v|tau)
      double Cdf(double u, double v, double tau);            // C(u,v|tau)
      double ConditionalCdf(double u, double v, double tau, int given);        // dC/dv or dC/du
      double InverseConditionalCdf(double w, double x, double tau, int given); // inverse Rosenblatt
      Vector Sample(double tau);                             // draw a (u,v) pair
  }
  ```
- `GaussianCopula.cs` — the workhorse (Appendix A):
  - `TauToTheta`: `θ = sin(π/2 · τ)`; `ThetaToTau`: `τ = (2/π) arcsin θ`.
  - `LogDensity` (eq. 12) using `a=Φ⁻¹(u)`, `b=Φ⁻¹(v)` via `MMath.NormalCdfInv`:
    `−½ ln(1−θ²) − (θ²(a²+b²) − 2θab)/(2(1−θ²))`.
  - `ConditionalCdf` (eqs. 13–14): `Φ((a − θb)/√(1−θ²))` / `Φ((b − θa)/√(1−θ²))`.
  - clamp `θ ∈ (−1+ε, 1−ε)`.
- `ClaytonCopula.cs`, `GumbelCopula.cs`, `FrankCopula.cs` — Table 1 maps and
  densities (later phases; Frank's τ↔θ needs a 1‑D root find on the Debye
  function — use `Quadrature` + a bisection helper).
- `CopulaFamily.cs` — enum/registry + factory used by the factor and example.

These are pure numeric classes — directly unit‑testable without the compiler.

### 3.2 The factor delegate — `src/Runtime/Factors/CopulaFactor.cs`

A thin factor method that, given the latent `score`, *generates* the observed
pair via `copula.Sample`. Output is observed, so the factor acts as a likelihood
on `score`:

```csharp
[ParameterNames("pair", "score", "copula")]
public static Vector BivariateCopula(double score, IBivariateCopula copula)
{
    // Generative form (used only for sampling/testing). tau = g(score).
    // EP never calls this; it calls the operator in BivariateCopulaOp.
}
```

(The `copula` is passed as observed data so a single factor / EP operator serves
every family; the per-family sampling lives on the copula classes.)

### 3.3 The EP message operator — `src/Runtime/Factors/BivariateCopulaOp.cs`

This is the core deliverable. Templated on `LogisticOp` (`Logistic.cs`) and
`SparseGPOp`. Annotated `[FactorMethod(typeof(CopulaFactor), "BivariateCopula")]` and
`[Quality(QualityBand.Experimental)]`. The factor graph: observed `pair=(u,v)`
and Gaussian‑distributed latent `score`. EP needs three things:

1. **Message to `score`** — `ScoreAverageConditional`:
   ```csharp
   public static Gaussian ScoreAverageConditional(
       Vector pair, [RequiredArgument] Gaussian score, IBivariateCopula copula)
   ```
   - Let the incoming message/cavity be `score ~ N(m, v)`.
   - Define the tilted integrand `t(s) = c(u, v | g(s))` with `g(s)=2Φ(s)−1`.
   - Place Gauss–Hermite nodes against the cavity:
     `Quadrature.GaussianNodesAndWeights(m, v, nodes, weights)`.
   - Compute moments `Z=Σ wᵢ t(sᵢ)`, `E[s]=Σ wᵢ sᵢ t(sᵢ)/Z`,
     `E[s²]=Σ wᵢ sᵢ² t(sᵢ)/Z`; form the projected Gaussian
     `q = Gaussian.FromMeanAndVariance(E[s], E[s²]−E[s]²)`.
   - Return `q / cavity` (Infer.NET `Gaussian` supports `/`), i.e. the outgoing
     message. Guard against negative/zero variance (damping / `SetToRatio` with
     `forceProper`), exactly as `LogisticOp` does.
2. **Evidence** — `LogAverageFactor` / `LogEvidenceRatio` returning `ln Z` from
   the same quadrature, so `engine.Infer<Bernoulli>(evidence).LogOdds` yields the
   per‑edge marginal likelihood used for hyperparameter tuning and family
   selection (Sec. 4).
3. **Init** — `ScoreAverageConditionalInit` returning `Gaussian.Uniform()`.

Numerical notes:
- Work in log space inside `t(s)` (`LogDensity`) and use a log‑sum‑exp over nodes
  for `ln Z` to avoid under/overflow.
- For heavy‑tailed integrands, fall back from fixed Gauss–Hermite to
  `Quadrature.AdaptiveClenshawCurtis` when the node estimate looks unstable.
- Clamp `τ` away from ±1 inside `g` to keep `1−θ²` bounded.

Registration is automatic: Infer.NET's `FactorManager` discovers `[FactorMethod]`
operators by assembly scan, so placing the op in the `Runtime` assembly is enough.

### 3.4 Regular‑vine layer — `src/Runtime/Distributions/Copulas/Vine/` (or a small new project)

Application logic, not inference. Could also live next to the example; keeping it
in a library makes it reusable and testable.

- `Pit.cs` — empirical PIT: rank/(n+1) per column → pseudo‑observations in (0,1).
- `VineEdge.cs` — conditioned set `C(e)`, conditioning set `D(e)`, the edge's
  `(u, v)` series, conditioning matrix `Z`, fitted copula + GP posterior.
- `KendallTau.cs` — empirical Kendall's τ (O(n log n) via merge‑sort inversions).
- `MaxSpanningTree.cs` — Prim's algorithm on the |τ| weight matrix (Sec. 2.1).
- `RegularVine.cs` — holds the `d−1` trees; methods:
  - `BuildFirstTree(u)` → `T_1` by MST on |τ|.
  - `BuildNextTree(prevTree)` → nodes = previous edges; connect only edges
    sharing a variable (proximity condition); new pseudo‑observations are the
    previous tree's fitted **h‑functions** (eq. 5); `D(e)` grows by one per level.
  - `LogLikelihood(x)` = Σ over edges of `Σ_i log c(...)` (eq. 4).
  - `Sample(n)` via the inverse Rosenblatt transform over the trees.

### 3.5 Driver / example — `src/Tutorials/GaussianProcessVine.cs`

Mirrors `GaussianProcessClassifier.cs` and carries the `[Example(...)]` attribute
so it auto‑registers in the Examples Browser. Responsibilities:

1. Generate or load data `X (n×d)`; compute pseudo‑observations via `Pit`.
2. Build `T_1` (unconditional copulas via Kendall‑τ MLE).
3. For each deeper tree: form conditioning vectors `z_i` from prior‑tree
   h‑functions; for each edge, build the GP+copula model (Sec. 2 pattern), set
   the `SparseGP` prior (`ConstantFunction` mean init from τ̂_MLE, `ARD` kernel,
   `basis` = inducing inputs), run EP, read back `SparseGP` posterior + evidence.
4. Predict `τ(z*)` via `SparseGP.Marginal(z*)` averaged through `g`.
5. Report total test log‑likelihood (reproduce Table 2/3 trend) and, optionally,
   the spatially varying‑τ visualization (Fig. 5).

A reduced sketch of one edge (conditional copula):

```csharp
Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
IfBlock block = Variable.If(evidence);

Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");
Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

VariableArray<Vector> z = Variable.Observed(condInputs).Named("z");
Range j = z.Range;
Variable<double> score = Variable.FunctionEvaluate(f, z[j]).Named("score");

VariableArray<Vector> uv = Variable.Observed(pairs, j).Named("uv");
uv[j] = BivariateCopula(score, new GaussianCopula());  // NEW (helper on the fitter)

block.CloseBlock();

var gp = new GaussianProcess(new ConstantFunction(meanInit), new ARD(logLen, logSig));
prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
SparseGP post = engine.Infer<SparseGP>(f);
```

### 3.6 Hyperparameter & inducing‑input optimization

Infer.NET fixes kernel hyperparameters and the `basis` for a given inference run.
The paper tunes them by maximizing the EP evidence (Sec. 3.1). Plan: an outer
optimization wrapper.

- `src/.../Copulas/Vine/EvidenceObjective.cs` — given `(logLengths, logSignal,
  basis)` builds the model, runs EP, returns `−logEvidence`.
- Optimize with a derivative‑free method first (Nelder–Mead / coordinate search)
  for robustness; the classifier tutorial already demonstrates a manual sweep
  over kernel settings comparing evidence. Initialize `basis` by k‑means/subset
  of the conditioning data (paper uses `n_0 = 20` inducing inputs).

---

## 4. EP factor — derivation detail

For observed `(u,v)` the term `t(s) = c(u, v | g(s))` is a smooth positive
function of the scalar `s = f(z)`. With cavity `N(s; m, v)` EP computes the tilted
moments:

```
Z      = ∫ N(s;m,v) t(s) ds
E[s]   = (1/Z) ∫ s   N(s;m,v) t(s) ds
E[s²]  = (1/Z) ∫ s²  N(s;m,v) t(s) ds
```

Using `Quadrature.GaussianNodesAndWeights(m, v, nodes, weights)` these become
weighted sums over nodes `sᵢ` with weights `wᵢ` (already incorporating the
Gaussian measure). The projected marginal is `N(E[s], E[s²]−E[s]²)` and the
outgoing message is `projected / cavity`. The evidence contribution is `ln Z`.

This is exactly the structure of `LogisticOp.LogisticAverageConditional` /
`XAverageConditional` (non‑conjugate likelihood, moment matching), so that file is
the concrete reference implementation to copy conventions from (proper‑message
guards, `LogEvidenceRatio`, init methods, `[Fresh]`/`[RequiredArgument]`
annotations).

Validation of the factor in isolation: with a *fixed* `score` distribution and a
known `g`, the EP message and `ln Z` must match a brute‑force fine‑grid
quadrature of the same integrals to ~1e‑6.

---

## 5. Phased roadmap

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Scaffolding: `Distributions/Copulas/` folder, `IBivariateCopula`, `GaussianCopula`, unit tests | density integrates to 1; h‑funcs in (0,1); τ↔θ round‑trips |
| 1 | `BivariateCopula` factor + `BivariateCopulaOp` (Gaussian only) | EP messages & `ln Z` match brute‑force quadrature on synthetic `(u,v,score)` |
| 2 | Single **conditional** copula end‑to‑end (one edge, known `g`) | recovers known `g(z)` like paper Fig. 4 on synthetic 3‑D data |
| 3 | Vine layer: PIT, Kendall τ, MST `T_1`, unconditional fit, log‑lik | `T_1` log‑lik matches a reference on a Gaussian‑dependent dataset |
| 4 | Deeper trees: h‑function recursion + per‑edge GP fit; `GaussianProcessVine.cs` | full GPVINE runs; test log‑lik increases with #trees (Fig. 3 trend) |
| 5 | Evidence‑based hyperparameter/inducing optimization (outer loop) | tuned model ≥ untuned on held‑out log‑lik |
| 6 | More families (Clayton/Gumbel/Frank) + Bayesian family selection via evidence | selection picks the generating family on synthetic data |
| 7 | Sampling (inverse Rosenblatt); optional Fig. 5 visualization | round‑trip sample → refit recovers dependence |

Phases 0–4 deliver the paper's core result (GPVINE with Gaussian copulas).

---

## 6. Testing strategy

Follow the repo's XUnit conventions (`test/Tests`, see root `CLAUDE.md`).

- **Copula math** (`test/Tests`, new `CopulaTests.cs`): density normalization,
  independence (τ=0 ⇒ density 1), h‑function range, τ↔θ inversion, gradient sanity.
- **Factor operator** (`OperatorTests` style): EP message + evidence vs.
  brute‑force grid quadrature; symmetry `c(u,v)=c(v,u)`; behaviour as τ→0.
- **GP integration**: on data drawn from a known `g(z)`, the inferred
  `SparseGP.Marginal` posterior mean of τ tracks the truth (Fig. 4 analogue).
- **Vine**: MST picks the heaviest edges; `T_1` has `d−1` edges; full‑vine
  test log‑likelihood beats an independence baseline and a simplified vine.
- Mark long/perf tests with the categories the build excludes (`Performance`,
  etc.) per `CLAUDE.md`.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| EP non‑convergence on sharp copula likelihoods (τ near ±1) | damping in the operator (as `LogisticOp_SJ99` does); clamp τ; `forceProper` messages |
| Quadrature inaccuracy in tails | adaptive fallback (`AdaptiveClenshawCurtis`/`ExpSinh`); increase node count; log‑sum‑exp |
| Custom factor not discovered / scheduling errors | place op in `Runtime` assembly with correct `[FactorMethod]`/`[ParameterNames]`; validate against `LogisticOp` signatures; check generated code via the Compiler's transform browser |
| Hyperparameter optimization slow (EP per evaluation) | start with coarse derivative‑free search; cache; subset inducing inputs (`n_0=20`) |
| Quadratic cost in `d` (and `n_0²·n` per edge) per the paper | matches paper complexity `O(d²·n_0²·n)`; truncate vine after `d' < d−1` trees (eq. 4 pruning) |
| C# port effort for Layer B | Layer B is closed‑form; port mechanically from the existing Python prototype in `~/git/gpvine` |

---

## 8. Build & run

Per the repo `CLAUDE.md` (macOS uses the Core configurations):

```bash
# Build core libraries (Runtime + Compiler) on macOS/Linux
dotnet build -c DebugCore Infer.sln

# Run the new example via the Examples Browser / Tutorials runner
#   (GaussianProcessVine carries [Example("Applications", ...)])

# Tests (exclude platform/long categories)
dotnet test Infer.sln -c DebugCore \
  --filter "Category!=BadTest&Category!=OpenBug&Category!=CompilerOptionsTest&Category!=Performance&Category!=Platform"
```

---

## 9. Summary of new/changed files

```
src/Runtime/Distributions/Copulas/
    ICopula.cs                  (new)  interface
    GaussianCopula.cs           (new)  eqs. 11-14, Table 1
    ClaytonCopula.cs            (new, phase 6)
    GumbelCopula.cs             (new, phase 6)
    FrankCopula.cs              (new, phase 6)
    CopulaFamily.cs             (new)  enum + factory
    Vine/Pit.cs                 (new)
    Vine/KendallTau.cs          (new)
    Vine/MaxSpanningTree.cs     (new)
    Vine/VineEdge.cs            (new)
    Vine/RegularVine.cs         (new)
    Vine/EvidenceObjective.cs   (new, phase 5)
src/Runtime/Factors/
    CopulaFactor.cs             (new)  thin BivariateCopula factor method
    BivariateCopulaOp.cs        (new)  EP operator — the core deliverable
src/Tutorials/
    GaussianProcessVine.cs      (new)  driver mirroring GaussianProcessClassifier
test/Tests/
    CopulaTests.cs              (new)
    BivariateCopulaOpTests.cs   (new)
    VineTests.cs                (new)
```

The only deep inference work is `BivariateCopulaOp.cs`; everything else is either
existing Infer.NET machinery or closed‑form application code ported from the
Python prototype.
