// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Distributions.Copulas.Vine;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Assert = Xunit.Assert;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Phase 2 tests: a single conditional bivariate copula fit end-to-end with a sparse GP
    /// latent and the <see cref="BivariateCopulaOp"/> EP factor (GPVINE, Lopez-Paz et al. 2013).
    /// </summary>
    public class GPVineTests
    {
        /// <summary>
        /// On synthetic data drawn from a known latent function g(z), the sparse-GP posterior
        /// mean of tau should track the truth (the Fig. 4 analogue of the paper).
        /// </summary>
        [Fact]
        public void SingleConditionalCopula_RecoversG()
        {
            var engine = new InferenceEngine(new ExpectationPropagation());
            engine.ShowProgress = false;
            engine.NumberOfIterations = 15;

            // --- Synthetic data: tau varies smoothly with a 1-D conditioning variable z. ---
            // True latent function g(z) = 0.6 sin(z), z ~ U[-3, 3]; (u, v) ~ Gaussian copula(tau).
            Rand.Restart(42);
            int n = 120;
            Vector[] z = new Vector[n];
            Vector[] pairs = new Vector[n];
            var copula = new GaussianCopula();
            for (int i = 0; i < n; i++)
            {
                double zi = -3.0 + 6.0 * Rand.Double();
                double tau = TrueTau(zi);
                double theta = copula.TauToTheta(tau);
                double a = Rand.Normal();
                double b = theta * a + System.Math.Sqrt(1.0 - theta * theta) * Rand.Normal();
                z[i] = Vector.FromArray(zi);
                pairs[i] = Vector.FromArray(MMath.NormalCdf(a), MMath.NormalCdf(b));
            }

            // --- One vine edge: GP prior on f, score = f(z), copula likelihood on (u,v). ---
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

            VariableArray<Vector> zVar = Variable.Observed(z).Named("z");
            Range j = zVar.Range.Named("j");
            Variable<double> score = Variable.FunctionEvaluate(f, zVar[j]).Named("score");

            VariableArray<Vector> uv = Variable.Observed(pairs, j).Named("uv");
            uv[j] = Variable.BivariateCopula(score, (int)CopulaFamily.Gaussian);

            block.CloseBlock();

            // Inducing inputs spread over the conditioning range; constant mean (tau ~ 0 init).
            int nBasis = 12;
            Vector[] basis = new Vector[nBasis];
            for (int b = 0; b < nBasis; b++)
                basis[b] = Vector.FromArray(-3.0 + 6.0 * b / (nBasis - 1));
            var gp = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(0.0));
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

            double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            Assert.False(double.IsNaN(logEvidence), "log evidence is NaN");
            SparseGP post = engine.Infer<SparseGP>(f);

            // --- Predict tau on a held-out grid and compare to the truth. ---
            int nTest = 25;
            double[] truth = new double[nTest];
            double[] pred = new double[nTest];
            for (int t = 0; t < nTest; t++)
            {
                double zt = -3.0 + 6.0 * t / (nTest - 1);
                Gaussian fpost = post.Marginal(Vector.FromArray(zt));
                truth[t] = TrueTau(zt);
                pred[t] = 2.0 * MMath.NormalCdf(fpost.GetMean()) - 1.0; // tau = g(E[f])
            }

            double corr = Correlation(truth, pred);
            double rmse = Rmse(truth, pred);
            Console.WriteLine($"logEvidence={logEvidence:g4} corr={corr:g4} rmse={rmse:g4}");
            Assert.True(corr > 0.8, $"correlation between recovered and true tau too low: {corr}");
            Assert.True(rmse < 0.25, $"RMSE between recovered and true tau too high: {rmse}");
        }

        /// <summary>
        /// The paper's central result (Sec. 4, Fig. 3): when the conditional copula genuinely
        /// depends on its conditioning variable, GPVINE (conditional deeper trees) achieves a
        /// higher held-out log-likelihood than SVINE (the simplifying-assumption baseline).
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        public void GPVine_BeatsSVine_WhenConditionalDependenceVaries()
        {
            double[][] train = ConditionalDependenceData(seed: 5, n: 300);
            double[][] test = ConditionalDependenceData(seed: 6, n: 300);

            // SVINE: deeper trees use the unconditional simplifying assumption.
            double llSvine = new RegularVine().Fit(train, nTrees: 2).LogLikelihood(test);

            // GPVINE: the T_2 conditional copula is fitted with a sparse GP + EP.
            var fitter = new GaussianProcessCopulaFitter { NumInducing = 15, NumberOfIterations = 15 };
            double llGpvine = new RegularVine().Fit(train, nTrees: 2, fitter).LogLikelihood(test);

            Console.WriteLine($"SVINE test ll={llSvine:g5}  GPVINE test ll={llGpvine:g5}");
            Assert.False(double.IsNaN(llGpvine));
            Assert.True(llGpvine > llSvine + 5.0,
                $"GPVINE did not beat SVINE on held-out data: gpvine={llGpvine}, svine={llSvine}");
        }

        /// <summary>
        /// End-to-end check that a non-Gaussian family flows through the real EP factor and
        /// compiler: a conditional Clayton copula whose tau varies with z is fitted with the
        /// sparse GP, and the recovered tau(z) tracks the truth.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        public void ClaytonConditionalCopula_FitsThroughEpFactor()
        {
            // z ~ N(0,1); tau(z) in (0,1) varies with z; (u, v) ~ Clayton copula given tau(z).
            Rand.Restart(17);
            int n = 250;
            double[] uu = new double[n], vv = new double[n];
            double[][] z = new double[n][];
            double[] trueTau = new double[n];
            for (int i = 0; i < n; i++)
            {
                double zi = Rand.Normal();
                double tau = 0.45 + 0.35 * System.Math.Sin(1.2 * zi); // in (0.1, 0.8)
                double theta = 2.0 * tau / (1.0 - tau);
                double u1 = Rand.Double();
                double p = Rand.Double();
                double u2 = System.Math.Pow(
                    System.Math.Pow(u1, -theta) * (System.Math.Pow(p, -theta / (1.0 + theta)) - 1.0) + 1.0,
                    -1.0 / theta);
                uu[i] = u1; vv[i] = u2; z[i] = new[] { zi }; trueTau[i] = tau;
            }

            // Raw (non-PIT) conditioning inputs, so use a unit length-scale (cf. the GP classifier).
            var fitter = new GaussianProcessCopulaFitter { NumInducing = 15, NumberOfIterations = 15, LogLengthScale = 0.0 };
            IConditionalCopulaPosterior post = fitter.Fit(uu, vv, z, CopulaFamily.Clayton);

            Assert.False(double.IsNaN(fitter.LastLogEvidence), "log evidence is NaN");
            double[] pred = new double[n];
            for (int i = 0; i < n; i++)
                pred[i] = post.TauAt(z[i]);
            double corr = Correlation(trueTau, pred);
            Console.WriteLine($"Clayton conditional fit: corr(tau_true, tau_pred)={corr:g4}");
            Assert.True(corr > 0.6, $"recovered Clayton tau(z) does not track the truth: corr={corr}");
        }

        /// <summary>
        /// Posterior-predictive sampling: a canonical GPVINE fitted with the sparse-GP conditional
        /// copula generates joint data that reproduces the (strong) hub dependence, with draws
        /// reflecting the GP posterior.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        public void GPVine_Sample_PosteriorPredictive_ReproducesHubDependence()
        {
            double[][] train = ConditionalDependenceData(seed: 8, n: 250);
            var fitter = new GaussianProcessCopulaFitter { NumInducing = 15, NumberOfIterations = 15 };
            var vine = new RegularVine().Fit(train, nTrees: 2, fitter, VineStructure.Canonical);

            double[][] s = vine.Sample(1000);
            Assert.Equal(1000, s.Length);
            foreach (double[] row in s)
                foreach (double val in row)
                    Assert.False(double.IsNaN(val) || double.IsInfinity(val));

            // Variables 0 and 1 (X, Y) each depend on variable 2 (Z): the hub edges are strong,
            // and posterior-predictive draws should reproduce them.
            foreach (int hub in new[] { 0, 1 })
            {
                double tauData = KendallTau.Compute(ColG(train, hub), ColG(train, 2));
                double tauSamp = KendallTau.Compute(ColG(s, hub), ColG(s, 2));
                Assert.True(System.Math.Abs(tauData - tauSamp) < 0.1,
                    $"hub ({hub},2): data tau={tauData}, sample tau={tauSamp}");
            }
        }

        /// <summary>
        /// Imputation / posterior-predictive given a partial observation: conditioning a GPVINE on
        /// the value of Z must reproduce the Z-dependent sign of the (X, Y) | Z copula in the
        /// imputed X, Y.
        /// </summary>
        [Fact]
        [Trait("Category", "Performance")]
        public void GPVine_SampleConditional_ImputesZDependentDependence()
        {
            double[][] train = ConditionalDependenceData(seed: 9, n: 300);
            var fitter = new GaussianProcessCopulaFitter { NumInducing = 15, NumberOfIterations = 15 };
            // Z is variable 2; make it the root so we can condition on it exactly.
            var vine = new RegularVine().Fit(train, nTrees: 2, fitter, VineStructure.Canonical, rootOrder: new[] { 2 });

            // rho(z) = 0.9 sin(1.5 z): positive conditional dependence at z=1, negative at z=-1.
            double[][] posZ = vine.SampleConditional(new Dictionary<int, double> { { 2, 1.0 } }, 2000);
            double[][] negZ = vine.SampleConditional(new Dictionary<int, double> { { 2, -1.0 } }, 2000);

            double tauPos = KendallTau.Compute(ColG(posZ, 0), ColG(posZ, 1));
            double tauNeg = KendallTau.Compute(ColG(negZ, 0), ColG(negZ, 1));
            Console.WriteLine($"imputed tau(X,Y | Z=1)={tauPos:g3}, tau(X,Y | Z=-1)={tauNeg:g3}");

            Assert.All(posZ, r => Assert.Equal(1.0, r[2], 9)); // Z held at the conditioned value
            Assert.True(tauPos > 0.3, $"expected positive conditional dependence at Z=1, got {tauPos}");
            Assert.True(tauNeg < -0.3, $"expected negative conditional dependence at Z=-1, got {tauNeg}");
        }

        private static double[] ColG(double[][] m, int j)
        {
            double[] c = new double[m.Length];
            for (int i = 0; i < m.Length; i++) c[i] = m[i][j];
            return c;
        }

        // Three variables (X, Y, Z): X and Y are each marginally dependent on Z (so T_1 selects
        // the hub edges X-Z, Y-Z), while the copula of (X, Y) | Z has a Kendall's tau that varies
        // with Z -- exactly the structure the simplifying assumption fails to capture.
        private static double[][] ConditionalDependenceData(int seed, int n)
        {
            Rand.Restart(seed);
            double[][] x = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double z = Rand.Normal();
                double rho = 0.9 * System.Math.Sin(1.5 * z);            // conditional correlation g(z)
                double ex = Rand.Normal();
                double ey = rho * ex + System.Math.Sqrt(1 - rho * rho) * Rand.Normal();
                double xx = 0.7 * z + 0.6 * ex;                          // marginal dependence on Z ...
                double yy = 0.7 * z + 0.6 * ey;                          // ... gives the hub structure
                x[i] = new[] { xx, yy, z };
            }
            return x;
        }

        private static double TrueTau(double z) => 0.6 * System.Math.Sin(z);

        private static double Correlation(double[] x, double[] y)
        {
            int n = x.Length;
            double mx = 0, my = 0;
            for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
            mx /= n; my /= n;
            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                sxy += (x[i] - mx) * (y[i] - my);
                sxx += (x[i] - mx) * (x[i] - mx);
                syy += (y[i] - my) * (y[i] - my);
            }
            return sxy / System.Math.Sqrt(sxx * syy);
        }

        private static double Rmse(double[] x, double[] y)
        {
            double s = 0;
            for (int i = 0; i < x.Length; i++) s += (x[i] - y[i]) * (x[i] - y[i]);
            return System.Math.Sqrt(s / x.Length);
        }
    }
}
