// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Copulas;
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
