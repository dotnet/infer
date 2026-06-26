// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Phase 1 tests for <see cref="BivariateCopulaOp"/>. Exit criterion: the EP message
    /// to <c>score</c> and the evidence <c>lnZ</c> match a brute-force fine-grid quadrature
    /// of the same integrals.
    /// </summary>
    public class BivariateCopulaOpTests
    {
        private readonly GaussianCopula copula = new GaussianCopula();
        private const int GaussianFamily = (int)CopulaFamily.Gaussian;

        public static TheoryData<double, double, double, double> Cases()
        {
            // m, v (cavity), u, v (observed pair)
            var data = new TheoryData<double, double, double, double>();
            foreach (double m in new[] { -0.6, 0.0, 0.9 })
                foreach (double cav in new[] { 0.4, 1.0 })
                    foreach (var uv in new[] { (0.7, 0.8), (0.2, 0.9), (0.5, 0.5), (0.85, 0.3) })
                        data.Add(m, cav, uv.Item1, uv.Item2);
            return data;
        }

        [Theory]
        [MemberData(nameof(Cases))]
        public void Message_MatchesBruteForce(double m, double cav, double u, double pv)
        {
            Gaussian cavity = Gaussian.FromMeanAndVariance(m, cav);
            Vector pair = Vector.FromArray(u, pv);

            Gaussian msg = BivariateCopulaOp.ScoreAverageConditional(pair, cavity, GaussianFamily, Gaussian.Uniform());
            Gaussian expected = BruteForceMessage(cavity, u, pv);

            Assert.True(System.Math.Abs(msg.Precision - expected.Precision) < 1e-3,
                $"precision: op={msg.Precision}, bf={expected.Precision}");
            Assert.True(System.Math.Abs(msg.MeanTimesPrecision - expected.MeanTimesPrecision) < 1e-3,
                $"meanTimesPrecision: op={msg.MeanTimesPrecision}, bf={expected.MeanTimesPrecision}");
        }

        [Theory]
        [MemberData(nameof(Cases))]
        public void LogAverageFactor_MatchesBruteForce(double m, double cav, double u, double pv)
        {
            Gaussian cavity = Gaussian.FromMeanAndVariance(m, cav);
            Vector pair = Vector.FromArray(u, pv);

            double logZ = BivariateCopulaOp.LogAverageFactor(pair, cavity, GaussianFamily, Gaussian.Uniform());
            double expected = System.Math.Log(BruteForceIntegral(cavity, u, pv, s => 1.0));

            Assert.True(System.Math.Abs(logZ - expected) < 1e-3, $"lnZ: op={logZ}, bf={expected}");
        }

        [Fact]
        public void Message_IsSymmetricInUV()
        {
            // The Gaussian copula is symmetric: c(u,v) = c(v,u), so the message must match.
            Gaussian cavity = Gaussian.FromMeanAndVariance(0.3, 0.7);
            Gaussian a = BivariateCopulaOp.ScoreAverageConditional(Vector.FromArray(0.8, 0.25), cavity, GaussianFamily, Gaussian.Uniform());
            Gaussian b = BivariateCopulaOp.ScoreAverageConditional(Vector.FromArray(0.25, 0.8), cavity, GaussianFamily, Gaussian.Uniform());
            Assert.True(System.Math.Abs(a.Precision - b.Precision) < 1e-9);
            Assert.True(System.Math.Abs(a.MeanTimesPrecision - b.MeanTimesPrecision) < 1e-9);
        }

        [Fact]
        public void Message_ConcordantDataPushesScorePositive()
        {
            // Strongly concordant (u,v) is evidence for positive tau, i.e. positive score.
            // The outgoing message should pull the mean upward relative to a zero-mean cavity.
            Gaussian cavity = Gaussian.FromMeanAndVariance(0.0, 1.0);
            Gaussian msg = BivariateCopulaOp.ScoreAverageConditional(Vector.FromArray(0.9, 0.92), cavity, GaussianFamily, Gaussian.Uniform());
            Gaussian post = msg * cavity;
            Assert.True(post.GetMean() > 0.0, $"posterior mean was {post.GetMean()}");

            // Discordant data pulls the other way.
            Gaussian msgDisc = BivariateCopulaOp.ScoreAverageConditional(Vector.FromArray(0.9, 0.08), cavity, GaussianFamily, Gaussian.Uniform());
            Gaussian postDisc = msgDisc * cavity;
            Assert.True(postDisc.GetMean() < 0.0, $"posterior mean was {postDisc.GetMean()}");
        }

        [Fact]
        public void Message_UniformOrPointMassCavity_IsUniform()
        {
            Vector pair = Vector.FromArray(0.7, 0.8);
            Assert.True(BivariateCopulaOp.ScoreAverageConditional(pair, Gaussian.Uniform(), GaussianFamily, Gaussian.Uniform()).IsUniform());
            Assert.True(BivariateCopulaOp.ScoreAverageConditional(pair, Gaussian.PointMass(0.5), GaussianFamily, Gaussian.Uniform()).IsUniform());
        }

        // --- Brute-force references (fine-grid trapezoid over the cavity) -------------------

        // Returns the projected/cavity Gaussian message computed from a fine grid.
        private Gaussian BruteForceMessage(Gaussian cavity, double u, double v)
        {
            double z = BruteForceIntegral(cavity, u, v, s => 1.0);
            double mean = BruteForceIntegral(cavity, u, v, s => s) / z;
            double mean2 = BruteForceIntegral(cavity, u, v, s => s * s) / z;
            Gaussian projected = Gaussian.FromMeanAndVariance(mean, mean2 - mean * mean);
            Gaussian msg = new Gaussian();
            msg.SetToRatio(projected, cavity, BivariateCopulaOp.ForceProper);
            return msg;
        }

        // int h(s) N(s;m,v) c(u,v | g(s)) ds via dense trapezoid over +-10 sigma.
        private double BruteForceIntegral(Gaussian cavity, double u, double v, Func<double, double> h)
        {
            cavity.GetMeanAndVariance(out double m, out double sd2);
            double sd = System.Math.Sqrt(sd2);
            double lo = m - 10 * sd, hi = m + 10 * sd;
            const int n = 40000;
            double step = (hi - lo) / n;
            double sum = 0.0;
            for (int i = 0; i <= n; i++)
            {
                double s = lo + i * step;
                double tau = 2.0 * MMath.NormalCdf(s) - 1.0;
                double integrand = System.Math.Exp(Gaussian.GetLogProb(s, m, sd2) + copula.LogDensity(u, v, tau)) * h(s);
                double w = (i == 0 || i == n) ? 0.5 : 1.0;
                sum += w * integrand;
            }
            return sum * step;
        }
    }
}
