// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Math;
using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Phase 0 unit tests for the bivariate copula math (no inference compiler).
    /// Exit criteria: density integrates to 1; tau=0 =&gt; density 1; h-functions in
    /// (0,1) and equal to the numerical conditional CDFs; tau&lt;-&gt;theta round-trips.
    /// </summary>
    public class CopulaTests
    {
        private readonly GaussianCopula gaussian = new GaussianCopula();

        [Fact]
        public void GaussianCopula_TauThetaRoundTrips()
        {
            for (double tau = -0.9; tau <= 0.9 + 1e-9; tau += 0.1)
            {
                double theta = gaussian.TauToTheta(tau);
                double tauBack = gaussian.ThetaToTau(theta);
                Assert.True(System.Math.Abs(tau - tauBack) < 1e-9,
                    $"tau round-trip failed: tau={tau}, theta={theta}, back={tauBack}");
            }

            // Known anchors: tau=0 -> theta=0; tau=1 -> theta=1 (clamped just below).
            Assert.Equal(0.0, gaussian.TauToTheta(0.0), 12);
            Assert.True(gaussian.TauToTheta(1.0) <= 1.0 && gaussian.TauToTheta(1.0) > 0.99);
            Assert.True(gaussian.TauToTheta(-1.0) >= -1.0 && gaussian.TauToTheta(-1.0) < -0.99);
        }

        [Fact]
        public void GaussianCopula_IndependenceHasUnitDensity()
        {
            // tau = 0 => theta = 0 => c(u,v) = 1 everywhere, i.e. log density 0.
            foreach (double u in new[] { 0.1, 0.5, 0.9 })
            {
                foreach (double v in new[] { 0.2, 0.5, 0.8 })
                {
                    Assert.Equal(0.0, gaussian.LogDensity(u, v, 0.0), 12);
                }
            }
        }

        [Fact]
        public void GaussianCopula_DensityIsSymmetric()
        {
            // c(u, v) = c(v, u) for the Gaussian copula.
            foreach (double tau in new[] { -0.6, -0.2, 0.3, 0.7 })
            {
                Assert.Equal(gaussian.LogDensity(0.3, 0.8, tau), gaussian.LogDensity(0.8, 0.3, tau), 12);
                Assert.Equal(gaussian.LogDensity(0.1, 0.6, tau), gaussian.LogDensity(0.6, 0.1, tau), 12);
            }
        }

        [Fact]
        public void GaussianCopula_DensityIntegratesToOne()
        {
            // Midpoint rule over (0,1)^2; the midpoint grid avoids the singular edges.
            // The copula density integrates to 1 by construction.
            foreach (double tau in new[] { -0.5, 0.0, 0.4 })
            {
                const int n = 800;
                double h = 1.0 / n;
                double integral = 0.0;
                for (int i = 0; i < n; i++)
                {
                    double u = (i + 0.5) * h;
                    for (int j = 0; j < n; j++)
                    {
                        double v = (j + 0.5) * h;
                        integral += System.Math.Exp(gaussian.LogDensity(u, v, tau));
                    }
                }
                integral *= h * h;
                Assert.True(System.Math.Abs(integral - 1.0) < 5e-3,
                    $"density integral = {integral} for tau={tau}, expected ~1");
            }
        }

        [Fact]
        public void GaussianCopula_HFunctionInUnitInterval()
        {
            foreach (double tau in new[] { -0.7, 0.0, 0.5 })
            {
                foreach (double u in new[] { 0.05, 0.5, 0.95 })
                {
                    foreach (double v in new[] { 0.1, 0.5, 0.9 })
                    {
                        double h0 = gaussian.HFunction(u, v, tau, 0);
                        double h1 = gaussian.HFunction(u, v, tau, 1);
                        Assert.True(h0 > 0.0 && h0 < 1.0, $"h(given=0)={h0} out of (0,1)");
                        Assert.True(h1 > 0.0 && h1 < 1.0, $"h(given=1)={h1} out of (0,1)");
                    }
                }
            }
        }

        [Fact]
        public void GaussianCopula_HFunctionMatchesNumericalConditionalCdf()
        {
            // C(u,v|theta) = Phi_2(Phi^{-1}(u), Phi^{-1}(v) | theta) = MMath.NormalCdf(a, b, theta).
            // h(given=0) = dC/du ; h(given=1) = dC/dv. Compare to a central difference.
            const double eps = 1e-5;
            foreach (double tau in new[] { -0.6, 0.3, 0.7 })
            {
                double theta = gaussian.TauToTheta(tau);
                foreach (double u in new[] { 0.3, 0.5, 0.7 })
                {
                    foreach (double v in new[] { 0.25, 0.55, 0.8 })
                    {
                        double dCdu = (Cdf(u + eps, v, theta) - Cdf(u - eps, v, theta)) / (2 * eps);
                        double dCdv = (Cdf(u, v + eps, theta) - Cdf(u, v - eps, theta)) / (2 * eps);
                        Assert.True(System.Math.Abs(dCdu - gaussian.HFunction(u, v, tau, 0)) < 1e-4,
                            $"dC/du mismatch at u={u},v={v},tau={tau}");
                        Assert.True(System.Math.Abs(dCdv - gaussian.HFunction(u, v, tau, 1)) < 1e-4,
                            $"dC/dv mismatch at u={u},v={v},tau={tau}");
                    }
                }
            }
        }

        [Fact]
        public void CopulaFactory_CreatesImplementedFamilies()
        {
            Assert.Equal("Gaussian", CopulaFactory.Create(CopulaFamily.Gaussian).Name);
            Assert.Equal("Clayton", CopulaFactory.Create(CopulaFamily.Clayton).Name);
            Assert.Equal("Gumbel", CopulaFactory.Create(CopulaFamily.Gumbel).Name);
            Assert.Throws<NotImplementedException>(() => CopulaFactory.Create(CopulaFamily.Frank));
        }

        // --- Clayton / Gumbel (Archimedean, positive-dependence) -------------------------------

        [Theory]
        [InlineData(CopulaFamily.Clayton)]
        [InlineData(CopulaFamily.Gumbel)]
        public void Archimedean_TauThetaRoundTrips(CopulaFamily family)
        {
            IBivariateCopula c = CopulaFactory.Create(family);
            for (double tau = 0.05; tau <= 0.9 + 1e-9; tau += 0.1)
            {
                double theta = c.TauToTheta(tau);
                double back = c.ThetaToTau(theta);
                Assert.True(System.Math.Abs(tau - back) < 1e-9,
                    $"{family} tau round-trip failed: tau={tau}, theta={theta}, back={back}");
            }
        }

        [Theory]
        [InlineData(CopulaFamily.Clayton)]
        [InlineData(CopulaFamily.Gumbel)]
        public void Archimedean_IndependenceHasUnitDensity(CopulaFamily family)
        {
            // tau = 0 is the independence boundary (Clayton theta->0, Gumbel theta=1).
            IBivariateCopula c = CopulaFactory.Create(family);
            foreach (double u in new[] { 0.2, 0.5, 0.8 })
                foreach (double v in new[] { 0.3, 0.5, 0.7 })
                    Assert.True(System.Math.Abs(c.LogDensity(u, v, 0.0)) < 1e-3,
                        $"{family} not ~independent at tau=0: logc({u},{v})={c.LogDensity(u, v, 0.0)}");
        }

        [Theory]
        [InlineData(CopulaFamily.Clayton)]
        [InlineData(CopulaFamily.Gumbel)]
        public void Archimedean_DensityIntegratesToOne(CopulaFamily family)
        {
            IBivariateCopula c = CopulaFactory.Create(family);
            foreach (double tau in new[] { 0.2, 0.4 })
            {
                const int n = 1200;
                double h = 1.0 / n;
                double integral = 0.0;
                for (int i = 0; i < n; i++)
                {
                    double u = (i + 0.5) * h;
                    for (int j = 0; j < n; j++)
                    {
                        double v = (j + 0.5) * h;
                        integral += System.Math.Exp(c.LogDensity(u, v, tau));
                    }
                }
                integral *= h * h;
                Assert.True(System.Math.Abs(integral - 1.0) < 2e-2,
                    $"{family} density integral = {integral} for tau={tau}, expected ~1");
            }
        }

        [Theory]
        [InlineData(CopulaFamily.Clayton)]
        [InlineData(CopulaFamily.Gumbel)]
        public void Archimedean_HFunctionInUnitIntervalAndMatchesNumericalCdf(CopulaFamily family)
        {
            IBivariateCopula c = CopulaFactory.Create(family);
            const double eps = 1e-6;
            foreach (double tau in new[] { 0.2, 0.5, 0.75 })
            {
                double theta = c.TauToTheta(tau);
                foreach (double u in new[] { 0.3, 0.5, 0.7 })
                {
                    foreach (double v in new[] { 0.25, 0.55, 0.8 })
                    {
                        double h0 = c.HFunction(u, v, tau, 0);
                        double h1 = c.HFunction(u, v, tau, 1);
                        Assert.True(h0 > 0.0 && h0 < 1.0, $"{family} h(given=0)={h0} out of (0,1)");
                        Assert.True(h1 > 0.0 && h1 < 1.0, $"{family} h(given=1)={h1} out of (0,1)");

                        double dCdu = (ArchimedeanCdf(family, u + eps, v, theta) - ArchimedeanCdf(family, u - eps, v, theta)) / (2 * eps);
                        double dCdv = (ArchimedeanCdf(family, u, v + eps, theta) - ArchimedeanCdf(family, u, v - eps, theta)) / (2 * eps);
                        Assert.True(System.Math.Abs(dCdu - h0) < 1e-4, $"{family} dC/du mismatch at u={u},v={v},tau={tau}");
                        Assert.True(System.Math.Abs(dCdv - h1) < 1e-4, $"{family} dC/dv mismatch at u={u},v={v},tau={tau}");
                    }
                }
            }
        }

        [Theory]
        [InlineData(CopulaFamily.Gaussian)]
        [InlineData(CopulaFamily.Clayton)]
        [InlineData(CopulaFamily.Gumbel)]
        public void InverseHFunction_RoundTrips(CopulaFamily family)
        {
            // InverseHFunction must invert HFunction: recover the unknown variable from its
            // conditional-CDF level. This is the core of inverse-Rosenblatt sampling.
            IBivariateCopula c = CopulaFactory.Create(family);
            foreach (double tau in new[] { 0.2, 0.5, 0.75 })
            {
                foreach (double u in new[] { 0.2, 0.5, 0.8 })
                {
                    foreach (double v in new[] { 0.3, 0.6, 0.85 })
                    {
                        // given = 1: unknown is u, known is v.
                        double w1 = c.HFunction(u, v, tau, 1);
                        double uBack = c.InverseHFunction(w1, v, tau, 1);
                        Assert.True(System.Math.Abs(uBack - u) < 1e-6,
                            $"{family} given=1: u={u}, recovered={uBack}");

                        // given = 0: unknown is v, known is u.
                        double w0 = c.HFunction(u, v, tau, 0);
                        double vBack = c.InverseHFunction(w0, u, tau, 0);
                        Assert.True(System.Math.Abs(vBack - v) < 1e-6,
                            $"{family} given=0: v={v}, recovered={vBack}");
                    }
                }
            }
        }

        // Gaussian copula CDF C(u,v|theta) via the bivariate normal CDF.
        private static double Cdf(double u, double v, double theta)
        {
            double a = MMath.NormalCdfInv(u);
            double b = MMath.NormalCdfInv(v);
            return MMath.NormalCdf(a, b, theta);
        }

        // Closed-form CDFs for the Archimedean families, used to check the h-functions.
        private static double ArchimedeanCdf(CopulaFamily family, double u, double v, double theta)
        {
            switch (family)
            {
                case CopulaFamily.Clayton:
                    return System.Math.Pow(System.Math.Pow(u, -theta) + System.Math.Pow(v, -theta) - 1.0, -1.0 / theta);
                case CopulaFamily.Gumbel:
                    double a = System.Math.Pow(-System.Math.Log(u), theta) + System.Math.Pow(-System.Math.Log(v), theta);
                    return System.Math.Exp(-System.Math.Pow(a, 1.0 / theta));
                default:
                    throw new ArgumentOutOfRangeException(nameof(family));
            }
        }
    }
}
