// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas
{
    /// <summary>
    /// The bivariate Gumbel copula (Lopez-Paz et al., 2013, Table 1) - an Archimedean family
    /// with upper-tail dependence, modelling positive dependence only.
    /// </summary>
    /// <remarks>
    /// Parametrised through Kendall's tau with <c>theta = 1 / (1 - tau)</c>, valid for
    /// <c>tau in [0, 1)</c> (theta &gt;= 1; theta = 1 is independence). As with Clayton, tau is
    /// clamped to the family's range because the GPVINE link spans (-1, 1). With
    /// <c>x = -ln u</c>, <c>y = -ln v</c>, <c>A = x^theta + y^theta</c>, <c>w = A^(1/theta)</c>,
    /// and <c>C(u,v) = exp(-w)</c>:
    /// <code>
    /// log c(u,v|tau) = -w + (theta-1)(ln x + ln y) - (ln u + ln v) + (1/theta - 2) ln A + ln(w + theta - 1)
    /// </code>
    /// Gumbel is exchangeable, so the h-functions for <c>given = 0</c> and <c>given = 1</c> are
    /// the same expression with the roles of u and v swapped.
    /// </remarks>
    [Serializable]
    public class GumbelCopula : IBivariateCopula
    {
        /// <summary>tau is clamped to <c>[0, 1 - TauEpsilon]</c> (theta in [1, large]).</summary>
        public const double TauEpsilon = 1e-6;

        /// <inheritdoc/>
        public string Name => "Gumbel";

        /// <inheritdoc/>
        public (double Min, double Max) TauRange => (0.0, 1.0);

        /// <inheritdoc/>
        public double TauToTheta(double tau)
        {
            double t = ClampTau(tau);
            return 1.0 / (1.0 - t);
        }

        /// <inheritdoc/>
        public double ThetaToTau(double theta)
        {
            double th = theta < 1.0 ? 1.0 : theta;
            return 1.0 - 1.0 / th;
        }

        /// <inheritdoc/>
        public double LogDensity(double u, double v, double tau)
        {
            double theta = TauToTheta(tau);
            // Work in log space: ln A and ln w avoid the overflow/underflow of x^theta when
            // theta is large (tau near 1), where x^theta would round to 0 or +Inf.
            double lnX = System.Math.Log(-System.Math.Log(u)); // ln x, x = -ln u
            double lnY = System.Math.Log(-System.Math.Log(v));
            double lnA = MMath.LogSumExp(theta * lnX, theta * lnY);
            double w = System.Math.Exp(lnA / theta);
            return -w
                + (theta - 1.0) * (lnX + lnY)
                - (System.Math.Log(u) + System.Math.Log(v))
                + (1.0 / theta - 2.0) * lnA
                + System.Math.Log(w + theta - 1.0);
        }

        /// <inheritdoc/>
        public double Cdf(double u, double v, double tau)
        {
            double theta = TauToTheta(tau);
            double lnX = System.Math.Log(-System.Math.Log(u));
            double lnY = System.Math.Log(-System.Math.Log(v));
            double lnA = MMath.LogSumExp(theta * lnX, theta * lnY);
            double w = System.Math.Exp(lnA / theta);
            return System.Math.Exp(-w);
        }

        /// <inheritdoc/>
        public double ConditionalCdf(double u, double v, double tau, int given)
        {
            double theta = TauToTheta(tau);
            double lnX = System.Math.Log(-System.Math.Log(u));
            double lnY = System.Math.Log(-System.Math.Log(v));
            double lnA = MMath.LogSumExp(theta * lnX, theta * lnY);
            double w = System.Math.Exp(lnA / theta);
            // given == 1: P(u|v) = dC/dv = exp(-w + (theta-1) ln y + (1/theta-1) ln A) / v
            // given == 0: P(v|u) = dC/du = exp(-w + (theta-1) ln x + (1/theta-1) ln A) / u
            double lnLeading = (given == 1) ? (theta - 1.0) * lnY : (theta - 1.0) * lnX;
            double denom = (given == 1) ? v : u;
            return System.Math.Exp(-w + lnLeading + (1.0 / theta - 1.0) * lnA) / denom;
        }

        /// <inheritdoc/>
        public double InverseConditionalCdf(double w, double x, double tau, int given)
        {
            // No closed form: the conditional CDF is monotone increasing in the unknown variable,
            // so solve ConditionalCdf(unknown | x) = w by bisection on (0, 1).
            double lo = 1e-12, hi = 1.0 - 1e-12;
            for (int it = 0; it < 60; it++)
            {
                double mid = 0.5 * (lo + hi);
                double h = (given == 1) ? ConditionalCdf(mid, x, tau, 1) : ConditionalCdf(x, mid, tau, 0);
                if (h < w) lo = mid; else hi = mid;
            }
            return 0.5 * (lo + hi);
        }

        /// <inheritdoc/>
        public Vector Sample(double tau)
        {
            // No closed-form bivariate sampler; use conditional sampling (inverse-Rosenblatt):
            // draw u uniform and a uniform conditional-CDF level, then invert P(v | u) for v.
            double u = Rand.Double();
            double p = Rand.Double();
            double v = InverseConditionalCdf(p, u, tau, 0);
            return Vector.FromArray(u, v);
        }

        private static double ClampTau(double tau)
        {
            if (tau < 0.0) return 0.0;
            if (tau > 1.0 - TauEpsilon) return 1.0 - TauEpsilon;
            return tau;
        }
    }
}
