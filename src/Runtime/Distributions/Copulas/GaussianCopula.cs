// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas
{
    /// <summary>
    /// The bivariate Gaussian copula (Lopez-Paz et al., 2013, Appendix A) - the
    /// workhorse family of the GPVINE method.
    /// </summary>
    /// <remarks>
    /// Parametrised through Kendall's tau with <c>theta = sin(pi/2 * tau)</c>
    /// (Table 1). Implements the log density (eq. 12) and the conditional CDFs /
    /// h-functions (eqs. 13-14). With <c>a = Phi^{-1}(u)</c> and <c>b = Phi^{-1}(v)</c>:
    /// <code>
    /// log c(u,v|tau) = -0.5 * log(1 - theta^2)
    ///                  - (theta^2 (a^2 + b^2) - 2 theta a b) / (2 (1 - theta^2))
    /// </code>
    /// </remarks>
    [Serializable]
    public class GaussianCopula : IBivariateCopula
    {
        /// <summary>
        /// theta is clamped to <c>(-1 + ThetaEpsilon, 1 - ThetaEpsilon)</c> to keep
        /// <c>1 - theta^2</c> strictly positive and the density finite as tau -&gt; +-1.
        /// </summary>
        public const double ThetaEpsilon = 1e-6;

        /// <inheritdoc/>
        public string Name => "Gaussian";

        /// <inheritdoc/>
        public (double Min, double Max) TauRange => (-1.0, 1.0);

        /// <inheritdoc/>
        public double TauToTheta(double tau) => ClampTheta(System.Math.Sin(0.5 * System.Math.PI * tau));

        /// <inheritdoc/>
        public double ThetaToTau(double theta) => (2.0 / System.Math.PI) * System.Math.Asin(ClampTheta(theta));

        /// <inheritdoc/>
        public double LogDensity(double u, double v, double tau)
        {
            double theta = TauToTheta(tau);
            double a = MMath.NormalCdfInv(u);
            double b = MMath.NormalCdfInv(v);
            double oneMinusT2 = 1.0 - theta * theta;
            double quad = (theta * theta * (a * a + b * b) - 2.0 * theta * a * b) / (2.0 * oneMinusT2);
            return -0.5 * System.Math.Log(oneMinusT2) - quad;
        }

        /// <inheritdoc/>
        public double Cdf(double u, double v, double tau)
        {
            // C(u, v | theta) = Phi_2(Phi^{-1}(u), Phi^{-1}(v) | theta), the bivariate normal CDF.
            double theta = TauToTheta(tau);
            return MMath.NormalCdf(MMath.NormalCdfInv(u), MMath.NormalCdfInv(v), theta);
        }

        /// <inheritdoc/>
        public double ConditionalCdf(double u, double v, double tau, int given)
        {
            double theta = TauToTheta(tau);
            double a = MMath.NormalCdfInv(u);
            double b = MMath.NormalCdfInv(v);
            double denom = System.Math.Sqrt(1.0 - theta * theta);
            // given == 1: P(u | v) = dC/dv (eq. 14); given == 0: P(v | u) = dC/du (eq. 13).
            double z = (given == 1) ? (a - theta * b) / denom : (b - theta * a) / denom;
            return MMath.NormalCdf(z);
        }

        /// <inheritdoc/>
        public double InverseConditionalCdf(double w, double x, double tau, int given)
        {
            // P(u|v) = Phi((Phi^{-1}(u) - theta Phi^{-1}(v))/sqrt(1-theta^2)) = w
            // => Phi^{-1}(u) = theta Phi^{-1}(x) + sqrt(1-theta^2) Phi^{-1}(w).
            // The Gaussian copula is exchangeable, so given 0 and 1 share this form.
            double theta = TauToTheta(tau);
            double a = MMath.NormalCdfInv(x);
            double score = theta * a + System.Math.Sqrt(1.0 - theta * theta) * MMath.NormalCdfInv(w);
            return MMath.NormalCdf(score);
        }

        /// <inheritdoc/>
        public Vector Sample(double tau)
        {
            // Sample (a, b) from a bivariate standard normal with correlation theta,
            // then map back to the unit square via the standard normal CDF.
            double theta = TauToTheta(tau);
            double a = Rand.Normal();
            double b = theta * a + System.Math.Sqrt(1.0 - theta * theta) * Rand.Normal();
            return Vector.FromArray(MMath.NormalCdf(a), MMath.NormalCdf(b));
        }

        private static double ClampTheta(double theta)
        {
            if (theta < -1.0 + ThetaEpsilon) return -1.0 + ThetaEpsilon;
            if (theta > 1.0 - ThetaEpsilon) return 1.0 - ThetaEpsilon;
            return theta;
        }
    }
}
