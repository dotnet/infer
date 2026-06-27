// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas
{
    /// <summary>
    /// The bivariate Clayton copula (Lopez-Paz et al., 2013, Table 1) - an Archimedean family
    /// with lower-tail dependence, modelling positive dependence only.
    /// </summary>
    /// <remarks>
    /// Parametrised through Kendall's tau with <c>theta = 2 tau / (1 - tau)</c>, valid for
    /// <c>tau in (0, 1)</c>. Because the GPVINE link <c>tau = 2*Phi(f) - 1</c> spans (-1, 1),
    /// tau is clamped to the family's range, so latent values implying negative dependence map
    /// to (near-)independence. With <c>C(u,v) = (u^-theta + v^-theta - 1)^(-1/theta)</c>:
    /// <code>
    /// log c(u,v|tau) = log(1+theta) - (theta+1)(ln u + ln v) - (1/theta + 2) ln(u^-theta + v^-theta - 1)
    /// </code>
    /// Clayton is exchangeable, so the h-functions for <c>given = 0</c> and <c>given = 1</c> are
    /// the same expression with the roles of u and v swapped.
    /// </remarks>
    [Serializable]
    public class ClaytonCopula : IBivariateCopula
    {
        /// <summary>tau is clamped to <c>[TauEpsilon, 1 - TauEpsilon]</c> to keep theta finite and positive.</summary>
        public const double TauEpsilon = 1e-6;

        // Below this theta the copula is numerically indistinguishable from independence.
        private const double IndependenceTheta = 1e-8;

        /// <inheritdoc/>
        public string Name => "Clayton";

        /// <inheritdoc/>
        public (double Min, double Max) TauRange => (0.0, 1.0);

        /// <inheritdoc/>
        public double TauToTheta(double tau)
        {
            double t = ClampTau(tau);
            return 2.0 * t / (1.0 - t);
        }

        /// <inheritdoc/>
        public double ThetaToTau(double theta)
        {
            double th = theta < 0.0 ? 0.0 : theta;
            return th / (th + 2.0);
        }

        /// <inheritdoc/>
        public double LogDensity(double u, double v, double tau)
        {
            double theta = TauToTheta(tau);
            if (theta < IndependenceTheta) return 0.0;
            double s = System.Math.Pow(u, -theta) + System.Math.Pow(v, -theta) - 1.0;
            return System.Math.Log(1.0 + theta)
                - (theta + 1.0) * (System.Math.Log(u) + System.Math.Log(v))
                - (1.0 / theta + 2.0) * System.Math.Log(s);
        }

        /// <inheritdoc/>
        public double HFunction(double u, double v, double tau, int given)
        {
            double theta = TauToTheta(tau);
            if (theta < IndependenceTheta) return given == 1 ? u : v; // independence: P(u|v)=u, P(v|u)=v
            double s = System.Math.Pow(u, -theta) + System.Math.Pow(v, -theta) - 1.0;
            double sPow = System.Math.Pow(s, -1.0 / theta - 1.0);
            // given == 1: P(u|v) = dC/dv = v^(-theta-1) s^(-1/theta-1)
            // given == 0: P(v|u) = dC/du = u^(-theta-1) s^(-1/theta-1)
            double leading = (given == 1)
                ? System.Math.Pow(v, -theta - 1.0)
                : System.Math.Pow(u, -theta - 1.0);
            return leading * sPow;
        }

        private static double ClampTau(double tau)
        {
            if (tau < TauEpsilon) return TauEpsilon;
            if (tau > 1.0 - TauEpsilon) return 1.0 - TauEpsilon;
            return tau;
        }
    }
}
