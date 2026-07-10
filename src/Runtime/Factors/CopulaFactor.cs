// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions.Copulas;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// The bivariate-copula likelihood factor of the GPVINE method (Lopez-Paz,
    /// Hernandez-Lobato and Ghahramani, ICML 2013).
    /// </summary>
    /// <remarks>
    /// This is only a thin hook for the inference compiler: it names the factor that
    /// <see cref="BivariateCopulaOp"/> attaches its Expectation Propagation messages to. The
    /// actual per-family sampling lives on the copula classes (<see cref="IBivariateCopula.Sample"/>),
    /// and the EP messages never call this method - they moment-match the copula likelihood
    /// against the Gaussian latent directly.
    /// </remarks>
    public static class CopulaFactor
    {
        /// <summary>
        /// Generates a bivariate copula pseudo-observation pair (u, v) whose dependence is
        /// governed by a latent score, via the paper's link tau = g(score) = 2*Phi(score) - 1
        /// (Sec. 3).
        /// </summary>
        /// <param name="score">Latent function value f(z).</param>
        /// <param name="copula">The copula family evaluator.</param>
        /// <returns>A length-2 vector (u, v) of pseudo-observations in (0, 1).</returns>
        [Stochastic]
        [ParameterNames("pair", "score", "copula")]
        public static Vector BivariateCopula(double score, IBivariateCopula copula)
        {
            double tau = 2.0 * MMath.NormalCdf(score) - 1.0;
            return copula.Sample(tau);
        }
    }
}
