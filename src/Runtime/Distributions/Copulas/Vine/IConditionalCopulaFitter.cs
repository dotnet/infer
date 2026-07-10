// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// A fitted conditional copula: maps a conditioning vector <c>z</c> to Kendall's tau,
    /// <c>tau = g(z)</c> (Lopez-Paz et al., 2013, Sec. 3).
    /// </summary>
    public interface IConditionalCopulaPosterior
    {
        /// <summary>Predicted Kendall's tau at a single conditioning vector <paramref name="z"/>.</summary>
        double TauAt(double[] z);

        /// <summary>
        /// A Kendall's tau drawn from the posterior at <paramref name="z"/> (a posterior-predictive
        /// draw that reflects the latent function's uncertainty), rather than the posterior mean.
        /// </summary>
        double SampleTau(double[] z);
    }

    /// <summary>
    /// Fits a single conditional bivariate copula <c>c(u, v | z)</c> for a vine edge,
    /// returning a posterior that can predict Kendall's tau at new conditioning vectors.
    /// </summary>
    /// <remarks>
    /// The fit itself (sparse GP + Expectation Propagation, Sec. 3) requires the inference
    /// engine, which lives above the Runtime assembly. This interface lets the vine recursion
    /// stay in Runtime while the EP-based implementation is injected from the modelling layer.
    /// </remarks>
    public interface IConditionalCopulaFitter
    {
        /// <summary>
        /// Fits <c>c(u, v | z)</c> to paired pseudo-observations conditioned on <paramref name="z"/>.
        /// </summary>
        /// <param name="u">First pseudo-observation series in (0, 1), length n.</param>
        /// <param name="v">Second pseudo-observation series in (0, 1), length n.</param>
        /// <param name="z">Conditioning matrix, shape [n][m] (m = size of the conditioning set).</param>
        /// <param name="copula">The bivariate copula family evaluator.</param>
        /// <returns>A fitted posterior over Kendall's tau as a function of <c>z</c>.</returns>
        IConditionalCopulaPosterior Fit(double[] u, double[] v, double[][] z, IBivariateCopula copula);
    }

    /// <summary>
    /// A conditional-copula posterior backed by a sparse GP over the latent function <c>f</c>,
    /// with <c>tau = g(z) = 2*Phi(E[f(z)]) - 1</c> (the link of Lopez-Paz et al., 2013, Sec. 3).
    /// </summary>
    public sealed class SparseGPCopulaPosterior : IConditionalCopulaPosterior
    {
        private readonly SparseGP gp;

        /// <summary>Wraps a fitted sparse-GP posterior over the latent function.</summary>
        public SparseGPCopulaPosterior(SparseGP gp)
        {
            this.gp = gp;
        }

        /// <summary>The underlying fitted sparse-GP posterior over <c>f</c>.</summary>
        public SparseGP Posterior => gp;

        /// <inheritdoc/>
        public double TauAt(double[] z)
        {
            double meanF = gp.Marginal(Vector.FromArray(z)).GetMean();
            return 2.0 * MMath.NormalCdf(meanF) - 1.0;
        }

        /// <inheritdoc/>
        public double SampleTau(double[] z)
        {
            double f = gp.Marginal(Vector.FromArray(z)).Sample();
            return 2.0 * MMath.NormalCdf(f) - 1.0;
        }
    }
}
