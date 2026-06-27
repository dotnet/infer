// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Copulas
{
    /// <summary>
    /// A single conditional bivariate copula density <c>c(u, v | tau)</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Implements the copula building block of Lopez-Paz, Hernandez-Lobato and
    /// Ghahramani, <i>Gaussian Process Vine Copulas for Multivariate Dependence</i>
    /// (ICML 2013). Every family is parametrised through Kendall's rank correlation
    /// <c>tau</c> so that a single latent function <c>g</c> can produce a valid
    /// parameter for any family (Sec. 3, Table 1). Concrete implementations convert
    /// <c>tau</c> to their native parameter <c>theta</c> internally.
    /// </para>
    /// <para>
    /// These are pure numeric classes with no dependency on the inference compiler:
    /// they are consumed both by the EP message operator (which moment-matches the
    /// copula likelihood) and by the regular-vine application layer.
    /// </para>
    /// </remarks>
    public interface IBivariateCopula
    {
        /// <summary>
        /// Human-readable family name (e.g. <c>"Gaussian"</c>).
        /// </summary>
        string Name { get; }

        /// <summary>
        /// The valid range <c>(min, max)</c> of Kendall's tau for this family.
        /// </summary>
        (double Min, double Max) TauRange { get; }

        /// <summary>
        /// Maps Kendall's tau to the family's native parameter theta (Table 1).
        /// </summary>
        double TauToTheta(double tau);

        /// <summary>
        /// Maps the family's native parameter theta back to Kendall's tau (Table 1).
        /// </summary>
        double ThetaToTau(double theta);

        /// <summary>
        /// Log copula density <c>log c(u, v | tau)</c> for a single observation
        /// (eq. 12 for the Gaussian family).
        /// </summary>
        /// <param name="u">First pseudo-observation in (0, 1).</param>
        /// <param name="v">Second pseudo-observation in (0, 1).</param>
        /// <param name="tau">Kendall's tau in <see cref="TauRange"/>.</param>
        double LogDensity(double u, double v, double tau);

        /// <summary>
        /// The conditional CDF (vine "h-function"), used to generate the
        /// pseudo-observations that condition the deeper trees of a vine (eq. 5).
        /// </summary>
        /// <param name="u">First pseudo-observation in (0, 1).</param>
        /// <param name="v">Second pseudo-observation in (0, 1).</param>
        /// <param name="tau">Kendall's tau in <see cref="TauRange"/>.</param>
        /// <param name="given">
        /// Conditioning variable: <c>1</c> returns <c>P(u | v) = dC/dv</c> (eq. 14);
        /// <c>0</c> returns <c>P(v | u) = dC/du</c> (eq. 13).
        /// </param>
        /// <returns>A conditional probability in (0, 1).</returns>
        double HFunction(double u, double v, double tau, int given);

        /// <summary>
        /// The inverse of <see cref="HFunction"/>: given a uniform <paramref name="w"/> and the
        /// known conditioning value <paramref name="x"/>, returns the variable value that maps to
        /// <paramref name="w"/> under the conditional CDF. This is the building block of the
        /// inverse-Rosenblatt transform used to sample from a vine.
        /// </summary>
        /// <param name="w">A uniform value in (0, 1) (the conditional-CDF level).</param>
        /// <param name="x">The known conditioning value in (0, 1).</param>
        /// <param name="tau">Kendall's tau in <see cref="TauRange"/>.</param>
        /// <param name="given">
        /// Matches <see cref="HFunction"/>: with <c>given = 1</c>, <paramref name="x"/> is v and the
        /// result is the u solving <c>P(u | v) = w</c>; with <c>given = 0</c>, <paramref name="x"/> is
        /// u and the result is the v solving <c>P(v | u) = w</c>.
        /// </param>
        /// <returns>The recovered variable value in (0, 1).</returns>
        double InverseHFunction(double w, double x, double tau, int given);
    }
}
