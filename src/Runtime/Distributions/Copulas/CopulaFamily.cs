// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas
{
    /// <summary>
    /// The bivariate copula families of Lopez-Paz et al. (2013), Table 1.
    /// </summary>
    /// <remarks>
    /// The enum is passed as an observed integer constant to the copula EP factor
    /// so that a single factor / operator can serve every family. Only
    /// <see cref="Gaussian"/> ships with a full evaluation implementation in the
    /// initial phase; the remaining members are placeholders for later phases.
    /// </remarks>
    public enum CopulaFamily
    {
        /// <summary>Bivariate Gaussian copula (Appendix A). <c>theta = sin(pi/2 tau)</c>.</summary>
        Gaussian = 0,

        /// <summary>Clayton copula. <c>theta = 2 tau / (1 - tau)</c>. (Not yet implemented.)</summary>
        Clayton = 1,

        /// <summary>Gumbel copula. <c>theta = 1 / (1 - tau)</c>. (Not yet implemented.)</summary>
        Gumbel = 2,

        /// <summary>Frank copula. theta via numerical inversion of the Debye function. (Not yet implemented.)</summary>
        Frank = 3,
    }

    /// <summary>
    /// Factory that maps a <see cref="CopulaFamily"/> to its
    /// <see cref="IBivariateCopula"/> implementation.
    /// </summary>
    public static class CopulaFactory
    {
        /// <summary>
        /// Creates the <see cref="IBivariateCopula"/> for the given family.
        /// </summary>
        /// <param name="family">The copula family.</param>
        /// <returns>A stateless copula evaluator for that family.</returns>
        /// <exception cref="NotImplementedException">
        /// Thrown for families not yet implemented in the current phase.
        /// </exception>
        public static IBivariateCopula Create(CopulaFamily family)
        {
            switch (family)
            {
                case CopulaFamily.Gaussian:
                    return new GaussianCopula();
                default:
                    throw new NotImplementedException($"Copula family '{family}' is not yet implemented.");
            }
        }
    }
}
