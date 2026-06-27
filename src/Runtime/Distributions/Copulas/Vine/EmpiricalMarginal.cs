// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// An empirical marginal: the inverse of the rank-based PIT (<see cref="Pit"/>). It maps a
    /// uniform value back to the data scale via the empirical quantile function, so a vine fitted
    /// on PIT pseudo-observations can generate samples in the original units.
    /// </summary>
    /// <remarks>
    /// Quantiles are interpolated at the plotting positions <c>k/(n+1)</c> used by the forward
    /// transform <c>u = rank/(n+1)</c>, so <c>Quantile</c> is a consistent inverse of the PIT.
    /// </remarks>
    public class EmpiricalMarginal
    {
        private readonly double[] sorted;

        /// <summary>Builds an empirical marginal from a column of raw training observations.</summary>
        public EmpiricalMarginal(double[] column)
        {
            if (column == null) throw new ArgumentNullException(nameof(column));
            if (column.Length == 0) throw new ArgumentException("No observations.", nameof(column));
            sorted = (double[])column.Clone();
            Array.Sort(sorted);
        }

        /// <summary>The number of training observations.</summary>
        public int Count => sorted.Length;

        /// <summary>
        /// The empirical quantile at probability <paramref name="p"/> in (0, 1): the inverse of
        /// <c>u = rank/(n+1)</c>, linearly interpolated and clamped to the observed range.
        /// </summary>
        public double Quantile(double p)
        {
            int n = sorted.Length;
            double pos = p * (n + 1); // 1-based fractional rank
            if (pos <= 1.0) return sorted[0];
            if (pos >= n) return sorted[n - 1];
            int lo = (int)System.Math.Floor(pos); // 1-based lower rank
            double frac = pos - lo;
            return sorted[lo - 1] + frac * (sorted[lo] - sorted[lo - 1]);
        }
    }
}
