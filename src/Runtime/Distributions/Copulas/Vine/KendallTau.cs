// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// Empirical Kendall's rank correlation coefficient tau between two paired series.
    /// </summary>
    /// <remarks>
    /// Kendall's tau measures the difference between the number of concordant and discordant
    /// pairs, normalized by the total number of pairs. In GPVINE (Lopez-Paz et al., 2013) it
    /// serves two roles: the absolute value <c>|tau|</c> is the edge weight for maximum
    /// spanning tree selection (Sec. 2.1), and the unconditional MLE is the fitted parameter
    /// of each first-tree copula (Sec. 4).
    /// </remarks>
    public static class KendallTau
    {
        /// <summary>
        /// Computes Kendall's tau-b between two equal-length series.
        /// </summary>
        /// <param name="a">First series.</param>
        /// <param name="b">Second series, paired with <paramref name="a"/>.</param>
        /// <returns>
        /// Kendall's tau in [-1, 1], or 0 if it is undefined (constant input / fewer than two
        /// points). The tau-b form corrects for ties; for tie-free continuous data it equals
        /// the simple (concordant - discordant) / (n(n-1)/2).
        /// </returns>
        public static double Compute(double[] a, double[] b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (a.Length != b.Length) throw new ArgumentException("Series must have equal length.");
            int n = a.Length;
            if (n < 2) return 0.0;

            long concordant = 0, discordant = 0, tiesA = 0, tiesB = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double da = a[i] - a[j];
                    double db = b[i] - b[j];
                    double prod = da * db;
                    if (prod > 0) concordant++;
                    else if (prod < 0) discordant++;
                    else
                    {
                        // At least one of the pairs is tied.
                        if (da == 0) tiesA++;
                        if (db == 0) tiesB++;
                    }
                }
            }

            long n0 = (long)n * (n - 1) / 2;
            double denom = System.Math.Sqrt((double)(n0 - tiesA) * (n0 - tiesB));
            if (denom <= 0) return 0.0;
            return (concordant - discordant) / denom;
        }
    }
}
