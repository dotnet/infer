// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// The empirical Probability Integral Transform (PIT): maps raw observations to uniform
    /// pseudo-observations in (0, 1), column by column.
    /// </summary>
    /// <remarks>
    /// A copula models dependence after the marginals have been removed (Lopez-Paz et al.,
    /// 2013, Sec. 2). Each marginal column is mapped to a uniform variable via its empirical
    /// CDF. The rank-based estimate <c>u = rank / (n + 1)</c> keeps values strictly inside
    /// (0, 1), avoiding the +-inf produced by <c>Phi^{-1}(0)</c> / <c>Phi^{-1}(1)</c> in
    /// Gaussian-copula evaluations.
    /// </remarks>
    public static class Pit
    {
        /// <summary>
        /// Transforms a single column of raw observations to pseudo-observations in (0, 1)
        /// using average ranks divided by <c>n + 1</c>.
        /// </summary>
        /// <param name="column">Raw, real-valued observations.</param>
        /// <returns>A new array of the same length with entries in (0, 1).</returns>
        public static double[] Transform(double[] column)
        {
            if (column == null) throw new ArgumentNullException(nameof(column));
            int n = column.Length;
            double[] ranks = AverageRanks(column);
            double[] u = new double[n];
            for (int i = 0; i < n; i++)
                u[i] = ranks[i] / (n + 1);
            return u;
        }

        /// <summary>
        /// Transforms a data matrix (rows = observations, columns = variables) to
        /// pseudo-observations in (0, 1), applying <see cref="Transform(double[])"/> per column.
        /// </summary>
        /// <param name="x">Raw data of shape [n][d].</param>
        /// <returns>A new [n][d] matrix of pseudo-observations in (0, 1).</returns>
        public static double[][] Transform(double[][] x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            int n = x.Length;
            if (n == 0) return new double[0][];
            int d = x[0].Length;
            double[][] u = new double[n][];
            for (int i = 0; i < n; i++)
                u[i] = new double[d];
            double[] col = new double[n];
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < n; i++)
                    col[i] = x[i][j];
                double[] uj = Transform(col);
                for (int i = 0; i < n; i++)
                    u[i][j] = uj[i];
            }
            return u;
        }

        /// <summary>
        /// Clamps a value into <c>[eps, 1 - eps]</c> for numerically safe inverse-CDFs.
        /// </summary>
        public static double ClampUnit(double u, double eps = 1e-6)
        {
            if (u < eps) return eps;
            if (u > 1.0 - eps) return 1.0 - eps;
            return u;
        }

        /// <summary>
        /// Average (fractional) ranks of <paramref name="values"/>, 1-based, with ties
        /// receiving the mean of the ranks they span (matching the standard "average" method).
        /// </summary>
        private static double[] AverageRanks(double[] values)
        {
            int n = values.Length;
            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            Array.Sort(order, (a, b) => values[a].CompareTo(values[b]));

            double[] ranks = new double[n];
            int k = 0;
            while (k < n)
            {
                int m = k + 1;
                while (m < n && values[order[m]] == values[order[k]]) m++;
                // Items order[k..m-1] are tied; their 1-based ranks are k+1 .. m.
                double avgRank = (k + 1 + m) / 2.0;
                for (int t = k; t < m; t++)
                    ranks[order[t]] = avgRank;
                k = m;
            }
            return ranks;
        }
    }
}
