// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// A regular vine (R-vine) copula model (Lopez-Paz et al., 2013, Sec. 2): a nested
    /// sequence of trees that factorizes a d-dimensional copula density into a product of
    /// bivariate (conditional) copulas (eq. 4).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class currently implements the first tree <c>T_1</c>, whose edges are
    /// <i>unconditional</i> copulas fitted by the Kendall's-tau MLE - the cheap, deterministic
    /// path the paper uses for the first tree (Sec. 4). Deeper trees (the h-function recursion
    /// of eq. 5 and the per-edge sparse-GP fit for conditional copulas) are future work.
    /// </para>
    /// <para>
    /// The model operates on pseudo-observations: callers pass raw data and the empirical PIT
    /// (<see cref="Pit"/>) is applied internally, so train and test data are transformed
    /// consistently.
    /// </para>
    /// </remarks>
    public class RegularVine
    {
        private readonly CopulaFamily family;
        private readonly IBivariateCopula copula;

        /// <summary>The trees of the vine (currently only <c>T_1</c> after <see cref="Fit"/>).</summary>
        public List<VineTree> Trees { get; } = new List<VineTree>();

        /// <summary>Creates a regular vine that uses the given bivariate copula family.</summary>
        /// <param name="family">The copula family for every edge.</param>
        public RegularVine(CopulaFamily family = CopulaFamily.Gaussian)
        {
            this.family = family;
            this.copula = CopulaFactory.Create(family);
        }

        /// <summary>The copula family used by this vine.</summary>
        public CopulaFamily Family => family;

        /// <summary>
        /// Fits the vine to raw data of shape [n][d]. Applies the empirical PIT, builds the
        /// first tree as the maximum spanning tree on the <c>|tau|</c> matrix, and fits each
        /// edge's unconditional Kendall's-tau MLE.
        /// </summary>
        /// <param name="x">Raw observations of shape [n][d] (rows = observations).</param>
        /// <returns>This vine, fitted.</returns>
        public RegularVine Fit(double[][] x)
        {
            double[][] u = Pit.Transform(x);
            Trees.Clear();
            Trees.Add(BuildFirstTree(u));
            return this;
        }

        /// <summary>
        /// Builds the first tree <c>T_1</c> from pseudo-observations of shape [n][d]: the
        /// maximum spanning tree on the absolute-Kendall's-tau weight matrix, with each edge's
        /// tau fitted to the empirical value.
        /// </summary>
        /// <param name="u">Pseudo-observations in (0, 1), shape [n][d].</param>
        /// <returns>The first tree with fitted edges.</returns>
        public VineTree BuildFirstTree(double[][] u)
        {
            if (u == null) throw new ArgumentNullException(nameof(u));
            int n = u.Length;
            if (n == 0) throw new ArgumentException("No observations.", nameof(u));
            int d = u[0].Length;

            // Extract columns once for repeated pairwise Kendall's tau.
            double[][] cols = new double[d][];
            for (int j = 0; j < d; j++)
            {
                cols[j] = new double[n];
                for (int i = 0; i < n; i++)
                    cols[j][i] = u[i][j];
            }

            double[,] weights = new double[d, d];
            double[,] tau = new double[d, d];
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    double t = KendallTau.Compute(cols[i], cols[j]);
                    tau[i, j] = tau[j, i] = t;
                    weights[i, j] = weights[j, i] = System.Math.Abs(t);
                }
            }

            var tree = new VineTree(1);
            foreach (var (i, j) in MaxSpanningTree.Prim(weights))
            {
                tree.Edges.Add(new VineEdge
                {
                    Conditioned = (i, j),
                    Conditioning = Array.Empty<int>(),
                    U = (double[])cols[i].Clone(),
                    V = (double[])cols[j].Clone(),
                    Z = new double[n][],
                    Weight = weights[i, j],
                    Tau = tau[i, j],
                });
            }
            return tree;
        }

        /// <summary>
        /// Total log copula density of raw data <paramref name="x"/> under the fitted vine
        /// (eq. 4). Currently sums only the first-tree contributions.
        /// </summary>
        /// <param name="x">Raw observations of shape [n][d].</param>
        /// <returns>The summed log copula density over all observations and fitted edges.</returns>
        public double LogLikelihood(double[][] x)
        {
            if (Trees.Count == 0) throw new InvalidOperationException("Call Fit() before LogLikelihood().");
            double[][] u = Pit.Transform(x);
            double total = 0.0;
            foreach (VineTree tree in Trees)
            {
                foreach (VineEdge edge in tree.Edges)
                {
                    int a = edge.Conditioned.I;
                    int b = edge.Conditioned.J;
                    for (int i = 0; i < u.Length; i++)
                        total += copula.LogDensity(u[i][a], u[i][b], edge.Tau);
                }
            }
            return total;
        }
    }
}
