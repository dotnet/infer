// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// Maximum spanning tree via Prim's algorithm, used to select each vine tree (Sec. 2.1
    /// of Lopez-Paz et al., 2013): nodes are variables (or previous-tree edges) and edge
    /// weights are the absolute empirical Kendall's tau between their pseudo-observations.
    /// </summary>
    public static class MaxSpanningTree
    {
        /// <summary>
        /// Returns the <c>m - 1</c> edges <c>(i, j)</c> of the maximum spanning tree of a
        /// complete graph with the given symmetric, non-negative weight matrix.
        /// </summary>
        /// <param name="weights">Symmetric <c>m x m</c> weight matrix; the diagonal is ignored.</param>
        /// <returns>The selected edges, each as a pair with <c>i &lt; j</c>.</returns>
        public static List<(int I, int J)> Prim(double[,] weights)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            int m = weights.GetLength(0);
            if (weights.GetLength(1) != m) throw new ArgumentException("Weight matrix must be square.");
            var edges = new List<(int, int)>();
            if (m <= 1) return edges;

            bool[] inTree = new bool[m];
            inTree[0] = true;
            int count = 1;
            while (count < m)
            {
                int bestI = -1, bestJ = -1;
                double bestW = double.NegativeInfinity;
                for (int i = 0; i < m; i++)
                {
                    if (!inTree[i]) continue;
                    for (int j = 0; j < m; j++)
                    {
                        if (inTree[j]) continue;
                        if (weights[i, j] > bestW)
                        {
                            bestW = weights[i, j];
                            bestI = i;
                            bestJ = j;
                        }
                    }
                }

                // bestJ is always assigned for a connected complete graph (m > 1).
                inTree[bestJ] = true;
                count++;
                edges.Add(bestI < bestJ ? (bestI, bestJ) : (bestJ, bestI));
            }
            return edges;
        }
    }
}
