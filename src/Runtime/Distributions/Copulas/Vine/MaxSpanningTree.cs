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
        /// Returns the edges <c>(i, j)</c> of the maximum spanning tree (or forest) of a graph
        /// with the given symmetric weight matrix.
        /// </summary>
        /// <param name="weights">
        /// Symmetric <c>m x m</c> weight matrix; the diagonal is ignored. Use
        /// <see cref="double.NegativeInfinity"/> to mark a disallowed pair (e.g. for the vine
        /// proximity condition); if the allowed edges do not connect all nodes the result is a
        /// spanning forest with fewer than <c>m - 1</c> edges.
        /// </param>
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

                // No allowed edge crosses the cut: the graph is disconnected under the
                // proximity constraint, so return the spanning forest found so far.
                if (bestJ < 0 || double.IsNegativeInfinity(bestW))
                    break;
                inTree[bestJ] = true;
                count++;
                edges.Add(bestI < bestJ ? (bestI, bestJ) : (bestJ, bestI));
            }
            return edges;
        }
    }
}
