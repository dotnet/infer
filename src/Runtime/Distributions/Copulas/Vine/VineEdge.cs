// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// One edge of a vine tree, i.e. one (conditional) bivariate copula in the factorization
    /// of the joint copula density (Lopez-Paz et al., 2013, eq. 4).
    /// </summary>
    /// <remarks>
    /// An edge carries the conditioned set <c>C(e)</c> (the two variables being coupled) and
    /// the conditioning set <c>D(e)</c> (variables conditioned on; empty in the first tree).
    /// It also holds the paired pseudo-observation series <c>(u, v)</c> that feed it and, for
    /// deeper trees, the conditioning matrix <c>Z</c>. After fitting it stores the copula
    /// parameter: for first-tree edges this is the single Kendall's-tau MLE in <see cref="Tau"/>.
    /// </remarks>
    public class VineEdge
    {
        /// <summary>The conditioned set C(e): the two variable indices being coupled (i &lt; j).</summary>
        public (int I, int J) Conditioned { get; set; }

        /// <summary>The conditioning set D(e): variable indices conditioned on (empty in T_1).</summary>
        public int[] Conditioning { get; set; } = System.Array.Empty<int>();

        /// <summary>First pseudo-observation series feeding this edge, length n.</summary>
        public double[] U { get; set; }

        /// <summary>Second pseudo-observation series feeding this edge, length n.</summary>
        public double[] V { get; set; }

        /// <summary>Conditioning matrix Z for this edge, shape [n][|D(e)|]; empty in T_1.</summary>
        public double[][] Z { get; set; }

        /// <summary>Edge weight used for tree selection: the absolute empirical Kendall's tau.</summary>
        public double Weight { get; set; }

        /// <summary>Fitted Kendall's tau (the unconditional MLE for first-tree edges).</summary>
        public double Tau { get; set; }

        /// <summary>A short label such as <c>"0,2|1,3"</c> (conditioned | conditioning).</summary>
        public string Label
        {
            get
            {
                string c = $"{Conditioned.I},{Conditioned.J}";
                return (Conditioning == null || Conditioning.Length == 0)
                    ? c
                    : c + "|" + string.Join(",", Conditioning);
            }
        }
    }

    /// <summary>
    /// One tree of a regular vine: a level index and its set of edges.
    /// </summary>
    public class VineTree
    {
        /// <summary>The tree level (1 for the first tree T_1).</summary>
        public int Level { get; set; }

        /// <summary>The edges of this tree.</summary>
        public List<VineEdge> Edges { get; } = new List<VineEdge>();

        /// <summary>Creates a tree at the given level.</summary>
        public VineTree(int level)
        {
            Level = level;
        }
    }
}
