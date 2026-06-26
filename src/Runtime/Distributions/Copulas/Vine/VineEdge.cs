// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// One edge of a vine tree, i.e. one (conditional) bivariate copula in the factorization
    /// of the joint copula density (Lopez-Paz et al., 2013, eq. 4).
    /// </summary>
    /// <remarks>
    /// An edge carries the conditioned set <c>C(e) = {Left, Right}</c> (the two variables being
    /// coupled) and the conditioning set <c>D(e)</c> (variables conditioned on; empty in the
    /// first tree). The pseudo-observation series <c>(U, V)</c> feeding the copula are the raw
    /// PIT columns in the first tree and the previous tree's h-functions thereafter (eq. 5).
    /// After fitting it stores either an unconditional Kendall's-tau MLE (<see cref="Tau"/>) or
    /// a conditional posterior (<see cref="Posterior"/>), and the per-conditioned-variable
    /// h-function series that seed the next tree.
    /// </remarks>
    public class VineEdge
    {
        /// <summary>The conditioned variable aligned with the <see cref="U"/> series.</summary>
        public int Left { get; set; }

        /// <summary>The conditioned variable aligned with the <see cref="V"/> series.</summary>
        public int Right { get; set; }

        /// <summary>The conditioning set D(e): variable indices conditioned on (empty in T_1), sorted.</summary>
        public int[] Conditioning { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Indices, within the parent tree's node set, of the two nodes this edge joins. For
        /// T_1 these are variable indices; for deeper trees they are previous-tree edge indices.
        /// Used to test the proximity condition when building the next tree.
        /// </summary>
        public (int A, int B) Endpoints { get; set; }

        /// <summary>First pseudo-observation series feeding this edge (aligned with <see cref="Left"/>).</summary>
        public double[] U { get; set; }

        /// <summary>Second pseudo-observation series feeding this edge (aligned with <see cref="Right"/>).</summary>
        public double[] V { get; set; }

        /// <summary>Conditioning matrix Z for this edge, shape [n][|D(e)|]; empty in T_1.</summary>
        public double[][] Z { get; set; }

        /// <summary>Edge weight used for tree selection: the absolute empirical Kendall's tau.</summary>
        public double Weight { get; set; }

        /// <summary>Fitted unconditional Kendall's tau (used when <see cref="Posterior"/> is null).</summary>
        public double Tau { get; set; }

        /// <summary>Fitted conditional posterior (null for unconditional / first-tree edges).</summary>
        public IConditionalCopulaPosterior Posterior { get; set; }

        /// <summary>Whether this edge was fitted as a conditional copula (GP-backed).</summary>
        public bool IsConditional => Posterior != null;

        /// <summary>
        /// Per-conditioned-variable h-function series computed at fit time on the training data:
        /// <c>HByVar[Left] = P(Left | Right, D)</c> and <c>HByVar[Right] = P(Right | Left, D)</c>.
        /// These become the pseudo-observations of the next tree (eq. 5).
        /// </summary>
        public Dictionary<int, double[]> HByVar { get; } = new Dictionary<int, double[]>();

        /// <summary>The complete set <c>N(e) = C(e) ∪ D(e)</c>, sorted.</summary>
        public int[] N()
        {
            var set = new SortedSet<int>(Conditioning) { Left, Right };
            return set.ToArray();
        }

        /// <summary>A short label such as <c>"0,2|1,3"</c> (conditioned | conditioning).</summary>
        public string Label
        {
            get
            {
                int lo = System.Math.Min(Left, Right), hi = System.Math.Max(Left, Right);
                string c = $"{lo},{hi}";
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
