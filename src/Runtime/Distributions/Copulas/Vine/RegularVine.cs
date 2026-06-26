// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>
    /// A regular vine (R-vine) copula model (Lopez-Paz et al., 2013, Sec. 2): a nested sequence
    /// of trees that factorizes a d-dimensional copula density into a product of bivariate
    /// (conditional) copulas (eq. 4).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The first tree <c>T_1</c> connects the raw (PIT-transformed) variables; each deeper tree
    /// <c>T_i</c> has the edges of <c>T_{i-1}</c> as its nodes and connects two of them only if
    /// they share a common node (the proximity condition, Sec. 2.1). The pseudo-observations of
    /// a deeper edge are the h-functions (conditional CDFs) of the previous tree's fitted
    /// copulas (eq. 5), and the conditioning set <c>D(e)</c> grows by one variable per level.
    /// </para>
    /// <para>
    /// First-tree edges are always fitted as <i>unconditional</i> copulas (Kendall's-tau MLE).
    /// For deeper trees, if an <see cref="IConditionalCopulaFitter"/> is supplied each edge is a
    /// <i>conditional</i> copula whose tau varies with the conditioning vector (the GPVINE model
    /// of the paper); otherwise deeper edges are also fitted unconditionally, giving the
    /// simplified-vine (SVINE) baseline.
    /// </para>
    /// </remarks>
    public class RegularVine
    {
        private readonly CopulaFamily family;
        private readonly IBivariateCopula copula;

        /// <summary>The trees of the vine, in order, populated by <see cref="Fit"/>.</summary>
        public List<VineTree> Trees { get; } = new List<VineTree>();

        /// <summary>Creates a regular vine that uses the given bivariate copula family.</summary>
        public RegularVine(CopulaFamily family = CopulaFamily.Gaussian)
        {
            this.family = family;
            this.copula = CopulaFactory.Create(family);
        }

        /// <summary>The copula family used by this vine.</summary>
        public CopulaFamily Family => family;

        /// <summary>
        /// Fits the vine to raw data of shape [n][d]. Applies the empirical PIT, builds the
        /// first tree by maximum spanning tree on the <c>|tau|</c> matrix, then constructs and
        /// fits deeper trees up to <paramref name="nTrees"/> levels.
        /// </summary>
        /// <param name="x">Raw observations of shape [n][d] (rows = observations).</param>
        /// <param name="nTrees">
        /// Number of trees to build; if &lt;= 0 or larger than <c>d - 1</c>, builds all <c>d - 1</c>.
        /// </param>
        /// <param name="fitter">
        /// Conditional-copula fitter for deeper trees (the GP fit). If null, deeper trees use the
        /// unconditional simplifying assumption (SVINE).
        /// </param>
        /// <returns>This vine, fitted.</returns>
        public RegularVine Fit(double[][] x, int nTrees = 0, IConditionalCopulaFitter fitter = null)
        {
            double[][] u = Pit.Transform(x);
            int n = u.Length;
            if (n == 0) throw new ArgumentException("No observations.", nameof(x));
            int d = u[0].Length;
            int maxTrees = d - 1;
            if (nTrees <= 0 || nTrees > maxTrees) nTrees = maxTrees;

            Trees.Clear();
            VineTree t1 = BuildFirstTree(u);
            FitTree(t1, u, fitter);
            Trees.Add(t1);

            for (int level = 2; level <= nTrees; level++)
            {
                VineTree next = BuildNextTree(Trees[level - 2], u, level);
                if (next.Edges.Count == 0) break;
                FitTree(next, u, fitter);
                Trees.Add(next);
            }
            return this;
        }

        /// <summary>
        /// Builds the first tree <c>T_1</c> from pseudo-observations [n][d]: the maximum spanning
        /// tree on the absolute-Kendall's-tau weight matrix, with each edge's tau set to the
        /// empirical value.
        /// </summary>
        public VineTree BuildFirstTree(double[][] u)
        {
            if (u == null) throw new ArgumentNullException(nameof(u));
            int n = u.Length;
            if (n == 0) throw new ArgumentException("No observations.", nameof(u));
            int d = u[0].Length;

            double[][] cols = new double[d][];
            for (int j = 0; j < d; j++)
                cols[j] = Column(u, j);

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
                    Left = i,
                    Right = j,
                    Conditioning = Array.Empty<int>(),
                    Endpoints = (i, j), // T_1 nodes are variables
                    U = (double[])cols[i].Clone(),
                    V = (double[])cols[j].Clone(),
                    Z = EmptyZ(n),
                    Weight = weights[i, j],
                    Tau = tau[i, j],
                });
            }
            return tree;
        }

        /// <summary>
        /// Builds tree <c>T_level</c> from the previous tree: candidate edges join two
        /// previous-tree edges that share a node (proximity condition); the new pseudo-
        /// observations are the previous edges' h-functions; the tree is the maximum spanning
        /// tree on the <c>|tau|</c> of those candidate series.
        /// </summary>
        public VineTree BuildNextTree(VineTree prev, double[][] u, int level)
        {
            int m = prev.Edges.Count;
            var tree = new VineTree(level);
            if (m < 2) return tree;

            var weights = new double[m, m];
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                    weights[a, b] = double.NegativeInfinity;

            var candidates = new Dictionary<(int, int), VineEdge>();
            for (int a = 0; a < m; a++)
            {
                for (int b = a + 1; b < m; b++)
                {
                    VineEdge cand = TryMakeCandidate(prev.Edges[a], prev.Edges[b], a, b, u);
                    if (cand == null) continue;
                    candidates[(a, b)] = cand;
                    weights[a, b] = weights[b, a] = cand.Weight;
                }
            }

            foreach (var (a, b) in MaxSpanningTree.Prim(weights))
            {
                if (candidates.TryGetValue((a, b), out VineEdge edge))
                    tree.Edges.Add(edge);
            }
            return tree;
        }

        /// <summary>
        /// Total log copula density of raw data <paramref name="x"/> under the fitted vine
        /// (eq. 4), summed over every edge of every tree. The vine structure is replayed on the
        /// PIT of <paramref name="x"/> so the result is a valid (held-out) log-likelihood.
        /// </summary>
        public double LogLikelihood(double[][] x)
        {
            if (Trees.Count == 0) throw new InvalidOperationException("Call Fit() before LogLikelihood().");
            double[][] u = Pit.Transform(x);
            int n = u.Length;
            double total = 0.0;

            // h-functions of the previous tree's edges on the test data, indexed by edge then by
            // conditioned variable; seeds the pseudo-observations of the next tree.
            List<Dictionary<int, double[]>> prevH = null;
            foreach (VineTree tree in Trees)
            {
                var thisH = new List<Dictionary<int, double[]>>(tree.Edges.Count);
                foreach (VineEdge edge in tree.Edges)
                {
                    double[] uu, vv;
                    if (tree.Level == 1)
                    {
                        uu = Column(u, edge.Left);
                        vv = Column(u, edge.Right);
                    }
                    else
                    {
                        uu = prevH[edge.Endpoints.A][edge.Left];
                        vv = prevH[edge.Endpoints.B][edge.Right];
                    }

                    double[] tau = TauSeries(edge, u, n);
                    var h = new Dictionary<int, double[]>(2)
                    {
                        [edge.Left] = new double[n],
                        [edge.Right] = new double[n],
                    };
                    for (int i = 0; i < n; i++)
                    {
                        total += copula.LogDensity(uu[i], vv[i], tau[i]);
                        h[edge.Left][i] = copula.HFunction(uu[i], vv[i], tau[i], 1);  // P(Left | Right, D)
                        h[edge.Right][i] = copula.HFunction(uu[i], vv[i], tau[i], 0); // P(Right | Left, D)
                    }
                    thisH.Add(h);
                }
                prevH = thisH;
            }
            return total;
        }

        // --- internals -----------------------------------------------------------------------

        /// <summary>Fits every edge of a tree and computes its training h-function series.</summary>
        private void FitTree(VineTree tree, double[][] u, IConditionalCopulaFitter fitter)
        {
            int n = u.Length;
            foreach (VineEdge edge in tree.Edges)
            {
                double[] tau;
                if (edge.Conditioning.Length == 0)
                {
                    // First tree: unconditional Kendall's-tau MLE (already set in BuildFirstTree).
                    tau = Constant(edge.Tau, n);
                }
                else if (fitter != null)
                {
                    edge.Posterior = fitter.Fit(edge.U, edge.V, edge.Z, family);
                    tau = TauSeries(edge, u, n);
                }
                else
                {
                    // Simplified vine: deeper edge fitted unconditionally too.
                    edge.Tau = KendallTau.Compute(edge.U, edge.V);
                    tau = Constant(edge.Tau, n);
                }

                edge.HByVar[edge.Left] = new double[n];
                edge.HByVar[edge.Right] = new double[n];
                for (int i = 0; i < n; i++)
                {
                    edge.HByVar[edge.Left][i] = copula.HFunction(edge.U[i], edge.V[i], tau[i], 1);
                    edge.HByVar[edge.Right][i] = copula.HFunction(edge.U[i], edge.V[i], tau[i], 0);
                }
            }
        }

        /// <summary>
        /// Forms the candidate edge joining previous-tree edges <paramref name="a"/> and
        /// <paramref name="b"/>, or null if they violate the proximity condition or would not
        /// yield a valid bivariate conditioned set.
        /// </summary>
        private VineEdge TryMakeCandidate(VineEdge a, VineEdge b, int aIdx, int bIdx, double[][] u)
        {
            // Proximity: the two edges must share a node in the previous tree.
            if (a.Endpoints.A != b.Endpoints.A && a.Endpoints.A != b.Endpoints.B &&
                a.Endpoints.B != b.Endpoints.A && a.Endpoints.B != b.Endpoints.B)
                return null;

            var Na = new HashSet<int>(a.N());
            var Nb = new HashSet<int>(b.N());
            var D = new SortedSet<int>(Na.Where(Nb.Contains));
            // Conditioned set = symmetric difference; must be exactly the two non-shared variables.
            int xa = OnlyNonShared(a, Nb);
            int xb = OnlyNonShared(b, Na);
            if (xa < 0 || xb < 0 || xa == xb) return null;
            if (Na.Count + Nb.Count - 2 * D.Count != 2) return null;

            double[] left = a.HByVar[xa];   // P(xa | D)
            double[] right = b.HByVar[xb];  // P(xb | D)
            int[] dvars = D.ToArray();
            return new VineEdge
            {
                Left = xa,
                Right = xb,
                Conditioning = dvars,
                Endpoints = (aIdx, bIdx),
                U = left,
                V = right,
                Z = BuildZ(u, dvars),
                Weight = System.Math.Abs(KendallTau.Compute(left, right)),
            };
        }

        /// <summary>
        /// Returns the conditioned variable of <paramref name="e"/> that is not in
        /// <paramref name="other"/>, or -1 if exactly one such variable does not exist.
        /// </summary>
        private static int OnlyNonShared(VineEdge e, HashSet<int> other)
        {
            bool leftIn = other.Contains(e.Left);
            bool rightIn = other.Contains(e.Right);
            if (leftIn == rightIn) return -1; // need exactly one shared
            return leftIn ? e.Right : e.Left;
        }

        /// <summary>Per-observation Kendall's tau for an edge (scalar, or GP posterior at z).</summary>
        private double[] TauSeries(VineEdge edge, double[][] u, int n)
        {
            if (edge.Posterior == null)
                return Constant(edge.Tau, n);
            double[][] z = BuildZ(u, edge.Conditioning);
            double[] tau = new double[n];
            for (int i = 0; i < n; i++)
                tau[i] = edge.Posterior.TauAt(z[i]);
            return tau;
        }

        private static double[] Column(double[][] m, int j)
        {
            int n = m.Length;
            double[] c = new double[n];
            for (int i = 0; i < n; i++) c[i] = m[i][j];
            return c;
        }

        private static double[][] BuildZ(double[][] u, int[] dvars)
        {
            int n = u.Length;
            double[][] z = new double[n][];
            for (int i = 0; i < n; i++)
            {
                z[i] = new double[dvars.Length];
                for (int k = 0; k < dvars.Length; k++)
                    z[i][k] = u[i][dvars[k]];
            }
            return z;
        }

        private static double[][] EmptyZ(int n)
        {
            double[][] z = new double[n][];
            for (int i = 0; i < n; i++) z[i] = Array.Empty<double>();
            return z;
        }

        private static double[] Constant(double value, int n)
        {
            double[] a = new double[n];
            for (int i = 0; i < n; i++) a[i] = value;
            return a;
        }
    }
}
