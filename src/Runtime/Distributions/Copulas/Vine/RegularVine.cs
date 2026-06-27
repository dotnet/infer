// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Rand = Microsoft.ML.Probabilistic.Math.Rand;

namespace Microsoft.ML.Probabilistic.Distributions.Copulas.Vine
{
    /// <summary>The tree-selection strategy for a <see cref="RegularVine"/>.</summary>
    public enum VineStructure
    {
        /// <summary>Maximum spanning tree on |tau| at each level (a general R-vine).</summary>
        Regular,

        /// <summary>
        /// A canonical vine (C-vine): each tree is a star centred on the variable with the
        /// strongest summed dependence. Required by <see cref="RegularVine.Sample"/>.
        /// </summary>
        Canonical,
    }

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

        /// <summary>
        /// The empirical marginals captured at <see cref="Fit"/> time, one per variable, used to
        /// map generated pseudo-observations back to the data scale.
        /// </summary>
        public EmpiricalMarginal[] Marginals { get; private set; }

        /// <summary>The tree-selection strategy used by the most recent <see cref="Fit"/>.</summary>
        public VineStructure Structure { get; private set; }

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
        /// <param name="structure">
        /// Tree-selection strategy. <see cref="VineStructure.Regular"/> (default) builds a maximum
        /// spanning tree per level; <see cref="VineStructure.Canonical"/> builds a C-vine and is
        /// required by <see cref="Sample"/>.
        /// </param>
        /// <param name="rootOrder">
        /// For a canonical fit, forces these variables (in order) to be the leading roots, so a
        /// chosen conditioning set becomes a prefix of the vine order and can be conditioned on
        /// exactly by <see cref="SampleConditional"/>. Ignored for a regular fit.
        /// </param>
        /// <returns>This vine, fitted.</returns>
        public RegularVine Fit(double[][] x, int nTrees = 0, IConditionalCopulaFitter fitter = null,
            VineStructure structure = VineStructure.Regular, int[] rootOrder = null)
        {
            double[][] u = Pit.Transform(x);
            int n = u.Length;
            if (n == 0) throw new ArgumentException("No observations.", nameof(x));
            int d = u[0].Length;
            int maxTrees = d - 1;
            if (nTrees <= 0 || nTrees > maxTrees) nTrees = maxTrees;
            Structure = structure;

            // Capture the marginals so generated samples can be returned on the data scale.
            Marginals = new EmpiricalMarginal[d];
            for (int j = 0; j < d; j++)
                Marginals[j] = new EmpiricalMarginal(Column(x, j));

            Trees.Clear();
            VineTree t1 = BuildFirstTree(u, structure, rootOrder);
            FitTree(t1, u, fitter);
            Trees.Add(t1);

            for (int level = 2; level <= nTrees; level++)
            {
                VineTree next = BuildNextTree(Trees[level - 2], u, level, structure, rootOrder);
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
        public VineTree BuildFirstTree(double[][] u) => BuildFirstTree(u, VineStructure.Regular, null);

        /// <summary>Builds the first tree using the given structure strategy and optional forced root.</summary>
        public VineTree BuildFirstTree(double[][] u, VineStructure structure, int[] rootOrder)
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

            int t1Center = (structure == VineStructure.Canonical && rootOrder != null && rootOrder.Length > 0)
                ? rootOrder[0] : -1;
            var tree = new VineTree(1);
            foreach (var (i, j) in SelectEdges(weights, structure, t1Center))
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
        public VineTree BuildNextTree(VineTree prev, double[][] u, int level) =>
            BuildNextTree(prev, u, level, VineStructure.Regular, null);

        /// <summary>Builds tree <c>T_level</c> using the given structure strategy and optional forced root.</summary>
        public VineTree BuildNextTree(VineTree prev, double[][] u, int level, VineStructure structure, int[] rootOrder)
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

            // For a forced root, centre the star on the previous-tree edge containing that
            // variable, making it the shared conditioned variable of this tree.
            int forcedCenter = -1;
            if (structure == VineStructure.Canonical && rootOrder != null && level - 1 < rootOrder.Length)
            {
                int rootVar = rootOrder[level - 1];
                for (int a = 0; a < m; a++)
                    if (prev.Edges[a].Left == rootVar || prev.Edges[a].Right == rootVar) { forcedCenter = a; break; }
                if (forcedCenter < 0)
                    throw new ArgumentException($"rootOrder variable {rootVar} is not available as a centre at tree {level}.", nameof(rootOrder));
            }

            foreach (var (a, b) in SelectEdges(weights, structure, forcedCenter))
            {
                if (candidates.TryGetValue((a, b), out VineEdge edge))
                    tree.Edges.Add(edge);
            }
            return tree;
        }

        /// <summary>
        /// Selects the edges of a tree from a symmetric weight matrix: a maximum spanning tree
        /// for <see cref="VineStructure.Regular"/>, or a maximum-weight star (all nodes joined to
        /// a single centre) for <see cref="VineStructure.Canonical"/>.
        /// </summary>
        private static List<(int, int)> SelectEdges(double[,] weights, VineStructure structure, int forcedCenter = -1)
        {
            if (structure == VineStructure.Regular)
                return MaxSpanningTree.Prim(weights);

            int m = weights.GetLength(0);
            var edges = new List<(int, int)>();
            if (m <= 1) return edges;

            // Centre = the forced node, else the node whose finite weights sum to the most.
            int center = forcedCenter;
            if (center < 0)
            {
                double bestSum = double.NegativeInfinity;
                for (int c = 0; c < m; c++)
                {
                    double sum = 0.0;
                    bool any = false;
                    for (int j = 0; j < m; j++)
                    {
                        if (j == c || double.IsNegativeInfinity(weights[c, j])) continue;
                        sum += weights[c, j];
                        any = true;
                    }
                    if (any && sum > bestSum) { bestSum = sum; center = c; }
                }
            }
            if (center < 0) return edges;
            for (int j = 0; j < m; j++)
            {
                if (j == center || double.IsNegativeInfinity(weights[center, j])) continue;
                edges.Add(center < j ? (center, j) : (j, center));
            }
            return edges;
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

        /// <summary>
        /// Draws <paramref name="n"/> samples from the fitted vine via the inverse-Rosenblatt
        /// transform, returned on the data scale (using the empirical marginals). Requires the
        /// vine to have been fitted with <see cref="VineStructure.Canonical"/>.
        /// </summary>
        /// <param name="n">Number of joint samples to draw.</param>
        /// <returns>An [n][d] array of samples in the original data units.</returns>
        public double[][] Sample(int n)
        {
            if (Trees.Count == 0) throw new InvalidOperationException("Call Fit() before Sample().");
            if (Structure != VineStructure.Canonical)
                throw new NotSupportedException(
                    "Sampling requires a canonical (C-vine) structure. Refit with structure: VineStructure.Canonical.");

            int d = Marginals.Length;
            int[] order = BuildCanonicalOrder(d);
            int[] posOf = new int[d];
            for (int p = 0; p < d; p++) posOf[order[p]] = p;
            var lookup = BuildEdgeLookup();

            double[][] result = new double[n][];
            for (int s = 0; s < n; s++)
                result[s] = SampleOne(d, order, posOf, lookup, 0, null);
            return result;
        }

        /// <summary>
        /// Conditional simulation: draws <paramref name="n"/> samples with the variables in
        /// <paramref name="known"/> fixed to the given data-scale values, sampling the rest from
        /// their conditional distribution. Useful for imputation and posterior-predictive
        /// inference given partial observations.
        /// </summary>
        /// <param name="known">Variable id -&gt; observed data-scale value for the conditioned variables.</param>
        /// <param name="n">Number of conditional samples to draw.</param>
        /// <returns>
        /// An [n][d] array on the data scale; the conditioned variables equal their given values.
        /// </returns>
        /// <remarks>
        /// Exact when the conditioned set is the leading roots of the canonical vine - fit with
        /// <c>rootOrder</c> set to the variables you intend to condition on. Otherwise a clear
        /// exception is thrown.
        /// </remarks>
        public double[][] SampleConditional(IDictionary<int, double> known, int n)
        {
            if (known == null) throw new ArgumentNullException(nameof(known));
            if (Trees.Count == 0) throw new InvalidOperationException("Call Fit() before SampleConditional().");
            if (Structure != VineStructure.Canonical)
                throw new NotSupportedException(
                    "Conditional simulation requires a canonical (C-vine) structure. Refit with structure: VineStructure.Canonical.");

            int d = Marginals.Length;
            int k = known.Count;
            if (k == 0) return Sample(n);
            if (k >= d) throw new ArgumentException("At least one variable must be left to sample.", nameof(known));

            int[] order = BuildCanonicalOrder(d);
            int[] posOf = new int[d];
            for (int p = 0; p < d; p++) posOf[order[p]] = p;

            // The conditioned set must be exactly the leading roots of the vine order.
            var prefix = new HashSet<int>();
            for (int p = 0; p < k; p++) prefix.Add(order[p]);
            if (!prefix.SetEquals(known.Keys))
                throw new NotSupportedException(
                    $"Conditioned set must be the leading roots of the canonical vine. Leading roots are [{string.Join(",", prefix)}]; " +
                    $"refit with rootOrder set to [{string.Join(",", known.Keys)}].");

            var lookup = BuildEdgeLookup();
            double[] fixedByPos = new double[k];
            for (int p = 0; p < k; p++)
                fixedByPos[p] = Marginals[order[p]].Cdf(known[order[p]]); // data -> pseudo-observation

            double[][] result = new double[n][];
            for (int s = 0; s < n; s++)
            {
                double[] row = SampleOne(d, order, posOf, lookup, k, fixedByPos);
                foreach (int varId in known.Keys) row[varId] = known[varId]; // return conditioned values exactly
                result[s] = row;
            }
            return result;
        }

        // One C-vine draw (Aas et al. 2009, Algorithm 2). Positions [0, fixedCount) are held at
        // fixedByPos (their pseudo-observations); the rest are sampled. Returned on the data scale.
        private double[] SampleOne(int d, int[] order, int[] posOf, Dictionary<string, VineEdge> lookup,
            int fixedCount, double[] fixedByPos)
        {
            double[] x = new double[d];      // pseudo-observations by position in the C-vine order
            double[][] v = new double[d][];
            for (int i = 0; i < d; i++) v[i] = new double[d];

            x[0] = fixedCount >= 1 ? fixedByPos[0] : Rand.Double();
            v[0][0] = x[0];
            for (int i = 1; i < d; i++)
            {
                if (i < fixedCount)
                {
                    v[i][0] = x[i] = fixedByPos[i]; // conditioned variable: held fixed
                }
                else
                {
                    v[i][0] = Rand.Double();
                    for (int k = i - 1; k >= 0; k--)
                    {
                        double tau = TauForCell(k, i, order, posOf, x, lookup);
                        v[i][0] = copula.InverseHFunction(v[i][0], v[k][k], tau, 1);
                    }
                    x[i] = v[i][0];
                }
                if (i == d - 1) break;
                for (int k = 0; k < i; k++)
                {
                    double tau = TauForCell(k, i, order, posOf, x, lookup);
                    v[i][k + 1] = copula.HFunction(v[i][k], v[k][k], tau, 1);
                }
            }

            // Map positions back to variable ids and through the empirical marginals.
            double[] row = new double[d];
            for (int p = 0; p < d; p++)
                row[order[p]] = Marginals[order[p]].Quantile(x[p]);
            return row;
        }

        // Kendall's tau for the C-vine cell coupling roots order[k] and order[i], conditioned on
        // order[0..k-1]. Missing edges (truncated vine) are independence (tau = 0).
        private double TauForCell(int k, int i, int[] order, int[] posOf, double[] x, Dictionary<string, VineEdge> lookup)
        {
            int[] condIds = new int[k];
            for (int p = 0; p < k; p++) condIds[p] = order[p];
            Array.Sort(condIds);
            if (!lookup.TryGetValue(EdgeKey(order[k], order[i], condIds), out VineEdge edge))
                return 0.0;
            if (edge.Posterior == null) return edge.Tau;
            double[] z = new double[condIds.Length];
            for (int p = 0; p < condIds.Length; p++) z[p] = x[posOf[condIds[p]]];
            return edge.Posterior.SampleTau(z); // posterior-predictive: propagate GP uncertainty
        }

        private Dictionary<string, VineEdge> BuildEdgeLookup()
        {
            var map = new Dictionary<string, VineEdge>();
            foreach (VineTree tree in Trees)
                foreach (VineEdge e in tree.Edges)
                    map[EdgeKey(e.Left, e.Right, e.Conditioning)] = e;
            return map;
        }

        private static string EdgeKey(int a, int b, int[] conditioning)
        {
            int lo = System.Math.Min(a, b), hi = System.Math.Max(a, b);
            return $"{lo},{hi}|{string.Join(",", conditioning)}";
        }

        // The C-vine diagonal order: the centre (shared conditioned variable) of each tree, then
        // any remaining variables (roots of unbuilt/independence trees plus the last variable).
        private int[] BuildCanonicalOrder(int d)
        {
            int[] order = new int[d];
            bool[] assigned = new bool[d];
            int filled = 0;
            foreach (VineTree tree in Trees)
            {
                int center = CanonicalCenter(tree, assigned);
                order[filled++] = center;
                assigned[center] = true;
            }
            for (int v = 0; v < d && filled < d; v++)
                if (!assigned[v]) { order[filled++] = v; assigned[v] = true; }
            return order;
        }

        // The variable common to every edge's conditioned set in a star tree (smallest unassigned
        // such variable when the tree has a single edge).
        private static int CanonicalCenter(VineTree tree, bool[] assigned)
        {
            HashSet<int> common = null;
            foreach (VineEdge e in tree.Edges)
            {
                var conditioned = new HashSet<int> { e.Left, e.Right };
                if (common == null) common = conditioned;
                else common.IntersectWith(conditioned);
            }
            if (common == null || common.Count == 0)
                throw new NotSupportedException("Vine is not canonical (no shared centre); sampling needs VineStructure.Canonical.");
            int best = -1;
            foreach (int c in common)
                if (!assigned[c] && (best < 0 || c < best)) best = c;
            if (best < 0) best = common.Min(); // all assigned (shouldn't happen for a canonical fit)
            return best;
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
