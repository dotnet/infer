// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Distributions.Copulas.Vine;
using Microsoft.ML.Probabilistic.Math;
using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Phase 3 tests for the regular-vine application layer (PIT, Kendall's tau, maximum
    /// spanning tree, first-tree construction and log-likelihood). Pure C#, no inference.
    /// </summary>
    public class VineTests
    {
        [Fact]
        public void Pit_ProducesRanksInUnitInterval()
        {
            double[] x = { 5.0, -2.0, 3.0, 100.0, 0.0 };
            double[] u = Pit.Transform(x);
            // Rank-based: smallest -> 1/(n+1), largest -> n/(n+1); strictly inside (0,1).
            Assert.All(u, ui => Assert.True(ui > 0.0 && ui < 1.0));
            int n = x.Length;
            Assert.Equal(1.0 / (n + 1), u[1], 12); // -2 is smallest
            Assert.Equal(5.0 / (n + 1), u[3], 12); // 100 is largest
            Assert.Equal(2.0 / (n + 1), u[4], 12); // 0 is second smallest
        }

        [Fact]
        public void Pit_TiesGetAverageRank()
        {
            double[] x = { 1.0, 1.0, 2.0 };
            double[] u = Pit.Transform(x);
            // The two tied 1's share ranks 1 and 2 -> average 1.5; the 2 gets rank 3.
            Assert.Equal(1.5 / 4.0, u[0], 12);
            Assert.Equal(1.5 / 4.0, u[1], 12);
            Assert.Equal(3.0 / 4.0, u[2], 12);
        }

        [Fact]
        public void KendallTau_KnownValues()
        {
            Assert.Equal(1.0, KendallTau.Compute(new[] { 1.0, 2, 3, 4 }, new[] { 10.0, 20, 30, 40 }), 12);
            Assert.Equal(-1.0, KendallTau.Compute(new[] { 1.0, 2, 3, 4 }, new[] { 4.0, 3, 2, 1 }), 12);
            // 4 points, one discordant pair out of 6: tau = (5 - 1) / 6.
            double tau = KendallTau.Compute(new[] { 1.0, 2, 3, 4 }, new[] { 1.0, 2, 4, 3 });
            Assert.Equal(4.0 / 6.0, tau, 12);
        }

        [Fact]
        public void MaxSpanningTree_PicksHeaviestEdges()
        {
            // 4 nodes; heaviest edges form the path 0-1-2-3.
            double[,] w =
            {
                { 0.0, 0.9, 0.2, 0.1 },
                { 0.9, 0.0, 0.8, 0.2 },
                { 0.2, 0.8, 0.0, 0.7 },
                { 0.1, 0.2, 0.7, 0.0 },
            };
            var edges = MaxSpanningTree.Prim(w);
            Assert.Equal(3, edges.Count); // d - 1
            var set = new HashSet<(int, int)>(edges);
            Assert.Contains((0, 1), set);
            Assert.Contains((1, 2), set);
            Assert.Contains((2, 3), set);
        }

        [Fact]
        public void FirstTree_RecoversKnownTau_AndStructure()
        {
            // AR(1) Gaussian chain: corr(x_k, x_{k+1}) = rho, so neighbours are the strongest
            // pairs and T_1 should select the chain 0-1-2-3.
            Rand.Restart(7);
            const double rho = 0.8;
            int n = 3000, d = 4;
            double[][] x = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double[] row = new double[d];
                row[0] = Rand.Normal();
                for (int k = 1; k < d; k++)
                    row[k] = rho * row[k - 1] + System.Math.Sqrt(1 - rho * rho) * Rand.Normal();
                x[i] = row;
            }

            var vine = new RegularVine(CopulaFamily.Gaussian).Fit(x, nTrees: 1);
            VineTree t1 = vine.Trees[0];
            Assert.Equal(3, t1.Edges.Count);

            var selected = new HashSet<(int, int)>(t1.Edges.Select(Conditioned));
            Assert.Equal(new HashSet<(int, int)> { (0, 1), (1, 2), (2, 3) }, selected);

            // For a Gaussian copula, tau = (2/pi) arcsin(rho).
            double expectedTau = 2.0 / System.Math.PI * System.Math.Asin(rho);
            foreach (var e in t1.Edges)
                Assert.True(System.Math.Abs(e.Tau - expectedTau) < 0.04,
                    $"edge {e.Label}: tau={e.Tau}, expected ~{expectedTau}");
        }

        [Fact]
        public void FirstTree_LogLikelihood_BeatsIndependenceBaseline()
        {
            Rand.Restart(11);
            const double rho = 0.7;
            int n = 2000, d = 3;

            double[][] dependent = new double[n][];
            double[][] independent = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double[] dep = new double[d];
                dep[0] = Rand.Normal();
                for (int k = 1; k < d; k++)
                    dep[k] = rho * dep[k - 1] + System.Math.Sqrt(1 - rho * rho) * Rand.Normal();
                dependent[i] = dep;
                independent[i] = new[] { Rand.Normal(), Rand.Normal(), Rand.Normal() };
            }

            double llDep = new RegularVine().Fit(dependent, nTrees: 1).LogLikelihood(dependent);
            double llIndep = new RegularVine().Fit(independent, nTrees: 1).LogLikelihood(independent);

            // Dependence is real -> the copula adds substantial likelihood; independence data
            // fits tau ~ 0 -> near-zero contribution.
            Assert.True(llDep > 100.0, $"dependent log-lik unexpectedly low: {llDep}");
            Assert.True(llDep > llIndep + 100.0, $"llDep={llDep} not clearly above llIndep={llIndep}");
            Assert.True(System.Math.Abs(llIndep) < 50.0, $"independent log-lik unexpectedly large: {llIndep}");
        }

        [Fact]
        public void DeeperTrees_HaveValidVineStructure()
        {
            // Equicorrelated Gaussian: every pair is dependent, so all d-1 trees are non-trivial.
            double[][] x = Equicorrelated(seed: 3, n: 1500, d: 4, rho: 0.6);
            var vine = new RegularVine().Fit(x); // all d-1 trees, SVINE (no fitter)

            int d = 4;
            Assert.Equal(d - 1, vine.Trees.Count);

            int totalEdges = 0;
            for (int level = 1; level <= vine.Trees.Count; level++)
            {
                VineTree tree = vine.Trees[level - 1];
                Assert.Equal(level, tree.Level);
                // Tree T_level has d - level edges.
                Assert.Equal(d - level, tree.Edges.Count);
                foreach (VineEdge e in tree.Edges)
                {
                    Assert.Equal(level - 1, e.Conditioning.Length); // |D(e)| grows by one per level
                    Assert.Equal(level + 1, e.N().Length);          // |N(e)| = |C| + |D| = 2 + (level-1)
                    Assert.NotEqual(e.Left, e.Right);
                    if (level >= 2)
                    {
                        // Proximity: the two joined previous-tree edges share a node.
                        VineEdge a = vine.Trees[level - 2].Edges[e.Endpoints.A];
                        VineEdge b = vine.Trees[level - 2].Edges[e.Endpoints.B];
                        Assert.True(ShareEndpoint(a, b), $"edge {e.Label} joins non-adjacent edges");
                    }
                }
                totalEdges += tree.Edges.Count;
            }
            Assert.Equal(d * (d - 1) / 2, totalEdges); // full vine has d(d-1)/2 edges
        }

        [Fact]
        public void SVine_HeldOutLogLikelihood_IncreasesWithTrees()
        {
            // Gaussian data: the simplifying assumption is exactly correct, so each deeper tree
            // captures real (constant) conditional dependence and raises the held-out likelihood.
            double[][] train = Equicorrelated(seed: 21, n: 2000, d: 4, rho: 0.6);
            double[][] test = Equicorrelated(seed: 22, n: 2000, d: 4, rho: 0.6);

            double ll1 = new RegularVine().Fit(train, nTrees: 1).LogLikelihood(test);
            double ll2 = new RegularVine().Fit(train, nTrees: 2).LogLikelihood(test);
            double ll3 = new RegularVine().Fit(train, nTrees: 3).LogLikelihood(test);

            Assert.True(ll2 > ll1 + 20.0, $"T_2 did not improve held-out log-lik: ll1={ll1}, ll2={ll2}");
            Assert.True(ll3 > ll2 + 20.0, $"T_3 did not improve held-out log-lik: ll2={ll2}, ll3={ll3}");
        }

        [Fact]
        public void Clayton_FitsClaytonDataBetterThanGaussian()
        {
            // Data drawn from a Clayton copula (lower-tail dependence). Fitting with the true
            // family should recover its tau and achieve a higher log-likelihood than a Gaussian
            // copula constrained to the same tau.
            const double trueTau = 0.5;
            double[][] x = ClaytonSample(seed: 31, n: 3000, tau: trueTau);

            var clayton = new RegularVine(CopulaFamily.Clayton).Fit(x); // d=2 -> single T_1 edge
            var gaussian = new RegularVine(CopulaFamily.Gaussian).Fit(x);

            double tauHat = clayton.Trees[0].Edges[0].Tau;
            Assert.True(System.Math.Abs(tauHat - trueTau) < 0.04, $"recovered tau={tauHat}, expected ~{trueTau}");

            double llClayton = clayton.LogLikelihood(x);
            double llGaussian = gaussian.LogLikelihood(x);
            Assert.True(llClayton > 0.0, $"Clayton log-lik unexpectedly low: {llClayton}");
            Assert.True(llClayton > llGaussian, $"true family did not win: clayton={llClayton}, gaussian={llGaussian}");
        }

        [Fact]
        public void Sample_RequiresCanonicalStructure()
        {
            var vine = new RegularVine().Fit(Equicorrelated(seed: 1, n: 500, d: 3, rho: 0.6)); // Regular
            Assert.Throws<NotSupportedException>(() => vine.Sample(10));
        }

        [Fact]
        public void CVine_Sample_ReproducesPairwiseDependenceAndMarginals()
        {
            double[][] x = Equicorrelated(seed: 50, n: 2000, d: 3, rho: 0.6);
            var vine = new RegularVine(CopulaFamily.Gaussian).Fit(x, structure: VineStructure.Canonical);
            double[][] s = vine.Sample(4000);

            Assert.Equal(4000, s.Length);
            Assert.Equal(3, s[0].Length);

            // The sampled joint reproduces the pairwise Kendall's tau of the training data ...
            for (int i = 0; i < 3; i++)
                for (int j = i + 1; j < 3; j++)
                {
                    double tauData = KendallTau.Compute(Col(x, i), Col(x, j));
                    double tauSamp = KendallTau.Compute(Col(s, i), Col(s, j));
                    Assert.True(System.Math.Abs(tauData - tauSamp) < 0.05,
                        $"pair ({i},{j}): data tau={tauData}, sample tau={tauSamp}");
                }

            // ... and the empirical marginals (data-scale draws via the inverse PIT).
            for (int j = 0; j < 3; j++)
                Assert.True(System.Math.Abs(Mean(Col(s, j)) - Mean(Col(x, j))) < 0.1,
                    $"marginal {j} mean mismatch");
        }

        [Fact]
        public void CVine_Sample_RoundTripRecoversTau()
        {
            double[][] x = Equicorrelated(seed: 71, n: 2000, d: 3, rho: 0.6);
            var v1 = new RegularVine(CopulaFamily.Gaussian).Fit(x, structure: VineStructure.Canonical);
            double[][] s = v1.Sample(4000);
            var v2 = new RegularVine(CopulaFamily.Gaussian).Fit(s, structure: VineStructure.Canonical);

            // First-tree dependences recovered after sample -> refit.
            double[] t1 = v1.Trees[0].Edges.Select(e => e.Tau).OrderBy(t => t).ToArray();
            double[] t2 = v2.Trees[0].Edges.Select(e => e.Tau).OrderBy(t => t).ToArray();
            for (int i = 0; i < t1.Length; i++)
                Assert.True(System.Math.Abs(t1[i] - t2[i]) < 0.05, $"tau[{i}]: {t1[i]} vs {t2[i]}");
        }

        [Fact]
        public void CVine_Sample_Clayton_ReproducesDependence()
        {
            double[][] x = ClaytonSample(seed: 60, n: 3000, tau: 0.5);
            var vine = new RegularVine(CopulaFamily.Clayton).Fit(x, structure: VineStructure.Canonical);
            double[][] s = vine.Sample(4000);
            double tauData = KendallTau.Compute(Col(x, 0), Col(x, 1));
            double tauSamp = KendallTau.Compute(Col(s, 0), Col(s, 1));
            Assert.True(System.Math.Abs(tauData - tauSamp) < 0.05, $"data tau={tauData}, sample tau={tauSamp}");
        }

        [Fact]
        public void CVine_SampleConditional_FixesValueAndShiftsConditional()
        {
            // Equicorrelated Gaussian: E[X_j | X_0 = x0] = rho * x0, so conditioning on a high vs
            // low X_0 must shift the sampled X_1, X_2 in the same direction.
            double[][] x = Equicorrelated(seed: 80, n: 2000, d: 3, rho: 0.6);
            var vine = new RegularVine(CopulaFamily.Gaussian)
                .Fit(x, structure: VineStructure.Canonical, rootOrder: new[] { 0 });

            double[][] hi = vine.SampleConditional(new Dictionary<int, double> { { 0, 2.0 } }, 2000);
            double[][] lo = vine.SampleConditional(new Dictionary<int, double> { { 0, -2.0 } }, 2000);

            // The conditioned variable is returned exactly.
            Assert.All(hi, r => Assert.Equal(2.0, r[0], 9));
            Assert.All(lo, r => Assert.Equal(-2.0, r[0], 9));

            // The sampled others shift with the conditioned value (E ~ rho*x0 = +-1.2).
            Assert.True(Mean(Col(hi, 1)) > 0.7 && Mean(Col(hi, 2)) > 0.7,
                $"hi means: {Mean(Col(hi, 1))}, {Mean(Col(hi, 2))}");
            Assert.True(Mean(Col(lo, 1)) < -0.7 && Mean(Col(lo, 2)) < -0.7,
                $"lo means: {Mean(Col(lo, 1))}, {Mean(Col(lo, 2))}");
        }

        [Fact]
        public void SampleConditional_RejectsNonRootOrRegular()
        {
            var canonical = new RegularVine(CopulaFamily.Gaussian)
                .Fit(Equicorrelated(seed: 2, n: 800, d: 3, rho: 0.6), structure: VineStructure.Canonical, rootOrder: new[] { 0 });
            // Variable 1 is not the leading root -> must throw with guidance.
            Assert.Throws<NotSupportedException>(() =>
                canonical.SampleConditional(new Dictionary<int, double> { { 1, 0.5 } }, 10));

            var regular = new RegularVine().Fit(Equicorrelated(seed: 3, n: 800, d: 3, rho: 0.6));
            Assert.Throws<NotSupportedException>(() =>
                regular.SampleConditional(new Dictionary<int, double> { { 0, 0.5 } }, 10));
        }

        // --- helpers ---------------------------------------------------------------------------

        private static double[] Col(double[][] m, int j)
        {
            double[] c = new double[m.Length];
            for (int i = 0; i < m.Length; i++) c[i] = m[i][j];
            return c;
        }

        private static double Mean(double[] a)
        {
            double s = 0; foreach (double v in a) s += v; return s / a.Length;
        }

        // Bivariate Clayton samples via conditional inversion; returned as raw (n x 2) data.
        private static double[][] ClaytonSample(int seed, int n, double tau)
        {
            Rand.Restart(seed);
            double theta = 2.0 * tau / (1.0 - tau);
            double[][] x = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double u1 = Rand.Double();
                double p = Rand.Double();
                double u2 = System.Math.Pow(
                    System.Math.Pow(u1, -theta) * (System.Math.Pow(p, -theta / (1.0 + theta)) - 1.0) + 1.0,
                    -1.0 / theta);
                x[i] = new[] { u1, u2 };
            }
            return x;
        }


        private static (int, int) Conditioned(VineEdge e) =>
            (System.Math.Min(e.Left, e.Right), System.Math.Max(e.Left, e.Right));

        private static bool ShareEndpoint(VineEdge a, VineEdge b) =>
            a.Endpoints.A == b.Endpoints.A || a.Endpoints.A == b.Endpoints.B ||
            a.Endpoints.B == b.Endpoints.A || a.Endpoints.B == b.Endpoints.B;

        // Equicorrelated Gaussian via a shared latent factor: corr(x_i, x_j) = rho for i != j.
        private static double[][] Equicorrelated(int seed, int n, int d, double rho)
        {
            Rand.Restart(seed);
            double sr = System.Math.Sqrt(rho), se = System.Math.Sqrt(1 - rho);
            double[][] x = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double w = Rand.Normal();
                double[] row = new double[d];
                for (int k = 0; k < d; k++)
                    row[k] = sr * w + se * Rand.Normal();
                x[i] = row;
            }
            return x;
        }
    }
}
