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

            var vine = new RegularVine(CopulaFamily.Gaussian).Fit(x);
            VineTree t1 = vine.Trees[0];
            Assert.Equal(3, t1.Edges.Count);

            var selected = new HashSet<(int, int)>(t1.Edges.Select(e => e.Conditioned));
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

            double llDep = new RegularVine().Fit(dependent).LogLikelihood(dependent);
            double llIndep = new RegularVine().Fit(independent).LogLikelihood(independent);

            // Dependence is real -> the copula adds substantial likelihood; independence data
            // fits tau ~ 0 -> near-zero contribution.
            Assert.True(llDep > 100.0, $"dependent log-lik unexpectedly low: {llDep}");
            Assert.True(llDep > llIndep + 100.0, $"llDep={llDep} not clearly above llIndep={llIndep}");
            Assert.True(System.Math.Abs(llIndep) < 50.0, $"independent log-lik unexpectedly large: {llIndep}");
        }
    }
}
