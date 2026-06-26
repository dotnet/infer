// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Distributions.Copulas.Vine;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    /// <summary>
    /// Gaussian Process Vine Copulas (GPVINE) of Lopez-Paz, Hernandez-Lobato and Ghahramani
    /// (ICML 2013). Builds a regular vine over synthetic data whose conditional copula varies
    /// with its conditioning variable, and compares the held-out log-likelihood of the full
    /// GPVINE (conditional deeper trees fitted with a sparse GP + EP) against the SVINE baseline
    /// (the simplifying assumption that ignores conditional dependence).
    /// </summary>
    [Example("Applications", "Gaussian Process vine copulas for multivariate dependence")]
    public class GaussianProcessVine
    {
        public void Run()
        {
            var engine = new InferenceEngine(new ExpectationPropagation());
            if (!(engine.Algorithm is ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            }
            engine.ShowProgress = false;

            // Synthetic data (X, Y, Z): X and Y are each dependent on Z, while the copula of
            // (X, Y) | Z has a Kendall's tau that varies with Z -- the structure that the
            // simplifying assumption fails to capture.
            double[][] train = MakeData(seed: 1, n: 300);
            double[][] test = MakeData(seed: 2, n: 300);

            Console.WriteLine("GPVINE on synthetic data with a Z-varying conditional copula\n");
            Console.WriteLine("trees | SVINE test log-lik | GPVINE test log-lik");
            Console.WriteLine("------+--------------------+--------------------");

            var fitter = new GaussianProcessCopulaFitter(engine) { NumInducing = 15, NumberOfIterations = 15 };
            for (int nTrees = 1; nTrees <= 2; nTrees++)
            {
                double svine = new RegularVine(CopulaFamily.Gaussian).Fit(train, nTrees).LogLikelihood(test);
                double gpvine = new RegularVine(CopulaFamily.Gaussian).Fit(train, nTrees, fitter).LogLikelihood(test);
                Console.WriteLine($"  {nTrees}   |     {svine,10:f2}     |     {gpvine,10:f2}");
            }

            Console.WriteLine("\nGPVINE should match SVINE at one tree (T_1 is unconditional) and");
            Console.WriteLine("overtake it once the conditional second-tree copula is modelled.");
        }

        // corr(X, Z), corr(Y, Z) make T_1 select the hub edges X-Z and Y-Z; rho(Z) gives the
        // (X, Y) | Z copula a conditioning-dependent Kendall's tau.
        private static double[][] MakeData(int seed, int n)
        {
            Rand.Restart(seed);
            double[][] x = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double z = Rand.Normal();
                double rho = 0.9 * System.Math.Sin(1.5 * z);
                double ex = Rand.Normal();
                double ey = rho * ex + System.Math.Sqrt(1 - rho * rho) * Rand.Normal();
                x[i] = new[] { 0.7 * z + 0.6 * ex, 0.7 * z + 0.6 * ey, z };
            }
            return x;
        }
    }
}
