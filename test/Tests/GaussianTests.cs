// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Xunit.Assert;

    /// <summary>
    /// Gaussian factor tests, all using MSL.
    /// </summary>
    public class GaussianTests
    {
        [Fact]
        [Trait("Category", "CsoftModel")]
        public void HelloGaussianTest()
        {
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca2 = engine.Compiler.Compile(HelloGaussianModel);
            ca2.Execute(1);
            Console.WriteLine("output=" + ca2.Marginal("x"));
            Gaussian expected = new Gaussian(0, 1);
            Gaussian actual = ca2.Marginal<Gaussian>("x");
            Assert.True(expected.MaxDiff(actual) < 1e-10);
        }

        private void HelloGaussianModel()
        {
            double x = Factor.Random(Gaussian.FromMeanAndVariance(0, 1));
            InferNet.Infer(x, nameof(x));
        }


        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ClippedGaussianTest()
        {
            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.Compile(ClippedGaussianModel, 0.0);
            Gaussian[] expected = new Gaussian[]
                {
                    new Gaussian(0.7979, 0.3634),
                    new Gaussian(0.8626, 0.3422),
                    new Gaussian(0.9294, 0.3221),
                    new Gaussian(0.9982, 0.3031),
                    new Gaussian(1.069, 0.2853),
                    new Gaussian(1.141, 0.2685),
                    new Gaussian(1.215, 0.2527),
                    new Gaussian(1.29, 0.238),
                    new Gaussian(1.367, 0.2241),
                    new Gaussian(1.446, 0.2112)
                };
            for (int i = 0; i < 10; i++)
            {
                double truncAt = ((double) i)/10; //truncAt = 0.5;
                ca.SetObservedValue("threshold", truncAt);
                ca.Execute(1);
                Console.WriteLine("Marginal for X, truncated at " + (float) truncAt + "=" + ca.Marginal("x"));
                Gaussian actual = ca.Marginal<Gaussian>("x");
                Assert.True(expected[i].MaxDiff(actual) < 5e-3);
            }
        }

        private void ClippedGaussianModel(double threshold)
        {
            double x = Factor.Random(new Gaussian(0, 1));
            //Attrib.Var(x, new Algorithm(new VariationalMessagePassing()));
            double diff = Factor.Difference(x, threshold);
            InferNet.Infer(x, nameof(x));
            bool h = Factor.IsPositive(diff);
            Constrain.Equal(true, h);
        }
    }
}