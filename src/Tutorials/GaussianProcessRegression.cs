// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "A Gaussian Process regression example")]
    class GaussianProcessRegression
    {
        public void Run()
        {
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            }

            // The data
            Vector[] inputs = new Vector[]
            {
                Vector.FromArray(new double[1] { -5 }),
                Vector.FromArray(new double[1] { -3 }),
                Vector.FromArray(new double[1] { -1 }),
                Vector.FromArray(new double[1] { 1 }),
                Vector.FromArray(new double[1] { 3 }),
                Vector.FromArray(new double[1] { 5 })
            };

            double[] outputs = { -5, -3, -1, 1, 3, 5 };

            // Open an evidence block to allow model scoring
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            // Set up the GP prior, a distribution over functions, which will be filled in later
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");

            // The sparse GP variable - a random function
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

            // The locations to evaluate the function
            VariableArray<Vector> x = Variable.Observed(inputs).Named("x");
            Range j = x.Range.Named("j");

            // The observation model
            VariableArray<double> y = Variable.Observed(outputs, j).Named("y");
            Variable<double> score = Variable.FunctionEvaluate(f, x[j]).Named("score");
            y[j] = Variable.GaussianFromMeanAndVariance(score, 0.1);

            // Close the evidence block
            block.CloseBlock();

            // The basis
            Vector[] basis = new Vector[]
            {
                Vector.FromArray(new double[1] { -5.0 }),
                Vector.FromArray(new double[1] { -3.0 }),
                Vector.FromArray(new double[1] { -1.0 }),
                Vector.FromArray(new double[1] { 1.0 }),
                Vector.FromArray(new double[1] { 3.0 }),
                Vector.FromArray(new double[1] { 5.0 }),
            };

            for (int trial = 0; trial < 2; trial++)
            {
                // The kernel
                IKernelFunction kf;
                if (trial == 0)
                {
                    kf = new SquaredExponential(-0.0);
                }
                else
                {
                    kf = new SquaredExponential(0.5);
                }

                // Fill in the sparse GP prior
                GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);
                prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

                // Model score
                double logLikeScore = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("{0} evidence = {1}", kf, logLikeScore.ToString("g4"));
            }

            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);

            // Check that training set is regressed on correctly
            Console.WriteLine("");
            Console.WriteLine("Predictions on training set:");
            double sqDiff = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                Gaussian post = sgp.Marginal(inputs[i]);
                double postMean = post.GetMean();
                sqDiff += System.Math.Pow(postMean - outputs[i], 2);
                Console.WriteLine("f({0}) = {1} (actual {2})", inputs[i], post, outputs[i]);
            }
            double rmse = System.Math.Sqrt(sqDiff / outputs.Length);
            Console.WriteLine("RMSE is: {0}", rmse);
        }
    }
}
