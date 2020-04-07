// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "A Gaussian Process classifier example")]
    public class GaussianProcessClassifier
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
                Vector.FromArray(new double[2] { 0, 0 }),
                Vector.FromArray(new double[2] { 0, 1 }),
                Vector.FromArray(new double[2] { 1, 0 }),
                Vector.FromArray(new double[2] { 0, 0.5 }),
                Vector.FromArray(new double[2] { 1.5, 0 }),
                Vector.FromArray(new double[2] { 0.5, 1.0 })
            };
            
            bool[] outputs = { true, true, false, true, false, false };

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
            VariableArray<bool> y = Variable.Observed(outputs, j).Named("y");
            Variable<double> score = Variable.FunctionEvaluate(f, x[j]).Named("score");
            y[j] = (Variable.GaussianFromMeanAndVariance(score, 0.1) > 0);

            // Close the evidence block
            block.CloseBlock();

            // The basis
            Vector[] basis = new Vector[]
            {
                Vector.FromArray(new double[2] { 0.2, 0.2 }),
                Vector.FromArray(new double[2] { 0.2, 0.8 }),
                Vector.FromArray(new double[2] { 0.8, 0.2 }),
                Vector.FromArray(new double[2] { 0.8, 0.8 })
            };

            for (int trial = 0; trial < 3; trial++)
            {
                // The kernel
                IKernelFunction kf;
                if (trial == 0)
                {
                    kf = new SquaredExponential(-0.0);
                }
                else if (trial == 1)
                {
                    kf = new SquaredExponential(-0.5);
                }
                else
                {
                    kf = new NNKernel(new double[] { 0.0, 0.0 }, -1.0);
                }

                // Fill in the sparse GP prior
                GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);
                prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

                // Model score
                double NNscore = engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("{0} evidence = {1}", kf, NNscore.ToString("g4"));
            }

            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);

            // Check that training set is classified correctly
            Console.WriteLine("");
            Console.WriteLine("Predictions on training set:");
            for (int i = 0; i < outputs.Length; i++)
            {
                Gaussian post = sgp.Marginal(inputs[i]);
                double postMean = post.GetMean();
                string comment = (outputs[i] == (postMean > 0.0)) ? "correct" : "incorrect";
                Console.WriteLine("f({0}) = {1} ({2})", inputs[i], post, comment);
            }
        }
    }
}
