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
    class GaussianProcessDataGenerator
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

            //double[] outputs = { -5, -3, -1, 1, 3, 5 };

            // Set up the GP prior, a distribution over functions, which will be filled in later
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");

            // The sparse GP variable - a random function
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

            // The locations to evaluate the function
            VariableArray<Vector> x = Variable.Observed(inputs).Named("x");
            Range j = x.Range.Named("j");

            // The observation model
            Variable<double> score = Variable.FunctionEvaluate(f, x[j]).Named("score");

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

            // The kernel
            IKernelFunction kf;
            kf = new SquaredExponential(1);

            // Fill in the sparse GP prior
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);
            var randomFunc = sgp.Sample();

            // Check that training set is regressed on correctly
            Console.WriteLine("");
            Console.WriteLine("Random function evaluations:");
            for (int i = 0; i < inputs.Length; i++)
            {
                double post = randomFunc.Evaluate(inputs[i]);
                Console.WriteLine("f({0}) = {1}", inputs[i], post);
            }
        }
    }
}
