// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using OxyPlot;
using OxyPlot.Wpf;
using System;
using System.Threading;

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

            // The points to evaluate
            Vector[] inputs = this.VectorLinSpace(-5, 5, 51);

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
            Vector[] basis = VectorLinSpace(-5, 5, 6);

            // The kernel
            IKernelFunction kf;
            kf = new SquaredExponential(1);

            // Fill in the sparse GP prior
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

            // Infer the posterior Sparse GP, and sample a random function from it
            SparseGP sgp = engine.Infer<SparseGP>(f);
            var randomFunc = sgp.Sample();

            // plotting boilerplate
            var p1 = new OxyPlot.Series.LineSeries
            {
                Title = "Random Function"
            };

            Console.WriteLine("");
            Console.WriteLine("Random function evaluations:");
            for (int i = 0; i < inputs.Length; i++)
            {
                double post = randomFunc.Evaluate(inputs[i]);
                Console.WriteLine("f({0}) = {1}", inputs[i], post);
                p1.Points.Add(new DataPoint(inputs[i][0], post));
            }

            var model = new PlotModel();
            model.Series.Add(p1);

            Thread thread = new Thread(() => DisplayPNG(model));
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();
        }

        private void DisplayPNG(PlotModel model)
        {
            var a = Thread.CurrentThread.GetApartmentState();
            var outputToFile = "test-oxyplot-file.png";
            PngExporter.Export(model, outputToFile, 600, 400, OxyColors.White);
        }

        private Vector[] VectorLinSpace(int min, int max, int len)
        {
            Vector[] inputs = new Vector[len];

            for (int i = 0; i < len; i++)
            {
                double num = i / (double)(len - 1);
                num = num * (max - min);
                num += min;
                inputs[i] = Vector.FromArray(new double[1] { num });
            }

            return inputs;
        }
    }
}
