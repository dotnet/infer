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
    /// <summary>
    /// This script generates a dataset which has the following pipeline:
    /// 1) randomly sample a 1D function from a GP;
    /// 2) pick a random subset of 'numData' points;
    /// 3) pick a further random proportion 'propCorrupt' of 'numData' to corrupt according to a uniform distribution with a range of -3 to 3
    /// </summary>
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

            // Number of datapoints
            int numData = 30;

            // The proportion of points to randomly corrupt
            double propCorrupt = 0.3;

            // The points to evaluate
            Vector[] inputs = this.VectorRange(-5, 5, numData, true);

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
            Vector[] basis = this.VectorRange(-5, 5, 6, false);

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
            var p1 = new OxyPlot.Series.ScatterSeries
            {
                Title = "Random Function"
            };

            Random rng = new Random();

            // get random data
            for (int i = 0; i < inputs.Length; i++)
            {
                double post = randomFunc.Evaluate(inputs[i]);
                // corrupt data point if it we haven't exceed the proportion we wish to corrupt
                if (i < propCorrupt * numData)
                {
                    double sign = rng.NextDouble() > 0.5 ? 1 : -1;
                    double distance = rng.NextDouble() * 3;
                    post = (sign * distance) + post;
                }

                p1.Points.Add(new OxyPlot.Series.ScatterPoint(inputs[i][0], post));
            }

            var model = new PlotModel();
            model.Series.Add(p1);

            Thread thread = new Thread(() => DisplayPNG(model));
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();

            Console.WriteLine("Plotting complete: Generated {0} points with {1} corrupted", numData, (int)System.Math.Ceiling(numData * propCorrupt));
        }

        private void DisplayPNG(PlotModel model)
        {
            var a = Thread.CurrentThread.GetApartmentState();
            var outputToFile = "test-oxyplot-file.png";
            PngExporter.Export(model, outputToFile, 600, 400, OxyColors.White);
        }

        /// <summary>
        /// Generates a 1D vector with length len having a min and max; data points are randomly distributed and ordered if specified
        /// </summary>
        private Vector[] VectorRange(double min, double max, int len, bool random)
        {
            Vector[] inputs = new Vector[len];
            Random rng = new Random();

            for (int i = 0; i < len; i++)
            {
                double num = new double();

                if (random)
                {
                    num = rng.NextDouble();
                }
                else
                {
                    num = i / (double)(len - 1);
                }
                num = num * (max - min);
                num += min;
                inputs[i] = Vector.FromArray(new double[1] { num });
            }

            return inputs;
        }
    }
}
