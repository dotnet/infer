// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using System;

namespace RobustGaussianProcess
{
    class GaussianProcessDataGenerator
    {
        // Path for the results plot
        private const string OutputPlotPath = @"..\..\..\TutorialData\GPRandomDataGenerator.png";

        public static (Vector[] dataX, double[] dataY) GenerateRandomData(int numData, double proportionCorrupt)
        {
            InferenceEngine engine = Utilities.GetInferenceEngine();

            // The points to evaluate
            Vector[] randomInputs = Utilities.VectorRange(0, 1, numData, true);

            var gaussianProcessGenerator = new GaussianProcessRegressor(randomInputs);
            gaussianProcessGenerator.Block.CloseBlock();

            // The basis
            Vector[] basis = Utilities.VectorRange(0, 1, 6, false);

            // The kernel
            var kf = new SquaredExponential(-1);

            // Fill in the sparse GP prior
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);
            gaussianProcessGenerator.Prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

            // Infer the posterior Sparse GP, and sample a random function from it
            SparseGP sgp = engine.Infer<SparseGP>(gaussianProcessGenerator.F);
            var randomFunc = sgp.Sample();

            Random rng = new Random();

            double[] randomOutputs = new double[randomInputs.Length];

            // get random data
            for (int i = 0; i < randomInputs.Length; i++)
            {
                double post = randomFunc.Evaluate(randomInputs[i]);
                // corrupt data point if it we haven't exceed the proportion we wish to corrupt
                if (i < proportionCorrupt * numData)
                {
                    double sign = rng.NextDouble() > 0.5 ? 1 : -1;
                    double distance = rng.NextDouble() * 3;
                    post = (sign * distance) + post;
                }

                randomOutputs[i] = post;
            }

            int numCorrupted = (int)System.Math.Ceiling(numData * proportionCorrupt);
            Console.WriteLine("Model complete: Generated {0} points with {1} corrupted", numData, numCorrupted);

            return (randomInputs, randomOutputs);
        }

#if oxyplot
    private void PlotScatter(Vector[] dataX, double[] dataY)
    {
        var scatterSeries = new OxyPlot.Series.ScatterSeries { Title = "Random data" };

        for (int i = 0; i < dataX.Length; i++)
        {
            scatterSeries.Points.Add(new OxyPlot.Series.ScatterPoint(dataX[i][0], dataY[i]));
        }

        var model = new PlotModel();
        model.Title = "Random Function drawn from a GP with corrupted data points";
        model.Series.Add(scatterSeries);
        model.Axes.Add(new OxyPlot.Axes.LinearAxis
        {
            Position = OxyPlot.Axes.AxisPosition.Bottom,
            Title = "x"
        });
        model.Axes.Add(new OxyPlot.Axes.LinearAxis
        {
            Position = OxyPlot.Axes.AxisPosition.Left,
            Title = "y"
        });

        Utilities.PlotModel(model, OutputPlotPath);

        Console.WriteLine("Saved PNG to {0}", OutputPlotPath);
    }
#endif
    }
}
