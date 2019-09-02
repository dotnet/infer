// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using System;
#if oxyplot
using OxyPlot;
using OxyPlot.Wpf;
using System.Threading;
#endif

namespace RobustGaussianProcess
{
    class GaussianProcessDataGenerator
    {
        // Path for the results plot
        private const string OutputPlotPath = @"..\..\..\TutorialData\GPRandomDataGenerator.png";

        public void Run()
        {
            (Vector[] dataX, double[] dataY) = GenerateRandomData(30, 0.3);
#if oxyplot
        PlotScatter(dataX, dataY);
#endif
        }

        public (Vector[] dataX, double[] dataY) GenerateRandomData(int numData, double proportionCorrupt)
        {
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is ExpectationPropagation))
            {
                throw new ArgumentException("This example only runs with Expectation Propagation");
            }

            // The points to evaluate
            Vector[] randomInputs = Utilities.VectorRange(-5, 5, numData, true);

            // Set up the GP prior, a distribution over functions, which will be filled in later
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");

            // The sparse GP variable - a random function
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

            // The locations to evaluate the function
            VariableArray<Vector> x = Variable.Observed(randomInputs).Named("x");
            Range j = x.Range.Named("j");

            // The observation model
            Variable<double> score = Variable.FunctionEvaluate(f, x[j]).Named("score");

            // The basis
            Vector[] basis = Utilities.VectorRange(-5, 5, 6, false);

            // The kernel
            IKernelFunction kf;
            kf = new SquaredExponential(1);

            // Fill in the sparse GP prior
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));

            // Infer the posterior Sparse GP, and sample a random function from it
            SparseGP sgp = engine.Infer<SparseGP>(f);
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

        // Required for plotting
        Thread thread = new Thread(() => PngExporter.Export(model, OutputPlotPath, 600, 400, OxyColors.White));
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();

        Console.WriteLine("Saved PNG to {0}", OutputPlotPath);
    }
#endif
    }
}
