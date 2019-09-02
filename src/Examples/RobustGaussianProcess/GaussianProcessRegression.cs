// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Collections.Generic;
using System.Linq;
using System;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
#if oxyplot
using OxyPlot.Wpf;
using OxyPlot;
using System.Threading;
#endif
// #define oxyplot
// Install OxPlot NuGet pcakages to enable graph plotting

namespace RobustGaussianProcess
{
    public class GaussianProcessRegression
    {
        public void Run()
        {
            var trainingData = Utilities.LoadAISDataset();
            var trainingInputs = trainingData.Select(tup => Vector.FromArray(new double[1] { tup.x })).ToArray();
            var trainingOutputs = trainingData.Select(tup => tup.y).ToArray();

            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            }

            // Modelling code
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
            VariableArray<Vector> x = Variable.Observed(trainingInputs).Named("x");
            Range j = x.Range.Named("j");
            VariableArray<double> y = Variable.Observed(trainingOutputs, j).Named("y");
            Variable<double> score = Variable.FunctionEvaluate(f, x[j]);

            // Add some noise to the score
            y[j] = Variable.GaussianFromMeanAndVariance(score, 0.5);
            block.CloseBlock();

            // Log length scale estimated as -1
            var kf = new SquaredExponential(-1);
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);

            // Convert SparseGP to full Gaussian Process by evaluating at all the training points
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, trainingInputs.ToArray()));
            double logOdds = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("{0} evidence = {1}", kf, logOdds.ToString("g4"));

            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);
            //PlotPredictions(sgp, trainingInputs, trainingOutputs);
        }
#if oxyplot
        private void PlotPredictions(SparseGP sgp, Vector[] trainingInputs, double[] trainingOutputs)
        {
            var meanSeries = new OxyPlot.Series.LineSeries { Title = "Mean function", Color = OxyColors.SkyBlue, };
            var scatterSeries = new OxyPlot.Series.ScatterSeries { Title = "Training points" };
            var areaSeries = new OxyPlot.Series.AreaSeries { Title = "\u00B1 2\u03C3", Color = OxyColors.PowderBlue };

            for (int i = 0; i < trainingInputs.Length; i++)
            {
                Gaussian post = sgp.Marginal(trainingInputs[i]);
                double postMean = post.GetMean();
                var xTrain = trainingInputs[i][0];
                meanSeries.Points.Add(new DataPoint(xTrain, postMean));
                scatterSeries.Points.Add(new OxyPlot.Series.ScatterPoint(xTrain, trainingOutputs[i]));

                var mean = 0.0;
                var precision = 0.0;
                post.GetMeanAndPrecision(out mean, out precision);
                var stdDev = System.Math.Sqrt(1 / precision);
                areaSeries.Points.Add(new DataPoint(xTrain, postMean + (2 * stdDev)));
                areaSeries.Points2.Add(new DataPoint(xTrain, postMean - (2 * stdDev)));
            }

            var model = new PlotModel();
            model.Title = "Gaussian Process trained on Auto Insurance in Sweden dataset";
            model.Series.Add(meanSeries);
            model.Series.Add(scatterSeries);
            model.Series.Add(areaSeries);
            model.Axes.Add(new OxyPlot.Axes.LinearAxis {
                Position = OxyPlot.Axes.AxisPosition.Bottom,
                Title = "x (number of claims)" });
            model.Axes.Add(new OxyPlot.Axes.LinearAxis {
                Position = OxyPlot.Axes.AxisPosition.Left,
                Title = "y (total payment for all the claims)" });

            // Required for plotting
            Thread thread = new Thread(() => PngExporter.Export(model, OutputPlotPath, 600, 400, OxyColors.White));
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();

            Console.WriteLine("Saved PNG to {0}", OutputPlotPath);
        }
#endif
    }
}
