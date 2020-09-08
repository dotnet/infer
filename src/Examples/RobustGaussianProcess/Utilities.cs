// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

#if WINDOWS
using OxyPlot.Wpf;
using OxyPlot;
using System.Threading;
using Microsoft.ML.Probabilistic.Distributions;
#endif

namespace RobustGaussianProcess
{
    class Utilities
    {
        // Train Gaussian Process on the small 'Auto Insurance in Sweden' dataset
        // The insurance.csv file can be found in the Data directory
        private const string AisCsvPath = @"Data\insurance.csv";

        // Path for the results plot
        private const string OutputPlotPath = @"Data";

        /// <summary>
        /// Generates a 1D vector with length len having a min and max; data points are randomly distributed and ordered if specified
        /// </summary>
        public static Vector[] VectorRange(double min, double max, int len, Random rng)
        {
            var inputs = new double[len];
            double num;

            for (int i = 0; i < len; i++)
            {
                if (rng != null)
                {
                    num = rng.NextDouble();
                }
                else
                {
                    num = i / (double)(len - 1);
                }
                num = num * (max - min);
                num += min;
                inputs[i] = num;
            }

            if (rng != null)
            {
                inputs = inputs.OrderBy(x => x).ToArray();
            }

            return inputs.Select(x => Vector.FromArray(new double[1] { x })).ToArray();
        }

        /// <summary>
        /// Read the Auto Insurance in Sweden dataset from its CSV file
        /// </summary>
        public static IEnumerable<(double x, double y)> LoadAISDataset()
        {
            var data = new List<(double x, double y)>();

            // Read CSV file
            using (var reader = new StreamReader(AisCsvPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    if (values.Length == 2)
                    {
                        data.Add((double.Parse(values[0]), double.Parse(values[1])));
                    }
                }
            }

            return PreprocessData(data).ToList();
        }

        private static IEnumerable<(double x, double y)> PreprocessData(
            IEnumerable<(double x, double y)> data)
        {
            var x = data.Select(tup => tup.x);
            var y = data.Select(tup => tup.y);

            // Shift targets so mean is 0
            var meanY = y.Sum() / y.Count();
            y = y.Select(val => val - meanY);

            // Scale data to lie between 1 and -1
            var absoluteMaxY = y.Select(val => Math.Abs(val)).Max();
            y = y.Select(val => val / absoluteMaxY);
            var maxX = x.Max();
            x = x.Select(val => val / maxX);
            var dataset = x.Zip(y, (a, b) => (a, b));

            // Order data by input value
            return dataset.OrderBy(tup => tup.Item1);
        }

        public static InferenceEngine GetInferenceEngine()
        {
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is ExpectationPropagation))
            {
                throw new ArgumentException("This example only runs with Expectation Propagation");
            }
            return engine;
        }

#if WINDOWS
        public static void PlotGraph(PlotModel model, string graphPath)
        {
            // Required for plotting
            Thread thread = new Thread(() => PngExporter.Export(model, graphPath, 800, 600, OxyColors.White));
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();
        }

        public static void PlotPredictions(SparseGP sgp, Vector[] trainingInputs, double[] trainingOutputs, bool useStudentT, string dataset)
        {
            var meanSeries = new OxyPlot.Series.LineSeries { Title = "Mean function", Color = OxyColors.SkyBlue, };
            var scatterSeries = new OxyPlot.Series.ScatterSeries { Title = "Training points" };
            var areaSeries = new OxyPlot.Series.AreaSeries { Title = "\u00B1 2\u03C3", Color = OxyColors.PowderBlue };

            double sqDiff = 0;
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                Gaussian post = sgp.Marginal(trainingInputs[i]);
                double postMean = post.GetMean();
                var xTrain = trainingInputs[i][0];
                meanSeries.Points.Add(new DataPoint(xTrain, postMean));
                scatterSeries.Points.Add(new OxyPlot.Series.ScatterPoint(xTrain, trainingOutputs[i]));

                var stdDev = Math.Sqrt(post.GetVariance());
                areaSeries.Points.Add(new DataPoint(xTrain, postMean + (2 * stdDev)));
                areaSeries.Points2.Add(new DataPoint(xTrain, postMean - (2 * stdDev)));
                sqDiff += Math.Pow(postMean - trainingOutputs[i], 2);
            }

            Console.WriteLine("RMSE is: {0}", Math.Sqrt(sqDiff / trainingOutputs.Length));

            var model = new PlotModel();
            string pngPath;
            if (!useStudentT)
            {
                model.Title = $"Gaussian Process trained on {dataset} dataset";
                pngPath = Path.Combine(OutputPlotPath, $"{dataset}.png");
            }
            else
            {
                model.Title = $"Gaussian Process trained on {dataset} dataset (Student-t likelihood)";
                pngPath = Path.Combine(OutputPlotPath, $"StudentT{dataset}.png");
            }

            model.Series.Add(meanSeries);
            model.Series.Add(scatterSeries);
            model.Series.Add(areaSeries);
            model.Axes.Add(new OxyPlot.Axes.LinearAxis {
                Position = OxyPlot.Axes.AxisPosition.Bottom,
                Title = "x" });
            model.Axes.Add(new OxyPlot.Axes.LinearAxis {
                Position = OxyPlot.Axes.AxisPosition.Left,
                Title = "y" });

            PlotGraph(model, pngPath);
            Console.WriteLine($"Saved PNG to {Path.GetFullPath(pngPath)}");
        }
#endif
    }
}
