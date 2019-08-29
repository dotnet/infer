using System.IO;
using System.Collections.Generic;
using System.Linq;
using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
#if oxyplot
using OxyPlot.Wpf;
using OxyPlot;
#endif
using System.Threading;
// #define oxyplot
// Install OxPlot NuGet pcakages to enable graph plotting

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "A Gaussian Process regressor example")]
    public class GaussianProcessRegressor
    {
        // Train Gaussian Process on the small 'Auto Insurance in Sweden' dataset
        // Add the insurance.csv file to a folder named Data
        private const string AisCsvPath = @"..\..\..\Data\insurance.csv";

        // Path for the results plot
        private const string OutputPlotPath = @"..\..\..\Data\GPRegressionPredictions.png";

        public void Run()
        {
            var trainingData = LoadAISDataset();
            var trainingInputs = trainingData.Select(tup => Vector.FromArray(new double[1] { tup.x })).ToArray();
            var trainingOutputs = trainingData.Select(tup => tup.y).ToArray();

            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
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

        private IEnumerable<(double x, double y)> LoadAISDataset()
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

        private IEnumerable<(double x, double y)> PreprocessData(
            IEnumerable<(double x, double y)> data)
        {
            var x = data.Select(tup => tup.x);
            var y = data.Select(tup => tup.y);

            // Shift targets so mean is 0
            var meanY = y.Sum() / y.Count();
            y = y.Select(val => val - meanY);

            // Scale data to lie between 1 and -1
            var absoluteMaxY = y.Select(val => System.Math.Abs(val)).Max();
            y = y.Select(val => val / absoluteMaxY);
            var maxX = x.Max();
            x = x.Select(val => val / maxX);
            var dataset = x.Zip(y, (a, b) => (a, b));

            // Order data by input value
            return dataset.OrderBy(tup => tup.Item1);
        }
    }
}