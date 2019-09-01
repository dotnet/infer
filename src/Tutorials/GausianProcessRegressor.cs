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
using System.Threading;
#endif
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
        private const string OutputPlotPath = @"..\..\..\Data\GPRegressionPredictions";

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

            for(var i = 0; i < 2; i++)
            {
                // Modelling code
                Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
                IfBlock block = Variable.If(evidence);
                Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");
                Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
                VariableArray<Vector> x = Variable.Observed(trainingInputs).Named("x");
                Range j = x.Range.Named("j");
                VariableArray<double> y = Variable.Observed(trainingOutputs, j).Named("y");

                if (i == 0)
                {
                    // Standard Gaussian Process
                    Console.WriteLine("Training a Gaussian Process regressor");
                    var score = GetScore(x, f, j);
                    y[j] = Variable.GaussianFromMeanAndVariance(score, 0.8);
                }
                else
                {
                    // Gaussian Process with Student-t likelihood
                    Console.WriteLine("Training a Gaussian Process regressor with Student-t likelihood");
                    var noisyScore = GetNoisyScore(x, f, j, trainingOutputs);
                    y[j] = Variable.GaussianFromMeanAndVariance(noisyScore[j], 0.8);
                }

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

                #if oxyplot
                PlotPredictions(sgp, trainingInputs, trainingOutputs, i != 0);
                #endif
            }
        }

        #if oxyplot
        private void PlotPredictions(SparseGP sgp, Vector[] trainingInputs, double[] trainingOutputs, bool useStudentT)
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

                var mean = 0.0;
                var precision = 0.0;
                post.GetMeanAndPrecision(out mean, out precision);
                var stdDev = System.Math.Sqrt(1 / precision);
                areaSeries.Points.Add(new DataPoint(xTrain, postMean + (2 * stdDev)));
                areaSeries.Points2.Add(new DataPoint(xTrain, postMean - (2 * stdDev)));
                sqDiff += System.Math.Pow(postMean - trainingOutputs[i], 2);
            }

            Console.WriteLine("RMSE is: {0}", System.Math.Sqrt(sqDiff / trainingOutputs.Length));

            var model = new PlotModel();
            string pngPath;
            if (!useStudentT)
            {
                model.Title = "Gaussian Process trained on AIS dataset";
                pngPath = string.Format("{0}.png", OutputPlotPath);
            }
            else
            {
                model.Title = "Gaussian Process trained on AIS dataset (Student-t likelhood)";
                pngPath = string.Format("{0}{1}.png", OutputPlotPath, "StudentT");
            }

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
            Thread thread = new Thread(() => PngExporter.Export(model, pngPath, 600, 400, OxyColors.White));
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();

            Console.WriteLine("Saved PNG to {0}", pngPath);
        }
        #endif

        private Variable<double> GetScore(VariableArray<Vector> x, Variable<IFunction> f, Range j)
        {
            return Variable.FunctionEvaluate(f, x[j]);
        }

        private VariableArray<double> GetNoisyScore(VariableArray<Vector> x, Variable<IFunction> f, Range j, double[] trainingOutputs)
        {
            // The student-t distribution arises as the mean of a normal distribution once an unknown precision is marginalised out
            Variable<double> score = GetScore(x, f, j);
            VariableArray<double> noisyScore = Variable.Observed(trainingOutputs, j).Named("noisyScore");
            using (Variable.ForEach(j))
            {
                // The precision of the Gaussian is modelled with a Gamma distribution
                var precision = Variable.GammaFromShapeAndRate(4, 1).Named("precision");
                noisyScore[j] = Variable.GaussianFromMeanAndPrecision(score, precision);
            }
            return noisyScore;
        }

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
