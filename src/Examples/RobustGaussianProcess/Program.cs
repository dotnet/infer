#define oxyplot
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RobustGaussianProcess
{
    class Program
    {
        static void Main()
        {
            FitDataset(true);
            FitDataset(false);
        }

        static void FitDataset(bool fitAIS)
        {
            Vector[] trainingInputs;
            double[] trainingOutputs;
            string datasetName;

            if (fitAIS)
            {
                datasetName = "AIS";
                var trainingData = Utilities.LoadAISDataset();
                trainingInputs = trainingData.Select(tup => Vector.FromArray(new double[1] { tup.x })).ToArray();
                trainingOutputs = trainingData.Select(tup => tup.y).ToArray();
            }
            else
            {
                datasetName = "Generated";
                (trainingInputs, trainingOutputs) = GaussianProcessDataGenerator.GenerateRandomData(30, 0.3);
            }

            InferenceEngine engine = Utilities.GetInferenceEngine();

            for (var i = 0; i < 2; i++)
            {
                var gaussianProcessRegressor = new GaussianProcessRegressor(trainingInputs, i != 0, trainingOutputs);
                gaussianProcessRegressor.Block.CloseBlock();

                // Log length scale estimated as -1
                var kf = new SquaredExponential(-1);
                GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);

                // Convert SparseGP to full Gaussian Process by evaluating at all the training points
                gaussianProcessRegressor.Prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, trainingInputs.ToArray()));
                double logOdds = engine.Infer<Bernoulli>(gaussianProcessRegressor.Evidence).LogOdds;
                Console.WriteLine("{0} evidence = {1}", kf, logOdds.ToString("g4"));

                // Infer the posterior Sparse GP
                SparseGP sgp = engine.Infer<SparseGP>(gaussianProcessRegressor.F);

#if oxyplot
                Utilities.PlotPredictions(sgp, trainingInputs, trainingOutputs, i != 0, datasetName);
#endif
            }
        }
    }
}
