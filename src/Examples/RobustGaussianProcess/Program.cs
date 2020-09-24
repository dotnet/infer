// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using System;
using System.Linq;

namespace RobustGaussianProcess
{
    class Program
    {
        /// <summary>
        /// Main for Gaussian Process regression example
        /// Fits two datasets (real and synthetic) using a standard Gaussian Process and a Robust Student-T Gaussian Process
        /// </summary>
        static void Main()
        {
            //FitDataset(useSynthetic: false);
            FitDataset(useSynthetic: true);
        }

        static void FitDataset(bool useSynthetic)
        {
            Vector[] trainingInputs;
            double[] trainingOutputs;

            if (!useSynthetic)
            {
                var trainingData = Utilities.LoadAISDataset();
                trainingInputs = trainingData.Select(tup => Vector.FromArray(new double[1] { tup.x })).ToArray();
                trainingOutputs = trainingData.Select(tup => tup.y).ToArray();
            }
            else
            {
                (trainingInputs, trainingOutputs) = GaussianProcessDataGenerator.GenerateRandomData(30, 0.3);
            }

            InferenceEngine engine = Utilities.GetInferenceEngine();

            // First fit standard GP, then fit Student-T GP
            foreach (var useStudentTLikelihood in new[] { false, true })
            {
                var gaussianProcessRegressor = new GaussianProcessRegressor(trainingInputs, useStudentTLikelihood, trainingOutputs);

                // Log length scale estimated as -1
                var noiseVariance = 0.8;
                var kf = new SummationKernel(new SquaredExponential(-1)) + new WhiteNoise(Math.Log(noiseVariance) / 2);
                GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), kf);

                // Convert SparseGP to full Gaussian Process by evaluating at all the training points
                gaussianProcessRegressor.Prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, trainingInputs.ToArray()));
                double logOdds = engine.Infer<Bernoulli>(gaussianProcessRegressor.Evidence).LogOdds;
                Console.WriteLine("{0} evidence = {1}", kf, logOdds.ToString("g4"));

                // Infer the posterior Sparse GP
                SparseGP sgp = engine.Infer<SparseGP>(gaussianProcessRegressor.F);
#if WINDOWS
                string datasetName = useSynthetic ? "Synthetic" : "AIS";
                Utilities.PlotPredictions(sgp, trainingInputs, trainingOutputs, useStudentTLikelihood, datasetName);
#endif
            }
        }
    }
}
