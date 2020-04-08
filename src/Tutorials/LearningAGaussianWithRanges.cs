// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Tutorials", "Efficient Bayesian learning of the mean and precision of some data, using ranges.", Prefix = "3b.")]
    public class LearningAGaussianWithRanges
    {
        public void Run()
        {
            // Sample data from standard Gaussian
            double[] data = new double[100];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Rand.Normal(0, 1);
            }

            // Create mean and precision random variables
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
            Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");

            Range dataRange = new Range(data.Length).Named("n");
            VariableArray<double> x = Variable.Array<double>(dataRange).Named("x");
            x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange);
            x.ObservedValue = data;

            InferenceEngine engine = new InferenceEngine();

            // Retrieve the posterior distributions
            Console.WriteLine("mean=" + engine.Infer(mean));
            Console.WriteLine("prec=" + engine.Infer(precision));
        }
    }
}
