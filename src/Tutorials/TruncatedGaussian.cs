// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Tutorials", "Inefficient implementation of a truncated Gaussian distribution.", Prefix = "2a.")]
    public class TruncatedGaussian
    {
        public void Run()
        {
            for (double thresh = 0; thresh <= 1; thresh += 0.1)
            {
                Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
                Variable.ConstrainTrue(x > thresh);
                InferenceEngine engine = new InferenceEngine();
                if (engine.Algorithm is Algorithms.ExpectationPropagation)
                {
                    Console.WriteLine("Dist over x given thresh of " + thresh + "=" + engine.Infer(x));
                }
                else
                {
                    Console.WriteLine("This example only runs with Expectation Propagation");
                }
            }
        }
    }
}
