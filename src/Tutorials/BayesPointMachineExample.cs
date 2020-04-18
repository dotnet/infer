// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Tutorials", "Demonstrates how to train and apply a Bayes Point Machine classifier.", Prefix = "4.")]
    public class BayesPointMachineExample
    {
        public void Run()
        {
            // data
            double[] incomes = { 63, 16, 28, 55, 22, 20 };
            double[] ages = { 38, 23, 40, 27, 18, 40 };
            bool[] willBuy = { true, false, true, true, false, false };

            // Create target y
            VariableArray<bool> y = Variable.Observed(willBuy).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))).Named("w");
            BayesPointMachine(incomes, ages, w, y);

            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.GibbsSampling))
            {
                VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
                Console.WriteLine("Dist over w=\n" + wPosterior);

                double[] incomesTest = { 58, 18, 22 };
                double[] agesTest = { 36, 24, 37 };
                VariableArray<bool> ytest = Variable.Array<bool>(new Range(agesTest.Length)).Named("ytest");
                BayesPointMachine(incomesTest, agesTest, Variable.Random(wPosterior).Named("w"), ytest);
                Console.WriteLine("output=\n" + engine.Infer(ytest));
            }
            else
            {
                Console.WriteLine("This model has a non-conjugate factor, and therefore cannot use Gibbs sampling");
            }
        }

        public void BayesPointMachine(double[] incomes, double[] ages, Variable<Vector> w, VariableArray<bool> y)
        {
            // Create x vector, augmented by 1
            Range j = y.Range.Named("person");
            Vector[] xdata = new Vector[incomes.Length];
            for (int i = 0; i < xdata.Length; i++)
            {
                xdata[i] = Vector.FromArray(incomes[i], ages[i], 1);
            }

            VariableArray<Vector> x = Variable.Observed(xdata, j).Named("x");

            // Bayes Point Machine
            double noise = 0.1;
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct"), noise) > 0;
        }
    }
}
