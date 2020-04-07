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
    [Example("Applications", "The Rats example from BUGS")]
    public class BugsRats
    {
        public void Run()
        {
            Rand.Restart(12347);

            // The model
            Range N = new Range(RatsHeightData.GetLength(0)).Named("N");
            Range T = new Range(RatsHeightData.GetLength(1)).Named("T");

            Variable<double> alphaC = Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("alphaC");
            Variable<double> alphaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("alphaTau");
            VariableArray<double> alpha = Variable.Array<double>(N).Named("alpha");
            alpha[N] = Variable.GaussianFromMeanAndPrecision(alphaC, alphaTau).ForEach(N);

            Variable<double> betaC = Variable.GaussianFromMeanAndPrecision(0.0, 1e-4).Named("betaC");
            Variable<double> betaTau = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("betaTau");
            VariableArray<double> beta = Variable.Array<double>(N).Named("beta");
            beta[N] = Variable.GaussianFromMeanAndPrecision(betaC, betaTau).ForEach(N);

            Variable<double> tauC = Variable.GammaFromShapeAndRate(1e-3, 1e-3).Named("tauC");
            VariableArray<double> x = Variable.Observed<double>(RatsXData, T).Named("x");
            Variable<double> xbar = Variable.Sum(x) / T.SizeAsInt;
            VariableArray2D<double> y = Variable.Observed<double>(RatsHeightData, N, T).Named("y");
            y[N, T] = Variable.GaussianFromMeanAndPrecision(alpha[N] + (beta[N] * (x[T] - xbar)), tauC);
            Variable<double> alpha0 = (alphaC - betaC * xbar).Named("alpha0");

            // Initialise with the mean of the prior (needed for Gibbs to converge quickly)
            alphaC.InitialiseTo(Gaussian.PointMass(0.0));
            tauC.InitialiseTo(Gamma.PointMass(1.0));
            alphaTau.InitialiseTo(Gamma.PointMass(1.0));
            betaTau.InitialiseTo(Gamma.PointMass(1.0));

            // Inference engine
            InferenceEngine ie = new InferenceEngine();
            Gaussian betaCMarg = ie.Infer<Gaussian>(betaC);
            Gaussian alpha0Marg = ie.Infer<Gaussian>(alpha0);
            Gamma tauCMarg = ie.Infer<Gamma>(tauC);

            // Inference
            Console.WriteLine("alpha0 = {0}[sd={1}]", alpha0Marg, System.Math.Sqrt(alpha0Marg.GetVariance()).ToString("g4"));
            Console.WriteLine("betaC = {0}[sd={1}]", betaCMarg, System.Math.Sqrt(betaCMarg.GetVariance()).ToString("g4"));
            Console.WriteLine("tauC = {0}", tauCMarg);
        }

        // Height data
        double[,] RatsHeightData = new double[,]
          {
       { 151, 199, 246, 283, 320 },
       { 145, 199, 249, 293, 354 },
       { 147, 214, 263, 312, 328 },
       { 155, 200, 237, 272, 297 },
       { 135, 188, 230, 280, 323 },
       { 159, 210, 252, 298, 331 },
       { 141, 189, 231, 275, 305 },
       { 159, 201, 248, 297, 338 },
       { 177, 236, 285, 350, 376 },
       { 134, 182, 220, 260, 296 },
       { 160, 208, 261, 313, 352 },
       { 143, 188, 220, 273, 314 },
       { 154, 200, 244, 289, 325 },
       { 171, 221, 270, 326, 358 },
       { 163, 216, 242, 281, 312 },
       { 160, 207, 248, 288, 324 },
       { 142, 187, 234, 280, 316 },
       { 156, 203, 243, 283, 317 },
       { 157, 212, 259, 307, 336 },
       { 152, 203, 246, 286, 321 },
       { 154, 205, 253, 298, 334 },
       { 139, 190, 225, 267, 302 },
       { 146, 191, 229, 272, 302 },
       { 157, 211, 250, 285, 323 },
       { 132, 185, 237, 286, 331 },
       { 160, 207, 257, 303, 345 },
       { 169, 216, 261, 295, 333 },
       { 157, 205, 248, 289, 316 },
       { 137, 180, 219, 258, 291 },
       { 153, 200, 244, 286, 324 }
          };

        // x data
        double[] RatsXData = { 8.0, 15.0, 22.0, 29.0, 36.0 };
    }
}
