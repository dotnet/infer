// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Test of Infer.NET port of Ralf Herbrich's RAE model
    /// </summary>
    public class RAETest
    {
        public RAETest()
        {
        }


        // This models a single output of a single researcher.
        public void RAEOutputModel(
            Gaussian inpScoreMean,
            Gamma inpScorePrec,
            Gaussian[][] inpThresh,
            bool[][] reviewObs)
        {
            // int n = reviewObs.Length;
            // double scoreMean = Model.NewVariable<double>(new Output());
            // double scorePrec = Model.NewVariable<double>(new Output());
            // scoreMean = Factor.Random(inpScoreMean);
            // scorePrec = Factor.Random(inpScorePrec);

            // double[] thresh0 = Model.NewArray<double>(n);
            // double[] thresh1 = Model.NewArray<double>(n);
            // double[] thresh2 = Model.NewArray<double>(n);
            // double[] thresh3 = Model.NewArray<double>(n);
            // double[] thresh4 = Model.NewArray<double>(n);
            // double[] thresh5 = Model.NewArray<double>(n);
            // double[] score = Model.NewArray<double>(n);
            // for (int i = 0; i < n; i++)
            // {
            //     thresh0[i] = Factor.Random<double>(inpThresh[i][0]);
            //     thresh1[i] = Factor.Random<double>(inpThresh[i][1]);
            //     thresh2[i] = Factor.Random<double>(inpThresh[i][2]);
            //     thresh3[i] = Factor.Random<double>(inpThresh[i][3]);
            //     thresh4[i] = Factor.Random<double>(inpThresh[i][4]);
            //     thresh5[i] = Factor.Random<double>(inpThresh[i][5]);

            //     score[i] = Factor.Gaussian(scoreMean, scorePrec);
            //     Constrain.Equal<bool>(reviewObs[i][0], Factor.IsBetween(score[i], thresh0[i], thresh1[i]));
            //     Constrain.Equal<bool>(reviewObs[i][1], Factor.IsBetween(score[i], thresh1[i], thresh2[i]));
            //     Constrain.Equal<bool>(reviewObs[i][2], Factor.IsBetween(score[i], thresh2[i], thresh3[i]));
            //     Constrain.Equal<bool>(reviewObs[i][3], Factor.IsBetween(score[i], thresh3[i], thresh4[i]));
            //     Constrain.Equal<bool>(reviewObs[i][4], Factor.IsBetween(score[i], thresh4[i], thresh5[i]));
            //}
            //Model.Infer(scoreMean);
            //Model.Infer(scorePrec);
        }

        //[Fact]
        public void TestRAEModel()
        {
            InferenceEngine engine = new InferenceEngine();

            Gaussian inputScoreMean = new Gaussian(3.0, 1.0);
            Gamma inputScorePrec = Gamma.FromShapeAndRate(2.0, 1.0);
            int nData = 2;
            Gaussian[][] threshMean = new Gaussian[nData][];
            bool[][] obs = new bool[nData][];
            for (int i = 0; i < nData; i++)
            {
                threshMean[i] = new Gaussian[6];
                obs[i] = new bool[6];
                for (int j = 0; j < 6; j++)
                {
                    threshMean[i][j] = new Gaussian((double) i, 1.0);
                    obs[i][j] = false;
                }
                obs[i][i + 1] = true;
            }

            //var ca = engine.Compiler.Compile(RAEOutputModel, inputScoreMean, inputScorePrec, threshMean, obs);
        }
    }
}