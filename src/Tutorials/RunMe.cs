// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    /// <summary>
    /// Use this class to run tutorials or to show the tutorial viewer.
    /// </summary>
    public class RunMe
    {
        [STAThread]
        public static void Main()
        {
            // When compiled with .NET 4.6.1 or higher, this will pop up a window for running examples, then exit.  
            // Otherwise it does nothing.
            ExamplesBrowser.RunBrowser();

            //Choose one of the algorithms
            InferenceEngine.DefaultEngine.Algorithm = new ExpectationPropagation();
            //InferenceEngine.DefaultEngine.Algorithm = new VariationalMessagePassing();
            //InferenceEngine.DefaultEngine.Algorithm = new GibbsSampling();

            //Options
            InferenceEngine.DefaultEngine.ShowProgress = true;
            InferenceEngine.DefaultEngine.ShowTimings = false;
            InferenceEngine.DefaultEngine.ShowMsl = false;
            InferenceEngine.DefaultEngine.ShowFactorGraph = false;
            InferenceEngine.DefaultEngine.ShowSchedule = false;

            //Tutorials
            //Uncomment one of these lines to run a particular tutorial in console application
            Console.OutputEncoding = System.Text.Encoding.Unicode;

            new FirstExample().Run();
            //new TruncatedGaussian().Run();
            //new TruncatedGaussianEfficient().Run();
            //new LearningAGaussian().Run();
            //new LearningAGaussianWithRanges().Run();
            //new BayesPointMachineExample().Run();
            //new ClinicalTrial().Run();
            //new MixtureOfGaussians().Run();

            //String tutorials

            //new HelloStrings().Run();
            //new StringFormat().Run();

            //Applications

            //new BayesianPCA().Run();
            //new BugsRats().Run();
            //new ChessAnalysis().Run();
            //new ClickModel().Run();
            //new DifficultyAbility().Run();
            //new GaussianProcessClassifier().Run();
            //new MultinomialRegression().Run();
            //new RecommenderSystem().Run();
            //new StudentSkills().Run();
            //new WetGrassSprinklerRain().Run();
        }
    }
}
