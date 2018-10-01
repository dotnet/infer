// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using System.IO;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

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
            Type tutorialClass = null;

            // ********** UNCOMMENT AND EDIT THIS LINE TO RUN A PARTICULAR TUTORIAL DIRECTLY *************
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.FirstExample);


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

#if NETCORE
            //Tutorials
            //Uncomment one of these lines to run a particular tutorial in console application

            tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.FirstExample);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.TruncatedGaussian);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.TruncatedGaussianEfficient);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.LearningAGaussian);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.LearningAGaussianWithRanges);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.BayesPointMachineExample);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.ClinicalTrial);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.MixtureOfGaussians);

            //String tutorials

            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.HelloStrings);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.StringFormat);

            //Applications

            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.BayesianPCA);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.BugsRats);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.ChessAnalysis);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.ClickModel);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.DifficultyAbility);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.GaussianProcessClassifier);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.MultinomialRegression);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.RecommenderSystem);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.StudentSkills);
            //tutorialClass = typeof(Microsoft.ML.Probabilistic.Tutorials.WetGrassSprinklerRain);
#endif
            if (tutorialClass != null)
            {
                // Run the specified tutorial
                RunTutorial(tutorialClass);
            }
#if NETFULL
            else
            {
                InferenceEngine.Visualizer = new WindowsVisualizer();
                // Show all tutorials, in a browser
                IAlgorithm[] algs = InferenceEngine.GetBuiltInAlgorithms();

                // Avoid max product in the examples browser, as none of the examples apply.
                List<IAlgorithm> algList = new List<IAlgorithm>(algs);
                algList.RemoveAll(alg => alg is MaxProductBeliefPropagation);
                ExamplesViewer tview = new ExamplesViewer(typeof(RunMe), algList.ToArray());
                tview.RunBrowser();
            }
#endif
        }

        /// <summary>
        /// Runs the tutorial contained in the supplied class.
        /// </summary>
        /// <param name="tutorialClass">The class containing the tutorial to be run</param>
        public static void RunTutorial(Type tutorialClass)  // Must not be called "Run"
        {
            if (tutorialClass == null)
            {
                return;
            }

            object obj = Activator.CreateInstance(tutorialClass);
            MethodInfo mi = tutorialClass.GetMethod("Run");
            if (mi == null)
            {
                return;
            }

            mi.Invoke(obj, new object[0]);
        }
    }
}
