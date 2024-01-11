// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class TutorialTests
    {
        [Fact]
        public void ChessAnalysis()
        {
            (new Tutorials.ChessAnalysis()).Run();
        }

        [Fact]
        public void BugsRats()
        {
            RunWithAlgorithms(new Tutorials.BugsRats(), ep, vmp, gibbs);
        }

        [Fact]
        public void MatchboxRecommender()
        {
            RunWithAlgorithms(new Tutorials.RecommenderSystem() { largeData = true }, ep);
        }

        // TODO: Does not work with broad prior on discrimination
        [Fact]
        public void DifficultyAbility()
        {
            RunWithAlgorithms(new Tutorials.DifficultyAbility(), ep);
        }

        [Fact]
        public void BayesPointMachineExample()
        {
            RunWithAlgorithms(new Tutorials.BayesPointMachineExample(), ep, vmp);
        }

        [Fact]
        //[DeploymentItem(@"data\ClickModel.txt", "data")]
        public void ClickModel()
        {
            RunWithAlgorithms(new Tutorials.ClickModel(), ep);
        }

        [Fact]
        public void FirstExample()
        {
            RunWithAlgorithms(new Tutorials.FirstExample(), ep, gibbs);
        }

        [Fact]
        public void LearningAGaussian()
        {
            RunWithAlgorithms(new Tutorials.LearningAGaussian(), ep, vmp, gibbs);
        }

        [Fact]
        public void LearningAGaussianWithRanges()
        {
            RunWithAlgorithms(new Tutorials.LearningAGaussianWithRanges(), ep, vmp, gibbs);
        }

        [Fact]
        public void TruncatedGaussian()
        {
            RunWithAlgorithms(new Tutorials.TruncatedGaussian(), ep);
        }

        [Fact]
        public void TruncatedGaussianEfficient()
        {
            RunWithAlgorithms(new Tutorials.TruncatedGaussianEfficient(), ep);
        }

        [Fact]
        public void MixtureOfGaussians()
        {
            RunWithAlgorithms(new Tutorials.MixtureOfGaussians(), ep, vmp, gibbs);
        }

        [Fact]
        public void ClinicalTrial()
        {
            RunWithAlgorithms(new Tutorials.ClinicalTrial(), ep, vmp);
        }

        [Fact]
        public void BayesianPCA()
        {
            RunWithAlgorithms(new Tutorials.BayesianPCA(), vmp);
        }

        [Fact]
        public void GaussianProcessClassifier()
        {
            RunWithAlgorithms(new Tutorials.GaussianProcessClassifier(), ep);
        }

        [Fact]
        public void StudentSkills()
        {
            RunWithAlgorithms(new Tutorials.StudentSkills(), ep);
        }

        // with VMP, answer depends on schedule
        [Fact]
        public void WetGrassSprinklerRain()
        {
            RunWithAlgorithms(new Tutorials.WetGrassSprinklerRain(), ep, vmp);
        }

        ExpectationPropagation ep = new ExpectationPropagation();
        VariationalMessagePassing vmp = new VariationalMessagePassing();
        GibbsSampling gibbs = new GibbsSampling();

        protected void RunWithAlgorithms(object tutorialClass, params IAlgorithm[] algorithms)
        {
            lock (InferenceEngine.DefaultEngine)
            {
                IAlgorithm oldAlgorithm = InferenceEngine.DefaultEngine.Algorithm;
                try
                {
                    foreach (var algorithm in algorithms)
                    {
                        InferenceEngine.DefaultEngine.Algorithm = algorithm;
                        RunTutorial(tutorialClass);
                    }
                }
                finally
                {
                    InferenceEngine.DefaultEngine.Algorithm = oldAlgorithm;
                }
            }
        }

        protected void RunTutorial(object tutorialClass)
        {
            Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeMember(tutorialClass.GetType(), "Run", 
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.InvokeMethod,
                tutorialClass);
        }
    }
}
