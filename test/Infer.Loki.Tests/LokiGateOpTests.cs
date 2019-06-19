// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Loki.Extensions;
using System.IO;
using Loki;
using Microsoft.ML.Probabilistic.Tests;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Xunit;

namespace Infer.Loki.Tests
{
    /// <summary>
    /// Tests for gate operations.
    /// </summary>
    public class LokiGateOpTests
    {
        /// <summary>
        /// Tests EP and BP gate exit ops for Bernoulli random variable for correctness.
        /// </summary>
        [Fact]
        public async Task Loki_BernoulliEnterTest()
        {
            var solutionPath = $@"{Directory.GetCurrentDirectory()}\..\..\..\..\..\Infer.sln";
            Settings settings = new Settings(solutionPath);
            settings.WriteGeneratedSource = true;

            // Examples
            settings.ExcludeProject(new Regex("^ClickThroughModel"));
            settings.ExcludeProject(new Regex("^ClinicalTrial"));
            settings.ExcludeProject(new Regex("^Crowdsourcing"));
            settings.ExcludeProject(new Regex("^CrowdsourcingWithWords"));
            settings.ExcludeProject(new Regex("^Image_Classifier"));
            settings.ExcludeProject(new Regex("^InferNET101"));
            settings.ExcludeProject(new Regex("^LDA"));
            settings.ExcludeProject(new Regex("^MontyHall"));
            settings.ExcludeProject(new Regex("^MotifFinder"));
            settings.ExcludeProject(new Regex("^ReviewerCalibration"));

            // Learners
            settings.ExcludeProject(new Regex("^ClassifierModels"));
            settings.ExcludeProject(new Regex("^Core"));
            settings.ExcludeProject(new Regex("^RecommenderModels"));
            settings.ExcludeProject(new Regex("^Recommender"));
            settings.ExcludeProject(new Regex("^Classifier"));
            settings.ExcludeProject(new Regex("^Common"));
            settings.ExcludeProject(new Regex("^CommandLine"));
            settings.ExcludeProject(new Regex("^Evaluator"));
            settings.ExcludeProject(new Regex("^LearnersTests"));
            settings.ExcludeProject(new Regex("^Microsoft.ML.Probabilistic.Learners.TestApp"));

            settings.ExcludeProject(new Regex(@"^FSharpWrapper"));
            settings.ExcludeProject(new Regex(@"^TestApp"));
            settings.ExcludeProject(new Regex(@"^TestFSharp"));
            settings.ExcludeProject(new Regex(@"^TestPublic"));
            settings.ExcludeProject(new Regex(@"^Tools\.BuildFactorDoc"));
            settings.ExcludeProject(new Regex(@"^Tools\.PrepareSource"));
            settings.ExcludeProject(new Regex("^Tutorials"));
            settings.ExcludeProject(new Regex("^Visualizers"));
            
            var testResult = await TestRunner.BuildAndRunTest(settings, GateOpTests.BernoulliEnterTest);

            Assert.True(testResult.DoublePrecisionPassed);
            Assert.True(testResult.HighPrecisionPassed);
        }
    }
}