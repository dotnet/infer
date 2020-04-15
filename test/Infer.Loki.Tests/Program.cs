using Loki;
using Microsoft.ML.Probabilistic.Tests;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Infer.Loki.Tests
{
    class Program
    {
        public static async Task Main()
        {
            var solutionPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "..", "Infer.sln"));
            var settings = new Settings(solutionPath);
            settings.WriteGeneratedSource = true;
            settings.GeneratedSourceDirectory = Path.Combine(Path.GetDirectoryName(solutionPath), "test", "LokiGeneratedSource");

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
            settings.ExcludeProject(new Regex("^RobustGaussianProcess"));

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
            settings.ExcludeProject(new Regex("^LearnersNuGet"));
            settings.ExcludeProject(new Regex(@"^Microsoft\.ML\.Probabilistic\.Learners\.TestApp"));

            settings.ExcludeProject(new Regex(@"^FSharpWrapper"));
            settings.ExcludeProject(new Regex(@"^TestApp"));
            settings.ExcludeProject(new Regex(@"^TestFSharp"));
            settings.ExcludeProject(new Regex(@"^TestPublic"));
            settings.ExcludeProject(new Regex(@"^Tools\.BuildFactorDoc"));
            settings.ExcludeProject(new Regex(@"^Tools\.GenerateSeries"));
            settings.ExcludeProject(new Regex(@"^Tools\.PrepareSource"));
            settings.ExcludeProject(new Regex("^Tutorials"));
            settings.ExcludeProject(new Regex(@"^Visualizers\.Windows"));

            settings.ExcludeProject(new Regex(@"^Infer\.Loki\.Tests"));

            settings.ExcludeProject(new Regex("^generated"));

            settings.PreserveNamespace(new Regex(@"^Infer\.Loki\.Mappings"));

            settings.Mappers.AddMap(new AssertionMap());
            settings.Mappers.AddMap(new SpecialFunctionsMap());
            settings.Mappers.AddMap(new TestHelpersMap());

            var runner = new TestRunner(settings);
            //var testResult = await TestRunner.BuildAndRunTest(settings, new SpecialFunctionsTests().GammaSpecialFunctionsTest);
            //var testResult = await TestRunner.BuildAndRunTest(settings, new SpecialFunctionsTests().LogisticGaussianTest);

            //Assert.True(testResult.DoublePrecisionPassed);
            //Assert.True(testResult.HighPrecisionPassed);
            await runner.Build();
            Environment.CurrentDirectory = Path.Combine(Path.GetDirectoryName(solutionPath), "test", "Tests");
            var gammaTestResult = runner.RunTest(new SpecialFunctionsTests().GammaUpperTest, 0);
            Console.WriteLine($"Test result: {gammaTestResult.TestPassed},\nMessage: {gammaTestResult.Message}");
        }
    }
}
