using Loki;
using Loki.Shared;
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
            //Infer.Loki.Mappings.SpecialFunctionsMethods.FindThresholds();
            //return;
            var solutionPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "..", "Infer.sln"));
            var settings = new Settings(solutionPath);
            settings.WriteGeneratedSource = true;
            settings.GeneratedSourceDirectory = Path.Combine(Path.GetDirectoryName(solutionPath), "test", "LokiGeneratedSource");
            //settings.OverwriteGeneratedProgram = true;

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

            settings.ExcludeProject(new Regex(@"^Loki\.Tests$"));
            settings.ExcludeProject(new Regex(@"^Loki$"));
            settings.ExcludeProject(new Regex(@"^Loki\.Mapping$"));

            settings.Mappers.AddMap(new AssertionMap());
            settings.Mappers.AddMap(new SpecialFunctionsMap());
            settings.Mappers.AddMap(new TestHelpersMap());

            var transformer = new TestTransformer(settings);
            await transformer.Build();
            Environment.CurrentDirectory = Path.Combine(Path.GetDirectoryName(solutionPath), "test", "Tests");
            //var testResult = await TestRunner.RunTestAsync(transformer.GetCorrespondingTransformedTest(new OperatorTests().ExpMinus1RatioMinus1RatioMinusHalf_IsIncreasing), 0);
            //Console.WriteLine($"Test result: {testResult.TestPassed},\nMessage: {testResult.Message}");

            var tests = transformer.EnumerateTransformedTests()
                .Where(ti => ti.Method.ReflectedType.Name == typeof(SpecialFunctionsTests).Name /*&& ti.Method.Name != "NormalCdfIntegralTest"*/)
                //.Skip(4)
                //.Skip(100)
                //.Take(100)
                ;

            ////var tests = new[] { TestInfo.FromDelegate(new SpecialFunctionsTests().LogisticGaussianTest) };

            var testRun = TestSuiteRunner.RunTestSuite(
                tests,
                new[] { 0ul, ulong.MaxValue },
                //"log2.csv",
                //true,
                TimeSpan.FromMinutes(0.5),
                4,
                new Progress<TestSuiteRunStatus>(s => Console.Title = $"Completed {s.CompletedTests} tests."));

            var testResultEnumerator = testRun.GetAsyncEnumerator();
            try
            {
                while (await testResultEnumerator.MoveNextAsync())
                {
                    var result = testResultEnumerator.Current;
                    Console.WriteLine($"{result.ContainingTypeFullName}.{result.TestName}, {result.StartingFuel} fuel units: {result.Outcome}\n{result.Message}");
                }
            }
            finally
            {
                await testResultEnumerator.DisposeAsync();
            }
            //var test = transformer.GetCorrespondingTransformedTest(new OperatorTests().ExpMinus1RatioMinus1RatioMinusHalf_IsIncreasing);
            //var result = await TestRunner.RunTestIsolatedAsync(test, 0);
            //Console.WriteLine($"{result.ContainingTypeFullName}.{result.TestName}, {result.StartingFuel} fuel units: {result.Outcome}");
            //Console.WriteLine(result.Message);
        }
    }
}
