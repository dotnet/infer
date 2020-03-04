// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Loki;
using Microsoft.ML.Probabilistic.Tests;
using Xunit;
using Xunit.Abstractions;

namespace Infer.Loki.Tests
{
    /// <summary>
    ///     Tests for gate operations.
    /// </summary>
    public class LokiGateOpTests : IDisposable
    {
        private readonly XunitTraceListener listener;
        private bool disposed;

        public LokiGateOpTests(ITestOutputHelper output)
        {
            this.listener = new XunitTraceListener(output);
            Trace.Listeners.Add(this.listener);
        }

        [Fact]
        public async Task Loki_Test()
        {
            var solutionPath = $@"{Directory.GetCurrentDirectory()}\..\..\..\..\..\Infer.sln";
            var settings = new Settings(solutionPath);
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

            settings.PreserveNamespace(new Regex(@"^Infer\.Loki\.Mappings"));

            settings.Mappers.AddMap(new AssertionMap());
            settings.Mappers.AddMap(new SpecialFunctionsMap());


            var testResult = await TestRunner.BuildAndRunTest(settings, new SpecialFunctionsTests().GammaSpecialFunctionsTest);
            //var testResult = await TestRunner.BuildAndRunTest(settings, new SpecialFunctionsTests().LogisticGaussianTest);

            Assert.True(testResult.DoublePrecisionPassed);
            Assert.True(testResult.HighPrecisionPassed);
        }

        // Public implementation of Dispose pattern callable by consumers.
        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        // Protected implementation of Dispose pattern.
        protected virtual void Dispose(bool disposing)
        {
            if (this.disposed)
            {
                return;
            }

            if (disposing)
            {
                Trace.Listeners.Remove(this.listener);
            }

            this.disposed = true;
        }
    }

    public class XunitTraceListener : TraceListener
    {
        private readonly ITestOutputHelper output;

        public XunitTraceListener(ITestOutputHelper output)
        {
            this.output = output;
        }

        public override void Write(string str)
        {
            this.output.WriteLine(str);
        }

        public override void WriteLine(string str)
        {
            this.output.WriteLine(str);
        }
    }
}