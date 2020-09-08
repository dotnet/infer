using Loki.Shared;
using System;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.ML.Probabilistic.Tests;
using System.IO;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.Probabilistic.Learners.Tests;
using Microsoft.ML.Probabilistic.Distributions;

namespace Loki.Generated
{
    class Program
    {
        static async Task Main(string[] args)
        {
            DoublePrecisionFuel.SetFuel(0);
            GaussianIsBetweenCRCC_IsMonotonicInXMean_IsolatedCase();
            //var logFilePath = Path.Combine(Environment.CurrentDirectory, "log.csv");

            //var result = await TestRunner.RunTestAsync(new OperatorTests().GaussianIsBetweenCRCC_IsMonotonicInXMean, 0/*, null, new CancellationTokenSource(TimeSpan.FromMinutes(10)).Token*/);
            //Console.WriteLine($"{result.ContainingTypeFullName}.{result.TestName}, {result.StartingFuel} fuel units: {result.Outcome}\n{result.Message}");
            //var categoriesToSkip = new[] { "CompilerOptionsTest", "Performance", "OpenBug", "BadTest" };
            //var tests = TestInfo.FromAssembly(typeof(Program).Assembly)
            //    .Where(ti => !ti.Traits.Any(trait => trait.Key == "Category" && categoriesToSkip.Contains(trait.Value)))
            //    //.Skip(100)
            //    //.Take(100)
            //    ;

            //////var tests = new TestInfo[] { TestInfo.FromDelegate(new BayesPointMachineTests().BayesPointEvidence2) }
            //////    .Where(ti => !ti.Traits.Any(trait => trait.Key == "Category" && trait.Value == "CompilerOptionsTest"))
            //////    //.Skip(100)
            //////    //.Take(100)
            //////    ;

            //var testRun = TestSuiteRunner.RunTestSuite(
            //    tests,
            //    new[] { ulong.MaxValue, 0ul },
            //    logFilePath,
            //    false,
            //    TimeSpan.FromMinutes(10),
            //    16,
            //    true,
            //    new Progress<TestSuiteRunStatus>(s => Console.Title = $"Completed {s.CompletedTests} tests. OpCounts: {string.Join(" | ", s.CurrentTestStatuses.Select(cts => cts.CurrentRunStatus.OperationCount.ToString()))}"));

            //var testResultEnumerator = testRun.GetAsyncEnumerator();
            //try
            //{
            //    while (await testResultEnumerator.MoveNextAsync())
            //    {
            //        var result = testResultEnumerator.Current;
            //        Console.WriteLine($"{result.ContainingTypeFullName}.{result.TestName}, {result.StartingFuel} fuel units: {result.Outcome}\n{result.Message}");
            //    }
            //}
            //catch (Exception ex)
            //{
            //    Console.WriteLine("A FATAL ERROR OCCURED:");
            //    Console.WriteLine(ex);
            //}
            //finally
            //{
            //    await testResultEnumerator.DisposeAsync();
            //}
        }

        static void GaussianIsBetweenCRCC_IsMonotonicInXMean_IsolatedCase()
        {
            DoubleWithTransformingPrecision meanMaxUlpError = 0;
            DoubleWithTransformingPrecision precMaxUlpError = 0;
            DoubleWithTransformingPrecision upperBound = 0;
            DoubleWithTransformingPrecision lowerBound = -1000;
            DoubleWithTransformingPrecision center = (lowerBound + upperBound) / 2;
            DoubleWithTransformingPrecision meanDelta = 1;
            Bernoulli isBetween = new Bernoulli(new global::Loki.Shared.DoubleWithTransformingPrecision("1.0"));
            Gaussian x = Gaussian.FromMeanAndPrecision(new DoubleWithTransformingPrecision("-0.238256490488795107321616978173267452037E-323228474"), new DoubleWithTransformingPrecision("0.1000000000000000000000000000000000000017E-21"));
            DoubleWithTransformingPrecision mx = x.GetMean();
            Gaussian toX = global::Microsoft.ML.Probabilistic.Factors.DoubleIsBetweenOp.XAverageConditional(isBetween, x, lowerBound, upperBound);
            Gaussian xPost;
            DoubleWithTransformingPrecision meanError;
            if (toX.IsPointMass)
            {
                xPost = toX;
            }
            else
            {
                xPost = toX * x;
            }
            DoubleWithTransformingPrecision mean = xPost.GetMean();

            DoubleWithTransformingPrecision mx2 = mx + meanDelta;
            if (MapDispatcher.System_Double_IsPositiveInfinity_System_Double(meanDelta)) mx2 = meanDelta;
            Gaussian x2 = Gaussian.FromMeanAndPrecision(mx2, x.Precision);
            Gaussian toX2 = global::Microsoft.ML.Probabilistic.Factors.DoubleIsBetweenOp.XAverageConditional(isBetween, x2, lowerBound, upperBound);
            Gaussian xPost2;
            DoubleWithTransformingPrecision meanError2;
            if (toX2.IsPointMass)
            {
                xPost2 = toX2;
            }
            else
            {
                xPost2 = toX2 * x2;
            }
            DoubleWithTransformingPrecision mean2 = xPost2.GetMean();
            // Increasing the prior mean should increase the posterior mean.
            if (mean2 < mean)
            {
                meanError = MapDispatcher.Microsoft_ML_Probabilistic_Math_MMath_Ulp_System_Double(mean);
                meanError2 = MapDispatcher.Microsoft_ML_Probabilistic_Math_MMath_Ulp_System_Double(mean2);
                DoubleWithTransformingPrecision meanUlpDiff = (mean - mean2) / MapDispatcher.System_Math_Max_System_Double_System_Double(meanError, meanError2);
                if (meanUlpDiff > meanMaxUlpError)
                {
                    Xunit.Assert.True(meanUlpDiff < new DoubleWithTransformingPrecision("1e16"));
                }
            }
            // When mx > center, increasing prior mean should increase posterior precision.
            if (mx > center && xPost2.Precision < xPost.Precision)
            {
                DoubleWithTransformingPrecision ulpDiff = OperatorTests.UlpDiff(xPost2.Precision, xPost.Precision);
                if (ulpDiff > precMaxUlpError)
                {
                    precMaxUlpError = ulpDiff;
                    Xunit.Assert.True(precMaxUlpError < new DoubleWithTransformingPrecision("1e11"));
                }
            }
        }
    }
}
