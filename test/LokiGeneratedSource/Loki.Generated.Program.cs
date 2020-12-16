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
using System.Collections.Immutable;
using Newtonsoft.Json;

namespace Loki.Generated
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            const StaticMixedPrecisionTuningTestExecutionManager.OperationAmbiguityProcessingBehavior MissingDescriptionBehavior =
                StaticMixedPrecisionTuningTestExecutionManager.OperationAmbiguityProcessingBehavior.UseDouble;

            var testInfo = TestInfo.FromDelegate(new OperatorTests().GaussianIsBetweenCRRR_NegativeUpperBoundTest);
            string lastStatus = null;
            var localizer = new StaticMixedPrecisionTuningLocalizer(testInfo)
            {
                Progress = new Progress<StaticMixedPrecisionTuningLocalizationStatus>(s =>
                {
                    Console.Title = s.ToString();
                    if (lastStatus != s.StatusString)
                    {
                        lastStatus = s.StatusString;
                        Console.WriteLine(s);
                    }
                }),
                MissingDescriptionBehavior = MissingDescriptionBehavior,
                CheckpointAutosavingFrequency = 50
            };

            bool isLocalizable = await localizer.CheckLocalizeability();

            if (isLocalizable)
            {
                var arities = localizer.GetOperationsAndArities();

                Console.WriteLine();
                Console.WriteLine("Participating operations:");
                Console.WriteLine("ID\t| Arity");
                Console.WriteLine("--------|--------");
                foreach (var kvp in arities)
                    Console.WriteLine($"{kvp.Key}\t| {kvp.Value}");
                Console.WriteLine();

                await localizer.LocalizeOnOperations();
                var operationDescriptions = localizer.GetOperationDescriptions();
                Console.WriteLine();
                Console.WriteLine("Resulting operation precisions");
                Console.WriteLine("ID\t| Precision");
                Console.WriteLine("--------|---------------");
                foreach (var kvp in operationDescriptions)
                    Console.WriteLine($"{kvp.Key}\t| {(kvp.Value.UseExtendedPrecisionOperation ? "Extended" : "Double")}");
                Console.WriteLine();

                await localizer.LocalizeOnOperands();
                arities = localizer.GetOperationsAndArities();
                operationDescriptions = localizer.GetOperationDescriptions();
                Console.WriteLine();
                Console.WriteLine("Resulting operation precisions");
                Console.WriteLine("ID\t| Precision\t| Operand Precisions");
                Console.WriteLine("--------|---------------|--------------------------");
                foreach (var kvp in operationDescriptions)
                {
                    Console.Write($"{kvp.Key}\t| ");
                    if (kvp.Value.UseExtendedPrecisionOperation)
                    {
                        Console.Write("Extended");
                        for (int i = 0; i < arities[kvp.Key]; ++i)
                            Console.Write($"\t| {(kvp.Value.RoundOperands[i] ? "Rounded" : "Preserved")}");
                    }
                    else
                    {
                        Console.Write("Double");
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
                Console.WriteLine();
                Console.WriteLine("Extended precision operations");
                Console.WriteLine("ID\t| Operand Precisions");
                Console.WriteLine("--------|--------------------------");
                foreach (var kvp in operationDescriptions)
                {
                    if (kvp.Value.UseExtendedPrecisionOperation)
                    {
                        Console.Write($"{kvp.Key}");
                        for (int i = 0; i < arities[kvp.Key]; ++i)
                            Console.Write($"\t| {(kvp.Value.RoundOperands[i] ? "Rounded" : "Preserved")}");
                        Console.WriteLine();
                    }
                }
                Console.WriteLine();
                Console.WriteLine("Done!");
            }
            else
            {
                Console.WriteLine("Test is not localizable");
            }
        }

        [Xunit.Fact]
        public static void GaussianIsBetweenCRCC_IsMonotonicInXMean_IsolatedCase()
        {
            DoubleWithTransformingPrecision upperBound = 0;
            DoubleWithTransformingPrecision lowerBound = -1000;
            DoubleWithTransformingPrecision center = (lowerBound + upperBound) / 2;
            DoubleWithTransformingPrecision meanDelta = 1;
            Bernoulli isBetween = new Bernoulli(DoubleWithTransformingPrecision.FromString("1.0"));
            Gaussian x = Gaussian.FromMeanAndPrecision(DoubleWithTransformingPrecision.FromString("-0.238256490488795107321616978173267452037E-323228474"), DoubleWithTransformingPrecision.FromString("0.1000000000000000000000000000000000000017E-21"));
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
            if (MapDispatcher.System_Double_IsPositiveInfinity_System_Double(TestExecutionManager.MissingOperationId, meanDelta)) mx2 = meanDelta;
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
                meanError = MapDispatcher.Microsoft_ML_Probabilistic_Math_MMath_Ulp_System_Double(TestExecutionManager.MissingOperationId, mean);
                meanError2 = MapDispatcher.Microsoft_ML_Probabilistic_Math_MMath_Ulp_System_Double(TestExecutionManager.MissingOperationId, mean2);
                DoubleWithTransformingPrecision meanUlpDiff = (mean - mean2) / MapDispatcher.System_Math_Max_System_Double_System_Double(TestExecutionManager.MissingOperationId, meanError, meanError2);
                Xunit.Assert.True(meanUlpDiff < DoubleWithTransformingPrecision.FromString("1e16"));
            }
            // When mx > center, increasing prior mean should increase posterior precision.
            if (mx > center && xPost2.Precision < xPost.Precision)
            {
                DoubleWithTransformingPrecision ulpDiff = OperatorTests.UlpDiff(xPost2.Precision, xPost.Precision);
                Xunit.Assert.True(ulpDiff < DoubleWithTransformingPrecision.FromString("1e11"));
            }
        }

        static void GaussianIsBetweenCRCC_IsMonotonicInXPrecision_IsolatedCase()
        {
            Bernoulli isBetween = new Bernoulli(DoubleWithTransformingPrecision.FromString("1.0"));
            DoubleWithTransformingPrecision lowerBound = DoubleWithTransformingPrecision.FromString("-10000");
            DoubleWithTransformingPrecision upperBound = DoubleWithTransformingPrecision.FromString("-0.999999999999999819999999999999999999997E+4");
            Gaussian x = Gaussian.FromMeanAndPrecision(DoubleWithTransformingPrecision.FromString("-0.999999999999999999999999999999999999987E+17"), DoubleWithTransformingPrecision.FromString("0.1000000000000000000000000000000000000012E-16"));
            DoubleWithTransformingPrecision mx = x.GetMean();
            Gaussian toX = global::Microsoft.ML.Probabilistic.Factors.DoubleIsBetweenOp.XAverageConditional(isBetween, x, lowerBound, upperBound);
            Gaussian xPost;
            DoubleWithTransformingPrecision meanError;
            if (toX.IsPointMass)
            {
                xPost = toX;
                meanError = 0;
            }
            else
            {
                xPost = toX * x;
                meanError = OperatorTests.GetProductMeanError(toX, x);
            }
            DoubleWithTransformingPrecision mean = xPost.GetMean();
            DoubleWithTransformingPrecision precisionDelta = DoubleWithTransformingPrecision.FromString("0.1000000000000000000000000000000000000008E-12");
            Gaussian x2 = Gaussian.FromMeanAndPrecision(mx, x.Precision + precisionDelta);
            Gaussian toX2 = Microsoft.ML.Probabilistic.Factors.DoubleIsBetweenOp.XAverageConditional(isBetween, x2, lowerBound, upperBound);
            Gaussian xPost2;
            DoubleWithTransformingPrecision meanError2;
            if (toX2.IsPointMass)
            {
                xPost2 = toX2;
                meanError2 = 0;
            }
            else
            {
                xPost2 = toX2 * x2;
                meanError2 = OperatorTests.GetProductMeanError(toX2, x2);
            }
            DoubleWithTransformingPrecision mean2 = xPost2.GetMean();
            DoubleWithTransformingPrecision meanUlpDiff = 0;
            if (mean > mx)
            {
                // Since mx < mean, increasing the prior precision should decrease the posterior mean.
                if (mean2 > mean)
                {
                    meanUlpDiff = (mean2 - mean) / MapDispatcher.System_Math_Max_System_Double_System_Double(TestExecutionManager.MissingOperationId, meanError, meanError2);
                }
            }
            else
            {
                if (mean2 < mean)
                {
                    meanUlpDiff = (mean - mean2) / MapDispatcher.System_Math_Max_System_Double_System_Double(TestExecutionManager.MissingOperationId, meanError, meanError2);
                }
            }
            Xunit.Assert.True(meanUlpDiff < DoubleWithTransformingPrecision.FromString("1e16"));
            // Increasing prior precision should increase posterior precision.
            if (xPost2.Precision < xPost.Precision)
            {
                DoubleWithTransformingPrecision ulpDiff = OperatorTests.UlpDiff(xPost2.Precision, xPost.Precision);
                Xunit.Assert.True(ulpDiff < DoubleWithTransformingPrecision.FromString("1e16"));
            }
        }
    }
}
