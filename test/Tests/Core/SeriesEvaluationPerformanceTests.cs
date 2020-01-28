using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.Probabilistic.Core.Maths;
using Microsoft.ML.Probabilistic.Math;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class SeriesEvaluationPerformanceTests
    {
        static Series series = new Series(53);

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double ExpMinus1Explicit(double x)
        {
            return x * (1 + x * (0.5 + x * (1.0 / 6 + x * (1.0 / 24))));
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static double ExpMinus1CompiledExpressionBased(double x)
        {
            return series.ExpMinus1(x);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static double ExpMinus1SwitchExplicit(double x)
        {
            //return MMath.ExpMinus1Explicit(x);
            return SeriesCollection.ExpMinus1At0(x);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double ExpMinus1RatioMinus1RatioMinusHalfExplicit(double x)
        {
            return x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0 / 120 + x * (1.0 / 720 +
                x * (1.0 / 5040 + x * (1.0 / 40320 + x * (1.0 / 362880 + x * (1.0 / 3628800 +
                x * (1.0 / 39916800 + x * (1.0 / 479001600 + x * (1.0 / 6227020800 + x * (1.0 / 87178291200))))))))))));
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double ExpMinus1RatioMinus1RatioMinusHalfCompiledExpressionBased(double x)
        {
            return series.ExpMinus1RatioMinus1RatioMinusHalf(x);
        }

        private static readonly double[] gammaTaylorCoefficients =
        {
            0.32246703342411320303,
            -0.06735230105319810201,
            0.020580808427784546416,
            -0.0073855510286739856768,
            0.0028905103307415229257,
            -0.0011927539117032610189,
            0.00050966952474304234172,
            -0.00022315475845357938579,
            9.945751278180853098e-05,
            -4.4926236738133142046e-05,
            2.0507212775670691067e-05,
            -9.4394882752683967152e-06,
            4.3748667899074873274e-06,
            -2.0392157538013666132e-06,
            9.551412130407419353e-07,
            -4.4924691987645661855e-07,
            2.1207184805554664645e-07,
            -1.0043224823968100408e-07,
            4.7698101693639803983e-08,
            -2.2711094608943166813e-08,
            1.0838659214896952939e-08,
            -5.1834750419700474714e-09,
            2.4836745438024780616e-09,
            -1.1921401405860913615e-09,
            5.7313672416788612175e-10,
        };

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static double GammaLnMidExplicit(double x)
        {
            // 1.5 <= x <= 2.5
            // Use Taylor series at x=2
            double result = 0;
            double dx = x - 2;
            double sum = 0;
            for (int i = gammaTaylorCoefficients.Length - 1; i >= 0; i--)
            {
                sum = dx * (gammaTaylorCoefficients[i] + sum);
            }
            sum = dx * (1 + MMath.Digamma1 + sum);
            result += sum;
            return result;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static double GammaLnMidCompiledExpressionBased(double x)
        {
            // 1.5 <= x <= 2.5
            // Use Taylor series at x=2
            double result = 0;
            double dx = x - 2;
            double sum = series.GammaAt2(dx);
            sum = dx * (1 + MMath.Digamma1 + sum);
            result += sum;
            return result;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static double Log1PlusShortExplicit(double x)
        {
            if (x > -1e-3 && x < 2e-3)
            {
                return x * (1 - x * (0.5 - x * (1.0 / 3 - x * (0.25 - x * (1.0 / 5)))));
            }
            else
            {
                return System.Math.Log(1 + x);
            }
        }

        private static Func<double, double> Log1PlusTruncated = SeriesEvaluatorFactory.GetCompiledExpressionSeriesEvaluator(n => n == 0 ? 0.0 : (n % 2 == 0 ? -1.0 : 1.0) / n, 5, 0, 5);

        [MethodImpl(MethodImplOptions.NoInlining)]
        static double Log1PlusShortCompiledExpressionBased(double x)
        {
            if (x > -1e-3 && x < 2e-3)
            {
                return Log1PlusTruncated(x);
            }
            else
            {
                return System.Math.Log(1 + x);
            }
        }

        static double Log1PlusInfiniteSeriesNaive(double x)
        {
            double sum = 0.0;
            double term = 1;
            double oldSum;
            int i = 1;
            while (true)
            {
                oldSum = sum;
                term *= x;
                sum += term / i;
                if (MMath.AreEqual(sum, oldSum))
                    break;
                ++i;

                oldSum = sum;
                term *= x;
                sum -= term / i;
                if (MMath.AreEqual(sum, oldSum))
                    break;
                ++i;
            }
            return sum;
        }


        [MethodImpl(MethodImplOptions.NoInlining)]
        static double Log1PlusNaiveInfiniteSeries(double x)
        {
            if (x > -1e-3 && x < 6e-2)
            {
                return Log1PlusInfiniteSeriesNaive(x);
            }
            else
            {
                return System.Math.Log(1 + x);
            }
        }

        private readonly ITestOutputHelper output;

        public SeriesEvaluationPerformanceTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        [Fact]
        [Trait("Category", "Performance")]
        public void CompiledExpressionOnExpMinus1PerformanceTest()
        {
            const int samples = 10;
            const int evaluations = 25000000;
            const double point = 1e-3;
            //double targetPerformanceCoefficient = 1.6;
            Assert.True(MMath.AreEqual(ExpMinus1Explicit(point), ExpMinus1CompiledExpressionBased(point)));

            double time1 = MeasureRunTime(() => MMath.ExpMinus1Explicit(point), samples, evaluations);
            output.WriteLine($"MMath.ExpMinus1Explicit time:\t{time1}ms.");
            double time2 = MeasureRunTime(() => ExpMinus1Explicit(point), samples, evaluations);
            output.WriteLine($"ExpMinus1Explicit time:\t{time2}ms.");
            double time3 = MeasureRunTime(() => ExpMinus1CompiledExpressionBased(point), samples, evaluations);
            output.WriteLine($"ExpMinus1CompiledExpressionBased time:\t{time3}ms.");
            double time4 = MeasureRunTime(() => ExpMinus1SwitchExplicit(point), samples, evaluations);
            output.WriteLine($"ExpMinus1SwitchExplicit time:\t{time4}ms.");
            double time5 = MeasureRunTime(() => MMath.ExpMinus1SwitchExplicit(point), samples, evaluations);
            output.WriteLine($"MMath.ExpMinus1SwitchExplicit time:\t{time5}ms.");
            //Assert.True(exprBasedTimeInms <= targetPerformanceCoefficient * explicitTimeInms, "Compiled expression-based approach shouldn't lose too much performance compared to inlined series.");
        }

        [Fact]
        [Trait("Category", "Performance")]
        public void CompiledExpressionOnExpMinus1RatioMinus1RatioMinusHalfPerformanceTest()
        {
            const int samples = 10;
            const int evaluations = 20000000;
            const double point = 1e-3;
            //double targetPerformanceCoefficient = 1.6;
            Assert.True(MMath.AreEqual(ExpMinus1RatioMinus1RatioMinusHalfExplicit(point), ExpMinus1RatioMinus1RatioMinusHalfCompiledExpressionBased(point)));
            double explicitTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) ExpMinus1RatioMinus1RatioMinusHalfExplicit(point); }, samples);
            output.WriteLine($"Explicit time:\t{explicitTimeInms}ms.");
            double exprBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) ExpMinus1RatioMinus1RatioMinusHalfCompiledExpressionBased(point); }, samples);
            output.WriteLine($"Compiled expression-based time:\t{exprBasedTimeInms}ms.");
            //Assert.True(exprBasedTimeInms <= targetPerformanceCoefficient * explicitTimeInms, "Compiled expression-based approach shouldn't lose too much performance compared to inlined series.");
        }

        [Fact]
        [Trait("Category", "Performance")]
        public void CompiledExpressionOnGammaLnPerformanceTest()
        {
            const int samples = 10;
            const int evaluations = 8000000;
            const double point = 2.2;
            //double targetPerformanceCoefficient = 0.9; // depending on platform, usually is better
            Assert.True(MMath.AreEqual(GammaLnMidExplicit(point), GammaLnMidCompiledExpressionBased(point)));
            double explicitTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) GammaLnMidExplicit(point); }, samples);
            output.WriteLine($"Explicit time:\t{explicitTimeInms}ms.");
            double exprBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) GammaLnMidCompiledExpressionBased(point); }, samples);
            output.WriteLine($"Compiled expression-based time:\t{exprBasedTimeInms}ms.");
            //Assert.True(exprBasedTimeInms <= targetPerformanceCoefficient * explicitTimeInms, "Compiled expression-based approach should perform better than evaluating polynomial using an array of coefficients.");
        }

        [Fact]
        [Trait("Category", "Performance")]
        public void CompiledExpressionOnLog1PlusPerformanceTest()
        {
            const int samples = 10;
            int evaluations = 20000000;
            double point = 1e-3;
            //            double targetPerformanceCoefficientForShortLog = 1.6;
            //            double targetPerformanceCoefficientForInfCompExprVSystemMathLog = 3.0;
            //            double targetPerformanceCoefficient =
            //#if NETFULL
            //                0.9;
            //#else
            //                0.7;
            //#endif
            output.WriteLine($"x = {point}, {evaluations} evaluations.");
            Assert.True(MMath.AbsDiff(Log1PlusShortExplicit(point), Log1PlusShortCompiledExpressionBased(point)) < 1e-15);
            Assert.True(MMath.AbsDiff(Log1PlusShortExplicit(point), MMath.Log1Plus(point)) < 1e-15);
            double shortExplicitTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) Log1PlusShortExplicit(point); }, samples);
            output.WriteLine($"Explicit (truncated series branch) time:\t{shortExplicitTimeInms}ms.");
            double shortExprBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) Log1PlusShortCompiledExpressionBased(point); }, samples);
            output.WriteLine($"Compiled expression-based (truncated) time:\t{shortExprBasedTimeInms}ms.");
            double explicitTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) Log1PlusNaiveInfiniteSeries(point); }, samples);
            output.WriteLine($"Naive infinite series time:\t{explicitTimeInms}ms.");
            double exprBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) MMath.Log1Plus(point); }, samples);
            output.WriteLine($"Compiled expression-based time:\t{exprBasedTimeInms}ms.");
            //Assert.True(shortExplicitTimeInms <= targetPerformanceCoefficientForShortLog * shortExprBasedTimeInms, "Compiled expression-based approach shouldn't lose too much performance compared to inlined series.");
            //Assert.True(exprBasedTimeInms <= targetPerformanceCoefficient * explicitTimeInms, "Compiled expression-based approach should perform better than naive summation.");
            evaluations = 8000000;
            point = 5e-2;
            Assert.True(MMath.AbsDiff(Log1PlusNaiveInfiniteSeries(point), MMath.Log1Plus(point)) < 1e-15);
            output.WriteLine($"x = {point}, {evaluations} evaluations.");
            shortExplicitTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) Log1PlusShortExplicit(point); }, samples);
            output.WriteLine($"Explicit (System.Math branch) time:\t{shortExplicitTimeInms}ms.");
            explicitTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) Log1PlusNaiveInfiniteSeries(point); }, samples);
            output.WriteLine($"Naive infinite series time:\t{explicitTimeInms}ms.");
            exprBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) MMath.Log1Plus(point); }, samples);
            output.WriteLine($"Compiled expression-based time:\t{exprBasedTimeInms}ms.");
            //Assert.True(exprBasedTimeInms <= targetPerformanceCoefficientForInfCompExprVSystemMathLog * shortExplicitTimeInms, "Compiled expression-based approach shouldn't lose too much performance compared to inlined series.");
            //Assert.True(exprBasedTimeInms <= targetPerformanceCoefficient * explicitTimeInms, "Compiled expression-based approach should perform better than naive summation.");
        }

        private double MeasureRunTime(Action action, int samplesCount)
        {
            var sw = new Stopwatch();
            double timeInms = double.PositiveInfinity;
            for (int i = 0; i < samplesCount; ++i)
            {
                sw.Restart();
                action();
                sw.Stop();
                // Expecting only positive noise
                //output.WriteLine($"Sampled {sw.Elapsed.TotalMilliseconds}ms.");
                timeInms = System.Math.Min(timeInms, sw.Elapsed.TotalMilliseconds);
            }
            return timeInms;
        }

        private double MeasureRunTime(Action action, int iterationCount, int invokeCount)
        {
            var sw = new Stopwatch();
            double timeInms = double.PositiveInfinity;
            for (int i = 0; i < iterationCount; ++i)
            {
                sw.Restart();
                for (int j = 0; j < invokeCount; j++)
                {
                    action();
                }
                sw.Stop();
                // Expecting only positive noise
                //output.WriteLine($"Sampled {sw.Elapsed.TotalMilliseconds}ms.");
                timeInms = System.Math.Min(timeInms, sw.Elapsed.TotalMilliseconds);
            }
            return timeInms;
        }
    }

    public class TraceOutputHelper : ITestOutputHelper
    {
        public void WriteLine(string message)
        {
            Trace.WriteLine(message);
        }

        public void WriteLine(string format, params object[] args)
        {
            Trace.WriteLine(string.Format(format, args));
        }
    }
}
