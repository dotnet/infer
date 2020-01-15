using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Microsoft.ML.Probabilistic.Core.Maths;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class SeriesEvaluationPerformanceTests
    {
        struct EnumerableBasedPowerSeries
        {
            readonly IEnumerable<double> coefGenerator;

            public EnumerableBasedPowerSeries(IEnumerable<double> coefGenerator)
            {
                this.coefGenerator = coefGenerator;
            }

            public double Evaluate(double x)
            {
                double sum = 0.0;
                double term = 1.0;
                double oldSum = double.NaN;
                foreach (double coefficient in coefGenerator)
                {
                    oldSum = sum;
                    if (coefficient != 0)
                    {
                        sum += term * coefficient;
                        if (sum == oldSum) break;
                    }
                    term *= x;
                }
                return sum;
            }
        }

        static IEnumerable<double> ExpMinus1Coefficient()
        {
            yield return 0.0;
            double coef = 1.0;
            int n = 1;
            do
            {
                coef /= n;
                yield return coef;
                ++n;
            }
            while (coef > 0);
        }

        static IEnumerable<double> ExpMinus1RatioMinus1RatioMinusHalfCoefficient()
        {
            yield return 0.0;
            double coef = 0.5;
            int n = 1;
            do
            {
                coef /= n + 2;
                yield return coef;
                ++n;
            }
            while (coef > 0);
        }

        private readonly ITestOutputHelper output;

        public SeriesEvaluationPerformanceTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        [Fact]
        [Trait("Category", "Performance")]
        public void CoefficientGenerationDelegateOutperformsEnumerationOnExpMinus1Test()
        {
            var expMinus1DelegateBased = new Series(53).ExpMinus1;
            var expMinus1EnumerableBased = new EnumerableBasedPowerSeries(ExpMinus1Coefficient());
            var sw = new Stopwatch();
            const int samples = 3;
            const int evaluations = 5000000;
            const double point = 0.5;
            double targetPerformanceCoefficient =
#if NETFULL
                Environment.Is64BitProcess ? 0.9 : 0.98;
#else
                0.9;
#endif
            Assert.Equal(expMinus1DelegateBased.Evaluate(point), expMinus1EnumerableBased.Evaluate(point));
            double delegateBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) expMinus1DelegateBased.Evaluate(point); }, samples);
            double enumerableBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) expMinus1EnumerableBased.Evaluate(point); }, samples);
            output.WriteLine($"Delegate-based time:\t{delegateBasedTimeInms}ms.");
            output.WriteLine($"Enumerable-based time:\t{enumerableBasedTimeInms}ms.");
            Assert.True(delegateBasedTimeInms <= targetPerformanceCoefficient * enumerableBasedTimeInms, "Delegate-based approach should be faster than enumerable-based one to justify its complexity.");
        }

        [Fact]
        [Trait("Category", "Performance")]
        public void CoefficientGenerationDelegateOutperformsEnumerationOnExpMinus1RatioMinus1RatioMinusHalfTest()
        {
            var expMinus1RatioMinus1RatioMinusHalfDelegateBased = new Series(53).ExpMinus1RatioMinus1RatioMinusHalf;
            var expMinus1RatioMinus1RatioMinusHalfEnumerableBased = new EnumerableBasedPowerSeries(ExpMinus1RatioMinus1RatioMinusHalfCoefficient());
            var sw = new Stopwatch();
            const int samples = 3;
            const int evaluations = 5000000;
            const double point = 0.5;
            double targetPerformanceCoefficient =
#if NETFULL
                Environment.Is64BitProcess ? 0.9 : 0.98;
#else
                0.9;
#endif
            Assert.Equal(expMinus1RatioMinus1RatioMinusHalfDelegateBased.Evaluate(point), expMinus1RatioMinus1RatioMinusHalfEnumerableBased.Evaluate(point));
            double delegateBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) expMinus1RatioMinus1RatioMinusHalfDelegateBased.Evaluate(point); }, samples);
            double enumerableBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) expMinus1RatioMinus1RatioMinusHalfEnumerableBased.Evaluate(point); }, samples);
            output.WriteLine($"Delegate-based time:\t{delegateBasedTimeInms}ms.");
            output.WriteLine($"Enumerable-based time:\t{enumerableBasedTimeInms}ms.");
            Assert.True(delegateBasedTimeInms <= targetPerformanceCoefficient * enumerableBasedTimeInms, "Delegate-based approach should be faster than enumerable-based one to justify its complexity.");
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
                timeInms = System.Math.Min(timeInms, sw.Elapsed.TotalMilliseconds);
            }
            return timeInms;
        }
    }
}
