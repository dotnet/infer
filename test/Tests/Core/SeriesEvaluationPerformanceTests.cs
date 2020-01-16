using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Microsoft.ML.Probabilistic.Core.Maths;
using Microsoft.ML.Probabilistic.Math;
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
                        if (MMath.AreEqual(sum, oldSum)) break;
                    }
                    term *= x;
                }
                return sum;
            }
        }

        static double ExpMinus1Explicit(double x)
        {
            return x * (1 + x * (0.5 + x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0/(24*5) + x * (1.0/(24*5*6) + x * (1.0/(24*5*6*7))))))));
        }

        public class ExpMinus1CoefficientEnumerator : IEnumerator<double>
        {
            int n = 0;

            public double Current { get; set; }

            object IEnumerator.Current => Current;

            public void Dispose()
            {                
            }

            public bool MoveNext()
            {
                if(n == 0)
                {
                }
                else if(n == 1)
                {
                    Current = 1.0;                   
                }
                else
                {
                    Current /= n;
                }
                n++;
                return true;
            }

            public void Reset()
            {
                n = 0;
                Current = 0.0;
            }
        }

        public class ExpMinus1CoefficientEnumerable : IEnumerable<double>
        {
            public IEnumerator<double> GetEnumerator()
            {
                return new ExpMinus1CoefficientEnumerator();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }

        static IEnumerable<double> ExpMinus1Coefficient2()
        {
            return new ExpMinus1CoefficientEnumerable();
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
            double explicitTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) ExpMinus1Explicit(point); }, samples);
            output.WriteLine($"Explicit time:\t{explicitTimeInms}ms.");
            double delegateBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) expMinus1DelegateBased.Evaluate(point); }, samples);
            output.WriteLine($"Delegate-based time:\t{delegateBasedTimeInms}ms.");
            double enumerableBasedTimeInms = MeasureRunTime(() => { for (int j = 0; j < evaluations; ++j) expMinus1EnumerableBased.Evaluate(point); }, samples);
            output.WriteLine($"Enumerable-based time:\t{enumerableBasedTimeInms}ms.");
            Assert.True(delegateBasedTimeInms <= targetPerformanceCoefficient * enumerableBasedTimeInms, "Delegate-based approach should be faster than enumerable-based one to justify its complexity.");
        }

        [Fact]
        [Trait("Category", "Performance")]
        public void CoefficientGenerationDelegateOutperformsEnumerationOnExpMinus1RatioMinus1RatioMinusHalfTest()
        {
            var expMinus1RatioMinus1RatioMinusHalfDelegateBased = new Series(53).ExpMinus1RatioMinus1RatioMinusHalf;
            var expMinus1RatioMinus1RatioMinusHalfEnumerableBased = new EnumerableBasedPowerSeries(ExpMinus1RatioMinus1RatioMinusHalfCoefficient());
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
