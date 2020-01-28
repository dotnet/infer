// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Probabilistic.Core.Maths;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Benchmarks
{
    public static class Log1PlusInfiniteSeriesFunctions
    {
        private static readonly Series series = new Series(53);
        private static Log1PlusInfiniteSeriesInstantiable instance = new Log1PlusInfiniteSeriesInstantiable();

        public static double Log1PlusInfiniteSeriesNaive(double x)
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

        public static double Log1PlusNaiveInfiniteSeriesExplicit(double x)
        {
            return Log1PlusInfiniteSeriesNaive(x);
        }

        public static double Log1PlusNaiveInfiniteSeriesDirectInstanceCall(double x)
        {
            return instance.Log1PlusNaiveInfiniteSeries(x);
        }

        public static double Log1PlusSystemMath(double x)
        {
            return System.Math.Log(1.0 + x);
        }

        public static double Log1PlusInfiniteSeriesCompiledExpression(double x)
        {
            return series.Log1Plus(x);
        }
    }

    public class Log1PlusInfiniteSeriesInstantiable
    {
        public double Log1PlusNaiveInfiniteSeries(double x)
        {
            return Log1PlusInfiniteSeriesFunctions.Log1PlusInfiniteSeriesNaive(x);
        }
    }

    public class Log1PlusInfiniteSeriesBenchmarks
    {
        [Params(5e-2, -1e-4)]
        public double x;

        [Benchmark]
        public double Log1PlusNaiveInfiniteSeriesExplicit() => Log1PlusInfiniteSeriesFunctions.Log1PlusNaiveInfiniteSeriesExplicit(x);

        [Benchmark]
        public double Log1PlusNaiveInfiniteSeriesDirectInstanceCall() => Log1PlusInfiniteSeriesFunctions.Log1PlusNaiveInfiniteSeriesDirectInstanceCall(x);

        [Benchmark]
        public double Log1PlusSystemMath() => Log1PlusInfiniteSeriesFunctions.Log1PlusSystemMath(x);

        [Benchmark]
        public double Log1PlusInfiniteSeriesCompiledExpression() => Log1PlusInfiniteSeriesFunctions.Log1PlusInfiniteSeriesCompiledExpression(x);
    }
}
