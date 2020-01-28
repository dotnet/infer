// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Probabilistic.Core.Maths;
using Microsoft.ML.Probabilistic.Math;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Probabilistic.Benchmarks
{
    public static class ExpMinus1RatioMinus1RatioMinusHalfSeriesFunctions
    {
        private static readonly Series series = new Series(53);
        private static ExpMinus1RatioMinus1RatioMinusHalfSeriesInstantiable instance = new ExpMinus1RatioMinus1RatioMinusHalfSeriesInstantiable();

        public static double ExpMinus1RatioMinus1RatioMinusHalfExplicit(double x)
        {
            return x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0 / 120 + x * (1.0 / 720 +
                x * (1.0 / 5040 + x * (1.0 / 40320 + x * (1.0 / 362880 + x * (1.0 / 3628800 +
                x * (1.0 / 39916800 + x * (1.0 / 479001600 + x * (1.0 / 6227020800 + x * (1.0 / 87178291200))))))))))));
        }

        public static double ExpMinus1RatioMinus1RatioMinusHalfDirectCall(double x)
        {
            return SeriesCollection.ExpMinus1RatioMinus1RatioMinusHalfExplicitAt0(x);
        }

        public static double ExpMinus1RatioMinus1RatioMinusHalfDirectCallSameAssembly(double x)
        {
            return ExpMinus1RatioMinus1RatioMinusHalfExplicit(x);
        }

        public static double ExpMinus1RatioMinus1RatioMinusHalfDirectInstanceCallSameAssembly(double x)
        {
            return instance.ExpMinus1RatioMinus1RatioMinusHalf(x);
        }

        public static double ExpMinus1RatioMinus1RatioMinusHalfCompiledExpression(double x)
        {
            return series.ExpMinus1RatioMinus1RatioMinusHalf(x);
        }
    }

    public class ExpMinus1RatioMinus1RatioMinusHalfSeriesInstantiable
    {
        public double ExpMinus1RatioMinus1RatioMinusHalf(double x)
        {
            return x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0 / 120 + x * (1.0 / 720 +
                x * (1.0 / 5040 + x * (1.0 / 40320 + x * (1.0 / 362880 + x * (1.0 / 3628800 +
                x * (1.0 / 39916800 + x * (1.0 / 479001600 + x * (1.0 / 6227020800 + x * (1.0 / 87178291200))))))))))));
        }
    }

    public class ExpMinus1RatioMinus1RatioMinusHalfSeriesBenchmarks
    {
        [Params(1e-3, 1e-8)]
        public double x;

        [Benchmark]
        public double ExpMinus1RatioMinus1RatioMinusHalfExplicit() => ExpMinus1RatioMinus1RatioMinusHalfSeriesFunctions.ExpMinus1RatioMinus1RatioMinusHalfExplicit(x);

        [Benchmark]
        public double ExpMinus1RatioMinus1RatioMinusHalfCompiledExpression() => ExpMinus1RatioMinus1RatioMinusHalfSeriesFunctions.ExpMinus1RatioMinus1RatioMinusHalfCompiledExpression(x);

        [Benchmark]
        public double ExpMinus1RatioMinus1RatioMinusHalfDirectCall() => ExpMinus1RatioMinus1RatioMinusHalfSeriesFunctions.ExpMinus1RatioMinus1RatioMinusHalfDirectCall(x);

        [Benchmark]
        public double ExpMinus1RatioMinus1RatioMinusHalfDirectCallSameAssembly() => ExpMinus1RatioMinus1RatioMinusHalfSeriesFunctions.ExpMinus1RatioMinus1RatioMinusHalfDirectCallSameAssembly(x);

        [Benchmark]
        public double ExpMinus1RatioMinus1RatioMinusHalfDirectInstanceCallSameAssembly() => ExpMinus1RatioMinus1RatioMinusHalfSeriesFunctions.ExpMinus1RatioMinus1RatioMinusHalfDirectInstanceCallSameAssembly(x);
    }
}
