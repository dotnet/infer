// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Probabilistic.Core.Maths;
using System;

namespace Microsoft.ML.Probabilistic.Benchmarks
{
    public static class Log1PlusShortSeriesFunctions
    {
        public static double Log1PlusShortExplicit(double x)
        {
            return x * (1 - x * (0.5 - x * (1.0 / 3 - x * (0.25 - x * (1.0 / 5)))));
        }

        private static Func<double, double> Log1PlusTruncated = SeriesEvaluatorFactory.GetCompiledExpressionSeriesEvaluator(n => n == 0 ? 0.0 : (n % 2 == 0 ? -1.0 : 1.0) / n, 5, 0, 5);

        public static double Log1PlusShortCompiledExpression(double x)
        {
            return Log1PlusTruncated(x);
        }

        public static double Log1PlusShortDirectCallSameAssembly(double x)
        {
            return Log1PlusShortExplicit(x);
        }
    }

    public class Log1PlusShortSeriesBenchmarks
    {
        [Params(5e-4, -3e-4)]
        public double x;

        [Benchmark]
        public double Log1PlusShortExplicit() => Log1PlusShortSeriesFunctions.Log1PlusShortExplicit(x);

        [Benchmark]
        public double Log1PlusShortCompiledExpression() => Log1PlusShortSeriesFunctions.Log1PlusShortCompiledExpression(x);

        [Benchmark]
        public double Log1PlusShortDirectCallSameAssembly() => Log1PlusShortSeriesFunctions.Log1PlusShortDirectCallSameAssembly(x);
    }
}
