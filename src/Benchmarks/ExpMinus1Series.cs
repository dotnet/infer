// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Probabilistic.Core.Maths;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Benchmarks
{
    public static class ExpMinus1SeriesFunctions
    {
        private static readonly Series series = new Series(53);

        public static double ExpMinus1Explicit(double x)
        {
            return x * (1 + x * (0.5 + x * (1.0 / 6 + x * (1.0 / 24))));
        }

        public static double ExpMinus1DirectCall(double x)
        {
            return SeriesCollection.ExpMinus1At0(x);
        }

        public static double ExpMinus1DirectCallWithSwitch(double x)
        {
            return SeriesCollection.ExpMinus1At0Switch(x);
        }

        public static double ExpMinus1DirectCallSameAssembly(double x)
        {
            return ExpMinus1Explicit(x);
        }

        public static double ExpMinus1CompiledExpression(double x)
        {
            return series.ExpMinus1(x);
        }
    }

    public class ExpMinus1SeriesBenchmarks
    {

        [Params(1e-3, 1e-8)]
        public double x;

        [Benchmark]
        public double ExpMinus1Explicit() => ExpMinus1SeriesFunctions.ExpMinus1Explicit(x);

        [Benchmark]
        public double ExpMinus1CompiledExpression() => ExpMinus1SeriesFunctions.ExpMinus1CompiledExpression(x);

        [Benchmark]
        public double ExpMinus1DirectCall() => ExpMinus1SeriesFunctions.ExpMinus1DirectCall(x);

        [Benchmark]
        public double ExpMinus1DirectCallWithSwitch() => ExpMinus1SeriesFunctions.ExpMinus1DirectCallWithSwitch(x);

        [Benchmark]
        public double ExpMinus1DirectCallSameAssembly() => ExpMinus1SeriesFunctions.ExpMinus1DirectCallSameAssembly(x);

        [Benchmark]
        public double ExpMinus1MMathExplicit() => MMath.ExpMinus1Explicit(x);

        [Benchmark]
        public double ExpMinus1DirectMMathCall() => MMath.ExpMinus1SwitchExplicit(x);
    }
}
