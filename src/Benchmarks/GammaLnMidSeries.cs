// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using Microsoft.ML.Probabilistic.Core.Maths;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Benchmarks
{
    public static class GammaLnMidSeriesFunctions
    {
        private static readonly Series series = new Series(53);
        private static GammaLnMidSeriesInstantiable instance = new GammaLnMidSeriesInstantiable();

        public static readonly double[] gammaTaylorCoefficients =
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

        public static double GammaLnMidExplicit(double x)
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

        public static double GammaLnMidCompiledExpression(double x)
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

        public static double GammaLnMidDirectCallSameAssembly(double x)
        {
            return GammaLnMidExplicit(x);
        }

        public static double GammaLnMidDirectInstanceCallSameAssembly(double x)
        {
            return instance.GammaLnMid(x);
        }
    }

    public class GammaLnMidSeriesInstantiable
    {
        public double GammaLnMid(double x)
        {
            // 1.5 <= x <= 2.5
            // Use Taylor series at x=2
            double result = 0;
            double dx = x - 2;
            double sum = 0;
            for (int i = GammaLnMidSeriesFunctions.gammaTaylorCoefficients.Length - 1; i >= 0; i--)
            {
                sum = dx * (GammaLnMidSeriesFunctions.gammaTaylorCoefficients[i] + sum);
            }
            sum = dx * (1 + MMath.Digamma1 + sum);
            result += sum;
            return result;
        }
    }

    public class GammaLnMidSeriesBenchmarks
    {
        [Params(1.8, 2.2)]
        public double x;

        [Benchmark]
        public double GammaLnMidExplicit() => GammaLnMidSeriesFunctions.GammaLnMidExplicit(x);

        [Benchmark]
        public double GammaLnMidCompiledExpression() => GammaLnMidSeriesFunctions.GammaLnMidCompiledExpression(x);

        [Benchmark]
        public double GammaLnMidDirectCallSameAssembly() => GammaLnMidSeriesFunctions.GammaLnMidDirectCallSameAssembly(x);

        [Benchmark]
        public double GammaLnMidDirectInstanceCallSameAssembly() => GammaLnMidSeriesFunctions.GammaLnMidDirectInstanceCallSameAssembly(x);
    }
}
