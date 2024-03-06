// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests
{
    using System.Diagnostics;
    using System.Threading;
    using System.Threading.Tasks;

    using Utilities;

    using Assert = AssertHelper;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArray2D = DistributionStructArray2D<Gaussian, double>;


    public class OperatorTests
    {
        [Fact]
        public void GaussianGetDerivativesTest()
        {
            Gaussian g = Gaussian.FromNatural(double.MaxValue, double.MaxValue);
            double x = 2;
            // this.Precision * x overflows
            g.GetDerivatives(x, out double dlogp, out double ddlogp);
            Assert.Equal(-double.MaxValue, dlogp);
        }

        [Fact]
        public void GaussianFromDerivativesTest()
        {
            double x = 2;
            double dlogp = -double.MaxValue;
            double ddlogp = -double.MaxValue;
            Gaussian g = Gaussian.FromDerivatives(x, dlogp, ddlogp, false);
            g.GetDerivatives(x, out double dlogp2, out double ddlogp2);
            Assert.True(MMath.AbsDiff(dlogp, dlogp2) < 1e-10);
            Assert.True(MMath.AbsDiff(ddlogp, ddlogp2) < 1e-10);
        }

        [Fact]
        public void GaussianPlusOpTest()
        {
            var testCases = new[]
            {
                (2, 0, -1, 1/double.MaxValue),
                (1e160, 0, 0, 1e-160),
                (double.MaxValue*1e-160, 1e-160, double.MaxValue*1e-160, 1e-160),
                (double.MaxValue*1e-159, 1e-159, double.MaxValue*1e-160, 1e-160),
                (double.MaxValue*1e-160, 1e-160, double.MaxValue*1e-159, 1e-159),
            };
            foreach(var testCase in testCases)
            {
                double AMean = testCase.Item1;
                double AVariance = testCase.Item2;
                double BMean = testCase.Item3;
                double BVariance = testCase.Item4;
                Gaussian expected = new Gaussian(AMean + BMean, AVariance + BVariance);
                if (AVariance == 0)
                {
                    Assert.True(DoublePlusOp.SumAverageConditional(AMean, new Gaussian(BMean, BVariance)).MaxDiff(expected) < 1e-14);
                }
                Assert.True(MaxUlpDiff(DoublePlusOp.SumAverageConditional(new Gaussian(AMean, AVariance), new Gaussian(BMean, BVariance)), expected) < 2);
                Assert.True(MaxUlpDiff(DoublePlusOp.AAverageConditional(new Gaussian(AMean, AVariance), new Gaussian(-BMean, BVariance)), expected) < 2);
                Assert.True(MaxUlpDiff(DoublePlusOp.BAverageConditional(new Gaussian(AMean, AVariance), new Gaussian(-BMean, BVariance)), expected) < 2);
            }

            Assert.False(DoublePlusOp.BAverageConditional(Gaussian.FromNatural(-8.6696467442044984E+102, 0.43834920434350727), Gaussian.FromNatural(2.193337045017726E+205, 2.193337045017726E+205)).MeanTimesPrecision < double.MinValue);
            Gaussian Sum = Gaussian.FromNatural(1, 1);
            double tolerance = 1e-10;
            foreach (var prec in DoublesGreaterThanZero().Where(x => !double.IsPositiveInfinity(x)))
            {
                Gaussian b = Gaussian.FromNatural(prec, prec);
                Gaussian a = DoublePlusOp.AAverageConditional(Sum, b);
                //Trace.WriteLine($"b = {b}: a = {a}");
                Assert.True(MMath.AbsDiff(a.GetMean(), 0) < tolerance);

                Gaussian a2 = DoublePlusOp.AAverageConditional(b, Sum);
                //Trace.WriteLine($"b = {b}: a2 = {a2}");
                Assert.True(MMath.AbsDiff(a2.GetMean(), 0) < tolerance);

                Gaussian sum2 = DoublePlusOp.SumAverageConditional(Sum, b);
                //Trace.WriteLine($"b = {b}: sum2 = {sum2}");
                Assert.True(MMath.AbsDiff(sum2.GetMean(), 2) < tolerance);
            }
        }

        [Fact]
        public void PlusGammaOpTest()
        {
            long count = 0;
            Parallel.ForEach(new[] {
                GammaPower.FromShapeAndRate(double.MaxValue, double.MaxValue, 100000000000000.0),
            }.Concat(GammaPowers(1000000)).Where(g => g.Power != 0 && g.Shape > 1), gammaPower =>
            {
                Assert.True(gammaPower.IsPointMass || gammaPower.IsProper());
                GammaPower gammaPower1 = GammaPower.FromShapeAndRate(gammaPower.Shape, double.Epsilon, gammaPower.Power);
                GammaPower gammaPower2 = GammaPower.FromShapeAndRate(gammaPower.Shape, double.MaxValue, gammaPower.Power);
                double mean1 = gammaPower1.GetMean();
                double mean2 = gammaPower2.GetMean();
                double largestMean = System.Math.Max(mean1, mean2);
                double smallestMean = System.Math.Min(mean1, mean2);
                double mode1 = gammaPower1.GetMode();
                double mode2 = gammaPower2.GetMode();
                double largestMode = System.Math.Max(mode1, mode2);
                double smallestMode = System.Math.Min(mode1, mode2);
                foreach (var shift in DoublesAtLeastZero().Where(x => !double.IsInfinity(x)))
                {
                    var result = PlusGammaOp.SumAverageConditional(gammaPower, shift);
                    Assert.True((gammaPower.IsPointMass && result.IsPointMass) || result.IsProper());
                    if (gammaPower.Power < 0)
                    {
                        double expected = gammaPower.GetMode() + shift;
                        expected = System.Math.Max(smallestMode, System.Math.Min(largestMode, expected));
                        double actual = result.GetMode();
                        Assert.False(double.IsNaN(actual));
                        double belowActual = GammaPower.FromShapeAndRate(result.Shape, MMath.PreviousDouble(result.Rate), result.Power).GetMode();
                        double aboveActual = GammaPower.FromShapeAndRate(result.Shape, MMath.NextDouble(result.Rate), result.Power).GetMode();
                        Assert.True(MMath.AbsDiff(expected, actual, 1) < double.PositiveInfinity ||
                            MMath.AbsDiff(expected, belowActual, 1e-8) < 1e-8 ||
                            MMath.AbsDiff(expected, aboveActual, 1e-8) < 1e-8);
                    }
                    else
                    {
                        double expected = gammaPower.GetMean() + shift;
                        expected = System.Math.Max(smallestMean, System.Math.Min(largestMean, expected));
                        double actual = result.GetMean();
                        Assert.False(double.IsNaN(actual));
                        double belowActual = GammaPower.FromShapeAndRate(result.Shape, MMath.PreviousDouble(result.Rate), result.Power).GetMean();
                        double aboveActual = GammaPower.FromShapeAndRate(result.Shape, MMath.NextDouble(result.Rate), result.Power).GetMean();
                        Assert.True(MMath.AbsDiff(expected, actual, 1) < double.PositiveInfinity ||
                            MMath.AbsDiff(expected, belowActual, 1e-8) < 1e-8 ||
                            MMath.AbsDiff(expected, aboveActual, 1e-8) < 1e-8);
                    }
                }
                Interlocked.Add(ref count, 1);
                if (count % 100000 == 0)
                    Trace.WriteLine($"{count} cases passed");
            });
            Trace.WriteLine($"{count} cases passed");
        }

        [Fact]
        public void AverageTest()
        {
            foreach (var a in Doubles())
            {
                foreach (var b in Doubles())
                {
                    if (double.IsNaN(a + b)) continue;
                    double midpoint = MMath.Average(a, b);
                    Assert.True(midpoint >= System.Math.Min(a, b));
                    Assert.True(midpoint <= System.Math.Max(a, b));
                }
            }
        }

        [Fact]
        public void AverageLongTest()
        {
            foreach (var a in Longs())
            {
                foreach (var b in Longs())
                {
                    double midpoint = MMath.Average(a, b);
                    Assert.True(midpoint >= System.Math.Min(a, b));
                    Assert.True(midpoint <= System.Math.Max(a, b));
                }
            }
        }

        [Fact]
        public void LargestDoubleRatioTest()
        {
            foreach (var denominator in DoublesGreaterThanZero())
            {
                if (double.IsPositiveInfinity(denominator)) continue;
                foreach (var ratio in Doubles())
                {
                    AssertLargestDoubleRatio(denominator, ratio);
                }
            }
        }

        private void AssertLargestDoubleRatio(double denominator, double ratio)
        {
            double numerator = MMath.LargestDoubleRatio(ratio, denominator);
            Assert.True((double)(numerator * denominator) <= ratio);
            Assert.True(double.IsPositiveInfinity(numerator) || (double)(MMath.NextDouble(numerator) * denominator) > ratio);
        }

        /// <summary>
        /// Tests an edge case involving subnormal numbers.
        /// </summary>
        [Fact]
        public void LargestDoubleProductTest2()
        {
            // This case needs 50 iterations
            MMath.LargestDoubleProduct(9.8813129168249309E-324, 1.7976931348623157E+308);
            MMath.LargestDoubleProduct(-4.94065645841247E-324, 1.7976931348623157E+308);
            MMath.LargestDoubleProduct(1.0000000000000166E-300, 1.0000000000000005E-09);
            MMath.LargestDoubleProduct(-1.0000000000000166E-300, 1.0000000000000005E-09);
            MMath.LargestDoubleProduct(4.9187693503017E-319, 0.00115249439895759);
            MMath.LargestDoubleProduct(-4.9187693503017E-319, 0.00115249439895759);
        }

        [Fact]
        public void LargestDoubleProductTest()
        {
            foreach (var denominator in DoublesGreaterThanZero())
            {
                if (double.IsPositiveInfinity(denominator)) continue;
                foreach (var ratio in Doubles())
                {
                    AssertLargestDoubleProduct(denominator, ratio);
                }
            }
        }

        private void AssertLargestDoubleProduct(double denominator, double ratio)
        {
            double numerator = MMath.LargestDoubleProduct(ratio, denominator);
            Assert.True((double)(numerator / denominator) <= ratio);
            Assert.True(double.IsPositiveInfinity(numerator) || (double)(MMath.NextDouble(numerator) / denominator) > ratio);
        }

        [Fact]
        public void LargestDoubleSumTest()
        {
            MMath.LargestDoubleSum(1, -1);
            foreach (var b in Doubles())
            {
                if (double.IsNegativeInfinity(b)) continue;
                if (double.IsPositiveInfinity(b)) continue;
                foreach (var sum in Doubles())
                {
                    AssertLargestDoubleSum(b, sum);
                }
            }
        }

        private void AssertLargestDoubleSum(double b, double sum)
        {
            double a = MMath.LargestDoubleSum(b, sum);
            Assert.True((double)(a - b) <= sum);
            Assert.True(double.IsPositiveInfinity(a) || MMath.NextDouble(a) - b > sum);
        }

        [Fact]
        public void PrecisionAverageConditional_Point_IsIncreasing()
        {
            foreach (var precision in DoublesAtLeastZero())
            {
                foreach (var yv in DoublesAtLeastZero())
                {
                    var result0 = GaussianOp.PrecisionAverageConditional_Point(0, yv, precision);
                    var result = GaussianOp.PrecisionAverageConditional_Point(MMath.NextDouble(0), yv, precision);
                    Assert.True(result.Rate >= result0.Rate);
                    //Trace.WriteLine($"precision={precision} yv={yv}: {result0.Rate} {result.Rate}");
                }
            }
        }

        [Fact]
        public void ExpMinus1_IsIncreasing()
        {
            IsIncreasing(MMath.ExpMinus1);
        }

        [Fact]
        public void ExpMinus1RatioMinus1RatioMinusHalf_IsIncreasing()
        {
            IsIncreasingForAtLeastZero(MMath.ExpMinus1RatioMinus1RatioMinusHalf);
        }

        [Fact]
        public void Log1MinusExp_IsDecreasing()
        {
            IsIncreasingForAtLeastZero(x => MMath.Log1MinusExp(-x));
        }

        [Fact]
        public void NormalCdfLn_IsIncreasing()
        {
            IsIncreasing(MMath.NormalCdfLn);
        }

        [Fact]
        public void NormalCdfRatio_IsIncreasing()
        {
            IsIncreasing(MMath.NormalCdfRatio);
        }

        [Fact]
        public void NormalCdfRatioDiff_Converges()
        {
            foreach (var x in DoublesAtLeastZero().Select(y => -y))
            {
                if (double.IsInfinity(x)) continue;
                foreach (var delta in Doubles().Select(y => y + System.Math.Abs(x)))
                {
                    if (System.Math.Abs(delta) <= System.Math.Abs(x) * 0.7 || System.Math.Abs(delta) <= 9.9)
                    {
                        double f = MMath.NormalCdfRatioDiff(x, delta);
                        Assert.False(double.IsInfinity(f));
                    }
                }
            }
        }

        [Fact]
        public void NormalCdfMomentRatio_IsIncreasing()
        {
            for (int i = 1; i < 10; i++)
            {
                IsIncreasing(x => MMath.NormalCdfMomentRatio(i, x));
            }
        }

        [Fact]
        public void LogExpMinus1_IsIncreasing()
        {
            IsIncreasingForAtLeastZero(MMath.LogExpMinus1);
        }

        [Fact]
        public void Log1Plus_IsIncreasing()
        {
            IsIncreasingForAtLeastZero(x => MMath.Log1Plus(x - 1));
            IsIncreasingForAtLeastZero(MMath.Log1Plus);
        }

        [Fact]
        public void OneMinusSqrtOneMinus_IsIncreasing()
        {
            IsIncreasingForAtLeastZero(x => MMath.OneMinusSqrtOneMinus(1 / (1 + 1 / x)));
        }

        [Fact]
        public void GammaLower_IsIncreasingInX()
        {
           Parallel.ForEach(DoublesGreaterThanZero(), a =>
           {
               IsIncreasingForAtLeastZero(x => MMath.GammaLower(a, x));
           });
        }

        [Fact]
        public void GammaLower_IsDecreasingInA()
        {
            Parallel.ForEach(DoublesAtLeastZero(), x =>
            {
                IsIncreasingForAtLeastZero(a => -MMath.GammaLower(a + double.Epsilon, x));
            });
        }

        [Fact]
        public void GammaUpper_IsDecreasingInX()
        {
            double tiny = 4.94065645841247E-324;
            bool debug = false;
            if (debug)
            {
                double x2 = 1.7976931348623466;
                x2 = 0.10000000000000024;
                //x2 = 0.099999999999923289;
                //x2 = 1 - 1e-15;
                double fp = double.NaN;
                bool anyIncreasing = false;
                Stopwatch watch = new Stopwatch();
                for (int i = 0; i < 100; i++)
                {
                    double f = MMath.GammaUpper(tiny, x2, false);
                    watch.Restart();
                    for (int j = 0; j < 10000; j++)
                    {
                        f = MMath.GammaUpper(tiny, x2, false);
                    }
                    var elapsed = watch.ElapsedMilliseconds;
                    string star = (f > fp) ? "increasing" : "";
                    anyIncreasing |= (f > fp);
                    Trace.WriteLine($"{x2:g17} {f:g17} {elapsed} {star}");
                    x2 = MMath.NextDouble(x2);
                    fp = f;
                }
                Assert.False(anyIncreasing);
                return;
            }
            Assert.True(MMath.GammaUpper(tiny, 0.10000000000000024, false) >= MMath.GammaUpper(tiny, 0.10000000000000026, false));
            Assert.True(MMath.GammaUpper(tiny, 1.7976931348623466, false) >= MMath.GammaUpper(tiny, 1.7976931348623475, false));
            Assert.True(MMath.GammaUpper(tiny, 1.7976931348623466, false) >= MMath.GammaUpper(tiny, 1.7976931348623484, false));
            Parallel.ForEach(DoublesGreaterThanZero(), a =>
            {
                IsIncreasingForAtLeastZero(x => -MMath.GammaUpper(a, x, false));
            });
        }

        [Fact]
        public void GammaUpperRegularized_IsDecreasingInX()
        {
            Parallel.ForEach(DoublesGreaterThanZero(), a =>
            {
                IsIncreasingForAtLeastZero(x => -MMath.GammaUpper(a, x, true));
            });
        }

        [Fact]
        public void GammaUpper_IsIncreasingInA()
        {
            Parallel.ForEach(DoublesAtLeastZero(), x =>
            {
                IsIncreasingForAtLeastZero(a => MMath.GammaUpper(a + double.Epsilon, x), 2);
            });
        }

        [Fact]
        public void GammaLn_IsIncreasingAbove2()
        {
            IsIncreasingForAtLeastZero(x => MMath.GammaLn(2 + x));
        }

        [Fact]
        public void GammaLn_IsDecreasingBelow1()
        {
            IsIncreasingForAtLeastZero(x => MMath.GammaLn(1 / (1 + x)));
        }

        [Fact]
        public void Digamma_IsIncreasing()
        {
            IsIncreasingForAtLeastZero(x => MMath.Digamma(x));
        }

        [Fact]
        public void Trigamma_IsDecreasing()
        {
            IsIncreasingForAtLeastZero(x => -MMath.Trigamma(x));
        }

        [Fact]
        public void Tetragamma_IsIncreasing()
        {
            IsIncreasingForAtLeastZero(x => MMath.Tetragamma(x));
        }

        [Fact]
        public void ReciprocalFactorialMinus1_IsDecreasingAbove1()
        {
            IsIncreasingForAtLeastZero(x => -MMath.ReciprocalFactorialMinus1(1 + x));
        }

        /// <summary>
        /// Asserts that a function is increasing over the entire domain of reals.
        /// </summary>
        /// <param name="func"></param>
        /// <returns></returns>
        public bool IsIncreasing(Func<double, double> func)
        {
            foreach (var x in Doubles())
            {
                double fx = func(x);
                foreach (var delta in DoublesGreaterThanZero())
                {
                    double x2 = x + delta;
                    if (double.IsPositiveInfinity(delta)) x2 = delta;
                    double fx2 = func(x2);
                    // The cast here is important when running in 32-bit, Release mode.
                    Assert.True((double)fx2 >= fx);
                }
            }
            return true;
        }

        /// <summary>
        /// Asserts that a function is increasing over the non-negative reals.
        /// </summary>
        /// <param name="func"></param>
        /// <returns></returns>
        public bool IsIncreasingForAtLeastZero(Func<double, double> func, double ulpError = 0)
        {
            double scale = 1 + 2e-16 * ulpError;
            foreach (var x in DoublesAtLeastZero())
            {
                double fx = func(x);
                double smallFx;
                if (fx >= 0) smallFx = fx / scale;
                else smallFx = fx * scale;
                foreach (var delta in DoublesGreaterThanZero())
                {
                    double x2 = x + delta;
                    if (double.IsPositiveInfinity(delta)) x2 = delta;
                    double fx2 = func(x2);
                    // The cast here is important when running in 32-bit, Release mode.
                    Assert.True((double)fx2 >= smallFx);
                }
            }
            return true;
        }

        // Used to debug IsBetweenOp
        internal static void GaussianIsBetween_Experiments()
        {
            double lowerBound = -10000.0;
            double upperBound = -9999.9999999999982;
            double mx = 0.5;
            //mx = 2 * 0.5 - mx;
            Gaussian X = new Gaussian(mx, 1 / 10.1);
            X = Gaussian.FromNatural(0.010000100000000001, 0.00010000100000000002);
            //X = Gaussian.FromNatural(0.010000000000000002, 0.00010000000000000002);
            X = Gaussian.FromNatural(10, 1);
            lowerBound = 9999.9999999999982;
            upperBound = 10000.0;
            X = Gaussian.FromNatural(1.0000000000000001E+73, 1E+69);
            bool largeDelta = false;
            if (largeDelta)
            {
                lowerBound = -990000;
                upperBound = 10000.0;
                X = Gaussian.FromNatural(0.00010000000000000002, 1.0000000000000014E-25);
                X = Gaussian.FromNatural(0.00010000000000000002, 1.0000000000000028E-49);
                // tests accuracy of drU
                X = Gaussian.FromNatural(0.00010000000000000002, 1.0000000000000006E-12);
                //upperBound = 990000;
                //lowerBound = -10000.0;
                //X = Gaussian.FromNatural(-0.00010000000000000002, 1.0000000000000014E-25);
            }
            bool zeroDelta = false;
            if (zeroDelta)
            {
                lowerBound = 9999.99999999999;
                upperBound = 10000.0;
                X = Gaussian.FromNatural(0.00010000000000000002, 1.0000000000000005E-08);
                // center and mx are not equal, have significant errors
                // upperBound*X.Precision has same error
                // instead of (mx>center), check (-zU-zL) > 0 or (zU+zL) < 0 
            }
            /*
from mpmath import *
mp.dps = 100; mp.pretty = True
tau=mpf('-0.00010000000000000002'); prec=mpf('1.0000000000000005E-08'); L=mpf('-10000.0'); U = mpf('-9999.99999999999'); 
mx = tau/prec; 
center = (L+U)/2;
mx
(tau - prec*center)
zU = (U - mx)*sqrt(prec)
zL = (L - mx)*sqrt(prec)
             */
            bool largezU = false;
            if (largezU)
            {
                lowerBound = 0;
                upperBound = 1e54;
                X = Gaussian.FromNatural(1.0000000000000013E-23, 1.0000000000000043E-77);
                mx = X.GetMean();
                X.SetMeanAndPrecision(mx, 1e-80);
            }
            bool smallzU = false;
            if (smallzU)
            {
                lowerBound = 0;
                upperBound = 2e-69;
                X = Gaussian.FromNatural(10000000000000, 1.0000000000000056E-100);
            }
            bool smallDiffs = false;
            if (smallDiffs)
            {
                lowerBound = 0;
                upperBound = 2e-69;
                X = Gaussian.FromNatural(1.0000000000000052E-93, 1.0000000000000014E-24);
            }
            mx = X.GetMean();
            bool showNeighborhood = true;
            if (showNeighborhood)
            {
                for (int i = 0; i < 100; i++)
                {
                    //XAverageConditional_Debug(X, lowerBound, upperBound);
                    //X = Gaussian.FromMeanAndPrecision(mx, X.Precision + 1.0000000000000011E-19);
                    Gaussian toX2 = IsBetweenGaussianOp.XAverageConditional(Bernoulli.PointMass(true), X, lowerBound, upperBound);
                    Gaussian xPost = X * toX2;
                    ////Console.WriteLine($"mx = {X.GetMean():g17} mp = {xPost.GetMean():g17} vp = {xPost.GetVariance():g17} toX = {toX2}");
                    //X.Precision *= 100;
                    //X.MeanTimesPrecision *= 0.999999;
                    //X.SetMeanAndPrecision(mx, X.Precision * 2);
                    X.SetMeanAndPrecision(mx, MMath.NextDouble(X.Precision));
                }
            }
            else
            {
                var mva = new MeanVarianceAccumulator();
                Rand.Restart(0);
                for (int i = 0; i < 10000000; i++)
                {
                    double sample = X.Sample();
                    if (sample > lowerBound && sample < upperBound)
                        mva.Add(sample);
                }
                Gaussian toX = IsBetweenGaussianOp.XAverageConditional(Bernoulli.PointMass(true), X, lowerBound, upperBound);
                ////Console.WriteLine($"expected mp = {mva.Mean}, vp = {mva.Variance}, {X * toX}");
                XAverageConditional_Debug(X, lowerBound, upperBound);
            }
        }

        private static void XAverageConditional_Debug(Gaussian X, double lowerBound, double upperBound)
        {
            double mx = X.GetMean();
            double sqrtPrec = System.Math.Sqrt(X.Precision);
            double zL = (lowerBound - mx) * sqrtPrec;
            double zU = (upperBound - mx) * sqrtPrec;
            double logZ = IsBetweenGaussianOp.LogProbBetween(X, lowerBound, upperBound);
            double logPhiL = X.GetLogProb(lowerBound);
            double ZR = System.Math.Exp(logZ - logPhiL);
            // Z/X.Prob(U)*sqrtPrec = NormalCdfRatio(zU) - NormalCdfRatio(zL)*X.Prob(L)/X.Prob(U) 
            // Z/X.Prob(L)*sqrtPrec = NormalCdfRatio(zU)*X.Prob(U)/X.Prob(L) - NormalCdfRatio(zL)
            //                      = NormalCdfRatio(zU)*exp(delta) - NormalCdfRatio(zL)
            double diff = upperBound - lowerBound;
            double diffs = zU - zL;
            diffs = diff * sqrtPrec;
            double center = (upperBound + lowerBound) / 2;
            double delta = diff * (X.MeanTimesPrecision - X.Precision * center);
            // -(zL+zU)/2 = delta/diffs
            double deltaOverDiffs = (-zL - zU) / 2;
            // This is more accurate than checking mx<center
            if (deltaOverDiffs < 0) throw new Exception("deltaOverDiffs < 0");
            delta = diffs * deltaOverDiffs;
            double rL = MMath.NormalCdfRatio(zL);
            double rU = MMath.NormalCdfRatio(zU);
            double r1L = MMath.NormalCdfMomentRatio(1, zL);
            double r1U = MMath.NormalCdfMomentRatio(1, zU);
            double r2U = MMath.NormalCdfMomentRatio(2, zU) * 2;
            double r2L = MMath.NormalCdfMomentRatio(2, zL) * 2;
            double r3L = MMath.NormalCdfMomentRatio(3, zL) * 6;
            double r3U = MMath.NormalCdfMomentRatio(3, zU) * 6;
            double r4L = MMath.NormalCdfMomentRatio(4, zL) * 24;
            double drU, drU2, drU3, drU4, dr1U, dr1U2, dr1U3, dr1U4, dr2U, dr2U2, dr2U3;
            if (diffs < 0.5 || true)
            {
                // drU is noisy - why?
                //drU = MMath.NormalCdfRatioDiff(zL, diffs);
                //drU2 = MMath.NormalCdfRatioDiff(zL, diffs, 2);
                //drU3 = MMath.NormalCdfRatioDiff(zL, diffs, 3);
                drU4 = MMath.NormalCdfRatioDiff(zL, diffs, 4);
                drU3 = diffs * (r3L / 6 + drU4);
                drU2 = diffs * (r2L / 2 + drU3);
                drU = diffs * (r1L + drU2);
                dr1U = MMath.NormalCdfMomentRatioDiff(1, zL, diffs);
                dr1U2 = MMath.NormalCdfMomentRatioDiff(1, zL, diffs, 2);
                dr1U3 = MMath.NormalCdfMomentRatioDiff(1, zL, diffs, 3);
                dr1U4 = MMath.NormalCdfMomentRatioDiff(1, zL, diffs, 4);
                dr2U = MMath.NormalCdfMomentRatioDiff(2, zL, diffs);
                dr2U2 = MMath.NormalCdfMomentRatioDiff(2, zL, diffs, 2);
                dr2U3 = MMath.NormalCdfMomentRatioDiff(2, zL, diffs, 3);
            }
            else
            {
                // drU = diffs*(r1L + drU2)
                drU = rU - rL;
                // drU2 = diffs*(r2L/2 + drU3)
                drU2 = drU / diffs - r1L;
                // drU3 = diffs*(r3L/6 + drU4)
                drU3 = drU2 / diffs - r2L / 2;
                drU4 = drU3 / diffs - r3L / 6;
                // dr1U = diffs*(r2L + dr1U2)
                dr1U = r1U - r1L;
                // dr1U2 = dr1U / diffs - r2L
                // dr1U2 = diffs*r3L/2 + diffs^2*r4L/6 + ...
                // dr1U2 = diffs*(r3L/2 + dr1U3)
                dr1U2 = dr1U / diffs - r2L;
                // dr1U3 = diffs*(r4L/6 + dr1U4)
                dr1U3 = dr1U2 / diffs - r3L / 2;
                dr1U4 = dr1U3 / diffs - r4L / 6;
                // dr2U = diffs*(r3L + dr2U2)
                dr2U = r2U - r2L;
                // dr2U2 = diffs*(r4L/2 + dr2U3)
                dr2U2 = dr2U / diffs - r3L;
                dr2U3 = dr2U2 / diffs - r4L / 2;
            }
            double expMinus1 = MMath.ExpMinus1(delta);
            double ZR2 = (rU * System.Math.Exp(delta) - rL) / sqrtPrec;
            double ZR2b = (drU + rU * expMinus1) / sqrtPrec;
            double ZR2c = (drU + rU * diffs * (-zL - zU) / 2 + rU * (expMinus1 - delta)) / sqrtPrec;
            List<double> ZRs = new List<double>() { ZR, ZR2, ZR2b, ZR2c };
            double ZR4 = (0.5 * diffs * diffs * (-zL) + diffs) / sqrtPrec;
            // delta = 0.0004 diffs = 0.0002: ZR6 is exact, but not ZR5
            // delta = 0.0002 diffs = 0.0002: ZR6 is exact, but not ZR5
            // delta = 0.0001 diffs = 0.0002: ZR6 is exact, but not ZR5
            // delta = 2.5E-05 diffs = 0.0001: ZR6 is exact, but not ZR5
            // delta = 1E-05 diffs = 6.32455532033676E-05: ZR6 is exact, but not ZR5
            // delta = 2.5E-06 diffs = 0.0001: ZR6 is exact, but not ZR5
            // delta = 6.25E-06 diffs = 5E-05: ZR5 is exact
            // delta = 6.25E-07 diffs = 5E-05: ZR5 is exact
            // delta = 0.02 diffs = 2E-05 diffs*zL = -0.0200000002: neither ZR5 nor ZR6 are exact
            // seems we need diffs and diffs*zL (or delta) to be small
            // delta = 0.0002 diffs = 2E-05 diffs*zL = -0.0002000002: ZR6 is exact, but not ZR5
            // delta = 2E-05 diffs = 2E-05 diffs*zL = -2.00002E-05: ZR5 is exact
            // delta = 2E-08 diffs = 2E-05 diffs*zL = -2.02E-08: ZR5 is exact
            // all we need is a good approx for (ZR/diff - 1)
            double ZR5 = (1.0 / 6 * diffs * diffs * diffs * (-1 + zL * zL) + 0.5 * diffs * diffs * (-zL) + diffs) / sqrtPrec;
            double ZR6 = (1.0 / 24 * diffs * diffs * diffs * diffs * (zL - zL * zL * zL + 2 * zL) + 1.0 / 6 * diffs * diffs * diffs * (-1 + zL * zL) + 0.5 * diffs * diffs * (-zL) + diffs) / sqrtPrec;
            //Console.WriteLine($"zL = {zL:g17} delta = {delta:g17} (-zL-zU)/2*diffs={(-zL - zU) / 2 * diffs:g17} diffs = {diffs:g17} diffs*zL = {diffs * zL}");
            //Console.WriteLine($"Z/N = {ZR} {ZR2} {ZR2b} {ZR2c} asympt:{ZRasympt} {ZR4} {ZR5} {ZR6}");
            // want to compute Z/X.Prob(L)/diffs + (exp(delta)-1)/delta
            double expMinus1RatioMinus1RatioMinusHalf = MMath.ExpMinus1RatioMinus1RatioMinusHalf(delta);
            // expMinus1RatioMinus1 = expMinus1Ratio - 1;
            double expMinus1RatioMinus1 = delta * (0.5 + expMinus1RatioMinus1RatioMinusHalf);
            double expMinus1Ratio = 1 + expMinus1RatioMinus1;
            //double expMinus1Ratio = expMinus1 / delta;
            double numer = ZR2b / diff - expMinus1Ratio;
            //WriteLast(new[] { diffs * (r1L + drU2), drU, rU, diffs, numer });
            // when delta < 254 and zU < -1e61, we can assume drU =approx 0
            // (drU + rU * expMinus1) / diffs - expMinus1Ratio
            // drU/diffs + rU*expMinus1Ratio*delta/diffs - expMinus1Ratio
            // = (r1L + drU2) + expMinus1Ratio*(rU*(-zU-zL)/2 - 1)
            // r1L = r1U - dr1U
            // zL = zU - diffs
            // drU = diffs*(r1U-dr1U+drU2)
            double numerSmallzU = drU2 - dr1U + r1U + expMinus1Ratio * (rU * (-zU + diffs / 2) - 1);
            // rU = (r1U-1)/zU
            double numerSmallzU2 = drU2 - dr1U + r1U + expMinus1Ratio * (rU * diffs / 2 - r1U);
            // when delta is small:
            // expMinus1Ratio = 1 + expMinus1RatioMinus1
            double numerSmallzU3 = drU2 - dr1U + rU * diffs / 2 + expMinus1RatioMinus1 * (rU * diffs / 2 - r1U);
            // expMinus1RatioMinus1 = delta * (0.5 + expMinus1RatioMinus1RatioMinusHalf)
            // delta = diffs*(-zL-zU)/2
            // zL = zU - diffs
            // r1U = (r2U - rU)/zU = r2U/zU - r1U/zU^2 + 1/zU^2
            double numerSmallzU4 = drU2 - dr1U + diffs / 2 * (r2U - diffs / 2 * r1U) + expMinus1RatioMinus1 * rU * diffs / 2 - delta * expMinus1RatioMinus1RatioMinusHalf * r1U;
            List<double> numerSmallzUs = new List<double>() { numerSmallzU, numerSmallzU2, numerSmallzU3, numerSmallzU4 };
            double numer1SmallzL = -(-rL * zL) + drU2 + rU * expMinus1RatioMinus1 * delta / diffs + rU * delta / diffs - expMinus1RatioMinus1;
            // rU*delta/diffs = rU*(-zL-zU)/2 = rU*(-zL)/2 - (r1U-1)/2 - rL*(-zL)/2 + rL*(-zL)/2 = drU*(-zL)/2 - (r1U-1)/2 - (r1L-1)/2
            double numerLargezL = drU2 + rU * expMinus1RatioMinus1 * delta / diffs + drU * (-zL) / 2 - r1U / 2 + r1L / 2 - expMinus1RatioMinus1;
            // drU2 - dr1U/2 = diffs^2*r3L*(1/6 - 1/4) + ... = diffs^2*r3L*(-1/12) + ...
            double numerLargezL2 = (drU2 - dr1U / 2) + rU * expMinus1RatioMinus1 * delta / diffs + drU * (-zL) / 2 - expMinus1RatioMinus1;
            // now substitute rU*delta/diffs again
            double numerLargezL3 = (drU2 - dr1U / 2) + (drU * (-zL) / 2 - r1U / 2 - r1L / 2) * expMinus1RatioMinus1 + drU * (-zL) / 2;
            // drU*zL = diffs*(r1L*zL + drU2*zL) = diffs*(r2L-rL + drU2*zL)
            // expMinus1RatioMinus1 =approx delta/2 = 0.5*diffs*(-zL-zU)/2
            // replace r1L = drU/diffs - drU2
            // replace r1U = dr1U + r1L
            double numerLargezL4 = (drU2 - dr1U / 2) + (drU * (-zL) / 2 - dr1U / 2 + drU2) * expMinus1RatioMinus1 - drU / diffs * expMinus1RatioMinus1 + drU * (-zL) / 2;
            double numerLargezL5 = (drU2 - dr1U / 2) * expMinus1Ratio + drU * (-zL) / 2 * expMinus1RatioMinus1 - drU / diffs * (expMinus1RatioMinus1 - delta / 2)
                - drU / diffs * delta / 2 + drU * (-zL) / 2;
            // delta/diffs + zL = -diffs/2
            // cancellation in (drU2 - dr1U / 2)
            // drU2 = diffs*(r2L/2 + drU3)
            // dr1U = diffs*(r2L + dr1U2)
            double numerLargezL6 = (drU2 - dr1U / 2) * expMinus1Ratio + drU * (
                (-zL) / 2 * expMinus1RatioMinus1
                - deltaOverDiffs * expMinus1RatioMinus1RatioMinusHalf
                + diffs / 4);
            // divide by diffs
            double numerLargezL7 = (drU3 - dr1U2 / 2) * expMinus1Ratio + (r1L + drU2) * (
                (-zL) / 2 * expMinus1RatioMinus1
                - deltaOverDiffs * expMinus1RatioMinus1RatioMinusHalf
                + diffs / 4);
            List<double> numerLargezLs = new List<double>() { numerLargezL, numerLargezL2, numerLargezL3, numerLargezL4, numerLargezL4, numerLargezL5, numerLargezL6, numerLargezL7 * diffs };
            //WriteLast(numerLargezLs);
            double numer1e = drU2 + (drU * (-zL) / 2 - r1U / 2 - r1L / 2) * expMinus1RatioMinus1 + (-rL * (zU - zL) / 2 + drU * (-zL - zU) / 2);
            double numer2 = (0.5 * diffs * (-zL) + 1.0 / 6 * diffs * diffs * (-1 + zL * zL) + 1.0 / 24 * diffs * diffs * diffs * (zL - zL * zL * zL + 2 * zL)) - expMinus1RatioMinus1;
            double numer3 = (0.5 * diffs * (-zL) - delta * 0.5) + 1.0 / 6 * (-diffs * diffs + (diffs * diffs * zL * zL - delta * delta)) - 1.0 / 24 * delta * delta * delta - 1.0 / 120 * delta * delta * delta * delta;
            // diffs*(-zL) - delta = diffs*(-zL + (zL+zU)/2) = diffs*diffs/2
            // (diffs*(-zL))^2 = (delta + diffs^2/2)^2 = delta^2 + delta*diffs^2 + diffs^4/4
            double numer4 = 0.5 * diffs * ((-zL) - (-zL - zU) * 0.5) + 1.0 / 6 * (-diffs * diffs + (diffs * diffs * zL * zL - delta * delta)) - 1.0 / 24 * delta * delta * delta - 1.0 / 120 * delta * delta * delta * delta;
            // delta = 0.2 diffs = 0.14142135623731: bad
            // delta = 0.02 diffs = 0.014142135623731: bad
            // delta = 0.002 diffs = 0.0014142135623731: bad
            // delta = 0.0002 diffs = 0.00014142135623731: bad
            // delta = 2E-08 diffs = 1.4142135623731E-08: good
            double numer5 = delta * diffs * diffs / 6 + diffs * diffs * diffs * diffs / 24 - 1.0 / 24 * delta * delta * delta - 1.0 / 120 * delta * delta * delta * delta + diffs * diffs / 12;
            //Console.WriteLine($"numer = {numer} smallzL:{numer1SmallzL} largezL:{numerLargezL} {numerLargezL2} {numerLargezL3} {numerLargezL4:g17} {numerLargezL5:g17} {numerLargezL6:g17} {numerLargezL7:g17} {numerLargezL8:g17} {numer1e} asympt:{numerAsympt} {numerAsympt2} {numer2} {numer3} {numer4} {numer5}");
            double mp = mx - System.Math.Exp(logPhiL - logZ) * expMinus1 / X.Precision;
            double mp2 = center + (delta / diff - System.Math.Exp(logPhiL - logZ) * expMinus1) / X.Precision;
            double mp3 = center + (delta / diff * ZR2b - expMinus1) * System.Math.Exp(logPhiL - logZ) / X.Precision;
            double alphaXcLprecDiffs = diffs / (ZR2b * X.Precision);
            //Console.WriteLine($"alpha = {alphaXcLprec}");
            double mpLargezL4 = center + numerLargezL4 * delta / (ZR2b * X.Precision);
            double mpLargezL5 = center + numerLargezL5 * delta / (ZR2b * X.Precision);
            double mpLargezL6 = center + numerLargezL6 * delta / diffs * alphaXcLprecDiffs;
            double mpLargezL7 = center + numerLargezL7 * delta * alphaXcLprecDiffs;
            // center = lowerBound + diff/2 = upperBound - diff/2
            // try adding -diffs/delta/2*(rU * deltaOverDiffs * expMinus1Ratio + r1L + drU2) to numer
            // -rU/2*expMinus1Ratio - (r1L+drU2)/2/deltaOverDiffs
            // when the first term dominates (large delta), this is -rU/2*expMinus1Ratio
            // -rU / 2 * expMinus1Ratio + r1L * (-zL) / 2 * expMinus1RatioMinus1
            // r1L = (r2L - rL)/zL
            // should not be used for delta<1e-2
            // same as above for delta=10
            double numerLargezL8 = (drU3 - dr1U2 / 2 - drU / 2 - r2L / 2 + drU2 * (-zL) / 2) * expMinus1Ratio
                + (r1L + drU2) * (zL / 2
                - deltaOverDiffs * expMinus1RatioMinus1RatioMinusHalf
                - 0.5 / deltaOverDiffs
                + diffs / 4);
            double mpLargezL8 = upperBound + numerLargezL8 * delta * alphaXcLprecDiffs;
            List<double> mpLargezLs = new List<double>() { mpLargezL4, mpLargezL5, mpLargezL6, mpLargezL7, mpLargezL8 };
            WriteLast(mpLargezLs);
            double alphaXcLprecSmallzU = 1 / (rU * expMinus1 * sqrtPrec);
            double mpSmallzU4 = center + numerSmallzU4 * delta * alphaXcLprecSmallzU;
            double mpSmallzU4b = center + ((drU2 - dr1U) / rU + diffs / 2 * (r2U / rU - diffs / 2 * r1U / rU) + expMinus1RatioMinus1 * diffs / 2 - delta * expMinus1RatioMinus1RatioMinusHalf * r1U / rU) * delta / (expMinus1 * sqrtPrec);
            // r1U/rU = (r2U/rU - 1)/zU =approx -1/zU
            //double mpSmallzU2b = center + (expMinus1RatioMinus1 * diffs / 2 - delta * expMinus1RatioMinus1RatioMinusHalf /(-zU)) * delta / (expMinus1 * sqrtPrec);
            List<double> mpSmallzUs = new List<double>() { mpSmallzU4, mpSmallzU4b };
            //WriteLast(mpSmallzUs);
            double mp5 = center + numer5 * delta / diffs * alphaXcLprecDiffs;
            //double mpBrute = Util.ArrayInit(10000000, i => X.Sample()).Where(sample => (sample > lowerBound) && (sample < upperBound)).Average();
            //Console.WriteLine($"mp = {mp} {mp2} {mp3} {mpLargezL4:g17} {mpLargezL5:g17} {mpLargezL6:g17} {mpLargezL7:g17} {mpLargezL8:g17} asympt:{mpAsympt} {mpAsympt2} {mp5}");
            double cL = -1 / expMinus1;
            // rU*diffs = rU*zU - rU*zL = r1U - 1 - rU*zL + rL*zL - rL*zL = r1U - 1 - drU*zL - (r1L-1) = dr1U - drU*zL
            // zL = -diffs/2 - delta/diffs
            // zU = diffs/2 - delta/diffs
            // rU = r2U - zU*r1U
            //if (zL != -diffs / 2 - delta / diffs) throw new Exception();
            //if (zU != diffs / 2 - delta / diffs) throw new Exception();
            double q = ((rU - drU * cL) * (rU - drU * cL) - r1U * (1 - cL) - r1L * cL + (1 - cL) * cL * diffs * drU) / cL / cL;
            double q2 = (-rU * expMinus1 - drU) * (-rU * expMinus1 - drU) - r1U * (expMinus1 + 1) * expMinus1 + r1L * expMinus1 - (expMinus1 + 1) * diffs * drU;
            double q3 = (rU * expMinus1 + drU) * (rU * expMinus1 + drU) + (r1L - r1U * (expMinus1 + 1)) * expMinus1 - (expMinus1 + 1) * diffs * drU;
            // r1L = r1U - dr1U
            double q4 = (rU * expMinus1) * (rU * expMinus1) + 2 * rU * expMinus1 * drU + drU * drU + (-dr1U - r1U * expMinus1) * expMinus1 - (expMinus1 + 1) * diffs * drU;
            double q5 = (rU * expMinus1) * (rU * expMinus1) + 2 * rU * expMinus1 * drU + drU * diffs * (r1L + drU2 - 1 - expMinus1) + (-dr1U - r1U * expMinus1) * expMinus1;
            double q6 = (rU * expMinus1) * (rU * expMinus1) + 2 * rU * expMinus1 * drU + drU * diffs * (zL * rL + drU2 - expMinus1) - dr1U * expMinus1 - r1U * expMinus1 * expMinus1;
            // replace drU and r1L
            // want to relate rU*drU with dr1U
            // rU*drU = approx r2U*diffs
            // rU = r2U - zU*r1U
            // drU = diffs*(r1L + drU2)
            // dr1U = diffs*(r2L + dr1U2)
            double q7 = rU * expMinus1 * rU * expMinus1 + rU * expMinus1 * drU + rU * diffs * (zL * rL + drU2) * expMinus1 + (dr2U - zU * r1U - dr1U2) * diffs * expMinus1 + drU * diffs * (zL * rL + drU2 - expMinus1) - r1U * expMinus1 * expMinus1;
            // replace zL
            double q8 = rU * expMinus1 * rU * expMinus1 + drU * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (-diffs / 2 * rL + drU2 - expMinus1)) + rU * diffs * (zL * rL + drU2) * expMinus1 + (dr2U - zU * r1U - dr1U2) * diffs * expMinus1 - r1U * expMinus1 * expMinus1;
            // replace zU
            double q9 = rU * expMinus1 * rU * expMinus1 + drU * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (-diffs / 2 * rL + drU2 - expMinus1)) + rU * diffs * (zL * rL + drU2) * expMinus1 + (dr2U - dr1U2) * diffs * expMinus1 - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * expMinus1;
            // replace zL
            // rU * diffs * (zL * rL + drU2) * expMinus1 => rU*expMinus1 *diffs*((-diffs/2-delta/diffs)*rL + drU2)
            double q10 = rU * expMinus1 * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (-diffs / 2 * rL + drU2))
                + drU * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (-diffs / 2 * rL + drU2 - expMinus1))
                + (dr2U - dr1U2) * diffs * expMinus1
                - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * expMinus1;
            // drU = diffs*(r1L + drU2)
            // drU2 = diffs*(r2L/2 + drU3)
            // -diffs/2 *rL + drU2 = diffs/2*(r2L-rL) + diffs*drU3 =  diffs/2*r1L*zL + diffs*drU3 = diffs/2*(1+zL*rL)*zL + diffs*drU3
            double q11 = rU * expMinus1 * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (diffs / 2 * r1L * zL + diffs * drU3))
                + drU * (rU * expMinus1RatioMinus1 * delta + diffs * ((rL * zL + drU2) * delta + diffs / 2 * rL * zL * zL + diffs * drU3 - expMinus1RatioMinus1 * delta))
                + drU * diffs * diffs / 2 * zL
                - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * delta
                + (dr2U - dr1U2) * diffs * expMinus1
                - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * expMinus1RatioMinus1 * delta;
            // combine terms 3 and 4
            double q12 = rU * expMinus1 * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (diffs / 2 * r1L * zL + diffs * drU3))
                + drU * (rU * expMinus1RatioMinus1 * delta + diffs * ((rL * zL + drU2) * delta + diffs / 2 * rL * zL * zL + diffs * drU3 - expMinus1RatioMinus1 * delta))
                + diffs * diffs / 2 * (diffs * (r1L + drU2) * zL - r1U * delta)
                + (dr2U - dr1U2) * diffs * delta
                + (dr2U - dr1U2) * diffs * expMinus1RatioMinus1 * delta
                - expMinus1RatioMinus1 * delta * r1U * delta
                - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * expMinus1RatioMinus1 * delta;
            // combine terms 3 and 4
            // replace zL
            // dr2U = diffs*(r3L + dr2U2)
            // dr1U2 = diffs*(r3L/2 + dr1U3)
            double q13 = rU * expMinus1 * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (diffs / 2 * r1L * zL + diffs * drU3))
                + drU * (rU * expMinus1RatioMinus1 * delta + diffs * ((rL * zL + drU2) * delta + diffs / 2 * rL * zL * zL + diffs * drU3 - expMinus1RatioMinus1 * delta))
                + diffs * diffs / 2 * (-diffs * (r1L + drU2) * diffs / 2 + delta * (r3L - r1U - r1L - drU2 + 2 * dr2U2 - 2 * dr1U3))
                + (dr2U - dr1U2) * diffs * expMinus1RatioMinus1 * delta
                - expMinus1RatioMinus1 * delta * r1U * delta
                - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * expMinus1RatioMinus1 * delta;
            // 3rd term:
            // r3L = zL*r2L + 2*r1L
            double q14 = rU * expMinus1 * (rU * expMinus1RatioMinus1 * delta + drU * delta + diffs * (diffs / 2 * r1L * zL + diffs * drU3))
                + drU * (rU * expMinus1RatioMinus1 * delta + diffs * ((rL * zL + drU2) * delta + diffs / 2 * rL * zL * zL + diffs * drU3 - expMinus1RatioMinus1 * delta))
                + diffs * diffs / 2 * (-diffs * (r1L + drU2) * diffs / 2 + delta * (zL * r2L - dr1U - drU2 + 2 * dr2U2 - 2 * dr1U3))
                + (dr2U - dr1U2) * diffs * expMinus1RatioMinus1 * delta
                - expMinus1RatioMinus1 * delta * r1U * delta
                - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * expMinus1RatioMinus1 * delta;
            // combine terms 1,3,5
            // drU*delta = diffs*(r1L + drU2)*delta = diffs*diffs/2*(r1L+drU2)*(-zL-zU)
            // rU*delta/2*delta = rU*diffs*diffs/2*(-zL-zU)/2*(-zL-zU)/2 = diffs*diffs/2*(-rU*zL-rU*zU)/2*(-zL-zU)/2 = diffs*diffs/2*(rU*(diffs-zU) + 1-r1U)/2*(-zL-zU)/2
            //                  = diffs*diffs/2*(rU*diffs + 2-2*r1U)/2*(-zL-zU)/2
            // diffs*diffs/2*r1L*(-zU) = diffs*diffs/2*r1L*(-diffs/2+delta/diffs) = -diffs*diffs/2*r1L*diffs/2 + diffs/2*r1L*delta
            // drU = diffs*(r1L + drU2)
            // dr1U = diffs*(r2L + dr1U2)
            double q15 = rU * expMinus1 * (rU * expMinus1RatioMinus1RatioMinusHalf * delta * delta - (rU / 2 + (r1L + drU2) * zL) * diffs * diffs / 2 * delta - diffs * diffs / 2 * r1L * diffs / 2 + diffs * diffs / 2 * drU2 * (-zL - zU) + diffs * diffs * drU3)
                + rU * expMinus1RatioMinus1 * delta * diffs / 2 * delta
                //+ delta*delta*((rU-r2L) * diffs / 2 - delta/2*r1U - expMinus1RatioMinus1RatioMinusHalf * delta * r1U)
                + delta * delta * ((dr2U - diffs * r1U / 2) * diffs / 2 - expMinus1RatioMinus1RatioMinusHalf * delta * r1U)
                - diffs * diffs / 2 * delta * diffs / 2 * r2L
                + diffs * diffs / 2 * (delta * (-dr1U - drU2 + 2 * dr2U2 - 2 * dr1U3) - diffs * (r1L + drU2) * diffs / 2)
                + drU * (rU * expMinus1RatioMinus1 * delta + diffs * ((rL * zL + drU2) * delta + diffs / 2 * rL * zL * zL + diffs * drU3 - expMinus1RatioMinus1 * delta))
                + (dr2U - dr1U2) * diffs * expMinus1RatioMinus1 * delta
                - (diffs / 2 * diffs + expMinus1RatioMinus1 * delta) * r1U * expMinus1RatioMinus1 * delta;
            // combine terms 1,2,3 and 8
            // zU = diffs/2 - delta/diffs
            // rU = r2U -r1U*zU = r2U - r1U*(diffs/2 - delta/diffs)
            double drU2r1U = IsBetweenGaussianOp.NormalCdfRatioSqrMinusDerivative(zU, rU, r1U, r3U);
            double q16 =
                delta * delta * (drU2r1U * expMinus1 * expMinus1RatioMinus1RatioMinusHalf + (r2U - r1U * diffs / 2) * expMinus1RatioMinus1 * diffs / 2)
                + rU * expMinus1 * (-(rU / 2 + (r1L + drU2) * zL) * diffs * diffs / 2 * delta - diffs * diffs / 2 * r1L * diffs / 2 + diffs * diffs / 2 * drU2 * (-zL - zU) + diffs * diffs * drU3)
                - delta * diffs / 2 * diffs * r1U * expMinus1RatioMinus1
                + delta * delta * (dr2U - diffs * r1U / 2) * diffs / 2
                - delta * diffs * diffs / 2 * diffs / 2 * r2L
                + diffs * diffs / 2 * (delta * (-dr1U - drU2 + 2 * dr2U2 - 2 * dr1U3) - diffs * (r1L + drU2) * diffs / 2)
                // cancellation inside
                + drU * (rU * expMinus1RatioMinus1 * delta + diffs * ((rL * zL + drU2) * delta + diffs / 2 * rL * zL * zL + diffs * drU3 - expMinus1RatioMinus1 * delta))
                + delta * (dr2U - dr1U2) * diffs * expMinus1RatioMinus1;
            double q17 =
                delta * delta * (drU2r1U * expMinus1 * expMinus1RatioMinus1RatioMinusHalf + (r2U - r1U * diffs / 2) * expMinus1RatioMinus1 * diffs / 2)
                + rU * expMinus1 * (-(rU / 2 + (r1L + drU2) * zL) * diffs * diffs / 2 * delta - diffs * diffs / 2 * r1L * diffs / 2 + diffs * diffs / 2 * drU2 * (-zL - zU) + diffs * diffs * drU3)
                - delta * diffs / 2 * diffs * r1U * expMinus1RatioMinus1
                + delta * delta * (dr2U - diffs * r1U / 2) * diffs / 2
                - delta * diffs * diffs / 2 * diffs / 2 * r2L
                + delta * (dr2U - dr1U2) * diffs * expMinus1RatioMinus1
                + diffs * diffs / 2 * (delta * (-dr1U - drU2 + 2 * dr2U2 - 2 * dr1U3) - diffs * (r1L + drU2) * diffs / 2)
                // cancellation inside
                // drU = diffs*(r1L + drU2)
                // drU2 = diffs*(r2L/2 + drU3)
                // r2L = zL*r1L + rL
                // zL = -diffs/2 - delta/diffs
                + drU * (rU * expMinus1RatioMinus1RatioMinusHalf * delta * delta
                    - rL * delta * delta * delta / 2
                    + diffs * diffs * drU3
                    + r2L / 2 * diffs * diffs * delta
                    + diffs * diffs * drU3 * delta
                    - diffs * expMinus1RatioMinus1RatioMinusHalf * delta * delta
                    + diffs * drU2 * delta * delta / 2
                    + rL * diffs * diffs / 2 * diffs / 2 * diffs / 2
                    - rL * diffs * diffs / 2 * delta * delta / 2);
            double q18 =
                delta * delta * (drU2r1U / diffs * expMinus1Ratio * deltaOverDiffs * expMinus1RatioMinus1RatioMinusHalf + (r2U / diffs - r1U / 2) * expMinus1RatioMinus1 / 2)
                // r3L = zL*r2L + 2*r1L
                + rU * expMinus1 * (-(rU / 2 + (r1L + drU2) * zL) / 2 * delta + diffs * (r3L / 6 + drU4 - r1L / 4) + drU2 / diffs * delta)
                + delta * expMinus1RatioMinus1 * (dr2U / diffs - dr1U2 / diffs - r1U / 2)
                + delta * delta / 2 * (dr2U / diffs - r1U / 2)
                // cancellation inside
                // dr1U = diffs*(r2L + dr1U2)
                // dr2U2 = diffs*(r4L/2 + dr2U3)
                // dr1U3 = diffs*(r4L/6 + dr1U4)
                // r4L = zL*r3L + 3*r2L
                + delta * diffs / 2 * (2 * zL * r3L / 3 + 2 * dr2U3 - 2 * dr1U4 - drU3 - dr1U2)
                + (r1L + drU2) * (delta * deltaOverDiffs * rU * expMinus1RatioMinus1RatioMinusHalf
                    - delta * delta * expMinus1RatioMinus1RatioMinusHalf
                    + delta * delta * drU2 / 2
                    - delta * delta * deltaOverDiffs / 2 * rL
                    + delta * diffs * r2L / 2
                    + delta * diffs * drU3
                    // drU3 = diffs*(r3L/6 + drU4)
                    + diffs * (drU3 - diffs / 4 + diffs * diffs * rL / 8)
                    - delta * delta / 2 * diffs / 2 * rL);
            List<double> qs = new List<double>() { q, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18 };
            //WriteLast(qs);
            // from q4:
            // (rU*rU - r1U) = drU2r1U = O(1/zU^4)
            double qSmallzU = drU2r1U * expMinus1 * expMinus1 + drU * drU + drU * (2 * rU * expMinus1 - (expMinus1 + 1) * diffs) - dr1U * expMinus1;
            // drU = diffs*(r1U-dr1U+drU2) = O(1/zU^3)
            // dr1U = diffs*(r2U - dr2U + dr1U2)
            double qSmallzU2 = drU2r1U * expMinus1 * expMinus1 - drU * diffs + ((r1U - dr1U + drU2) * 2 * rU - r2U + dr2U - dr1U2) * diffs * expMinus1 - drU * diffs * expMinus1 + drU * drU;
            // rU = (r1U-1)/zU
            // r2U = (r3U - 2r1U)/zU
            // expMinus1 = delta*(1 + expMinus1RatioMinus1)
            double qSmallzU3 = drU2r1U * expMinus1 * expMinus1 - drU * diffs + (2 * r1U * r1U - r3U) / zU * diffs * expMinus1 + ((-dr1U + drU2) * 2 * rU + dr2U - dr1U2) * diffs * expMinus1 - drU * diffs * expMinus1 + drU * drU;
            double qSmallzU4 = drU2r1U * delta * (1 + expMinus1RatioMinus1) * delta * (1 + expMinus1RatioMinus1) - diffs * (r1U - dr1U + drU2) * diffs + (2 * r1U * r1U - r3U) / zU * diffs * expMinus1 + ((-dr1U + drU2) * 2 * rU + dr2U - dr1U2) * diffs * expMinus1 - drU * diffs * expMinus1 + drU * drU;
            // every term is O(1/zU^4)
            double qSmallzU5 =
                delta * delta * (drU2r1U * expMinus1 * expMinus1RatioMinus1RatioMinusHalf + (r2U - r1U * diffs / 2) * expMinus1RatioMinus1 * diffs / 2)
                + delta * diffs * diffs / 2 * rU * rU / 2 * expMinus1
                - delta * diffs * diffs / 2 * r1U * (expMinus1RatioMinus1 + delta / 2)
                + drU * (delta * delta * (rU - diffs) * expMinus1RatioMinus1RatioMinusHalf
                    - delta * delta * delta / 2 * rL);
            List<double> qSmallzUs = new List<double>() { q4, qSmallzU, qSmallzU2, qSmallzU3, qSmallzU4, qSmallzU5 };
            //WriteLast(qSmallzUs);
            double vp15 = q15 / diffs / diffs * alphaXcLprecDiffs * alphaXcLprecDiffs;
            double vp16 = q16 / diffs / diffs * alphaXcLprecDiffs * alphaXcLprecDiffs;
            double vp17 = q17 / diffs / diffs * alphaXcLprecDiffs * alphaXcLprecDiffs;
            double vp18 = q18 * alphaXcLprecDiffs * alphaXcLprecDiffs;
            double vpSmallzU =
                (r1U * expMinus1 * expMinus1RatioMinus1RatioMinusHalf
                + (2 * rU - diffs / 2) * expMinus1RatioMinus1 * diffs / 2
                + diffs * diffs / 2 / 2 * expMinus1Ratio
                - diffs / 2 * rU * (expMinus1RatioMinus1 + delta / 2)
                + diffs * rU * ((1 - delta) * expMinus1RatioMinus1RatioMinusHalf - delta / 2))
                / (expMinus1Ratio * expMinus1Ratio * X.Precision);
            List<double> vps = new List<double>() { vp15, vp16, vp17, vp18, vpSmallzU };
            //WriteLast(vps);
        }

        private static void WriteLast(ICollection<double> doubles)
        {
            int maxCount = 5;
            foreach (var item in doubles.Skip(System.Math.Max(0, doubles.Count - maxCount)))
            {
                Console.Write(item.ToString("r").PadRight(24));
            }
            Console.WriteLine();
        }

        [Fact]
        public void GaussianIsBetweenTest()
        {
            Assert.True(IsBetweenGaussianOp.LogProbBetween(Gaussian.FromNatural(3172.868479466179, 1.5806459147637875E-06), -0.18914271233981969, 0.18914271233981969) < 0);
            Assert.True(IsBetweenGaussianOp.LogProbBetween(new Gaussian(-49.13, 1.081), new Gaussian(47, 1.25), new Gaussian(48, 1.25)) < -10);
            Assert.True(!double.IsNaN(IsBetweenGaussianOp.LogZ(Bernoulli.PointMass(true), new Gaussian(-49.13, 1.081), new Gaussian(47, 1.25), new Gaussian(48, 1.25))));

            IsBetweenGaussianOp.LogProbBetween(Gaussian.FromNatural(-1.9421168441678062E-32, 3.0824440518369444E-35),
                Gaussian.FromNatural(-9.4928010569516363, 0.0084055128328109872),
                Gaussian.FromNatural(0.67290512665940949, 0.0042043926136280212));

            double actual, expected;
            Gaussian x = new Gaussian(0, 1);

            double lp = IsBetweenGaussianOp.LogProbBetween(x, 100, 100.1);
            double lp2 = IsBetweenGaussianOp.LogProbBetween(x, -100.1, -100);
            Assert.True(MMath.AbsDiff(lp, lp2, 1e-8) < 1e-8);
            ////Console.WriteLine("{0} {1}", lp, lp2);

            Gaussian lowerBound = Gaussian.PointMass(0);
            Gaussian upperBound = Gaussian.PointMass(1);
            Assert.True(System.Math.Abs(IsBetweenGaussianOp.LogProbBetween(new Gaussian(1, 3e-5), 0.5, 1.1)) < 1e-10);
            Assert.True(
                System.Math.Abs(IsBetweenGaussianOp.LogProbBetween(new Gaussian(1, 1e-6), Gaussian.FromMeanAndVariance(0.5, 1e-10), Gaussian.FromMeanAndVariance(1.1, 1e-10))) < 1e-10);
            // test in matlab: x=randn(1e7,1); [mean(0<x & x<1) normcdf(1)-normcdf(0)]
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, lowerBound, upperBound).GetProbTrue();
            expected = MMath.NormalCdf(1) - MMath.NormalCdf(0);
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, lowerBound.Point, upperBound.Point).GetProbTrue();
            expected = MMath.NormalCdf(1) - MMath.NormalCdf(0);
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);
            Assert.Throws<AllZeroException>(() =>
            {
                Assert.Equal(1.0, IsBetweenGaussianOp.IsBetweenAverageConditional(Gaussian.PointMass(0.5), 1, 0).GetProbTrue());
            });
            Assert.Throws<AllZeroException>(() =>
            {
                Assert.Equal(1.0, IsBetweenGaussianOp.IsBetweenAverageConditional(new Gaussian(0, 1), 1, 0).GetProbTrue());
                Assert.True(false, "Did not throw exception");
            });
            Assert.Equal(1.0, IsBetweenGaussianOp.IsBetweenAverageConditional(Gaussian.PointMass(0), 0, 1).GetProbTrue());
            Assert.Equal(0.0, IsBetweenGaussianOp.IsBetweenAverageConditional(Gaussian.PointMass(-1), Gaussian.PointMass(0), Gaussian.FromMeanAndVariance(1, 1)).GetProbTrue());
            Assert.Equal(0.0, IsBetweenGaussianOp.IsBetweenAverageConditional(Gaussian.PointMass(1), 0, 1).GetProbTrue());
            Assert.Equal(0.0, IsBetweenGaussianOp.IsBetweenAverageConditional(Gaussian.PointMass(1), Gaussian.FromMeanAndVariance(0, 1), Gaussian.PointMass(1)).GetProbTrue());

            Assert.Equal(1.0, IsBetweenGaussianOp.IsBetweenAverageConditional(new Gaussian(0, 1), Double.NegativeInfinity, Double.PositiveInfinity).GetProbTrue());
            Assert.Equal(0.0, IsBetweenGaussianOp.IsBetweenAverageConditional(new Gaussian(0, 1), 0.0, Double.PositiveInfinity).LogOdds);
            Assert.Equal(0.0, IsBetweenGaussianOp.IsBetweenAverageConditional(new Gaussian(0, 1), Double.NegativeInfinity, 0.0).LogOdds);
            Assert.Equal(0.0, IsBetweenGaussianOp.IsBetweenAverageConditional(new Gaussian(0, 1), Gaussian.PointMass(Double.NegativeInfinity), new Gaussian()).LogOdds);

            lowerBound = new Gaussian();
            upperBound = Gaussian.PointMass(1);
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, lowerBound, upperBound).GetProbTrue();
            expected = MMath.NormalCdf(1) * 0.5;
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);

            lowerBound = Gaussian.PointMass(0);
            upperBound = new Gaussian();
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, lowerBound, upperBound).GetProbTrue();
            expected = MMath.NormalCdf(0) * 0.5;
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);

            lowerBound = new Gaussian(1, 8);
            upperBound = new Gaussian(3, 3);
            // test in matlab: x=randn(1e7,1); mean(randnorm(1e7,1,[],8)'<x & x<randnorm(1e7,3,[],3)')
            // bvnl(-1/3,3/2,-1/2/3) = 0.33632
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, lowerBound, upperBound).GetProbTrue();
            expected = MMath.NormalCdf(-1.0 / 3, 3.0 / 2, -1.0 / 2 / 3);
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);

            // uniform x
            x.SetToUniform();
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, lowerBound, upperBound).GetProbTrue();
            expected = 0;
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, 0, 1).GetProbTrue();
            expected = 0;
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, new Gaussian(), upperBound).GetProbTrue();
            expected = 0.25;
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);
            actual = IsBetweenGaussianOp.IsBetweenAverageConditional(x, new Gaussian(), new Gaussian()).GetProbTrue();
            expected = 0.25;
            Assert.True(System.Math.Abs(expected - actual) < 1e-10);
        }

        // Test that the operator behaves correctly for arguments with small variance
        [Fact]
        public void GaussianIsBetween_PointLowerBound()
        {
            Bernoulli isBetween = Bernoulli.PointMass(true);
            Gaussian x = new Gaussian(0, 1);
            Gaussian upperBound;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                    upperBound = Gaussian.PointMass(1);
                else
                    upperBound = new Gaussian(1, 1e-8);
                ////Console.WriteLine($"upperBound = {upperBound}");
                Gaussian lowerBound = Gaussian.PointMass(-1);
                Gaussian result2 = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                ////Console.WriteLine($"{lowerBound}: {result2}");
                for (int i = 6; i < 30; i++)
                {
                    double v = System.Math.Pow(0.1, i);
                    lowerBound = Gaussian.FromMeanAndVariance(-1, v);
                    Gaussian result = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                    double error = result.MaxDiff(result2);
                    ////Console.WriteLine($"{lowerBound}: {result} {error}");
                    Assert.True(error < 1e-6);
                }
            }
        }

        // Test that the operator behaves correctly for arguments with small variance
        [Fact]
        public void GaussianIsBetween_PointUpperBound()
        {
            Bernoulli isBetween = Bernoulli.PointMass(true);
            Gaussian x = new Gaussian(0, 1);
            Gaussian lowerBound;
            for (int trial = 0; trial < 2; trial++)
            {
                if (trial == 0)
                    lowerBound = Gaussian.PointMass(-1);
                else
                    lowerBound = new Gaussian(-1, 1e-8);
                ////Console.WriteLine($"lowerBound = {lowerBound}");
                Gaussian upperBound = Gaussian.PointMass(1);
                Gaussian result2 = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                ////Console.WriteLine($"{upperBound}: {result2}");
                for (int i = 6; i < 300; i++)
                {
                    double v = System.Math.Pow(0.1, i);
                    upperBound = Gaussian.FromMeanAndVariance(1, v);
                    Gaussian result = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                    double error = result.MaxDiff(result2);
                    ////Console.WriteLine($"{upperBound}: {result} {error}");
                    Assert.True(error < 1e-6);
                }
            }
        }

        public static double UlpDiff(double a, double b)
        {
            if (a == b) return 0; // avoid infinity-infinity
            double diff = System.Math.Abs(a - b);
            if (diff == 0) return diff;  // avoid 0/0
            return diff / System.Math.Max(MMath.Ulp(a), MMath.Ulp(b));
        }

        public static double MaxUlpDiff(Gaussian a, Gaussian b)
        {
            return System.Math.Max(UlpDiff(a.MeanTimesPrecision, b.MeanTimesPrecision), UlpDiff(a.Precision, b.Precision));
        }

        /// <summary>
        /// Generates a representative set of int64 numbers for testing purposes.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<long> Longs()
        {
            yield return long.MaxValue;
            yield return 0L;
            yield return long.MinValue;
            for (int i = 0; i < 63; i++)
            {
                double bigValue = System.Math.Pow(2, i);
                yield return -(long)bigValue;
                yield return (long)bigValue;
            }
        }

        /// <summary>
        /// Generates a representative set of double-precision numbers for testing purposes.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<double> Doubles()
        {
            yield return double.NegativeInfinity;
            yield return MMath.NextDouble(double.NegativeInfinity);
            yield return MMath.PreviousDouble(0);
            yield return 0.0;
            yield return MMath.NextDouble(0);
            yield return MMath.PreviousDouble(double.PositiveInfinity);
            yield return double.PositiveInfinity;
            for (int i = 0; i <= 323; i++)
            {
                if (i <= 308)
                {
                    double bigValue = System.Math.Pow(10, i);
                    yield return -bigValue;
                    yield return bigValue;
                }
                if (i != 0)
                {
                    double smallValue = System.Math.Pow(0.1, i);
                    yield return -smallValue;
                    yield return smallValue;
                }
            }
        }

        public static IEnumerable<double> Doubles(int count)
        {
            return Doubles().TakeRandom(count);
        }

        public static IEnumerable<double> DoublesGreaterThanZero()
        {
            return Doubles().Where(value => value > 0);
        }

        public static IEnumerable<double> DoublesGreaterThanZero(int count)
        {
            return DoublesGreaterThanZero().TakeRandom(count);
        }

        public static IEnumerable<double> DoublesLessThanZero()
        {
            return Doubles().Where(value => value < 0);
        }

        public static IEnumerable<double> DoublesAtLeastZero()
        {
            return Doubles().Where(value => value >= 0);
        }

        public static IEnumerable<double> DoublesAtLeastZero(int count)
        {
            return DoublesAtLeastZero().TakeRandom(count);
        }

        public static IEnumerable<Bernoulli> Bernoullis()
        {
            foreach (var logOdds in Doubles())
            {
                yield return Bernoulli.FromLogOdds(logOdds);
            }
        }

        /// <summary>
        /// Generates a representative set of proper Gamma distributions.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<Gamma> Gammas()
        {
            foreach (var shape in DoublesGreaterThanZero())
            {
                foreach (var rate in DoublesGreaterThanZero())
                {
                    yield return Gamma.FromShapeAndRate(shape, rate);
                }
            }
        }

        /// <summary>
        /// Generates a representative set of proper Gamma distributions.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<Gamma> Gammas(int count)
        {
            int half = (int)System.Math.Sqrt(count);
            foreach (var shape in DoublesGreaterThanZero(half))
            {
                foreach (var rate in DoublesGreaterThanZero(half))
                {
                    yield return Gamma.FromShapeAndRate(shape, rate);
                }
            }
        }

        /// <summary>
        /// Generates a representative set of proper GammaPower distributions.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<GammaPower> GammaPowers(int count)
        {
            int half = (int)System.Math.Sqrt(count);
            foreach (var gamma in Gammas(half))
            {
                foreach (var power in Doubles(half).Where(x => !double.IsInfinity(x)))
                {
                    yield return GammaPower.FromGamma(gamma, power);
                }
            }
        }

        public static IEnumerable<GammaPower> GammaPowers()
        {
            foreach (var gamma in Gammas())
            {
                foreach (var power in Doubles().Where(x => !double.IsInfinity(x)))
                {
                    yield return GammaPower.FromGamma(gamma, power);
                }
            }
        }

        /// <summary>
        /// Generates a representative set of proper TruncatedGamma distributions with infinite upper bound.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<TruncatedGamma> LowerTruncatedGammas(int count)
        {
            int half = (int)System.Math.Sqrt(count);
            foreach (var gamma in Gammas(half))
            {
                foreach (var lowerBound in DoublesAtLeastZero(half))
                {
                    if (gamma.IsPointMass && gamma.Point < lowerBound) continue;
                    yield return new TruncatedGamma(gamma, lowerBound, double.PositiveInfinity);
                }
            }
        }

        /// <summary>
        /// Generates a representative set of proper TruncatedGamma distributions.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<TruncatedGamma> TruncatedGammas(int count)
        {
            int half = (int)System.Math.Sqrt(count);
            int quarter = (int)System.Math.Sqrt(half);
            foreach (var gamma in Gammas(quarter))
            {
                foreach (var lowerBound in DoublesAtLeastZero(quarter))
                {
                    foreach (var gap in DoublesGreaterThanZero(half))
                    {
                        double upperBound = lowerBound + gap;
                        if (upperBound == lowerBound) continue;
                        if (gamma.IsPointMass && (gamma.Point < lowerBound || gamma.Point > upperBound)) continue;
                        yield return new TruncatedGamma(gamma, lowerBound, upperBound);
                    }
                }
            }
        }

        /// <summary>
        /// Generates a representative set of proper TruncatedGamma distributions.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<TruncatedGamma> TruncatedGammas()
        {
            foreach (var gamma in Gammas())
            {
                foreach (var lowerBound in DoublesAtLeastZero())
                {
                    foreach (var gap in DoublesGreaterThanZero())
                    {
                        double upperBound = lowerBound + gap;
                        if (upperBound == lowerBound) continue;
                        if (gamma.IsPointMass && (gamma.Point < lowerBound || gamma.Point > upperBound)) continue;
                        yield return new TruncatedGamma(gamma, lowerBound, upperBound);
                    }
                }
            }
        }

        /// <summary>
        /// Generates a representative set of proper Gaussian distributions.
        /// </summary>
        /// <returns></returns>
        public static IEnumerable<Gaussian> Gaussians()
        {
            foreach (var tau in Doubles())
            {
                foreach (var precision in DoublesGreaterThanZero())
                {
                    yield return Gaussian.FromNatural(tau, precision);
                }
            }
        }

        public static IEnumerable<Gaussian> Gaussians(int count)
        {
            int half = (int)System.Math.Sqrt(count);
            foreach (var tau in Doubles(half))
            {
                foreach (var precision in DoublesGreaterThanZero(half))
                {
                    yield return Gaussian.FromNatural(tau, precision);
                }
            }
        }

        public static IEnumerable<double> UpperBounds(double lowerBound)
        {
            HashSet<double> set = new HashSet<double>();
            foreach (var diff in DoublesGreaterThanZero())
            {
                double upperBound = lowerBound + diff;
                if (double.IsPositiveInfinity(diff))
                    upperBound = diff;
                if (!set.Contains(upperBound))
                {
                    set.Add(upperBound);
                    yield return upperBound;
                }
            }
        }

        [Fact]
        public void GaussianIsBetweenCRCC_IsSymmetricInXMean()
        {
            IsBetweenGaussianOp.XAverageConditional(new Bernoulli(0), Gaussian.FromNatural(-1E+153, 1e308), -1.0000000000000085E-153, 1.0000000000000085E-153);
            double meanMaxUlpError = 0;
            double meanMaxUlpErrorLowerBound = 0;
            double meanMaxUlpErrorUpperBound = 0;
            Bernoulli meanMaxUlpErrorIsBetween = new Bernoulli();
            double precMaxUlpError = 0;
            double precMaxUlpErrorLowerBound = 0;
            double precMaxUlpErrorUpperBound = 0;
            Bernoulli precMaxUlpErrorIsBetween = new Bernoulli();
            foreach (var isBetween in new[] { Bernoulli.PointMass(true), Bernoulli.PointMass(false), new Bernoulli(0.1) })
            {
                Parallel.ForEach(DoublesLessThanZero(), lowerBound =>
                {
                    //Console.WriteLine($"isBetween = {isBetween}, lowerBound = {lowerBound:g17}");
                    foreach (var upperBound in new[] { -lowerBound }.Concat(UpperBounds(lowerBound)).Take(1))
                    {
                        //Console.WriteLine($"lowerBound = {lowerBound:g17}, upperBound = {upperBound:g17}");
                        double center = MMath.Average(lowerBound, upperBound);
                        if (double.IsNegativeInfinity(lowerBound) && double.IsPositiveInfinity(upperBound))
                            center = 0;
                        if (double.IsInfinity(center)) continue;
                        foreach (var x in Gaussians(10000))
                        {
                            double mx = x.GetMean();
                            if (double.IsInfinity(mx)) continue;
                            Gaussian toX = IsBetweenGaussianOp.XAverageConditional(isBetween, x, lowerBound, upperBound);
                            //Gaussian x2 = Gaussian.FromMeanAndPrecision(2 * center - mx, x.Precision);
                            Gaussian x2 = Gaussian.FromNatural(-x.MeanTimesPrecision, x.Precision);
                            Gaussian toX2 = IsBetweenGaussianOp.XAverageConditional(isBetween, x2, lowerBound, upperBound);
                            double precUlpDiff = UlpDiff(toX2.Precision, toX.Precision);
                            Assert.Equal(0, precUlpDiff);
                            if (precUlpDiff > precMaxUlpError)
                            {
                                precMaxUlpError = precUlpDiff;
                                precMaxUlpErrorLowerBound = lowerBound;
                                precMaxUlpErrorUpperBound = upperBound;
                                precMaxUlpErrorIsBetween = isBetween;
                            }
                            Gaussian xPost = toX.IsPointMass ? toX : toX * x;
                            Gaussian xPost2 = toX2.IsPointMass ? toX2 : toX2 * x2;
                            double mean = xPost.GetMean();
                            double mean2 = 2 * center - xPost2.GetMean();
                            double meanUlpDiff = UlpDiff(mean, mean2);
                            Assert.Equal(0, meanUlpDiff);
                            if (meanUlpDiff > meanMaxUlpError)
                            {
                                meanMaxUlpError = meanUlpDiff;
                                meanMaxUlpErrorLowerBound = lowerBound;
                                meanMaxUlpErrorUpperBound = upperBound;
                                meanMaxUlpErrorIsBetween = isBetween;
                            }
                        }
                    }
                });
            }
            ////Console.WriteLine($"meanMaxUlpError = {meanMaxUlpError}, lowerBound = {meanMaxUlpErrorLowerBound:g17}, upperBound = {meanMaxUlpErrorUpperBound:g17}, isBetween = {meanMaxUlpErrorIsBetween}");
            ////Console.WriteLine($"precMaxUlpError = {precMaxUlpError}, lowerBound = {precMaxUlpErrorLowerBound:g17}, upperBound = {precMaxUlpErrorUpperBound:g17}, isBetween = {precMaxUlpErrorIsBetween}");
            Assert.True(meanMaxUlpError == 0);
            Assert.True(precMaxUlpError == 0);
        }

        /// <summary>
        /// Used to debug LogProbBetween
        /// </summary>
        internal void LogProbBetweenCRCC_Experiments()
        {
            double lowerBound = -0.010000000000000002;
            double upperBound = -0.01;
            Gaussian x = Gaussian.FromNatural(-1E+34, 1E+36);
            for (int i = 0; i < 1000; i++)
            {
                double logProb = IsBetweenGaussianOp.LogProbBetween(x, lowerBound, upperBound);
                Console.WriteLine($"{x.Precision:g17} {logProb:g17}");
                x = Gaussian.FromMeanAndPrecision(x.GetMean(), x.Precision + 1000000000000 * MMath.Ulp(x.Precision));
            }
        }

        [Fact]
        public void LogProbBetweenCRCC_IsMonotonicInXPrecision()
        {
            double maxUlpError = 0;
            double maxUlpErrorLowerBound = 0;
            double maxUlpErrorUpperBound = 0;
            // maxUlpError = 22906784576, lowerBound = -0.010000000000000002, upperBound = -0.01
            bool trace = false;
            foreach (double lowerBound in new double[] { 0 }.Concat(Doubles()).Take(1))
            {
                foreach (double upperBound in new double[] { 1 }.Concat(UpperBounds(lowerBound)).Take(1))
                {
                    if (trace) Trace.WriteLine($"lowerBound = {lowerBound:g17}, upperBound = {upperBound:g17}");
                    Parallel.ForEach(Gaussians(100000), x =>
                    {
                        if (x.IsPointMass) return;
                        double mx = x.GetMean();
                        bool isBetween = Factor.IsBetween(mx, lowerBound, upperBound);
                        if (!isBetween)
                        {
                            // If mx is too close, LogProbBetween is not monotonic.
                            double distance;
                            if (mx < lowerBound) distance = System.Math.Abs(mx - lowerBound);
                            else distance = System.Math.Abs(mx - upperBound);
                            if (distance < 1 / System.Math.Sqrt(x.Precision)) return;
                        }
                        double logProbBetween = IsBetweenGaussianOp.LogProbBetween(x, lowerBound, upperBound);
                        foreach (var precisionDelta in DoublesGreaterThanZero())
                        {
                            Gaussian x2 = Gaussian.FromMeanAndPrecision(mx, x.Precision + precisionDelta);
                            if (x2.Equals(x)) continue;
                            if (x2.GetMean() != mx) continue;
                            double logProbBetween2 = IsBetweenGaussianOp.LogProbBetween(x2, lowerBound, upperBound);
                            if ((isBetween && logProbBetween2 < logProbBetween) ||
                                (!isBetween && logProbBetween2 > logProbBetween))
                            {
                                double ulpDiff = UlpDiff(logProbBetween2, logProbBetween);
                                if (ulpDiff > maxUlpError)
                                {
                                    maxUlpError = ulpDiff;
                                    maxUlpErrorLowerBound = lowerBound;
                                    maxUlpErrorUpperBound = upperBound;
                                    Assert.True(maxUlpError < 1e10);
                                }
                            }
                        }
                    });
                    if (trace) Trace.WriteLine($"maxUlpError = {maxUlpError}, lowerBound = {maxUlpErrorLowerBound:g17}, upperBound = {maxUlpErrorUpperBound:g17}");
                }
            }
            Assert.True(maxUlpError < 1e3);
        }

        [Fact]
        public void GaussianIsBetweenCRCC_IsMonotonicInXPrecision()
        {
            double meanMaxUlpError = 0;
            double meanMaxUlpErrorLowerBound = 0;
            double meanMaxUlpErrorUpperBound = 0;
            double precMaxUlpError = 0;
            double precMaxUlpErrorLowerBound = 0;
            double precMaxUlpErrorUpperBound = 0;
            Bernoulli isBetween = new Bernoulli(1.0);
            bool trace = false;
            foreach (double lowerBound in new[] { -10000.0 }.Concat(Doubles()).Take(1))
            {
                foreach (double upperBound in new[] { -9999.9999999999982 }.Concat(UpperBounds(lowerBound)).Take(1))
                {
                    if (trace) Trace.WriteLine($"lowerBound = {lowerBound:g17}, upperBound = {upperBound:g17}");
                    Parallel.ForEach(Gaussians(100000).Where(g => !g.IsPointMass), x =>
                    {
                        double mx = x.GetMean();
                        Gaussian toX = IsBetweenGaussianOp.XAverageConditional(isBetween, x, lowerBound, upperBound);
                        Gaussian xPost;
                        double meanError;
                        if (toX.IsPointMass)
                        {
                            xPost = toX;
                            meanError = 0;
                        }
                        else
                        {
                            xPost = toX * x;
                            meanError = GetProductMeanError(toX, x);
                        }
                        double mean = xPost.GetMean();
                        double variance = xPost.GetVariance();
                        double precError = MMath.Ulp(xPost.Precision);
                        double varianceError = 2 * precError * variance * variance;
                        foreach (var precisionDelta in DoublesGreaterThanZero())
                        {
                            Gaussian x2 = Gaussian.FromMeanAndPrecision(mx, x.Precision + precisionDelta);
                            if (x2.Equals(x)) continue;
                            if (x2.GetMean() != mx) continue;
                            Gaussian toX2 = IsBetweenGaussianOp.XAverageConditional(isBetween, x2, lowerBound, upperBound);
                            Gaussian xPost2;
                            double meanError2;
                            if (toX2.IsPointMass)
                            {
                                xPost2 = toX2;
                                meanError2 = 0;
                            }
                            else
                            {
                                xPost2 = toX2 * x2;
                                meanError2 = GetProductMeanError(toX2, x2);
                            }
                            double mean2 = xPost2.GetMean();
                            double meanUlpDiff = 0;
                            if (mean > mx)
                            {
                                // Since mx < mean, increasing the prior precision should decrease the posterior mean.
                                if (mean2 > mean)
                                {
                                    meanUlpDiff = (mean2 - mean) / System.Math.Max(meanError, meanError2);
                                }
                            }
                            else
                            {
                                if (mean2 < mean)
                                {
                                    meanUlpDiff = (mean - mean2) / System.Math.Max(meanError, meanError2);
                                }
                            }
                            if (meanUlpDiff > meanMaxUlpError)
                            {
                                meanMaxUlpError = meanUlpDiff;
                                meanMaxUlpErrorLowerBound = lowerBound;
                                meanMaxUlpErrorUpperBound = upperBound;
                                Assert.True(meanUlpDiff < 1e16);
                            }
                            double variance2 = xPost2.GetVariance();
                            double precError2 = MMath.Ulp(xPost2.Precision);
                            // 1 / (xPost.Precision - precError) - 1 / (xPost.Precision + precError) =approx 2*precError*variance*variance
                            double varianceError2 = 2 * precError2 * variance2 * variance2;
                            // Increasing prior precision should increase posterior precision.
                            if (xPost2.Precision < xPost.Precision)
                            {
                                double ulpDiff = UlpDiff(xPost2.Precision, xPost.Precision);
                                if (ulpDiff > precMaxUlpError)
                                {
                                    precMaxUlpError = ulpDiff;
                                    precMaxUlpErrorLowerBound = lowerBound;
                                    precMaxUlpErrorUpperBound = upperBound;
                                    Assert.True(precMaxUlpError < 1e16);
                                }
                            }
                        }
                    });
                }
                if (trace)
                {
                    Trace.WriteLine($"meanMaxUlpError = {meanMaxUlpError}, lowerBound = {meanMaxUlpErrorLowerBound:g17}, upperBound = {meanMaxUlpErrorUpperBound:g17}");
                    Trace.WriteLine($"precMaxUlpError = {precMaxUlpError}, lowerBound = {precMaxUlpErrorLowerBound:g17}, upperBound = {precMaxUlpErrorUpperBound:g17}");
                }
            }
            // meanMaxUlpError = 4271.53318407361, lowerBound = -1.0000000000000006E-12, upperBound = inf
            // precMaxUlpError = 5008, lowerBound = 1E+40, upperBound = 1.00000001E+40
            Assert.True(meanMaxUlpError < 3);
            Assert.True(precMaxUlpError < 1e16);
        }

        [Fact]
        public void GaussianIsBetweenCRCC_IsMonotonicInXMean()
        {
            double meanMaxUlpError = 0;
            double meanMaxUlpErrorLowerBound = 0;
            double meanMaxUlpErrorUpperBound = 0;
            double precMaxUlpError = 0;
            double precMaxUlpErrorLowerBound = 0;
            double precMaxUlpErrorUpperBound = 0;
            Bernoulli isBetween = new Bernoulli(1.0);
            bool trace = false;
            foreach (double lowerBound in new[] { -1000.0 }.Concat(Doubles()).Take(1))
            {
                foreach (double upperBound in new[] { 0.0 }.Concat(UpperBounds(lowerBound)).Take(1))
                {
                    if (trace) Console.WriteLine($"lowerBound = {lowerBound:g17}, upperBound = {upperBound:g17}");
                    double center = (lowerBound + upperBound) / 2;
                    if (double.IsNegativeInfinity(lowerBound) && double.IsPositiveInfinity(upperBound))
                        center = 0;
                    Parallel.ForEach(Gaussians(100000), x =>
                    {
                        double mx = x.GetMean();
                        Gaussian toX = IsBetweenGaussianOp.XAverageConditional(isBetween, x, lowerBound, upperBound);
                        Gaussian xPost;
                        double meanError;
                        if (toX.IsPointMass)
                        {
                            xPost = toX;
                            meanError = 0;
                        }
                        else
                        {
                            xPost = toX * x;
                            meanError = GetProductMeanError(toX, x);
                        }
                        double mean = xPost.GetMean();
                        foreach (var meanDelta in DoublesGreaterThanZero())
                        {
                            double mx2 = mx + meanDelta;
                            if (double.IsPositiveInfinity(meanDelta)) mx2 = meanDelta;
                            if (mx2 == mx) continue;
                            Gaussian x2 = Gaussian.FromMeanAndPrecision(mx2, x.Precision);
                            Gaussian toX2 = IsBetweenGaussianOp.XAverageConditional(isBetween, x2, lowerBound, upperBound);
                            Gaussian xPost2;
                            double meanError2;
                            if (toX2.IsPointMass)
                            {
                                xPost2 = toX2;
                                meanError2 = 0;
                            }
                            else
                            {
                                xPost2 = toX2 * x2;
                                meanError2 = GetProductMeanError(toX2, x2);
                            }
                            double mean2 = xPost2.GetMean();
                            // Increasing the prior mean should increase the posterior mean.
                            if (mean2 < mean)
                            {
                                meanError = MMath.Ulp(mean);
                                meanError2 = MMath.Ulp(mean2);
                                double meanUlpDiff = (mean - mean2) / System.Math.Max(meanError, meanError2);
                                if (meanUlpDiff > meanMaxUlpError)
                                {
                                    meanMaxUlpError = meanUlpDiff;
                                    meanMaxUlpErrorLowerBound = lowerBound;
                                    meanMaxUlpErrorUpperBound = upperBound;
                                    Assert.True(meanUlpDiff < 1e16);
                                }
                            }
                            // When mx > center, increasing prior mean should increase posterior precision.
                            if (mx > center && xPost2.Precision < xPost.Precision)
                            {
                                double ulpDiff = UlpDiff(xPost2.Precision, xPost.Precision);
                                if (ulpDiff > precMaxUlpError)
                                {
                                    precMaxUlpError = ulpDiff;
                                    precMaxUlpErrorLowerBound = lowerBound;
                                    precMaxUlpErrorUpperBound = upperBound;
                                    Assert.True(precMaxUlpError < 1e11);
                                }
                            }
                        }
                    });
                }
                if (trace)
                {
                    Console.WriteLine($"meanMaxUlpError = {meanMaxUlpError}, lowerBound = {meanMaxUlpErrorLowerBound:g17}, upperBound = {meanMaxUlpErrorUpperBound:g17}");
                    Console.WriteLine($"precMaxUlpError = {precMaxUlpError}, lowerBound = {precMaxUlpErrorLowerBound:g17}, upperBound = {precMaxUlpErrorUpperBound:g17}");
                }
            }
            // meanMaxUlpError = 104.001435643838, lowerBound = -1.0000000000000022E-37, upperBound = 9.9000000000000191E-36
            // precMaxUlpError = 4960, lowerBound = -1.0000000000000026E-47, upperBound = -9.9999999000000263E-48
            Assert.True(meanMaxUlpError < 1e16);
            Assert.True(precMaxUlpError < 1e11);
        }

        [Fact]
        public void GaussianIsBetweenCRCC_IsMonotonicInUpperBound()
        {
            // Test the symmetric version of a corner case that is tested below.
            IsBetweenGaussianOp.XAverageConditional(true, Gaussian.FromNatural(-1.7976931348623157E+308, 4.94065645841247E-324), double.NegativeInfinity, 0);
            double meanMaxUlpError = 0;
            double meanMaxUlpErrorLowerBound = 0;
            double meanMaxUlpErrorUpperBound = 0;
            double precMaxUlpError = 0;
            double precMaxUlpErrorLowerBound = 0;
            double precMaxUlpErrorUpperBound = 0;
            bool trace = false;
            foreach (double lowerBound in new[] { 0.0 }.Concat(Doubles()).Take(1))
            {
                foreach (double upperBound in new[] { 1.0 }.Concat(UpperBounds(lowerBound)).Take(1))
                {
                    if (trace) Console.WriteLine($"lowerBound = {lowerBound:g17}, upperBound = {upperBound:g17}");
                    Parallel.ForEach(Gaussians(100000), x =>
                    {
                        Gaussian toX = IsBetweenGaussianOp.XAverageConditional(true, x, lowerBound, upperBound);
                        Gaussian xPost;
                        double meanError;
                        if (toX.IsPointMass)
                        {
                            xPost = toX;
                            meanError = 0;
                        }
                        else
                        {
                            xPost = toX * x;
                            meanError = GetProductMeanError(toX, x);
                        }
                        double mean = xPost.GetMean();
                        foreach (double delta in DoublesGreaterThanZero())
                        {
                            double upperBound2 = upperBound + delta;
                            if (delta > double.MaxValue) upperBound2 = delta;
                            if (upperBound2 == upperBound) continue;
                            Gaussian toX2 = IsBetweenGaussianOp.XAverageConditional(true, x, lowerBound, upperBound2);
                            Gaussian xPost2;
                            double meanError2;
                            if (toX2.IsPointMass)
                            {
                                xPost2 = toX2;
                                meanError2 = 0;
                            }
                            else
                            {
                                xPost2 = toX2 * x;
                                meanError2 = GetProductMeanError(toX2, x);
                            }
                            if (toX2.IsUniform()) meanError2 = 0;
                            double mean2 = xPost2.GetMean();
                            // When adding a new point x to a population, the mean increases iff x is greater than the old mean.
                            // Increasing the upper bound adds new points that are larger than all existing points, so it must increase the mean.
                            if (mean > mean2)
                            {
                                double ulpDiff = (mean - mean2) / System.Math.Max(meanError, meanError2);
                                Assert.True(ulpDiff < 1e15);
                                if (ulpDiff > meanMaxUlpError)
                                {
                                    meanMaxUlpError = ulpDiff;
                                    meanMaxUlpErrorLowerBound = lowerBound;
                                    meanMaxUlpErrorUpperBound = upperBound;
                                }
                            }
                            if (toX2.Precision > toX.Precision)
                            {
                                double ulpDiff = UlpDiff(toX.Precision, toX2.Precision);
                                Assert.True(ulpDiff < 1e15);
                                if (ulpDiff > precMaxUlpError)
                                {
                                    precMaxUlpError = ulpDiff;
                                    precMaxUlpErrorLowerBound = lowerBound;
                                    precMaxUlpErrorUpperBound = upperBound;
                                }
                            }
                        }
                    });
                }
                if (trace)
                {
                    Console.WriteLine($"meanMaxUlpError = {meanMaxUlpError}, lowerBound = {meanMaxUlpErrorLowerBound:g17}, upperBound = {meanMaxUlpErrorUpperBound:g17}");
                    Console.WriteLine($"precMaxUlpError = {precMaxUlpError}, lowerBound = {precMaxUlpErrorLowerBound:g17}, upperBound = {precMaxUlpErrorUpperBound:g17}");
                }
            }
            // meanMaxUlpError = 33584, lowerBound = -1E+30, upperBound = 9.9E+31
            // precMaxUlpError = 256, lowerBound = -1, upperBound = 0
            Assert.True(meanMaxUlpError < 1e2);
            Assert.True(precMaxUlpError < 1e2);
        }

        [Fact]
        public void NormalCdfRatioSqrMinusDerivative_IsIncreasing()
        {
            double maxUlpError = 0;
            foreach (var z in Doubles())
            {
                if (z > 20) continue;
                double vp = NormalCdfRatioSqrMinusDerivativeRatio(z);
                //Console.WriteLine($"z = {z}: vp = {vp}");
                foreach (var zDelta in DoublesGreaterThanZero())
                {
                    double zU2 = z - zDelta;
                    double vp2 = NormalCdfRatioSqrMinusDerivativeRatio(zU2);
                    if (vp2 > vp)
                    {
                        double ulpDiff = UlpDiff(vp2, vp);
                        if (ulpDiff > maxUlpError)
                        {
                            maxUlpError = ulpDiff;
                        }
                    }
                }
            }
            //Console.WriteLine($"maxUlpError = {maxUlpError}");
            Assert.True(maxUlpError <= 3);
        }

        [Fact]
        public void NormalCdfRatioSqrMinusDerivative_EqualsExact()
        {
            Assert.Equal(0.0855952047653234157436344334434, NormalCdfRatioSqrMinusDerivative(-1), 1e-15);
            Assert.Equal(0.0000000000009609728081846573866852515941776879, NormalCdfRatioSqrMinusDerivative(-1009.9999999999991), 1e-27);
        }

        private static double NormalCdfRatioSqrMinusDerivative(double z)
        {
            double r = MMath.NormalCdfRatio(z);
            double r1 = MMath.NormalCdfMomentRatio(1, z);
            double r3 = MMath.NormalCdfMomentRatio(3, z) * 6;
            double vp = IsBetweenGaussianOp.NormalCdfRatioSqrMinusDerivative(z, r, r1, r3);
            return vp;
        }

        private static double NormalCdfRatioSqrMinusDerivativeRatio(double z)
        {
            double vp = NormalCdfRatioSqrMinusDerivative(z);
            if (vp != 0)
            {
                double r = MMath.NormalCdfRatio(z);
                vp /= (r * r);
            }
            return vp;
        }

        public static double GetPrecisionError(Gaussian gaussian)
        {
            if (double.IsInfinity(gaussian.Precision)) return 0;
            else return MMath.Ulp(gaussian.Precision);
        }

        public static double GetMeanError(Gaussian gaussian)
        {
            if (gaussian.IsPointMass) return 0;
            else if (double.IsInfinity(gaussian.MeanTimesPrecision)) return 0;
            else return MMath.Ulp(gaussian.MeanTimesPrecision) / (gaussian.Precision - GetPrecisionError(gaussian));
        }

        public static double GetSumError(double a, double b)
        {
            return MMath.Ulp(a) + MMath.Ulp(b);
        }

        public static double GetProductMeanError(Gaussian a, Gaussian b)
        {
            if (a.IsPointMass || b.IsPointMass) return 0;
            return GetSumError(a.MeanTimesPrecision, b.MeanTimesPrecision) /
                (a.Precision + b.Precision - GetSumError(a.Precision, b.Precision));
        }

        [Fact]
        public void GaussianIsBetween_CRRR_UncertainXTest()
        {
            Gaussian lowerBound = Gaussian.FromNatural(17.111433288915187, 0.66938434508155731);
            Gaussian upperBound = Gaussian.FromNatural(7.7694959349146462, 0.49485730932861044);
            lowerBound = Gaussian.FromNatural(-54.625321469620474, 24.217302563266891);
            upperBound = Gaussian.FromNatural(54.406085493265941, 24.077474982519977);
            double previousLogProb = double.PositiveInfinity;
            for (int i = 5; i < 200; i++)
            {
                Gaussian X = Gaussian.FromNatural(2.2204460492503131E-16, System.Math.Pow(0.1, i));
                X = Gaussian.FromMeanAndPrecision(-60, System.Math.Pow(0.1, i));
                double logProb = IsBetweenGaussianOp.LogProbBetween(X, lowerBound, upperBound);
                Trace.WriteLine($"X={X}: logProb={logProb}");
                Assert.True(!double.IsNaN(logProb));
                Assert.True(!double.IsInfinity(logProb));
                Assert.True(logProb < previousLogProb);
                previousLogProb = logProb;
            }
        }

        // Test that the operator behaves correctly for arguments with small variance
        [Fact]
        public void GaussianIsBetween_PointX_Test()
        {
            foreach (Gaussian lowerBound in new[] { Gaussian.PointMass(1), Gaussian.FromMeanAndVariance(1, 1) })
            {
                foreach (Gaussian upperBound in new[] { Gaussian.PointMass(3), Gaussian.FromMeanAndVariance(3, 1) })
                {
                    GaussianIsBetween_PointX(lowerBound, upperBound);
                }
            }
        }

        internal void GaussianIsBetween_PointX(Gaussian lowerBound, Gaussian upperBound)
        {
            Gaussian x = Gaussian.PointMass(-1000);
            Gaussian Xpost = new Gaussian();
            Bernoulli isBetween = Bernoulli.PointMass(true);
            Gaussian toXExpected = IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
            Gaussian toLowerExpected = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
            Gaussian toUpperExpected = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
            if (double.IsNaN(toXExpected.Precision)) throw new Exception();
            ////Console.WriteLine($"expected toX={toXExpected} toLower={toLowerExpected} toUpper={toUpperExpected}");
            Gaussian previousXpost = new Gaussian();
            Gaussian previousToLower = new Gaussian();
            Gaussian previousToUpper = new Gaussian();
            for (int i = -10; i < 30; i++)
            {
                x.SetMeanAndVariance(1, System.Math.Pow(0.1, i));
                Gaussian toX = IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                Gaussian toLower = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                Gaussian toUpper = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                Assert.True(toX.Precision >= 0);
                Xpost.SetToProduct(x, toX);
                ////Console.WriteLine($"{x}: {toX} {Xpost} toLower={toLower} toUpper={toUpper}");
                if (i > 0)
                {
                    Assert.True(Xpost.GetVariance() < previousXpost.GetVariance());
                    Assert.True(Xpost.GetMean() <= previousXpost.GetMean() + 1e-20);
                }
                previousXpost = Xpost;
                if (i > 0)
                {
                    Assert.True(toLower.GetMean() <= previousToLower.GetMean() || toLower.GetVariance() <= previousToLower.GetVariance());
                }
                previousToLower = toLower;
                if (i > 0)
                {
                    Assert.True(toUpper.GetVariance() >= previousToUpper.GetVariance() - 1e-20);
                    Assert.True(toUpper.IsUniform() || toUpper.GetMean() <= previousToUpper.GetMean());
                }
                previousToUpper = toUpper;

                // check making diffs smaller

                if (lowerBound.Precision == upperBound.Precision && lowerBound.MeanTimesPrecision == -upperBound.MeanTimesPrecision)
                {
                    // check the flipped case
                    x.SetMeanAndVariance(-x.GetMean(), System.Math.Pow(0.1, i));
                    Gaussian toX2 = IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                    Gaussian toLower2 = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                    Gaussian toUpper2 = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound);
                    Assert.True(toX.Precision == toX2.Precision);
                    Assert.True(toX.MeanTimesPrecision == -toX2.MeanTimesPrecision);
                    Assert.True(toLower2.Precision == toUpper.Precision);
                    Assert.True(toLower2.MeanTimesPrecision == -toUpper.MeanTimesPrecision);
                    Assert.True(toUpper2.Precision == toLower.Precision);
                    Assert.True(toUpper2.MeanTimesPrecision == -toLower.MeanTimesPrecision);
                }
            }
        }

        /// <summary>
        /// Test that the operator behaves correctly for contexts far outside of the bounds
        /// </summary>
        [Fact]
        public void GaussianIsBetween_EqualsExact()
        {
            /* Anaconda Python script to generate a true value (must not be indented):
from mpmath import *
mp.dps = 100; mp.pretty = True
prec=mpf('1.00001E+36'); tau = mpf('1.0000099999999998E+34'); mx = tau/prec; l=mpf('0.010000000000000002'); u=mpf('0.010000000000000004'); p0=mpf('0');
qu = (u-mx)*sqrt(prec);  ql = (l-mx)*sqrt(prec);
Z = (1-2*p0)*(ncdf(qu) - ncdf(ql))+p0;
#Z = ncdf(-ql) - ncdf(-qu);
alphaU = (1-2*p0)*npdf(qu)/Z*sqrt(prec);
alphaL = (1-2*p0)*npdf(ql)/Z*sqrt(prec);
alphaX = alphaL - alphaU;
betaX = alphaX*alphaX + qu*alphaU*sqrt(prec) - ql*alphaL*sqrt(prec);
mx + alphaX/prec
1/prec - betaX/prec/prec
weight = betaX / (prec - betaX);
prec * weight
weight * (tau + alphaX) + alphaX
            */
            //Assert.Equal(DoubleIsBetweenOp.LogProbBetween(Gaussian.FromNatural(1.0000099999999998E+34, 1.00001E+36), 0.010000000000000002, 0.010000000000000004), -10.360132636204013435798441, 1e-3);
            Gaussian expected;
            Gaussian result2;
            result2 = IsBetweenGaussianOp.XAverageConditional(true, Gaussian.FromNatural(100, 0.6), -0.0025, 0.0025);
            expected = Gaussian.FromNatural(0.83382446386897639746, 486015.0838217901083);
            Assert.True(MaxUlpDiff(expected, result2) < 1e5);
            result2 = IsBetweenGaussianOp.XAverageConditional(true, Gaussian.FromNatural(1000, 0.6), -0.0025, 0.0025);
            // exact posterior mean = 0.00153391785173542665
            // exact posterior variance = 0.0000008292583427621911323272374
            expected = Gaussian.FromNatural(849.7466623321181468, 1205896.221814396816657);
            Assert.True(MaxUlpDiff(expected, result2) < 1e6);
            result2 = IsBetweenGaussianOp.XAverageConditional(true, Gaussian.FromNatural(10000, 0.6), -0.0025, 0.0025);
            expected = Gaussian.FromNatural(229999.9352600054688757876, 99999973.0000021996482);
            Assert.True(MaxUlpDiff(expected, result2) < 1e2);
            Gaussian result = IsBetweenGaussianOp.XAverageConditional_Slow(Bernoulli.PointMass(true), new Gaussian(-48.26, 1.537), new Gaussian(-12.22, 0.4529), new Gaussian(-17.54, 0.3086));
            result2 = IsBetweenGaussianOp.XAverageConditional(true, Gaussian.FromNatural(1, 1e-19), -10, 0);
            // exact posterior mean = -0.99954598008990312211948
            // exact posterior variance = 0.99545959476495245821845
            expected = Gaussian.FromNatural(-2.00410502379648622469036, 1.0045611145433980376101655346945);
            Assert.True(MaxUlpDiff(expected, result2) < 2);
            result2 = IsBetweenGaussianOp.XAverageConditional(true, Gaussian.FromNatural(1, 1e19), -10, 0);
            // exact posterior mean = -0.00000000025231325216567798206492
            // exact posterior variance = 0.00000000000000000003633802275634766987678763433333
            expected = Gaussian.FromNatural(-6943505261.522269414985891, 17519383944062174805.8794215);
            Assert.True(MaxUlpDiff(expected, result2) <= 7);
        }

        /// <summary>
        /// Fails because betaL is wrong because logPhiR is wrong.  TODO: Use mpmath to determine which variable has the wrong value.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void GaussianIsBetweenCRRR_IsMonotonicInX()
        {
            Gaussian previousToX = new Gaussian();
            Gaussian previousToLowerBound = new Gaussian();
            Gaussian previousToUpperBound = new Gaussian();
            Gaussian lowerBound = Gaussian.FromNatural(-200, 100);
            Gaussian upperBound = Gaussian.FromNatural(255, 147);
            Bernoulli isBetween = Bernoulli.PointMass(true);
            double xMeanMaxUlpError = 0;
            double xPrecisionMaxUlpError = 0;
            double uMeanMaxUlpError = 0;
            double uPrecisionMaxUlpError = 0;
            double lMeanMaxUlpError = 0;
            for (int i = 10; i < 3000; i++)
            {
                Gaussian X = Gaussian.FromNatural(1.2, System.Math.Pow(10, -i * 0.1));
                var toX = IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, X, lowerBound, upperBound);
                var toLowerBound = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, X, lowerBound, upperBound);
                var toUpperBound = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, X, lowerBound, upperBound);
                Trace.WriteLine($"{X}: {toX} {toLowerBound} {toUpperBound}");
                if (!previousToX.IsUniform())
                {
                    if (previousToX.GetMean() > toX.GetMean())
                    {
                        xMeanMaxUlpError = System.Math.Max(xMeanMaxUlpError, UlpDiff(previousToX.GetMean(), toX.GetMean()));
                        //Assert.True(xMeanMaxUlpError < 1e13);
                    }
                    if (previousToX.Precision > toX.Precision)
                    {
                        xPrecisionMaxUlpError = System.Math.Max(xPrecisionMaxUlpError, UlpDiff(previousToX.Precision, toX.Precision));
                        //Assert.True(xPrecisionMaxUlpError <= 0);
                    }
                    if (previousToLowerBound.GetMean() < toLowerBound.GetMean())
                    {
                        lMeanMaxUlpError = System.Math.Max(lMeanMaxUlpError, UlpDiff(previousToLowerBound.GetMean(), toLowerBound.GetMean()));
                    }
                    //Assert.True(previousToLowerBound.GetVariance() <= toLowerBound.GetVariance());
                    if (previousToUpperBound.GetMean() > toUpperBound.GetMean())
                    {
                        uMeanMaxUlpError = System.Math.Max(uMeanMaxUlpError, UlpDiff(previousToUpperBound.GetMean(), toUpperBound.GetMean()));
                    }
                    if (previousToUpperBound.Precision < toUpperBound.Precision)
                    {
                        uPrecisionMaxUlpError = System.Math.Max(uPrecisionMaxUlpError, UlpDiff(previousToUpperBound.Precision, toUpperBound.Precision));
                    }
                }
                previousToX = toX;
                previousToLowerBound = toLowerBound;
                previousToUpperBound = toUpperBound;
            }
            Trace.WriteLine($"xMeanMaxUlpError = {xMeanMaxUlpError}, xPrecisionMaxUlpError = {xPrecisionMaxUlpError}, uMeanMaxUlpError = {uMeanMaxUlpError}, uPrecisionMaxUlpError = {uPrecisionMaxUlpError}, lMeanMaxUlpError = {lMeanMaxUlpError}");
            // TODO: tighten these thresholds
            Assert.True(xMeanMaxUlpError < 1e15);
            Assert.True(xPrecisionMaxUlpError < 1e15);
            Assert.True(uPrecisionMaxUlpError < 1);
            Assert.True(uMeanMaxUlpError < 1);
            Assert.True(lMeanMaxUlpError < 1);
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GaussianIsBetweenCRRR_IsMonotonicInX2()
        {
            Gaussian previousToX = new Gaussian();
            Gaussian previousToLowerBound = new Gaussian();
            Gaussian previousToUpperBound = new Gaussian();
            Gaussian upperBound = Gaussian.FromNatural(200, 100);
            Gaussian lowerBound = Gaussian.FromNatural(-255, 147);
            Bernoulli isBetween = Bernoulli.PointMass(true);
            IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, Gaussian.FromNatural(-1.2, 6.3095734448019427E-17), lowerBound, upperBound);
            double xMeanMaxUlpError = 0;
            double xPrecisionMaxUlpError = 0;
            double lMeanMaxUlpError = 0;
            double lPrecisionMaxUlpError = 0;
            double uMeanMaxUlpError = 0;
            for (int i = 10; i < 3000; i++)
            {
                Gaussian X = Gaussian.FromNatural(-1.2, System.Math.Pow(10, -i * 0.1));
                //Gaussian toXExpected = new Gaussian(-0.3047, 0.5397);
                //SolveAlphaBeta(X, toXExpected, out double alpha, out double beta);
                //Trace.WriteLine($"expected alpha = {alpha}, beta = {beta}");
                var toX = IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, X, lowerBound, upperBound);
                var toLowerBound = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, X, lowerBound, upperBound);
                var toUpperBound = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, X, lowerBound, upperBound);
                Trace.WriteLine($"{X}: {toX} {toLowerBound} {toUpperBound}");
                if (!previousToX.IsUniform())
                {
                    if (previousToX.GetMean() < toX.GetMean())
                    {
                        xMeanMaxUlpError = System.Math.Max(xMeanMaxUlpError, UlpDiff(previousToX.GetMean(), toX.GetMean()));
                        //Assert.True(xMeanMaxUlpError < 1e11);
                    }
                    if (previousToX.Precision > toX.Precision)
                    {
                        xPrecisionMaxUlpError = System.Math.Max(xPrecisionMaxUlpError, UlpDiff(previousToX.Precision, toX.Precision));
                        //Assert.True(xPrecisionMaxUlpError <= 0);
                    }
                    if (previousToLowerBound.GetMean() < toLowerBound.GetMean())
                    {
                        lMeanMaxUlpError = System.Math.Max(lMeanMaxUlpError, UlpDiff(previousToLowerBound.GetMean(), toLowerBound.GetMean()));
                    }
                    if (previousToLowerBound.Precision < toLowerBound.Precision)
                    {
                        lPrecisionMaxUlpError = System.Math.Max(lPrecisionMaxUlpError, UlpDiff(previousToLowerBound.Precision, toLowerBound.Precision));
                    }
                    if (previousToUpperBound.GetMean() > toUpperBound.GetMean())
                    {
                        uMeanMaxUlpError = System.Math.Max(uMeanMaxUlpError, UlpDiff(previousToUpperBound.GetMean(), toUpperBound.GetMean()));
                    }
                    //Assert.True(previousToUpperBound.GetVariance() <= toUpperBound.GetVariance());
                }
                previousToX = toX;
                previousToLowerBound = toLowerBound;
                previousToUpperBound = toUpperBound;
            }
            Trace.WriteLine($"xMeanMaxUlpError = {xMeanMaxUlpError}, xPrecisionMaxUlpError = {xPrecisionMaxUlpError}, lMeanMaxUlpError = {lMeanMaxUlpError}, lPrecisionMaxUlpError = {lPrecisionMaxUlpError}, uMeanMaxUlpError={uMeanMaxUlpError}");
            // TODO: tighten these thresholds
            Assert.True(xMeanMaxUlpError < 1e15);
            Assert.True(xPrecisionMaxUlpError < 1e15);
            Assert.True(lMeanMaxUlpError < 1);
            Assert.True(lPrecisionMaxUlpError < 1);
            Assert.True(uMeanMaxUlpError < 1);
        }

        private static void SolveAlphaBeta(Gaussian prior, Gaussian msg, out double alpha, out double beta)
        {
            if (prior.IsPointMass)
            {
                beta = msg.Precision;
                alpha = msg.MeanTimesPrecision - prior.Point * beta;
                return;
            }
            if (msg.IsPointMass)
            {
                beta = prior.Precision;
                alpha = (msg.Point - prior.GetMean()) * beta;
                return;
            }
            beta = 1 / (1 / msg.Precision + 1 / prior.Precision);
            double weight = beta / (prior.Precision - beta);
            // weight*tau + (weight + 1)*alpha = msg.MeanTimesPrecision
            alpha = (msg.MeanTimesPrecision - weight * prior.MeanTimesPrecision) / (1 + weight);
        }

        [Fact]
        public void GaussianIsBetweenCRRR_NegativeUpperBoundTest()
        {
            Gaussian X = Gaussian.FromNatural(813.982758311301, 1.0594806725507477);
            Gaussian previousToX = new Gaussian();
            Gaussian previousXpost = new Gaussian();
            double tolerance = 1e-10;
            for (int i = 8; i < 100; i++)
            {
                // seems like answer should always be Gaussian(m/v=-814, 1/v=0)
                Gaussian upperBound = Gaussian.FromNatural(-System.Math.Pow(10, i), 9);
                Gaussian toX = IsBetweenGaussianOp.XAverageConditional_Slow(Bernoulli.PointMass(true), X, Gaussian.PointMass(0), upperBound);
                Gaussian Xpost = X * toX;
                Trace.WriteLine($"{upperBound}: {toX} {toX.MeanTimesPrecision} {Xpost}");
                // lowerBound is point mass, so cannot be violated.
                Assert.True(Xpost.GetMean() >= 0 - tolerance);
                if (!previousToX.IsUniform())
                {
                    // upperBound is decreasing, so posterior mean should be decreasing.
                    Assert.True(Xpost.GetMean() <= previousXpost.GetMean() + tolerance);
                }
                previousToX = toX;
                previousXpost = Xpost;
            }
        }

        [Fact]
        public void GaussianIsBetweenCRRR_LowerBoundTest()
        {
            Gaussian X = Gaussian.FromNatural(898.71395259259464, 1.4308788553248037);
            Gaussian lowerBound = Gaussian.FromNatural(17028358.45574614, 9);
            Gaussian upperBound = Gaussian.FromNatural(412820.08287991461, 423722.55474045349);
            for (int i = -10; i <= 0; i++)
            {
                lowerBound = Gaussian.FromNatural(17028358.45574614 * System.Math.Pow(2, i), 9);
                Gaussian toLowerBound = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), X, lowerBound, upperBound);
                Trace.WriteLine($"{lowerBound}: {toLowerBound.MeanTimesPrecision} {toLowerBound.Precision}");
                Assert.False(toLowerBound.IsPointMass);
            }
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GaussianIsBetweenCRRR_SmallXPrecisionTest()
        {
            Gaussian lowerBound = Gaussian.FromNatural(-102.3311202057678, 91.572320438929935);
            Gaussian upperBound = Gaussian.FromNatural(102.27224205502382, 91.541070478258376);
            foreach (var mean in new[] { 1e7, -1e7 })
            {
                Gaussian toLowerBoundPrev = Gaussian.FromNatural(double.MaxValue, double.MaxValue);
                Gaussian toXPrev = Gaussian.FromNatural(double.MaxValue, double.MaxValue);
                double xMeanTimesPrecisionMaxUlpError = 0;
                double lowerBoundMeanTimesPrecisionMaxUlpError = 0;
                for (int i = 0; i < 200; i++)
                {
                    Gaussian X = Gaussian.FromMeanAndPrecision(mean, System.Math.Pow(2, -i * 1 - 20));
                    Gaussian toX = IsBetweenGaussianOp.XAverageConditional_Slow(Bernoulli.PointMass(true), X, lowerBound, upperBound);
                    Gaussian toLowerBound = toLowerBoundPrev;// DoubleIsBetweenOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), X, lowerBound, upperBound);
                    Trace.WriteLine($"{i} {X}: {toX.MeanTimesPrecision:g17} {toX.Precision:g17} {toLowerBound.MeanTimesPrecision:g17} {toLowerBound.Precision:g17}");
                    Assert.False(toLowerBound.IsPointMass);
                    if ((mean > 0 && toLowerBound.MeanTimesPrecision > toLowerBoundPrev.MeanTimesPrecision) ||
                        (mean < 0 && toLowerBound.MeanTimesPrecision < toLowerBoundPrev.MeanTimesPrecision))
                    {
                        lowerBoundMeanTimesPrecisionMaxUlpError = System.Math.Max(lowerBoundMeanTimesPrecisionMaxUlpError, UlpDiff(toLowerBound.MeanTimesPrecision, toLowerBoundPrev.MeanTimesPrecision));
                        //Assert.True(meanTimesPrecisionMaxUlpError < 5);
                    }
                    if ((mean > 0 && toX.MeanTimesPrecision > toXPrev.MeanTimesPrecision) ||
                        (mean < 0 && toX.MeanTimesPrecision < toXPrev.MeanTimesPrecision))
                    {
                        xMeanTimesPrecisionMaxUlpError = System.Math.Max(xMeanTimesPrecisionMaxUlpError, UlpDiff(toX.MeanTimesPrecision, toXPrev.MeanTimesPrecision));
                        //Assert.True(xMeanTimesPrecisionMaxUlpError < 1e12);
                    }
                    toLowerBoundPrev = toLowerBound;
                    toXPrev = toX;
                }
                Trace.WriteLine($"xMeanTimesPrecisionMaxUlpError = {xMeanTimesPrecisionMaxUlpError} lowerBoundMeanTimesPrecisionMaxUlpError = {lowerBoundMeanTimesPrecisionMaxUlpError}");
                Assert.True(xMeanTimesPrecisionMaxUlpError < 1e12);
                Assert.True(lowerBoundMeanTimesPrecisionMaxUlpError < 1e4);
            }
        }

        [Fact]
        public void IsBetweenGaussian_WithImprobableInputs_Throws()
        {
            Bernoulli isBetween = new Bernoulli(1);
            Gaussian X = Gaussian.FromMeanAndVariance(double.MinValue / 2, 1);
            Gaussian lowerBound = Gaussian.FromMeanAndVariance(double.MaxValue / 2, 1);
            Gaussian upperBound = Gaussian.FromNatural(0, 4);
            Assert.Throws<AllZeroException>(() =>
                IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, X, lowerBound, upperBound));
            Assert.Throws<AllZeroException>(() =>
                IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, X, lowerBound, upperBound));
        }

        [Fact]
        public void GaussianIsBetweenTest2()
        {
            Bernoulli isBetween = new Bernoulli(1);
            Assert.False(double.IsNaN(IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, Gaussian.FromNatural(83.257525699629014, 0.22985792223206492), Gaussian.PointMass(double.NegativeInfinity), Gaussian.FromNatural(0, 4)).MeanTimesPrecision));
            Assert.False(double.IsNaN(IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, Gaussian.FromNatural(0.9106071714590378, 5.9521837280027985E-11), Gaussian.FromNatural(-49.9894026120194, 107.30343404076896), Gaussian.FromNatural(49.051818445888259, 107.26846525506932)).MeanTimesPrecision));
            Assert.False(double.IsNaN(IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, Gaussian.FromNatural(2.2204460492503131E-16, 6.9388939039072284E-18), Gaussian.FromNatural(17.111433288915187, 0.66938434508155731), Gaussian.FromNatural(7.7694959349146462, 0.49485730932861044)).MeanTimesPrecision));
            Assert.False(IsBetweenGaussianOp.LowerBoundAverageConditional(isBetween, Gaussian.FromNatural(1253646736.336942, 0.193405698968261), Gaussian.PointMass(-1.0000002501122697E-08), Gaussian.PointMass(1.0000002501122697E-08), -4.0630398894972073E+18).IsPointMass);
            IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, Gaussian.FromNatural(0, 0.0038937777431664196), Gaussian.PointMass(double.NegativeInfinity), Gaussian.FromNatural(-1.6, 0.8));
            IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, Gaussian.FromNatural(1.156293233217532E-25, 6.162975822039154E-33), Gaussian.FromNatural(-102.3311202057678, 91.572320438929935), Gaussian.FromNatural(102.27224205502382, 91.541070478258376));
            Assert.False(double.IsNaN(IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, Gaussian.FromNatural(980.18122429721575, 1.409544490082087), Gaussian.FromNatural(17028174.685026139, 837.26675043005957), Gaussian.FromNatural(412820.4122154137, 423722.54499249317)).MeanTimesPrecision));
            Assert.False(double.IsNaN(IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, Gaussian.FromNatural(-5.3548456213550253E-41, 4.61370960061741E-81), Gaussian.FromNatural(-15848812.635800883, 13451.362337266379), Gaussian.FromNatural(-22204349.280881952, 389690.00236138358)).MeanTimesPrecision));
            Assert.False(double.IsNaN(IsBetweenGaussianOp.XAverageConditional(isBetween, new Gaussian(1, 2), double.PositiveInfinity, double.PositiveInfinity).MeanTimesPrecision));
            Gaussian x = new Gaussian(0, 1);
            Gaussian lowerBound = new Gaussian(1, 8);
            Gaussian upperBound = new Gaussian(3, 3);
            Assert.True(IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, Gaussian.PointMass(0), Gaussian.PointMass(-1), Gaussian.PointMass(1)).IsUniform());
            Assert.True(IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, Gaussian.PointMass(double.NegativeInfinity), upperBound).IsUniform());
            Assert.True(IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, Gaussian.PointMass(double.PositiveInfinity)).IsUniform());
            // test in matlab: x=randn(1e7,1); l=randnorm(1e7,1,[],8)'; u=randnorm(1e7,3,[],3)';
            // ok=(l<x & x<u); [mean(l(ok)) var(l(ok)); mean(u(ok)) var(u(ok)); mean(x(ok)) var(x(ok))]
            Gaussian LpostTrue = new Gaussian(-1.7784, 2.934);
            Gaussian UpostTrue = new Gaussian(3.2694, 2.3796);
            Gaussian XpostTrue = new Gaussian(0.25745, 0.8625);
            Gaussian Lpost = new Gaussian();
            Gaussian Upost = new Gaussian();
            Gaussian Xpost = new Gaussian();
            Lpost.SetToProduct(lowerBound, IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound));
            Assert.True(LpostTrue.MaxDiff(Lpost) < 1e-4);
            Upost.SetToProduct(upperBound, IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound));
            Assert.True(UpostTrue.MaxDiff(Upost) < 1e-4);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, x, lowerBound, upperBound));
            Assert.True(XpostTrue.MaxDiff(Xpost) < 1e-4);
            // special case for uniform X
            x = new Gaussian();
            LpostTrue.SetMeanAndVariance(-1.2741, 5.3392);
            UpostTrue.SetMeanAndVariance(3.8528, 2.6258);
            XpostTrue.SetMeanAndVariance(1.2894, 5.1780);
            Lpost.SetToProduct(lowerBound, IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound));
            Assert.True(LpostTrue.MaxDiff(Lpost) < 1e-4);
            Upost.SetToProduct(upperBound, IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(isBetween, x, lowerBound, upperBound));
            Assert.True(UpostTrue.MaxDiff(Upost) < 1e-4);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional_Slow(isBetween, x, lowerBound, upperBound));
            Assert.True(XpostTrue.MaxDiff(Xpost) < 1e-4);

            Upost = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(new Bernoulli(0), Gaussian.PointMass(1.0), Gaussian.PointMass(0.0), new Gaussian());
            Assert.True(new Gaussian(0.5, 1.0 / 12).MaxDiff(Upost) < 1e-4);
            Upost = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(new Bernoulli(0), new Gaussian(1.0, 0.5), Gaussian.PointMass(0.0), new Gaussian());
            UpostTrue = IsBetweenGaussianOp.XAverageConditional_Slow(Bernoulli.PointMass(true), new Gaussian(), Gaussian.PointMass(0.0), new Gaussian(1.0, 0.5));
            Assert.True(UpostTrue.MaxDiff(Upost) < 1e-4);
            Upost = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(new Bernoulli(0.1), Gaussian.PointMass(1.0), Gaussian.PointMass(0.0), new Gaussian());
            Assert.True(new Gaussian().MaxDiff(Upost) < 1e-10);
            Upost = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(new Bernoulli(1), Gaussian.PointMass(1.0), Gaussian.PointMass(0.0), new Gaussian());
            Assert.True(new Gaussian().MaxDiff(Upost) < 1e-10);

            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(new Bernoulli(0), Gaussian.PointMass(0.0), new Gaussian(), Gaussian.PointMass(1.0));
            Assert.True(new Gaussian(0.5, 1.0 / 12).MaxDiff(Lpost) < 1e-4);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(new Bernoulli(0), new Gaussian(0.0, 0.5), new Gaussian(), Gaussian.PointMass(1.0));
            LpostTrue = IsBetweenGaussianOp.XAverageConditional_Slow(Bernoulli.PointMass(true), new Gaussian(), new Gaussian(0.0, 0.5), Gaussian.PointMass(1.0));
            Assert.True(LpostTrue.MaxDiff(Lpost) < 1e-4);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(new Bernoulli(0.1), Gaussian.PointMass(0.0), new Gaussian(), Gaussian.PointMass(1.0));
            Assert.True(new Gaussian().MaxDiff(Lpost) < 1e-10);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(new Bernoulli(1), Gaussian.PointMass(0.0), new Gaussian(), Gaussian.PointMass(1.0));
            Assert.True(new Gaussian().MaxDiff(Lpost) < 1e-10);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional(Bernoulli.PointMass(true), new Gaussian(-2.839e+10, 39.75), Gaussian.PointMass(0.05692), Gaussian.PointMass(double.PositiveInfinity), -1.0138308906207461E+19);
            Assert.True(!double.IsNaN(Lpost.MeanTimesPrecision));

            /* test in matlab: 
               s = [0 0 0];
               for iter = 1:100
                 x=randn(1e7,1); l=1; u=3;
                 ok=(l<x & x<u); 
                 s = s + [sum(ok) sum(x(ok)) sum(x(ok).^2)];
               end
               m = s(2)/s(1); v = s(3)/s(1) - m*m; [m v]
             */
            x = new Gaussian(0, 1);
            Assert.True(IsBetweenGaussianOp.XAverageConditional_Slow(new Bernoulli(0.5), x, Gaussian.PointMass(Double.NegativeInfinity), Gaussian.Uniform()).IsUniform());
            Assert.True(IsBetweenGaussianOp.XAverageConditional_Slow(Bernoulli.PointMass(true), x, Gaussian.PointMass(Double.NegativeInfinity), Gaussian.Uniform()).IsUniform());
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(true, x, 1, 3));
            Assert.True(Xpost.MaxDiff(Gaussian.FromMeanAndVariance(1.5101, 0.17345)) < 1e-3);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(false, x, 1, 3));
            Assert.True(Xpost.MaxDiff(Gaussian.FromMeanAndVariance(-0.28189, 0.64921)) < 1e-3);
            Assert.True(IsPositiveOp.XAverageConditional(true, x).MaxDiff(
              IsBetweenGaussianOp.XAverageConditional(true, x, 0.0, Double.PositiveInfinity)) < 1e-8);
            Assert.True(IsPositiveOp.XAverageConditional(false, x).MaxDiff(
              IsBetweenGaussianOp.XAverageConditional(true, x, Double.NegativeInfinity, 0)) < 1e-8);
            /* test in matlab:
               z=0;s1=0;s2=0;
               for iter = 1:100
                 x=randn(1e7,1); l=1; u=3;
                 ok=(l<x & x<u); 
                 z=z + 0.6*sum(ok)+0.4*sum(~ok); 
                 s1=s1 + 0.6*sum(x(ok))+0.4*sum(x(~ok));
                 s2=s2 + 0.6*sum(x(ok).^2)+0.4*sum(x(~ok).^2);
               end
               m=s1/z; v=s2/z - m*m; [m v]
             */
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(new Bernoulli(0.6), x, 1, 3));
            if (IsBetweenGaussianOp.ForceProper)
                Assert.True(Xpost.MaxDiff(Gaussian.FromMeanAndVariance(0.1101, 1.0)) < 1e-3);
            else
                Assert.True(Xpost.MaxDiff(Gaussian.FromMeanAndVariance(0.11027, 1.0936)) < 1e-3);
            // special case for uniform X
            x = new Gaussian();
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(true, x, 1, 3));
            Assert.True(Xpost.MaxDiff(Gaussian.FromMeanAndVariance(2, 4.0 / 12)) < 1e-10);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(false, x, 1, 3));
            Assert.True(Xpost.MaxDiff(new Gaussian()) < 1e-10);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(new Bernoulli(0.6), x, 1, 3));
            Assert.True(Xpost.MaxDiff(new Gaussian()) < 1e-10);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(true, x, 1, Double.PositiveInfinity));
            Assert.True(Xpost.MaxDiff(new Gaussian()) < 1e-10);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(false, x, 1, Double.PositiveInfinity));
            Assert.True(Xpost.MaxDiff(new Gaussian()) < 1e-10);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(true, x, Double.NegativeInfinity, 1));
            Assert.True(Xpost.MaxDiff(new Gaussian()) < 1e-10);
            Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(false, x, Double.NegativeInfinity, 1));
            Assert.True(Xpost.MaxDiff(new Gaussian()) < 1e-10);

            x = new Gaussian(0, 1);
            double[] lowerBounds = { 2, 10, 100, 1000 };
            for (int i = 0; i < lowerBounds.Length; i++)
            {
                double m, v;
                Gaussian x2 = new Gaussian(-lowerBounds[i], 1);
                Xpost.SetToProduct(x2, IsPositiveOp.XAverageConditional(true, x2));
                //Console.WriteLine(Xpost);
                Xpost.GetMeanAndVariance(out m, out v);
                Xpost.SetToProduct(x, IsBetweenGaussianOp.XAverageConditional(true, x, lowerBounds[i], Double.PositiveInfinity));
                Assert.True(Xpost.MaxDiff(new Gaussian(m + lowerBounds[i], v)) < 1e-5 / v);
                //Xpost.SetToProduct(x, DoubleIsBetweenOp.XAverageConditional(true, x, new Gaussian(lowerBounds[i], 1e-80), Double.PositiveInfinity));
                //Assert.True(Xpost.MaxDiff(new Gaussian(m+lowerBounds[i], v)) < 1e-8);
            }


            Gaussian Lexpected = IsPositiveOp.XAverageConditional(false, lowerBound);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.FromMeanAndVariance(0, 1e-10), lowerBound, Gaussian.FromMeanAndVariance(-1, 1));
            Assert.True(Lpost.MaxDiff(Lexpected) < 1e-10);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.PointMass(0), lowerBound, Gaussian.FromMeanAndVariance(-1, 1));
            Assert.True(Lpost.MaxDiff(Lexpected) < 1e-10);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.PointMass(1), lowerBound, Gaussian.PointMass(0));
            Assert.True(Lpost.MaxDiff(Lexpected) < 1e-10);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.PointMass(0), lowerBound, Gaussian.PointMass(0));
            Assert.True(Lpost.MaxDiff(Lexpected) < 1e-10);
            Lpost = IsBetweenGaussianOp.LowerBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.PointMass(0), lowerBound, Gaussian.PointMass(1));
            Assert.True(Lpost.MaxDiff(Lexpected) < 1e-10);
            //Lpost = DoubleIsBetweenOp.LowerBoundAverageConditional(true,Gaussian.Uniform(),lowerBound,0);
            //Assert.True(Lpost.MaxDiff(IsPositiveOp.XAverageConditional(false,lowerBound)) < 1e-3);
            Gaussian Uexpected = IsPositiveOp.XAverageConditional(true, upperBound);
            Upost = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.PointMass(-1), Gaussian.PointMass(0), upperBound);
            Assert.True(Upost.MaxDiff(Uexpected) < 1e-10);
            Upost = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.PointMass(0), Gaussian.PointMass(0), upperBound);
            Assert.True(Upost.MaxDiff(Uexpected) < 1e-10);
            Upost = IsBetweenGaussianOp.UpperBoundAverageConditional_Slow(Bernoulli.PointMass(true), Gaussian.PointMass(0), Gaussian.PointMass(-1), upperBound);
            Assert.True(Upost.MaxDiff(Uexpected) < 1e-10);
            //Upost = DoubleIsBetweenOp.UpperBoundAverageConditional(true,Gaussian.Uniform(),0,upperBound);
            //Assert.True(Upost.MaxDiff(IsPositiveOp.XAverageConditional(true,upperBound)) < 1e-3);
        }


        /// <summary>
        /// Tests for the message operators for <see cref="Factor.CountTrue"/>.
        /// </summary>
        [Fact]
        public void CountTrueOpTest()
        {
            const double Tolerance = 1e-10;

            // Test forward message
            {
                var array = new[] { new Bernoulli(0.0), new Bernoulli(0.1), new Bernoulli(0.3), new Bernoulli(0.5), new Bernoulli(1.0) };
                double[,] forwardPassBuffer = CountTrueOp.PoissonBinomialTable(array);
                Discrete count = CountTrueOp.CountAverageConditional(forwardPassBuffer);
                Assert.Equal(6, count.Dimension);
                Assert.Equal(0, count[0], Tolerance);
                Assert.Equal(0.9 * 0.7 * 0.5, count[1], Tolerance);
                Assert.Equal((0.1 * 0.7 * 0.5) + (0.9 * 0.3 * 0.5) + (0.9 * 0.7 * 0.5), count[2], Tolerance);
                Assert.Equal((0.1 * 0.3 * 0.5) + (0.1 * 0.7 * 0.5) + (0.9 * 0.3 * 0.5), count[3], Tolerance);
                Assert.Equal(0.1 * 0.3 * 0.5, count[4], Tolerance);
                Assert.Equal(0, count[5], Tolerance);
            }

            // Test backward message
            {
                var array = new[] { new Bernoulli(1.0 / 3.0), new Bernoulli(1.0 / 2.0), new Bernoulli(0.0) };
                var count = new Discrete(1.0 / 9.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 9.0);
                var resultArray = new Bernoulli[3];
                double[,] forwardPassBuffer = CountTrueOp.PoissonBinomialTable(array);
                CountTrueOp.ArrayAverageConditional(array, count, forwardPassBuffer, resultArray);
                Assert.Equal(3.0 / 5.0, resultArray[0].GetProbTrue(), Tolerance);
                Assert.Equal(9.0 / 14.0, resultArray[1].GetProbTrue(), Tolerance);
                Assert.Equal(8.0 / 15.0, resultArray[2].GetProbTrue(), Tolerance);
            }

            // Test corner cases for forward message
            {
                const double Prob = 0.666;
                double[,] forwardPassBuffer = CountTrueOp.PoissonBinomialTable(new[] { new Bernoulli(Prob) });
                Discrete count = CountTrueOp.CountAverageConditional(forwardPassBuffer);
                Assert.Equal(2, count.Dimension);
                Assert.Equal(1.0 - Prob, count[0], Tolerance);
                Assert.Equal(Prob, count[1], Tolerance);
            }

            {
                double[,] forwardPassBuffer = CountTrueOp.PoissonBinomialTable(new Bernoulli[] { });
                Discrete count = CountTrueOp.CountAverageConditional(forwardPassBuffer);
                Assert.Equal(1, count.Dimension);
                Assert.Equal(1.0, count[0]);
            }

            // Test corner cases for backward message
            {
                const double Prob = 0.666, OtherProb = 0.777;
                var array = new[] { new Bernoulli(OtherProb) };
                var resultArray = new Bernoulli[1];
                double[,] forwardPassBuffer = CountTrueOp.PoissonBinomialTable(array);
                CountTrueOp.ArrayAverageConditional(array, new Discrete(1 - Prob, Prob), forwardPassBuffer, resultArray);
                Assert.Equal(Prob, resultArray[0].GetProbTrue(), Tolerance);
            }

            {
                var array = new Bernoulli[] { };
                var resultArray = new Bernoulli[] { };
                double[,] forwardPassBuffer = CountTrueOp.PoissonBinomialTable(array);
                CountTrueOp.ArrayAverageConditional(array, new Discrete(1.0), forwardPassBuffer, resultArray);
            }
        }

        [Fact]
        public void VariablePointOp_RpropGammaTest()
        {
            using (TestUtils.TemporarilyUseMeanPointGamma)
            {
                var buffer = VariablePointOp_RpropGamma.Buffer0Init();
                Gamma g = Gamma.FromShapeAndRate(0.6, 1);
                Gamma to_marginal = Gamma.PointMass(1);
                for (int i = 0; i < 1000; i++)
                {
                    buffer = VariablePointOp_RpropGamma.Buffer0(g, g, to_marginal, buffer);
                    //Console.WriteLine(buffer.nextPoint);
                    to_marginal = Gamma.Uniform();
                }
            }
        }

        [Fact]
        public void ArrayFromVectorOpTest()
        {
            bool verbose = false;
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(new double[,] {
                { 4.008640513161180,  1.303104352135630, - 2.696380025254830, - 2.728465435435790 },
                { 1.303104352135630,  4.024136989099960, - 2.681070246787840, - 2.713155656968810 },
                { -2.696380025254830, - 2.681070246787840, 4.136120496920130,  1.403451295855420 },
                { -2.728465435435790, - 2.713155656968810, 1.403451295855420,  4.063123100392480 }
            });
            Vector arrayVariance = Vector.FromArray(1, 0, 3, 4);
            Vector arrayMean = Vector.FromArray(1, 2, 3, 4);

            VectorGaussianMoments vg = new VectorGaussianMoments(Vector.FromArray(6, 5, 4, 3), A);
            Gaussian[] array = Util.ArrayInit(arrayMean.Count, i => new Gaussian(arrayMean[i], arrayVariance[i]));
            var result = ArrayFromVectorOp.ArrayAverageConditional(array, vg, new Gaussian[array.Length]);
            Vector varianceExpected = Vector.FromArray(0.699231932932321, 0, 0.946948669226297, 0.926496676481940);
            Vector varianceActual = Vector.FromArray(Util.ArrayInit(result.Length, i => (result[i] * array[i]).GetVariance()));
            if (verbose) Console.WriteLine($"variance = {varianceActual} should be {varianceExpected}");
            Assert.True(varianceExpected.MaxDiff(varianceActual) < 1e-4);
            Vector meanExpected = Vector.FromArray(2.640276200841019, 2.000000014880260, 6.527941507328482, 6.908179339051594);
            Vector meanActual = Vector.FromArray(Util.ArrayInit(result.Length, i => (result[i] * array[i]).GetMean()));
            if (verbose) Console.WriteLine($"mean = {meanActual} should be {meanExpected}");
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-4);

            arrayVariance.SetTo(Vector.FromArray(1, double.PositiveInfinity, 3, 4));
            array = Util.ArrayInit(arrayMean.Count, i => new Gaussian(arrayMean[i], arrayVariance[i]));
            result = ArrayFromVectorOp.ArrayAverageConditional(array, vg, new Gaussian[array.Length]);
            varianceExpected = Vector.FromArray(0.703202829692760, 2.371534574978036, 1.416576208767429, 1.566923316764467);
            varianceActual = Vector.FromArray(Util.ArrayInit(result.Length, i => (result[i] * array[i]).GetVariance()));
            if (verbose) Console.WriteLine($"variance = {varianceActual} should be {varianceExpected}");
            Assert.True(varianceExpected.MaxDiff(varianceActual) < 1e-4);
            meanExpected = Vector.FromArray(2.495869677904531, 5.528904975989440, 4.957577684071078, 5.074347889058121);
            meanActual = Vector.FromArray(Util.ArrayInit(result.Length, i => (result[i] * array[i]).GetMean()));
            if (verbose) Console.WriteLine($"mean = {meanActual} should be {meanExpected}");
            Assert.True(meanExpected.MaxDiff(meanActual) < 1e-4);

            bool testImproper = false;
            if (testImproper)
            {
                array = new Gaussian[]
                {
                new Gaussian(0.5069, 27.43),
                new Gaussian(0.649, 38.94),
                Gaussian.FromNatural(-0.5753, 0),
                new Gaussian(0.1064, 10.15)
                };
                PositiveDefiniteMatrix B = new PositiveDefiniteMatrix(new double[,]
                {
                { 0.7158,  -0.1356, -0.2979, -0.2993 },
                { -0.1356, 0.7294,  -0.2975, -0.2989 },
                { -0.2979, -0.2975, 0.7625,  -0.1337 },
                { -0.2993, -0.2989, -0.1337, 0.7203 },
                });
                vg = new VectorGaussianMoments(Vector.FromArray(0.3042, 0.4795, -0.2114, -0.5748), B);
                result = ArrayFromVectorOp.ArrayAverageConditional(array, vg, new Gaussian[array.Length]);
                Gaussian[] resultExpected = new Gaussian[] {
                    new Gaussian(0.4589, 0.707),
                    new Gaussian(0.6338, 0.7204),
                    new Gaussian(-0.2236, 0.7553),
                    new Gaussian(-0.4982, 0.7148),
                };
                if (verbose) Console.WriteLine(StringUtil.ArrayToString(result));
                for (int i = 0; i < result.Length; i++)
                {
                    Assert.True(resultExpected[i].MaxDiff(result[i]) < 1e-4);
                }
            }
        }

        internal void VectorFromArrayOp_SpeedTest()
        {
            int dim = 20;
            int n = 100000;
            VectorGaussian vector = new VectorGaussian(dim);
            Matrix random = new Matrix(dim, dim);
            for (int i = 0; i < dim; i++)
            {
                vector.MeanTimesPrecision[i] = Rand.Double();
                for (int j = 0; j < dim; j++)
                {
                    random[i, j] = Rand.Double();
                }
            }
            vector.Precision.SetToOuter(random);
            Gaussian[] array = Util.ArrayInit(dim, i => Gaussian.FromMeanAndVariance(i, i));
            Gaussian[] result = new Gaussian[dim];
            for (int i = 0; i < n; i++)
            {
                VectorFromArrayOp.ArrayAverageConditional(vector, array, result);
            }
        }

        [Fact]
        public void VectorFromArrayOp_PointMassTest()
        {
            VectorFromArrayOp_HandlesPointMass(false);
            VectorFromArrayOp_HandlesPointMass(true);
        }

        private void VectorFromArrayOp_HandlesPointMass(bool partial)
        {
            int dim = 2;
            VectorGaussian to_vector = new VectorGaussian(dim);
            VectorGaussian vector = VectorGaussian.FromNatural(Vector.FromArray(2.0, 3.0), new PositiveDefiniteMatrix(new double[,] { { 5.0, 1.0 }, { 1.0, 6.0 } }));
            GaussianArray expected = null;
            double lastError = double.PositiveInfinity;
            for (int i = 0; i < 10; i++)
            {
                double variance = System.Math.Exp(-i);
                if (i == 0)
                    variance = 0;
                Gaussian[] array = Util.ArrayInit(dim, j => new Gaussian(j, (j > 0 && partial) ? j : variance));
                to_vector = VectorFromArrayOp.VectorAverageConditional(array, to_vector);
                var to_array = new GaussianArray(dim);
                to_array = VectorFromArrayOp.ArrayAverageConditional(vector, array, to_array);
                if (i == 0)
                    expected = to_array;
                else
                {
                    double error = expected.MaxDiff(to_array);
                    Assert.True(error < lastError);
                    lastError = error;
                }
                ////Console.WriteLine(to_array);
            }
        }


        [Fact]
        public void MatrixTimesVectorOpTest()
        {
            Matrix[] ms = new Matrix[3];
            ms[0] = new Matrix(new double[,] { { 0.036, -0.036, 0 }, { 0.036, 0, -0.036 } });
            ms[1] = new Matrix(new double[,] { { -0.036, 0.036, 0 }, { 0, 0.036, -0.036 } });
            ms[2] = new Matrix(new double[,] { { -0.036, 0, 0.036 }, { 0, -0.036, 0.036 } });
            VectorGaussian product = new VectorGaussian(Vector.FromArray(new double[] { 1, 1 }), PositiveDefiniteMatrix.IdentityScaledBy(2, 0.01));
            Matrix A = ms[0];
            VectorGaussian result = MatrixVectorProductOp.BAverageConditional(product, A, new VectorGaussian(3));
            Assert.False(result.Precision.IsPositiveDefinite());

            Vector BMean = Vector.FromArray(1, 2, 3);
            PositiveDefiniteMatrix BVariance = new PositiveDefiniteMatrix(new double[,]
            {
                { 3,2,1 },
                { 2,3,2 },
                { 1,2,3 }
            });
            GaussianArray2D to_A = new GaussianArray2D(A.Rows, A.Cols);
            MatrixVectorProductOp.AAverageConditional(product, GaussianArray2D.PointMass(A.ToArray()), BMean, BVariance, to_A);
            double[,] dExpected = new double[,]
            {
                { 542.014350893474, -34.0730521080406, -261.179402050882 },
                { 449.735198014871, -31.4263853641523, -215.272881589687 },
            };
            for (int i = 0; i < to_A.GetLength(0); i++)
            {
                for (int j = 0; j < to_A.GetLength(1); j++)
                {
                    Gaussian dist = to_A[i, j];
                    double dlogp, ddlogp;
                    dist.GetDerivatives(A[i, j], out dlogp, out ddlogp);
                    Assert.True(System.Math.Abs(dExpected[i, j] - dlogp) < 1e-8);
                }
            }
        }

        [Fact]
        public void SumOpTest()
        {
            Gaussian sum_F = new Gaussian(0, 1);
            Gaussian sum_B = Gaussian.FromMeanAndPrecision(0, 1e-310);
            GaussianArray array = new GaussianArray(2, i => new Gaussian(0, 1));
            var result = FastSumOp.ArrayAverageConditional(sum_B, sum_F, array, new GaussianArray(array.Count));
            Assert.True(!double.IsNaN(result[0].MeanTimesPrecision));

            sum_B = new Gaussian(0, 1);
            array[0] = Gaussian.FromMeanAndPrecision(0, 0);
            sum_F = FastSumOp.SumAverageConditional(array);
            result = FastSumOp.ArrayAverageConditional(sum_B, sum_F, array, new GaussianArray(array.Count));
            Assert.True(result[0].MaxDiff(new Gaussian(0, 2)) < 1e-10);
            Assert.True(result[1].IsUniform());
            for (int i = 0; i < 1030; i++)
            {
                array[0] = Gaussian.FromMeanAndPrecision(0, System.Math.Pow(2, -i));
                sum_F = FastSumOp.SumAverageConditional(array);
                result = FastSumOp.ArrayAverageConditional(sum_B, sum_F, array, new GaussianArray(array.Count));
                Assert.True(result[0].MaxDiff(new Gaussian(0, 2)) < 1e-10);
            }
        }


        [Fact]
        public void LogisticOpTest()
        {
            for (int trial = 0; trial < 2; trial++)
            {
                double xMean = (trial == 0) ? -1 : 1;
                Gaussian x = Gaussian.FromMeanAndVariance(xMean, 0);
                Gaussian result2 = BernoulliFromLogOddsOp.LogOddsAverageConditional(true, x);
                ////Console.WriteLine("{0}: {1}", x, result2);
                for (int i = 8; i < 30; i++)
                {
                    double v = System.Math.Pow(0.1, i);
                    x = Gaussian.FromMeanAndVariance(xMean, v);
                    Gaussian result = BernoulliFromLogOddsOp.LogOddsAverageConditional(true, x);
                    ////Console.WriteLine("{0}: {1} maxDiff={2}", x, result, result2.MaxDiff(result));
                    Assert.True(result2.MaxDiff(result) < 1e-6);
                }
            }

            for (int i = 0; i < 10; i++)
            {
                Gaussian falseMsg = LogisticOp.FalseMsg(new Beta(0.2, 1.8), new Gaussian(0, 97.0 * (i + 1)), new Gaussian());
                ////Console.WriteLine(falseMsg);
            }
            Gaussian toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, new Gaussian(-4662, 1314));
            ////Console.WriteLine(toLogOdds);
            toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, new Gaussian(2249, 2.5));
            ////Console.WriteLine(toLogOdds);
            Gaussian logOdds = new Gaussian(100, 100);
            toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, logOdds);
            ////Console.WriteLine(toLogOdds * logOdds);
            // test m =approx 1.5*v for increasing v
            for (int i = 0; i < 10; i++)
            {
                double v = System.Math.Pow(2, i + 5);
                Gaussian g = new Gaussian(37 + 1.5 * v, v);
                toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, g);
                ////Console.WriteLine("{0}: {1}", g, toLogOdds);
                Gaussian actualPost = toLogOdds * g;
                Gaussian expectedPost = new Gaussian(0, 1);
                if (i == 0)
                {
                    // for Gaussian(85,32)
                    expectedPost = new Gaussian(53, 32);
                    Assert.True(expectedPost.MaxDiff(actualPost) < 1e-3);
                }
                else if (i == 1)
                {
                    // for Gaussian(133,64)
                    expectedPost = new Gaussian(69, 64);
                    Assert.True(expectedPost.MaxDiff(actualPost) < 1e-3);
                }
                else if (i == 4)
                {
                    // for Gaussian(805,512) the posterior should be (293, 511.81)
                    expectedPost = new Gaussian(293, 511.81);
                    Assert.True(expectedPost.MaxDiff(actualPost) < 1e-3);
                }
                else if (i == 5)
                {
                    // for Gaussian(1573,1024)
                    expectedPost = new Gaussian(549, 1023.9);
                    Assert.True(expectedPost.MaxDiff(actualPost) < 1e-3);
                }
            }
            // test m =approx v for increasing v
            for (int i = 0; i < 10; i++)
            {
                double v = System.Math.Pow(2, i + 5);
                Gaussian g = new Gaussian(v - 1, v);
                toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, g);
                ////Console.WriteLine("{0}: {1}", g, toLogOdds);
                Gaussian actualPost = toLogOdds * g;
                Gaussian expectedPost = new Gaussian(0, 1);
                if (i == 0)
                {
                    // for Gaussian(31,32)
                    expectedPost = new Gaussian(3.8979, 12.4725);
                    Assert.True(expectedPost.MaxDiff(actualPost) < 1e-3);
                }
                else if (i == 6)
                {
                    // for Gaussian(2047,2048)
                    expectedPost = new Gaussian(35.7156, 736.2395);
                    Assert.True(expectedPost.MaxDiff(actualPost) < 1e-3);
                }
            }
            for (int i = 0; i < 10; i++)
            {
                toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, new Gaussian(54.65 / 10 * (i + 1), 8.964));
                ////Console.WriteLine(toLogOdds);
            }
            toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, new Gaussian(9900, 10000) ^ 0.1);
            ////Console.WriteLine(toLogOdds);
            Gaussian falseMsg2 = LogisticOp.FalseMsg(new Beta(0.9, 0.1), Gaussian.FromNatural(-10.097766458353044, 0.000011644704327819733),
              Gaussian.FromNatural(-0.0010832099815010626, 0.000010092906656322242));
            ////Console.WriteLine(falseMsg2);
        }

        [Fact]
        public void BernoulliFromLogOddsTest()
        {
            double tolerance = 1e-8;
            Assert.True(MMath.AbsDiff(BernoulliFromLogOddsOp.LogAverageFactor(true, new Gaussian(0, 2)), System.Math.Log(0.5)) < tolerance);
            Assert.True(MMath.AbsDiff(BernoulliFromLogOddsOp.LogAverageFactor(true, new Gaussian(2, 0)), MMath.LogisticLn(2)) < tolerance);
            Assert.True(MMath.AbsDiff(BernoulliFromLogOddsOp.LogAverageFactor(true, new Gaussian(1, 1)), System.Math.Log(1 - 0.5 * System.Math.Exp(-0.5))) < tolerance);
            Assert.True(MMath.AbsDiff(BernoulliFromLogOddsOp.LogAverageFactor(true, new Gaussian(10, 10)), System.Math.Log(1 - 0.5 * System.Math.Exp(-5))) < tolerance);
            Assert.True(MMath.AbsDiff(BernoulliFromLogOddsOp.LogAverageFactor(true, new Gaussian(-1, 1)), System.Math.Log(0.303265329856008)) < tolerance);
            Assert.True(MMath.AbsDiff(BernoulliFromLogOddsOp.LogAverageFactor(true, new Gaussian(-5, 3)), System.Math.Log(0.023962216989475)) < tolerance);
            Assert.True(MMath.AbsDiff(BernoulliFromLogOddsOp.LogAverageFactor(true, new Gaussian(-5, 10)), System.Math.Log(0.084396512538678)) < tolerance);
        }


        [Fact]
        public void OrTest()
        {
            Assert.Equal(1.0, BooleanOrOp.AAverageConditional(true, false).GetProbTrue());
            Assert.Equal(1.0, BooleanOrOp.OrAverageConditional(Bernoulli.PointMass(true), false).GetProbTrue());

            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(Bernoulli.PointMass(false), Bernoulli.PointMass(false)).LogOdds);
            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(false, Bernoulli.PointMass(false)).LogOdds);
            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(Bernoulli.PointMass(false), false).LogOdds);
            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(false, false).LogOdds);

            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(Bernoulli.PointMass(false), Bernoulli.PointMass(true)).GetProbTrue());
            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(false, Bernoulli.PointMass(true)).GetProbTrue());
            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(Bernoulli.PointMass(false), true).GetProbTrue());
            Assert.Equal(0.0, BooleanAndOp.AAverageConditional(false, true).GetProbTrue());

            Assert.Throws<AllZeroException>(() =>
            {
                BooleanAndOp.AAverageConditional(true, false);
            });
            Assert.Throws<AllZeroException>(() =>
            {
                BooleanAndOp.AAverageConditional(Bernoulli.PointMass(true), false);
            });
            Assert.Throws<AllZeroException>(() =>
            {
                BooleanAndOp.AAverageConditional(true, Bernoulli.PointMass(false));
            });
            Assert.Throws<AllZeroException>(() =>
            {
                BooleanAndOp.AAverageConditional(Bernoulli.PointMass(true), Bernoulli.PointMass(false));
            });
        }

        [Fact]
        public void BernoulliFromDiscreteTest()
        {
            double[] probTrue = { 0.4, 0.7 };
            Discrete indexDist = new Discrete(0.8, 0.2);
            Bernoulli sampleDist = new Bernoulli(0.8 * 0.4 + 0.2 * 0.7);
            Assert.True(sampleDist.MaxDiff(BernoulliFromDiscreteOp.SampleAverageConditional(indexDist, probTrue)) < 1e-4);
            sampleDist.SetProbTrue(0.2);
            double p = (0.8 * 0.3 + 0.2 * 0.7) / (0.8 * 0.3 + 0.2 * 0.7 + 0.8 * 0.6 + 0.2 * 0.4);
            Vector probs = indexDist.GetWorkspace();
            probs[0] = 1 - p;
            probs[1] = p;
            indexDist.SetProbs(probs);
            Assert.True(indexDist.MaxDiff(BernoulliFromDiscreteOp.IndexAverageConditional(sampleDist, probTrue, Discrete.Uniform(indexDist.Dimension))) < 1e-4);
        }

        [Fact]
        public void BernoulliFromBooleanTest()
        {
            double[] probTrue = { 0.4, 0.7 };
            Bernoulli choiceDist = new Bernoulli(0.2);
            Bernoulli sampleDist = new Bernoulli(0.8 * 0.4 + 0.2 * 0.7);
            Assert.True(sampleDist.MaxDiff(BernoulliFromBooleanArray.SampleAverageConditional(choiceDist, probTrue)) < 1e-4);
            sampleDist.SetProbTrue(0.2);
            choiceDist.SetProbTrue((0.8 * 0.3 + 0.2 * 0.7) / (0.8 * 0.3 + 0.2 * 0.7 + 0.8 * 0.6 + 0.2 * 0.4));
            Assert.True(choiceDist.MaxDiff(BernoulliFromBooleanArray.ChoiceAverageConditional(sampleDist, probTrue)) < 1e-4);
        }


        // test the limit of an incoming message with small precision
        [Fact]
        public void IsPositiveGaussianTest2()
        {
            for (int trial = 0; trial < 2; trial++)
            {
                bool isPositive = (trial == 0);
                Gaussian x = Gaussian.FromNatural(isPositive ? -2 : 2, 0);
                Gaussian result = IsPositiveOp.XAverageConditional(isPositive, x);
                for (int i = 10; i < 20; i++)
                {
                    x.Precision = System.Math.Pow(0.1, i);
                    Gaussian result2 = IsPositiveOp.XAverageConditional(isPositive, x);
                    //Console.WriteLine($"{x}: {result2} maxDiff={result.MaxDiff(result2)}");
                    Assert.True(result.MaxDiff(result2) < 1e-6);
                }
            }
        }

        [Fact]
        public void IsPositiveGaussianTest()
        {
            for (int trial = 0; trial < 2; trial++)
            {
                double xMean = (trial == 0) ? -1 : 1;
                Gaussian x = Gaussian.FromMeanAndVariance(xMean, 0);
                Gaussian result2 = IsPositiveOp.XAverageConditional(true, x);
                //Console.WriteLine("{0}: {1}", x, result2);
                if (trial == 0)
                    Assert.True(result2.IsPointMass && result2.Point == 0.0);
                else
                    Assert.True(result2.IsUniform());
                for (int i = 8; i < 30; i++)
                {
                    double v = System.Math.Pow(0.1, i);
                    x = Gaussian.FromMeanAndVariance(xMean, v);
                    Gaussian result = IsPositiveOp.XAverageConditional(true, x);
                    //Console.WriteLine("{0}: {1} maxDiff={2}", x, result, result2.MaxDiff(result));
                    if (trial == 0)
                    {
                        Assert.True(MMath.AbsDiff(result2.GetMean(), result.GetMean()) < 1e-6);
                    }
                    else
                        Assert.True(result2.MaxDiff(result) < 1e-6);
                }
            }
            for (int i = 0; i < 20; i++)
            {
                Assert.True(IsPositiveOp.XAverageConditional(true, Gaussian.FromNatural(-System.Math.Pow(10, i), 0.1)).IsProper());
            }
            Assert.True(IsPositiveOp.XAverageConditional(true, Gaussian.FromNatural(-2.3287253734154412E+107, 0.090258824802119691)).IsProper());
            Assert.True(IsPositiveOp.XAverageConditional(false, Gaussian.FromNatural(2.3287253734154412E+107, 0.090258824802119691)).IsProper());
            //Assert.True(IsPositiveOp.XAverageConditional(Bernoulli.FromLogOdds(-4e-16), new Gaussian(490, 1.488e+06)).IsProper());
            Assert.True(IsPositiveOp.XAverageConditional(true, new Gaussian(-2.03e+09, 5.348e+09)).IsProper());

            Gaussian uniform = new Gaussian();
            Assert.Equal(0.0, IsPositiveOp.IsPositiveAverageConditional(Gaussian.FromMeanAndVariance(0, 1)).LogOdds);
            Assert.True(MMath.AbsDiff(MMath.NormalCdfLogit(2.0 / System.Math.Sqrt(3)), IsPositiveOp.IsPositiveAverageConditional(Gaussian.FromMeanAndVariance(2, 3)).LogOdds, 1e-10) <
                          1e-10);
            Assert.Equal(0.0, IsPositiveOp.IsPositiveAverageConditional(uniform).LogOdds);
            Assert.Equal(0.0, IsPositiveOp.IsPositiveAverageConditional(Gaussian.PointMass(0.0)).GetProbTrue());

            Bernoulli isPositiveDist = new Bernoulli();
            Gaussian xDist = new Gaussian(0, 1);
            Gaussian xMsg = new Gaussian();
            Assert.True(IsPositiveOp.XAverageConditional(isPositiveDist, xDist).Equals(uniform));
            isPositiveDist = new Bernoulli(0);
            Gaussian xBelief = new Gaussian();
            xMsg = IsPositiveOp.XAverageConditional(isPositiveDist, xDist);
            xBelief.SetToProduct(xMsg, xDist);
            Assert.True(xBelief.MaxDiff(Gaussian.FromMeanAndVariance(-0.7979, 0.3634)) < 1e-1);
            isPositiveDist = new Bernoulli(1);
            xMsg = IsPositiveOp.XAverageConditional(isPositiveDist, xDist);
            xBelief.SetToProduct(xMsg, xDist);
            Assert.True(xBelief.MaxDiff(Gaussian.FromMeanAndVariance(0.7979, 0.3634)) < 1e-1);
            isPositiveDist = new Bernoulli(0.6);
            xMsg = IsPositiveOp.XAverageConditional(isPositiveDist, xDist);
            xBelief.SetToProduct(xMsg, xDist);
            Assert.True(xBelief.MaxDiff(Gaussian.FromMeanAndVariance(0.15958, 0.97454)) < 1e-1);
            Assert.True(IsPositiveOp.XAverageConditional(isPositiveDist, uniform).Equals(uniform));

            Assert.True(IsPositiveOp.XAverageConditional(true, new Gaussian(127, 11)).Equals(uniform));
            Assert.True(IsPositiveOp.XAverageConditional(false, new Gaussian(-127, 11)).Equals(uniform));
            Assert.True(IsPositiveOp.XAverageConditional(true, new Gaussian(-1e5, 10)).IsProper());
            Assert.True(IsPositiveOp.XAverageConditional(false, new Gaussian(1e5, 10)).IsProper());
        }
    }
}