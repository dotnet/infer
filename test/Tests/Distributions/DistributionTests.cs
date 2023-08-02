// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using GaussianArray2D = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray2D<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
using System.Threading.Tasks;
using System.Threading;
using Microsoft.ML.Probabilistic.Factors;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;

    public class DistributionTests
    {
        public const int nsamples = 50000;

        [Fact]
        public void BinomialTest()
        {
            Binomial a = new Binomial(10, 0.2);
            Binomial b = new Binomial(10, 0.3);

            ProductWithUniformTest(a);
            ProductWithUniformTest(b);

            RatioWithUniformTest(a);
            RatioWithUniformTest(b);
            SettableToRatioTest(a, b);

            SettableToPowerTest(a);
            SettableToPowerTest(b);

            SettableToTest(a);
            SettableToTest(b);

            int value = 3;
            ProductWithPointMassTest(a, value);
            RatioWithPointMassTest(a, value);
            PointMassPowerTest(a, value);
            PointMassSampleTest(a, value);
            PointMassGetLogProbTest(a, value);

            SamplingTest(a, 3);
        }

        [Fact]
        public void GammaPowerFromMeanAndMeanLogTest()
        {
            var testCases = new[]
            {
                (3, 0.4, 5),
                (2.7196413092151412, 1, 2),
                (0.82657135035599683, -0.19053040661796108, -1.0),
                (0.78123008687766349, -0.24698718364292091, -1.0),
                (0.65867320393062079, -0.41774280264746583, -1.0),
                (32.016744802449665, 3.4078368553200575, -1.0),
                (1.2739417498500658, -0.9778858306365934, -1.0),
                (System.Math.Exp(1.0), 1.0, -1.0),
                (System.Math.Exp(1.25), 1.0, -1.0),
            };
            foreach (var testCase in testCases)
            {
                var (mean, meanLog, power) = testCase;
                GammaPower gammaPower = GammaPower.FromMeanAndMeanLog(mean, meanLog, power);
                Assert.Equal(mean, gammaPower.GetMean(), 1e-10);
                Assert.Equal(meanLog, gammaPower.GetMeanLog(), 1e-10);
            }
            GammaPower.FromMeanAndMeanLog(0.82657135035599683, -0.19053040661796108, -1.0);
            GammaPower.FromMeanAndMeanLog(0.78123008687766349, -0.24698718364292091, -1.0);
        }

        [Fact]
        public void Gamma_GetMode_MaximizesGetLogProb()
        {
            Parallel.ForEach(OperatorTests.Gammas()/*.Take(100000)*/, gamma =>
            {
                double max = double.NegativeInfinity;
                foreach (var x in OperatorTests.DoublesAtLeastZero())
                {
                    double logProb = gamma.GetLogProb(x);
                    Assert.False(double.IsNaN(logProb));
                    if (logProb > max)
                    {
                        max = logProb;
                    }
                }
                double mode = gamma.GetMode();
                Assert.False(double.IsNaN(mode));
                double logProbBelowMode = gamma.GetLogProb(MMath.PreviousDouble(mode));
                Assert.False(double.IsNaN(logProbBelowMode));
                double logProbAboveMode = gamma.GetLogProb(MMath.NextDouble(mode));
                Assert.False(double.IsNaN(logProbAboveMode));
                double logProbAtMode = gamma.GetLogProb(mode);
                Assert.False(double.IsNaN(logProbAtMode));
                logProbAtMode = System.Math.Max(System.Math.Max(logProbAtMode, logProbAboveMode), logProbBelowMode);
                const double smallestNormalized = 1e-308;
                Assert.True(logProbAtMode >= max ||
                    MMath.AbsDiff(logProbAtMode, max, 1e-8) < 1e-4 ||
                    (mode == 0 && gamma.GetLogProb(smallestNormalized) >= max)
                    );
            });
        }

        [Fact]
        public void GammaPower_GetMode_MaximizesGetLogProb()
        {
            Assert.False(double.IsNaN(GammaPower.FromShapeAndRate(1E+306, 1E+226, 1.7976931348623157E+308).GetLogProb(0)));
            Assert.True(GammaPower.FromShapeAndRate(1E+17, 1E+17, -10000000000).GetLogProb(double.Epsilon) >= GammaPower.FromShapeAndRate(1E+17, 1E+17, -10000000000).GetLogProb(MMath.NextDouble(double.Epsilon)));
            Assert.False(double.IsPositiveInfinity(GammaPower.FromShapeAndRate(1.00000000000001E-152, 1E-88, -1E-06).GetLogProb(0)));
            long count = 0;
            Parallel.ForEach(new[] {
                GammaPower.FromShapeAndRate(1.7976931348623157E+308, 1.7976931348623157E+308, -1.7976931348623157E+308),
            }.Concat(OperatorTests.GammaPowers(1000000)), gammaPower =>
            {
                double argmax = double.NaN;
                double max = double.NegativeInfinity;
                foreach (var x in OperatorTests.DoublesAtLeastZero())
                {
                    double logProb = gammaPower.GetLogProb(x);
                    if (double.IsNaN(logProb))
                    {
                        throw new Exception($"GammaPower.FromShapeAndRate({gammaPower.Shape}, {gammaPower.Rate}, {gammaPower.Power}).GetLogProb({x:g17}) is NaN");
                    }
                    if (logProb > max)
                    {
                        max = logProb;
                        argmax = x;
                    }
                }
                double mode = gammaPower.GetMode();
                Assert.False(double.IsNaN(mode));
                double logProbBelowMode = gammaPower.GetLogProb(MMath.PreviousDouble(mode));
                Assert.False(double.IsNaN(logProbBelowMode));
                double logProbAboveMode = gammaPower.GetLogProb(MMath.NextDouble(mode));
                Assert.False(double.IsNaN(logProbAboveMode));
                double logProbAtMode = gammaPower.GetLogProb(mode);
                Assert.False(double.IsNaN(logProbAtMode));
                logProbAtMode = System.Math.Max(System.Math.Max(logProbAtMode, logProbAboveMode), logProbBelowMode);
                const double smallestNormalized = 1e-308;
                bool logProbAtModeIsMax = logProbAtMode >= max ||
                    MMath.AbsDiff(logProbAtMode, max, 1e-8) < 1e-8 ||
                    (mode <= double.Epsilon && gammaPower.GetLogProb(smallestNormalized) >= max);
                if (!logProbAtModeIsMax)
                {
                    throw new Exception($"GammaPower.FromShapeAndRate({gammaPower.Shape}, {gammaPower.Rate}, {gammaPower.Power}) logProbAtMode={logProbAtMode} max={max} mode={mode} logProbAtMode={logProbAtMode} logProbBelowMode={logProbBelowMode} logProbAboveMode={logProbAboveMode}");
                }
                Interlocked.Add(ref count, 1);
                if (count % 100000 == 0)
                    Trace.WriteLine($"{count} cases passed");
            });
            Trace.WriteLine($"{count} cases passed");
        }

        [Fact]
        public void TruncatedGamma_GetMode_MaximizesGetLogProb()
        {
            Assert.False(double.IsNaN(new TruncatedGamma(9.9999999999999999e+306, 9.9999999999999996e-70, 1e+149, double.PositiveInfinity).GetLogProb(1.7976931348623157e+308)));
            Assert.False(double.IsNaN(new TruncatedGamma(9.9999999999999999e+306, 1e-108, 9.9999999999999995e+123, 1e+270).GetLogProb(9.9999999999999995e+123)));
            Assert.False(double.IsNaN(new TruncatedGamma(1.0000000000000005e-09, 9.9999999999999129e+156, 4.9406564584124654e-324, 9.8813129168249309e-324).GetLogProb(4.9406564584124654e-324)));
            Assert.False(double.IsNaN(new TruncatedGamma(Gamma.FromShapeAndRate(1.7976931348623157E+308, 4.94065645841247E-324), 1E+25, double.PositiveInfinity).GetLogProb(1.7976931348623157E+308)));
            Assert.False(double.IsNaN(new TruncatedGamma(1.0000000000000032E-58, 1e-77, 1E+113, 1E+232).GetLogProb(1E+232)));
            Assert.False(double.IsNaN(new TruncatedGamma(1E+230, 1e-35, 4.94065645841247E-324, 1.0000000000000015E-28).GetLogProb(4.94065645841247E-324)));
            long count = 0;
            Parallel.ForEach(OperatorTests.TruncatedGammas(1000000), dist =>
            {
                double argmax = double.NaN;
                double max = double.NegativeInfinity;
                foreach (var x in OperatorTests.DoublesAtLeastZero())
                {
                    double logProb = dist.GetLogProb(x);
                    if (double.IsNaN(logProb))
                    {
                        throw new Exception($"TruncatedGamma({dist.Gamma.Shape:g17}, {dist.Gamma.GetScale():g17}, {dist.LowerBound:g17}, {dist.UpperBound:g17}).GetLogProb({x:g17}) is NaN");
                    }
                    if (logProb > max)
                    {
                        max = logProb;
                        argmax = x;
                    }
                }
                double mode = dist.GetMode();
                Assert.False(double.IsNaN(mode));
                double logProbBelowMode = dist.GetLogProb(MMath.PreviousDouble(mode));
                Assert.False(double.IsNaN(logProbBelowMode));
                double logProbAboveMode = dist.GetLogProb(MMath.NextDouble(mode));
                Assert.False(double.IsNaN(logProbAboveMode));
                double logProbAtMode = dist.GetLogProb(mode);
                Assert.False(double.IsNaN(logProbAtMode));
                logProbAtMode = System.Math.Max(System.Math.Max(logProbAtMode, logProbAboveMode), logProbBelowMode);
                const double smallestNormalized = 1e-308;
                Assert.True(logProbAtMode >= max ||
                    MMath.AbsDiff(logProbAtMode, max, 1e-8) < 1e-8 ||
                    (mode <= double.Epsilon && dist.GetLogProb(smallestNormalized) >= max)
                    );
                Interlocked.Add(ref count, 1);
                if (count % 100000 == 0)
                    Trace.WriteLine($"{count} cases passed");
            });
            Trace.WriteLine($"{count} cases passed");
        }

        [Fact]
        public void GammaPowerTest()
        {
            foreach (var gammaPower in new[] {
                GammaPower.FromShapeAndRate(3, 2, -4.0552419045546273),
                new GammaPower(0.04591, 19.61, -1),
            })
            {
                gammaPower.GetMeanAndVariance(out double mean, out double variance);
                Assert.False(double.IsNaN(mean));
                Assert.False(double.IsNaN(variance));
                Assert.False(mean < 0);
                Assert.False(variance < 0);
                Assert.Equal(variance, gammaPower.GetVariance());
            }

            Assert.Equal(0, GammaPower.FromShapeAndRate(2, 0, -1).GetMean());
            Assert.Equal(0, GammaPower.FromShapeAndRate(2, 0, -1).GetVariance());
            Assert.True(GammaPower.FromShapeAndRate(2, double.PositiveInfinity, -1).IsPointMass);

            GammaPower g = new GammaPower(1, 1, -1);
            g.ToString();
            Gamma gamma = new Gamma(1, 1);
            double expectedProbLessThan = gamma.GetProbLessThan(2);
            Assert.Equal(expectedProbLessThan, 1 - g.GetProbLessThan(0.5), 1e-10);
            Assert.Equal(2, gamma.GetQuantile(expectedProbLessThan), 1e-10);
            Assert.Equal(0.5, g.GetQuantile(1 - expectedProbLessThan), 1e-10);

            Assert.Equal(0, g.GetProbLessThan(0));
            Assert.Equal(0, g.GetProbLessThan(double.MinValue));
            Assert.Equal(0, g.GetProbLessThan(double.NegativeInfinity));

            g = GammaPower.FromMeanAndVariance(3, double.PositiveInfinity, -1);
            Assert.Equal(2, g.Shape);
            Assert.Equal(3, g.Rate);

            GammaPowerMomentTest(1);
            GammaPowerMomentTest(-1);
            GammaPowerMomentTest(2);
        }

        private void GammaPowerMomentTest(double power)
        {
            GammaPower g = new GammaPower(9.9, 1, power);
            GammaPower g2 = new GammaPower(4.4, 3.3, power);
            DistributionTest(g, g2);
            PointMassTest(g, 7.7);
            UniformTest(g, 7.7);
            SetMomentTest(g, 1.1, 2.2);
            PointMassMomentTest(g, 7.7, 4.4, 5.5);
            SamplingTest(g, 7.7);
            g.SetToUniform();
        }

        [Fact]
        public void GammaPowerMeanAndVarianceFuzzTest()
        {
            Assert.False(GammaPower.FromShapeAndRate(1E+49, 1E+66, 9.99999999999997E-311).GetVariance() < 0);
            foreach (var gammaPower in OperatorTests.GammaPowers(10000000))
            {
                gammaPower.GetMeanAndVariance(out double mean, out double variance);
                Assert.False(double.IsNaN(mean));
                Assert.False(double.IsNaN(variance));
                Assert.False(mean < 0);
                Assert.False(variance < 0);
            }
        }

        [Fact]
        public void GammaMeanAndVarianceFuzzTest()
        {
            foreach (var gamma in OperatorTests.Gammas())
            {
                gamma.GetMeanAndVariance(out double mean, out double variance);
                Assert.False(double.IsNaN(mean));
                Assert.False(double.IsNaN(variance));
                Assert.False(mean < 0);
                Assert.False(variance < 0);
            }
        }

        //[Fact]
        internal void WrappedGaussianTest()
        {
            WrappedGaussian g = new WrappedGaussian(2.2, 3.3);
            WrappedGaussian g2 = new WrappedGaussian(4.4, 5.5);
            DistributionTest(g, g2);
            PointMassTest(g, 7.7);
            UniformTest(g, 7.7);
        }

        [Fact]
        public void TruncatedGaussianTest()
        {
            double lowerBound = 5;
            double upperBound = 10;
            var g = new TruncatedGaussian(2.2, 3.3, lowerBound, upperBound);
            double m, v;
            g.GetMeanAndVariance(out m, out v);
            //Assert.Equal(2.2, m);
            //Assert.Equal(3.3, v);

            foreach (double x in new[] { 1.1, 6.4 })
            {
                double probLessThan = g.GetProbLessThan(x);
                Assert.Equal(probLessThan, g.GetProbBetween(double.NegativeInfinity, x));
                double quantile = g.GetQuantile(probLessThan);
                Assert.Equal(System.Math.Max(lowerBound, x), quantile, 1e-4);
            }

            var g2 = new TruncatedGaussian(4.4, 5.5, lowerBound, upperBound);
            DistributionTest(g, g2);
            PointMassTest(g, 7.7);
            UniformTest(g, 7.7);
            //SetMomentTest(g, 1.1, 2.2);
            PointMassMomentTest(g, 7.7, 4.4, 5.5);
            SamplingTest(g, 7.7);

            var ratio = g / g2;
            Assert.Throws<DivideByZeroException>(() =>
            {
                ratio = g / new TruncatedGaussian(4.4, 5.5, lowerBound + 1, upperBound);
            });
            ratio = TruncatedGaussian.PointMass(lowerBound) / new TruncatedGaussian(4.4, 5.5, lowerBound, upperBound);
            Assert.Throws<DivideByZeroException>(() =>
            {
                ratio = TruncatedGaussian.PointMass(2) / new TruncatedGaussian(4.4, 5.5, lowerBound + 1, upperBound);
            });

            g.SetToUniform();
            //GetAndSetMomentTest(g, 0.0, Double.PositiveInfinity);

            g = new TruncatedGaussian(2, 1, 100, double.PositiveInfinity);
            SamplingTest(g, 7.7);
            foreach (double variance in new[] { 1e8, 1e17, double.MaxValue, double.PositiveInfinity })
            {
                g = new TruncatedGaussian(2, variance, 100, 101);
                SamplingTest(g, 7.7);
            }
        }

        [Fact]
        public void TruncatedGammaTest()
        {
            TruncatedGamma tg = new TruncatedGamma(Gamma.FromShapeAndRate(1.000000000000004E-72, 1.0000000000000144E-260), 1.0000000000000162E-290, double.PositiveInfinity);
            AssertAlmostEqual(1.2664218011467302E-69, tg.Gamma.GetProbBetween(tg.LowerBound, tg.UpperBound));
            //TruncatedGammaMomentTest(tg);
            double lowerBound = 5;
            double upperBound = 10;
            var g = new TruncatedGamma(2.2, 3.3, lowerBound, upperBound);
            g.GetMeanAndVariance(out double m, out double v);
            /* in matlab:
               x = linspace(5,10,1e6);
               inc = x(2)-x(1);
               lp = log(gampdf(x,2.2,3.3)); % gampdfln(x,2.2,1/3.3);
               m = sum(x.*exp(lp))/sum(exp(lp))
               v = sum((x-m).^2.*exp(lp))/sum(exp(lp))
               m2 = sum(x.*x.*exp(lp))/sum(exp(lp))
               ilp = log(gampdf(1./x,2.2,3.3)./x./x);
               tpm = sum(x.*exp(ilp))/sum(exp(ilp))
             */
            Assert.True(MMath.AbsDiff(7.2174, m) < 1e-4);
            Assert.True(MMath.AbsDiff(1.9969, v) < 1e-4);
            double m2 = g.GetMeanPower(2);
            Assert.True(MMath.AbsDiff(54.0874, m2) < 1e-4);
            //TruncatedGammaMomentTest(g);
            double truncatedPowerMean = PlusGammaOp.TruncatedGammaPowerGetMean(new GammaPower(2.2, 3.3, -1), lowerBound, upperBound);
            Assert.True(MMath.AbsDiff(6.6277, truncatedPowerMean) < 1e-4);

            foreach (double x in new[] { 1.1, 6.4 })
            {
                double probLessThan = g.GetProbLessThan(x);
                Assert.Equal(probLessThan, g.GetProbBetween(double.NegativeInfinity, x));
                double quantile = g.GetQuantile(probLessThan);
                Assert.Equal(System.Math.Max(lowerBound, x), quantile, 1e-4);
            }

            var g2 = new TruncatedGamma(4.4, 5.5, lowerBound, upperBound);
            DistributionTest(g, g2);
            PointMassTest(g, 7.7);
            // TODO
            //UniformTest(g, 7.7);
            //SetMomentTest(g, 1.1, 2.2);
            PointMassMomentTest(g, 7.7, 4.4, 5.5);
            // TODO
            //SamplingTest(g, 7.7);
            g.SetToUniform();
            //GetAndSetMomentTest(g, 0.0, Double.PositiveInfinity);

            g = new TruncatedGamma(2, 1, 100, double.PositiveInfinity);
            //SamplingTest(g, 7.7);
            g = new TruncatedGamma(2, 1e8, 100, 101);
            //SamplingTest(g, 7.7);

            g = new TruncatedGamma(2, 1, 3, 3);
            Assert.True(g.IsPointMass);
            Assert.Equal(3.0, g.Point);

            g = new TruncatedGamma(Gamma.FromShapeAndRate(4.94065645841247E-324, 4.94065645841247E-324), 0, 1e14);
            Assert.True(g.Sample() >= 0);
        }

        /// <summary>
        /// Checks that TruncatedGamma.GetMean does not return infinity or NaN for proper distributions.
        /// </summary>
        [Fact]
        public void TruncatedGamma_GetMean()
        {
            double target = 0.1;
            double oldDiff = double.PositiveInfinity;
            for (int i = 1; i < 20; i++)
            {
                TruncatedGamma g = new TruncatedGamma(1, System.Math.Exp(-i), target, double.PositiveInfinity);
                var mean = g.GetMean();
                //Trace.WriteLine($"GetNormalizer = {g.GetNormalizer()} GetMean = {g.GetMean()}");
                Assert.False(double.IsInfinity(mean));
                Assert.False(double.IsNaN(mean));
                double diff = System.Math.Abs(mean - target);
                Assert.True(diff == 0 || diff < oldDiff);
                oldDiff = diff;
                double mean2, variance;
                g.GetMeanAndVariance(out mean2, out variance);
                Assert.Equal(mean, mean2);
            }
            oldDiff = double.PositiveInfinity;
            for (int i = 0; i < 20; i++)
            {
                TruncatedGamma g = new TruncatedGamma(System.Math.Exp(i), 1, 0, target);
                var mean = g.GetMean();
                //Trace.WriteLine($"GetNormalizer = {g.GetNormalizer()} GetMean = {g.GetMean()}");
                Assert.False(double.IsInfinity(mean));
                Assert.False(double.IsNaN(mean));
                double diff = System.Math.Abs(mean - target);
                Assert.True(diff == 0 || diff < oldDiff);
                oldDiff = diff;
                double mean2, variance;
                g.GetMeanAndVariance(out mean2, out variance);
                Assert.Equal(mean, mean2);
            }
        }

        /// <summary>
        /// Checks that TruncatedGamma.GetMeanPower does not return infinity or NaN for proper distributions.
        /// </summary>
        [Fact]
        public void TruncatedGamma_GetMeanPower()
        {
            double shape = 1;
            TruncatedGamma g = new TruncatedGamma(shape, 1, 1, double.PositiveInfinity);
            for (int i = 0; i < 100; i++)
            {
                var meanPower = g.GetMeanPower(-i);
                //Trace.WriteLine($"GetMeanPower({-i}) = {meanPower}");
                Assert.False(double.IsNaN(meanPower));
                Assert.False(double.IsInfinity(meanPower));
                if (i == 1) Assert.Equal(MMath.GammaUpper(shape - 1, 1, false) / MMath.GammaUpper(shape, 1, false), meanPower, 1e-8);
            }
        }

        [Fact]
        public void TruncatedGamma_GetMeanAndVariance_WithinBounds()
        {
            Assert.True(new TruncatedGamma(1.0000000000000083E-150, 1 / 1.0000000000000169E-305, 1E+307, double.PositiveInfinity).GetVariance() >= 0);
            long count = 0;
            Parallel.ForEach(OperatorTests.LowerTruncatedGammas(100000), dist =>
            {
                dist.GetMeanAndVariance(out double mean, out double variance);
                // Compiler.Quoter.Quote(dist)
                Assert.True(mean >= dist.LowerBound);
                Assert.True(mean <= dist.UpperBound);
                Assert.Equal(mean, dist.GetMean());
                Assert.True(variance >= 0);
                Interlocked.Add(ref count, 1);
                if (count % 100000 == 0)
                    Trace.WriteLine($"{count} cases passed");
            });
            Trace.WriteLine($"{count} cases passed");
        }

        [Fact]
        public void TruncatedGamma_GetMeanPower_WithinBounds()
        {
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1000.0, 9.8813129168249309E-323), 1.0000000000000014E-25, double.PositiveInfinity).GetMeanPower(-10000) <= double.PositiveInfinity);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000001E+209, 1E+206), 1000.0, double.PositiveInfinity).GetMeanPower(-100) <= 1E-300);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000001E+23, 1.0000000000000001E+23), 1.0, double.PositiveInfinity).GetMeanPower(10000) >= 1);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(9.9999999999999992E+80, 1.0000000000000149E-269), 9.9999999999999995E+246, double.PositiveInfinity).GetMeanPower(4.94065645841247E-324) >= 1);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000011E-19, 1.0000000000000101E-181), 1.0000000000000165E-298, double.PositiveInfinity).GetMeanPower(0.010000000000000002) >= 0.0010471285480508983);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000002306925374E-317, 1.0000000000000159E-286), 1.0000000000000053E-95, double.PositiveInfinity).GetMeanPower(4.94065645841247E-324) >= 1);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000106E-191, 4.9406564584124654E-324), 1.0000000000000096E-173, double.PositiveInfinity).GetMeanPower(-0.00010000000000000002) <= 1.0406387653975053);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000112E-203, 1.0000000000000149E-269), 4.9406564584124654E-324, double.PositiveInfinity).GetMeanPower(-0.1) <= 2.1410239937243691E+32);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000048E-87, 1.0000000000000112E-201), 1.0000000000000001E+190, double.PositiveInfinity).GetMeanPower(-100) <= 0);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1E+266, 1E+110), 9.9801260459931802E-322, double.PositiveInfinity).GetMeanPower(-1e267) <= double.PositiveInfinity);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000063E-113, 1.0000000000000113E-205), 1.0000000000000154E-278, double.PositiveInfinity).GetMeanPower(-0.00010000000000000002) <= 1.0661050486467796);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000118E-212, 1.000000000000007E-127), 1.0, double.PositiveInfinity).GetMeanPower(double.NegativeInfinity) <= 1);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.000000000000004E-72, 1.0000000000000144E-260), 1.0000000000000162E-290, double.PositiveInfinity).GetMeanPower(0.1) >= 9.9999999999999784E-30);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000078E-140, 1.0000000000000123E-221), 1.0000000000000004E-07, double.PositiveInfinity).GetMeanPower(-10) <= 9.9999999999999655E+69);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(1.0000000000000008E-16, 1.0000000000000155E-280), 1.0000000000000098E-177, double.PositiveInfinity).GetMeanPower(-1) <= 9.9999999999999022E+176);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(9.9999999999999994E+236, 1.0000000000000001E+301), 9.8813129168249309E-324, double.PositiveInfinity).GetMeanPower(double.NegativeInfinity) >= 0);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(4.94065645841247E-324, 4.94065645841247E-324), 0, 100).GetMeanPower(4.94065645841247E-324) <= 100);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(4.94065645841247E-324, 4.94065645841247E-324), 0, 1e9).GetMeanPower(1.7976931348623157E+308) <= 1e9);
            Assert.True(new TruncatedGamma(Gamma.FromShapeAndRate(4.94065645841247E-324, 4.94065645841247E-324), 0, 1e6).GetMeanPower(1.7976931348623157E+308) <= 1e6);
            var g = new TruncatedGamma(Gamma.FromShapeAndRate(4.94065645841247E-324, 4.94065645841247E-324), 0, 1e14);
            Assert.True(g.GetMean() <= g.UpperBound);
            for (int i = 0; i < 308; i++)
            {
                double power = System.Math.Pow(10, i);
                //Trace.WriteLine($"GetMeanPower({power}) = {g.GetMeanPower(power)}");
                Assert.True(g.GetMeanPower(power) <= g.UpperBound);
            }
            Assert.True(g.GetMeanPower(1.7976931348623157E+308) <= g.UpperBound);

            long count = 0;
            Parallel.ForEach(OperatorTests.LowerTruncatedGammas(100000), dist =>
            {
                foreach (var power in OperatorTests.Doubles())
                {
                    if (dist.Gamma.Shape <= -power && dist.LowerBound == 0) continue;
                    double meanPower = dist.GetMeanPower(power);
                    if (power >= 0)
                    {
                        // Compiler.Quoter.Quote(dist)
                        Assert.True(meanPower >= System.Math.Pow(dist.LowerBound, power) || OperatorTests.UlpDiff(meanPower, System.Math.Pow(dist.LowerBound, power)) < 1000);
                        Assert.True(meanPower <= System.Math.Pow(dist.UpperBound, power));
                    }
                    else
                    {
                        Assert.True(meanPower <= System.Math.Pow(dist.LowerBound, power) || OperatorTests.UlpDiff(meanPower, System.Math.Pow(dist.LowerBound, power)) < 1000);
                        Assert.True(meanPower >= System.Math.Pow(dist.UpperBound, power));
                    }
                    if (power == 1)
                    {
                        Assert.Equal(meanPower, dist.GetMean());
                    }
                }
                Interlocked.Add(ref count, 1);
                if (count % 100000 == 0)
                    Trace.WriteLine($"{count} cases passed");
            });
            Trace.WriteLine($"{count} cases passed");
        }

        [Fact]
        public void GaussianTest()
        {
            Gaussian g = new Gaussian();
            g.SetMeanAndVariance(2.2, 3.3);
            double m, v;
            g.GetMeanAndVariance(out m, out v);
            Assert.Equal(2.2, m);
            Assert.Equal(3.3, v);

            foreach (double x in new[] { 1.1, 4.4 })
            {
                double probLessThan = g.GetProbLessThan(x);
                Assert.Equal(MMath.NormalCdf((x - m) / System.Math.Sqrt(v)), probLessThan, 1e-4);
                Assert.Equal(probLessThan, g.GetProbBetween(double.NegativeInfinity, x));
                Assert.Equal(probLessThan, g.GetProbBetween(2*m - x, double.PositiveInfinity));
            }

            Gaussian g2 = new Gaussian(4.4, 5.5);
            /* Test in matlab:
                x = linspace(-100,100,1e6);
                inc = x(2)-x(1);
                m1 = 2.2;
                v1 = 3.3;
                m2 = 4.4;
                v2 = 5.5;
                logsumexp(normpdfln(x,m1,[],v1)+normpdfln(x,m2,[],v2),2)+log(inc)
                    normpdfln(m1,m2,[],v1+v2)
                logsumexp(normpdfln(x,m1,[],v1)-normpdfln(x,m2,[],v2),2)+log(inc)
                    log(v2/(v2-v1)) - normpdfln(m1,m2,[],v2-v1)
                v2 = -5.5
                logsumexp(normpdfln(x,m1,[],v1) -0.5*x.^2/v2+x*(m2/v2),2)+log(inc)
                    -0.5*(m1-m2)^2/(v1+v2) +0.5*log(v2/(v1+v2))
                logsumexp(normpdfln(x,m1,[],v1) +0.5*x.^2/v2-x*(m2/v2),2)+log(inc)
                    0.5*(m1-m2)^2/(v2-v1) +0.5*log(v2/(v2-v1))
                logsumexp(0.5*x.^2/v2-x*(m2/v2),2)+log(inc)
                v1 = -6.6
                logsumexp(-0.5*x.^2/v1+x*m1/v1 +0.5*x.^2/v2-x*(m2/v2),2)+log(inc)
                    0.5*(m1-m2)^2/(v2-v1) +0.5*log(2*pi*v2*v1/(v2-v1))
             */
            // (proper,proper)
            Assert.True(System.Math.Abs(g.GetLogAverageOf(g2) - (-2.28131439394676)) < 1e-8);
            Assert.True(System.Math.Abs(g.GetLogAverageOfPower(g2, -1) - 3.32945794526097) < 1e-8);
            // (proper,improper)
            g2.SetMeanAndVariance(4.4, -5.5);
            Assert.True(System.Math.Abs(g.GetLogAverageOf(g2) - (-0.201854634084064)) < 1e-8);
            Assert.True(System.Math.Abs(g.GetLogAverageOfPower(g2, -1) - 1.249998185356082) < 1e-8);
            // (uniform,improper)
            g.SetToUniform();
            Assert.True(System.Math.Abs(g.GetLogAverageOfPower(g2, -1) - 3.531312579302817) < 1e-8);
            // (improper,improper)
            g.SetMeanAndVariance(2.2, -6.6);
            Assert.True(System.Math.Abs(g.GetLogAverageOfPower(g2, -1) - 6.260525647249976) < 1e-8);
            // (proper,uniform)
            g.SetMeanAndVariance(2.2, 3.3);
            Assert.True(g.GetLogAverageOf(Gaussian.Uniform()) == 0);

            // Checks for infinite means
            Assert.Equal(new Gaussian(double.NegativeInfinity, 2.2), Gaussian.PointMass(double.NegativeInfinity));
            Assert.Equal(new Gaussian(double.PositiveInfinity, 2.2), Gaussian.PointMass(double.PositiveInfinity));
            g.SetMeanAndVariance(double.NegativeInfinity, 2.2);
            Assert.Equal(g, Gaussian.PointMass(double.NegativeInfinity));
            g.SetMeanAndVariance(double.PositiveInfinity, 2.2);
            Assert.Equal(g, Gaussian.PointMass(double.PositiveInfinity));
            g.SetMeanAndPrecision(double.NegativeInfinity, 2.2);
            Assert.Equal(g, Gaussian.PointMass(double.NegativeInfinity));
            g.SetMeanAndPrecision(double.PositiveInfinity, 2.2);
            Assert.Equal(g, Gaussian.PointMass(double.PositiveInfinity));

            double mean = 1024;
            double precision = double.MaxValue / mean * 2;
            // precision * mean > double.MaxValue
            g.SetMeanAndPrecision(mean, precision);
            g2 = Gaussian.FromMeanAndPrecision(mean, double.MaxValue / mean);
            Assert.Equal(g2, g);
            Assert.Equal(g2, new Gaussian(mean, 1 / precision));
            double inverseIsInfinity = 0.5 / double.MaxValue;
            Assert.Equal(Gaussian.PointMass(mean), new Gaussian(mean, inverseIsInfinity));
            Gaussian.FromNatural(1, inverseIsInfinity).GetMeanAndVarianceImproper(out m, out v);
            if (v > double.MaxValue)
                Assert.Equal(0, m);
            Gaussian.Uniform().GetMeanAndVarianceImproper(out m, out v);
            Assert.Equal(0, m);
            Assert.True(double.IsPositiveInfinity(v));

            g.SetMeanAndVariance(2.2, 3.3);
            g2.SetMeanAndVariance(4.4, 5.5);
            DistributionTest(g, g2);
            PointMassTest(g, 7.7);
            UniformTest(g, 7.7);
            SetMomentTest(g, 1.1, 2.2);
            PointMassMomentTest(g, 7.7, 4.4, 5.5);
            SamplingTest(g, 7.7);
            g.SetToUniform();
            Assert.True(g.GetVariance() == double.PositiveInfinity);
            Assert.Throws<ImproperDistributionException>(() => g.GetMean());

            Gaussian g3 = new Gaussian();
            g3.SetToSum(1.0, g, double.PositiveInfinity, g2);
            Assert.True(g3.Equals(g2));
        }

        /// <summary>
        /// Tests that a product of high-precision Gaussians produces a point mass.
        /// </summary>
        [Fact]
        public void GaussianSetToProduct_ProducesPointMassTest()
        {
            GaussianSetToProduct_ProducesPointMass(Gaussian.FromMeanAndPrecision(1, double.MaxValue));
            GaussianSetToProduct_ProducesPointMass(Gaussian.FromMeanAndPrecision(0.9, double.MaxValue));
            GaussianSetToProduct_ProducesPointMass(Gaussian.FromMeanAndPrecision(10, double.MaxValue / 10));
        }

        private void GaussianSetToProduct_ProducesPointMass(Gaussian g)
        {
            Gaussian expected = Gaussian.PointMass(g.GetMean());
            Assert.Equal(expected, g * g);
            Assert.Equal(expected, g ^ 2);
        }

        /// <summary>
        /// Tests that a product of high-precision Gaussians produces a point mass.
        /// </summary>
        [Fact]
        public void GammaSetToProduct_ProducesPointMassTest()
        {
            GammaSetToProduct_ProducesPointMass(Gamma.FromShapeAndRate(double.MaxValue, double.MaxValue));
            GammaSetToProduct_ProducesPointMass(Gamma.FromShapeAndRate(double.MaxValue / 10, double.MaxValue));
            GammaSetToProduct_ProducesPointMass(Gamma.FromShapeAndRate(double.MaxValue, double.MaxValue / 10));
        }

        private void GammaSetToProduct_ProducesPointMass(Gamma g)
        {
            Gamma expected = Gamma.PointMass(g.GetMode());
            Assert.Equal(expected, g * g);
            Assert.Equal(expected, g ^ 2);
        }

        [Fact]
        public void VectorGaussianTest()
        {
            Rand.Restart(12347); // Restart random number generator so as to get a consistent test
            int d = 3;
            VectorGaussian g = new VectorGaussian(d);
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(new double[,] { { 2, 2, 2 }, { 2, 3, 3 }, { 2, 3, 4 } });
            Vector b = Vector.FromArray(new double[] { 1, 2, 3 });
            g.Precision.SetTo(A);
            g.MeanTimesPrecision.SetTo(b);
            Console.WriteLine(StringUtil.JoinColumns("g = ", g));

            // test Equals
            VectorGaussian g1 = new VectorGaussian(d);
            g1.Precision.SetTo(A);
            g1.MeanTimesPrecision.SetTo(b);
            Assert.True(g1.Equals(g));
            Assert.True(g1.GetHashCode() == g.GetHashCode());

            // test Sample
            Vector x = null;
            VectorMeanVarianceAccumulator mva = new VectorMeanVarianceAccumulator(d);
            for (int i = 0; i < nsamples; i++)
            {
                x = g.Sample();
                mva.Add(x);
            }
            //Console.WriteLine("x = {0}", x);
            //Console.WriteLine("eval(x) = {0}", g.GetLogProb(x));
            // x = [-1 1 2]';
            // b = [1 2 3]';
            // a = [2 2 2; 2 3 3; 2 3 4];
            // normpdfln(x, a\b, 'inv', a)
            x = Vector.FromArray(new double[] { -1, 1, 2 });
            Assert.True(System.Math.Abs(g.GetLogProb(x) - (-7.160242009334)) < 1e-10);
            PositiveDefiniteMatrix prec = new PositiveDefiniteMatrix(3, 3);
            prec.SetToInverse(mva.Variance);
            Console.WriteLine(StringUtil.JoinColumns("precision = ", prec.ToString()));
            Console.WriteLine(" (error {0})", prec.MaxDiff(A));
            Assert.True(prec.MaxDiff(A) < 0.1);
            Vector pm = prec * mva.Mean;
            Console.WriteLine("precision * mean = {0}", pm);
            Console.WriteLine(" (error {0})", pm.MaxDiff(b));
            Assert.True(pm.MaxDiff(b) < 0.1);

            // test SetToProduct
            VectorGaussian g2 = new VectorGaussian(d);
            A = new PositiveDefiniteMatrix(new double[,] { { 3, 1, 1 }, { 1, 5, 1 }, { 1, 1, 6 } });
            b = Vector.FromArray(new double[] { 3, 2, 1 });
            g2.Precision.SetTo(A);
            g2.MeanTimesPrecision.SetTo(b);
            Console.WriteLine(StringUtil.JoinColumns("g2 = ", g2));

            g.SetToProduct(g1, g2);
            Console.WriteLine(StringUtil.JoinColumns("g1*g2 = ", g));

            VectorGaussian dx = VectorGaussian.PointMass(x);
            g.SetToProduct(g1, dx);
            Console.WriteLine(StringUtil.JoinColumns("g1*dx = ", g));
            Assert.True(g.IsPointMass);

            g.SetToProduct(dx, g2);
            Console.WriteLine(StringUtil.JoinColumns("dx*g2 = ", g));
            Assert.True(g.IsPointMass);

            DistributionTest(g1, g2);
            PointMassTest(g1, b);
            UniformTest(g1, b);
            SetMomentTest(g2, b, A);
            PointMassMomentTest(g2, x, b, A);
            g2.SetToUniform();
            Assert.True(g2.GetMean().EqualsAll(0.0));
            Assert.True(Double.IsPositiveInfinity(g2.GetVariance()[0, 0]));
            SetMomentTest(g, g.GetMean(), g.GetVariance());

            g = new VectorGaussian(Vector.FromArray(2, 3), new PositiveDefiniteMatrix(new double[,] { { 1, 0 }, { 0, 0 } }));
            Assert.True(Double.IsPositiveInfinity(g.Precision[1, 1]));
            Assert.Equal(3.0, g.Point[1]);

            g = new VectorGaussian(Vector.FromArray(-15.01, 45.69), new PositiveDefiniteMatrix(new double[,] { { 9.067, -13.74 }, { -13.74, 26.56 } }));
            g2 = new VectorGaussian(Vector.FromArray(-14.8, 45.78), new PositiveDefiniteMatrix(new double[,] { { 9.059, -13.69 }, { -13.69, 26.51 } }));
            g.SetToRatio(g, g2, forceProper: true);
            Assert.True(g.Precision.EqualsAll(0.0));
        }

        [Fact]
        public void VectorGaussian_GetMarginal_ReturnsUniform()
        {
            VectorGaussian vg = VectorGaussian.FromNatural(Vector.FromArray(0, 2), new PositiveDefiniteMatrix(new double[,] { { 0, 0 }, { 0, 3 } }));
            Gaussian g = vg.GetMarginal(0);
            Assert.True(g.IsUniform());
            VectorGaussian vg1 = new VectorGaussian(1);
            vg1 = vg.GetMarginal(0, vg1);
            Assert.True(vg1.IsUniform());
            VectorGaussian vg2 = new VectorGaussian(2);
            vg2 = vg.GetMarginal(0, vg2);
            Assert.True(vg.MaxDiff(vg2) < 1e-10);
        }

        [Fact]
        public void VectorGaussianMoments_SetToProduct_Singular()
        {
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(new double[,] {
                { 4.008640513161180,  1.303104352135630, - 2.696380025254830, - 2.728465435435790 },
                { 1.303104352135630,  4.024136989099960, - 2.681070246787840, - 2.713155656968810 },
                { -2.696380025254830, - 2.681070246787840, 4.136120496920130,  1.403451295855420 },
                { -2.728465435435790, - 2.713155656968810, 1.403451295855420,  4.063123100392480 }
            });
            PositiveDefiniteMatrix B = new PositiveDefiniteMatrix(4, 4);
            B.SetToDiagonal(Vector.FromArray(1, 0, 3, 4));

            VectorGaussianMoments vg1 = new VectorGaussianMoments(Vector.FromArray(6, 5, 4, 3), A);
            VectorGaussianMoments vg2 = new VectorGaussianMoments(Vector.FromArray(1, 2, 3, 4), B);
            var product = vg1 * vg2;

            PositiveDefiniteMatrix vExpected = new PositiveDefiniteMatrix(new double[,]
            {
                { 0.699231932932321,  -0.000000000000000 , -0.330187594961638,  -0.385553717718764 },
                { -0.000000000000000,   0.000000000000000,  -0.000000000000002,   0.000000000000002 },
                { -0.330187594961638,  -0.000000000000002,   0.946948669226297,  -0.569413640803344 },
                { -0.385553717718764,   0.000000000000002,  -0.569413640803344 ,  0.926496676481940 },
            });
            Assert.True(vExpected.MaxDiff(product.Variance) < 1e-4);
            Vector meanExpected = Vector.FromArray(2.640276200841019, 2.000000014880260, 6.527941507328482, 6.908179339051594);
            Assert.True(meanExpected.MaxDiff(product.Mean) < 1e-4);

            vg2.Variance.SetToDiagonal(Vector.FromArray(1, double.PositiveInfinity, 3, 4));
            product = vg1 * vg2;
            vExpected.SetTo(new double[,] {
                { 0.703202829692760,  -0.097044359196042,  -0.287002295003263,  -0.335123696183070 },
                { -0.097044359196042,   2.371534574978036,  -1.055338156807388,  -1.232393390393070 },
                { -0.287002295003263,  -1.055338156807388,   1.416576208767429,  -0.020995915719158 },
                { -0.335123696183070,  -1.232393390393070,  -0.020995915719158,   1.566923316764467 },
            });
            Assert.True(vExpected.MaxDiff(product.Variance) < 1e-4);
            meanExpected.SetTo(new double[] { 2.495869677904531, 5.528904975989440, 4.957577684071078, 5.074347889058121 });
            Assert.True(meanExpected.MaxDiff(product.Mean) < 1e-4);
        }

        [Fact]
        public void GammaTest()
        {
            Rand.Restart(12347); // Restart random number generator so as to get a consistent test
            Gamma w = Gamma.FromMeanAndVariance(3, 7);
            Assert.True(MMath.AbsDiff(w.GetMean(), 3, 1e-6) < 1e-10);
            Assert.True(MMath.AbsDiff(w.GetVariance(), 7, 1e-6) < 1e-10);

            GammaMomentTest(Gamma.FromShapeAndRate(1.0/70, System.Math.Exp(-170)));
            double a = 0.7;
            Gamma g = new Gamma();
            g.Shape = a;
            g.Rate = 2;
            GammaMomentTest(g);
            GammaMomentTest(Gamma.FromShapeAndRate(1.7, 2));
            GammaMomentTest(Gamma.PointMass(2.3));

            Assert.Equal(g.GetProbLessThan(0.1), g.GetProbBetween(0, 0.1));
            Assert.Equal(g.GetProbBetween(0, 0.2), g.GetProbBetween(-1, 0.2));

            DistributionTest(g, new Gamma(10, 11));
            PointMassTest(g, 7.7);
            UniformTest(g, 7.7);
            SetMomentTest(g, 2.2, 3.3);
            PointMassMomentTest(g, 7.7, 1.1, 2.2);
            g.SetToUniform();
            GetAndSetMomentTest(g, Double.PositiveInfinity, Double.PositiveInfinity);

            // test for round-off errors
            g.SetShapeAndRate(1e-20, 1e-20);
            GammaRoundoffTest(g);
            //DistributionTest(g, g);
            g.SetShapeAndRate(1e20, 1e20);
            GammaRoundoffTest(g);
            DistributionTest(g, Gamma.FromShapeAndRate(2e20, 2e20));

            Assert.Equal(Gamma.FromMeanAndVariance(double.MaxValue, 1), Gamma.PointMass(double.MaxValue));
            Assert.Equal(Gamma.FromMeanAndVariance(double.MaxValue / 1e100, 1e-100), Gamma.PointMass(double.MaxValue / 1e100));
            Assert.Equal(Gamma.FromMeanAndVariance(1, double.Epsilon), Gamma.PointMass(1));
            Assert.Equal(Gamma.PointMass(0), Gamma.FromShapeAndRate(2.5, double.PositiveInfinity));
            Assert.Equal(Gamma.PointMass(0), Gamma.FromShapeAndScale(2.5, 1e-320));
            Assert.Equal(Gamma.PointMass(0), new Gamma(2.5, 1e-320));
            Assert.Equal(Gamma.PointMass(0), new Gamma(2.5, 0));
            Assert.Equal(Gamma.PointMass(1e-300), Gamma.FromShapeAndRate(2, 1e300) ^ 1e10);

            ProductWithUniformTest(g);
            Gamma g2 = new Gamma();
            g2.SetToPower(g, 1.0);
            Assert.True(g2.MaxDiff(g) < 1e-12);

            Assert.Equal(double.NegativeInfinity, g.GetLogProb(double.PositiveInfinity));
            Assert.Equal(double.NegativeInfinity, g.GetLogAverageOf(Gamma.PointMass(double.PositiveInfinity)));
            Assert.Equal(double.NegativeInfinity, g.GetLogProb(0));
            g.Shape = 1;
            g.Rate = 2;
            Assert.Equal(-g.GetLogNormalizer(), g.GetLogProb(0));
            g.Rate = 0;
            Assert.Equal(0, g.GetLogProb(double.PositiveInfinity));
            Assert.Equal(0, g.GetLogProb(0));
        }

        private static void GammaRoundoffTest(Gamma a)
        {
            ProductWithUniformTest(a);
            RatioWithUniformTest(a);
            Gamma c = (a / a);
            Assert.True(c.IsUniform());
            c.SetToPower(a, 1.0);
            Assert.True(a.MaxDiff(c) < 1e-12);
            c.SetToPower(a, 0.0);
            Assert.True(c.IsUniform());
            LogAverageOfPowerTest(a, a);
        }

        [Fact]
        public void GammaSetToPowerTest()
        {
            double[,] gammaSetToPower_pairs = {
                { 1e-20, 1e-20, 0.99999999999999999999 },
                { 1e-20, 1e-10, 0.999999999899999999999999996 },
                { 1e-20, 0.1, 0.89999999999999999 },
                { 1e-20, 1, 1e-20 },
                { 1e-20, 1.1, -0.1000000000000000888 },
                { 1e-20, 1e4, -9998.9999999999999999 },
                { 1e-20, 1e16, -9999999999999998.9999 },
                { 1e-20, -1, 1.99999999999999999999 },
                { 1, 1e-20, 1 },
                { 1, 1e-10, 1 },
                { 1, 1, 1 },
                { 1, 1e4, 1 },
                { 1, 1e10, 1 },
                { 1, 1e16, 1 },
                { 1, 1e20, 1 },
                { 1, -1e20, 1 },
                { 1, -1e-20, 1 },
                { 1e4, 1e4, 99990001.0 },
                { 1e4, 1e20, 999900000000000000000001.0 },
            };

            for (int i = 0; i < gammaSetToPower_pairs.GetLength(0); i++)
            {
                double a = gammaSetToPower_pairs[i, 0];
                double e = gammaSetToPower_pairs[i, 1];
                double a2 = gammaSetToPower_pairs[i, 2];
                Gamma g = Gamma.FromShapeAndRate(a, 1);
                g.SetToPower(g, e);
                Assert.True(MMath.AbsDiff(a2, g.Shape, 1e-10) < 1e-10);
            }
        }

        private void GammaMomentTest(Gamma g)
        {
            Rand.Restart(12347); // Restart random number generator so as to get a consistent test
            double m = 0, s = 0, minv = 0, m2 = 0;
            for (int i = 0; i < nsamples; i++)
            {
                double x = g.Sample();
                m = m + x;
                s = s + System.Math.Log(x);
                minv = minv + 1.0 / x;
                m2 = m2 + x * x;
            }
            m = m / nsamples;
            s = s / nsamples;
            minv = minv / nsamples;
            m2 = m2 / nsamples;
            double mTrue = g.GetMean();
            double sTrue = g.GetMeanLog();
            double m2True = g.GetMeanPower(2);
            double mError = MMath.AbsDiff(m, mTrue, 1e-6);
            Assert.True(mError < 0.05, $"|m - mTrue| = {mError}");
            if (g.Shape > 0.1)
            {
                double sError = MMath.AbsDiff(s, sTrue, 1e-6);
                Assert.True(sError < 0.05, $"|s - sTrue| = {sError}");
            }
            if (g.Shape > 1)
            {
                double minvTrue = g.GetMeanInverse();
                double minvError = MMath.AbsDiff(minv, minvTrue, 1e-6);
                Assert.True(minvError < 0.05, $"|minv - minvTrue| = {minvError}");
                Assert.True(System.Math.Abs(g.GetMeanPower(-1) - minvTrue) < 1e-10);
            }
            double m2Error = MMath.AbsDiff(m2, m2True, 1e-6);
            Assert.True(m2Error < 0.05, $"|m2 - m2True| = {m2Error}");
        }

        private void TruncatedGammaMomentTest(TruncatedGamma g)
        {
            Rand.Restart(123478); // Restart random number generator so as to get a consistent test
            double m = 0, s = 0, minv = 0, m2 = 0;
            for (int i = 0; i < nsamples; i++)
            {
                double x = g.Sample();
                m = m + x;
                s = s + System.Math.Log(x);
                minv = minv + 1.0 / x;
                m2 = m2 + x * x;
            }
            m = m / nsamples;
            s = s / nsamples;
            minv = minv / nsamples;
            m2 = m2 / nsamples;
            double mTrue = g.GetMean();
            //double sTrue = g.GetMeanLog();
            double m2True = g.GetMeanPower(2);
            Console.WriteLine("|m - mTrue| = {0}", System.Math.Abs(m - mTrue));
            Assert.True(System.Math.Abs(m - mTrue) < 0.05);
            //Console.WriteLine("|s - sTrue| = {0}", System.Math.Abs(s - sTrue));
            //Assert.True(System.Math.Abs(s - sTrue) < 0.05);
            if (g.Gamma.Shape > 1)
            {
                double minvTrue = g.GetMeanPower(-1);
                Console.WriteLine("|minv - minvTrue| = {0}", System.Math.Abs(minv - minvTrue));
                Assert.True(System.Math.Abs(minv - minvTrue) < 0.05);
                Assert.True(System.Math.Abs(g.GetMeanPower(-1) - minvTrue) < 1e-10);
            }
            Console.WriteLine("|m2 - m2True| = {0}", System.Math.Abs(m2 - m2True));
            Assert.True(System.Math.Abs(m2 - m2True) < 0.05);
        }

        [Fact]
        public void WishartTest()
        {
            Rand.Restart(12347); // Restart random number generator so as to get a consistent test
            int d = 3;
            double a = 1.7 + 2;
            Wishart g = Wishart.Uniform(d);
            Assert.True(g.IsUniform());
            PositiveDefiniteMatrix B = new PositiveDefiniteMatrix(new double[,] { { 2, 1, 1 }, { 1, 3, 1 }, { 1, 1, 4 } });
            g.Rate = B;
            g.Shape = a;

            // test Equals
            Wishart g1 = new Wishart(d);
            g1.Rate = B;
            g1.Shape = a;
            Assert.True(g1.Equals(g));
            Assert.True(g1.GetHashCode() == g.GetHashCode());

            // test Sample and WishartEstimator
            PositiveDefiniteMatrix X = null;
            MatrixMeanVarianceAccumulator mva = new MatrixMeanVarianceAccumulator(d, d);
            WishartEstimator est = new WishartEstimator(g.Dimension);
            double s = 0;
            for (int i = 0; i < nsamples; i++)
            {
                X = g.Sample();
                mva.Add(X);
                s = s + X.LogDeterminant();
                est.Add(X);
            }
            double sTrue = g.GetMeanLogDeterminant();
            s = s / nsamples;

            Console.WriteLine(StringUtil.JoinColumns("X = ", X));
            Console.WriteLine("eval(dx) = {0}", g.GetLogProb(X));
            Console.WriteLine("|s - sTrue| = {0}", System.Math.Abs(s - sTrue));
            Assert.True(System.Math.Abs(s - sTrue) < 0.05);
            PositiveDefiniteMatrix mTrue = new PositiveDefiniteMatrix(d, d);
            PositiveDefiniteMatrix vTrue = new PositiveDefiniteMatrix(d, d);
            g.GetMeanAndVariance(mTrue, vTrue);
            Assert.True(mTrue.MaxDiff(g.GetMean()) < 1e-10);
            Assert.True(vTrue.MaxDiff(g.GetVariance()) < 1e-10);
            Console.WriteLine("|m - mTrue| = {0}", mTrue.MaxDiff(mva.Mean));
            Assert.True(mTrue.MaxDiff(mva.Mean) < 0.05);
            Assert.True(vTrue.MaxDiff(mva.Variance) < 0.05);
            est.GetDistribution(g1);
            Console.WriteLine(g1);
            Assert.True(g.MaxDiff(g1) < 0.05);

            // test SetToProduct
            Wishart g2 = new Wishart(d);
            B = new PositiveDefiniteMatrix(new double[,] { { 3, 1, 1 }, { 1, 3, 1 }, { 1, 1, 3 } });
            g2.Rate = B;
            g2.Shape = 3.3 + 2;
            Console.WriteLine(StringUtil.JoinColumns("g2 = ", g2));

            g.SetToProduct(g1, g2);
            Console.WriteLine(StringUtil.JoinColumns("g1*g2 = ", g));

            Wishart dx = Wishart.PointMass(B);
            g.SetToProduct(g1, dx);
            Console.WriteLine(StringUtil.JoinColumns("g1*dx = ", g));
            Assert.True(g.IsPointMass);

            g.SetToProduct(dx, g2);
            Console.WriteLine(StringUtil.JoinColumns("dx*g2 = ", g));
            Assert.True(g.IsPointMass);

            // test Get/SetMeanAndVariance for a point mass
            PositiveDefiniteMatrix V = new PositiveDefiniteMatrix(d, d);
            g.GetMeanAndVariance(mTrue, vTrue);
            g.SetMeanAndVariance(mTrue, vTrue);
            g.GetMeanAndVariance(B, V);
            Assert.True(mTrue.MaxDiff(B) < 1e-10);
            Assert.True(vTrue.MaxDiff(V) < 1e-10);

            g.Rate = B;
            g.Shape = a;
            g.GetMeanAndVariance(mTrue, vTrue);
            g.SetMeanAndVariance(mTrue, vTrue);
            g.GetMeanAndVariance(B, V);
            Assert.True(mTrue.MaxDiff(B) < 1e-10);
            Assert.True(vTrue.MaxDiff(V) < 1e-10);

            DistributionTest(g, g2);
            PointMassTest(g, B);
            UniformTest(g, B);
            SetMomentTest(g, mTrue, vTrue);
            PointMassMomentTest(g, B, B, V);
            g.SetToUniform();
            Assert.True(g.GetMean().EqualsAll(Double.PositiveInfinity));
            Assert.True(Double.IsPositiveInfinity(g.GetVariance()[0, 0]));
            SetMomentTest(g, g.GetMean(), g.GetVariance());

            // test for round-off errors
            DistributionTest(new Wishart(1e-20, 1e+20), new Wishart(10, 11));
        }

        [Fact]
        public void DistributionArrayConversionTest()
        {
            Gaussian[] ga = new Gaussian[] { new Gaussian(0, 1) };
            IDistribution<double[]> dga = Distribution<double>.Array(ga);
            Gaussian[] ga2 = Distribution.ToArray<Gaussian[]>(dga);
            IDistribution<double[]> dga2 = Distribution<double>.Array(ga2);
            Assert.Equal(dga, dga2);

            Gaussian[,] ga2D = new Gaussian[,] { { new Gaussian(0, 1) } };
            IDistribution<double[,]> dga2D = Distribution<double>.Array(ga2D);
            Gaussian[,] ga2D2 = Distribution.ToArray<Gaussian[,]>(dga2D);
            IDistribution<double[,]> dga2D2 = Distribution<double>.Array(ga2D2);
            Assert.Equal(dga2D, dga2D2);

            Gaussian[][] gaa = new Gaussian[][] { new Gaussian[] { new Gaussian(0, 1) } };
            IDistribution<double[][]> dgaa = Distribution<double>.Array(gaa);
            Gaussian[][] gaa2 = Distribution.ToArray<Gaussian[][]>(dgaa);
            IDistribution<double[][]> dgaa2 = Distribution<double>.Array(gaa2);
            Assert.Equal(dgaa, dgaa2);

            Gaussian[,][] ga2Da = new Gaussian[,][] { { new Gaussian[] { new Gaussian(0, 1) } } };
            IDistribution<double[,][]> dga2Da = Distribution<double>.Array(ga2Da);
            Gaussian[,][] ga2Da2 = Distribution.ToArray<Gaussian[,][]>(dga2Da);
            IDistribution<double[,][]> dga2Da2 = Distribution<double>.Array(ga2Da2);
            Assert.Equal(dga2Da, dga2Da2);

            Gaussian[][,] gaa2D = new Gaussian[][,] { new Gaussian[,] { { new Gaussian(0, 1) } } };
            IDistribution<double[][,]> dgaa2D = Distribution<double>.Array(gaa2D);
            Gaussian[][,] gaa2D2 = Distribution.ToArray<Gaussian[][,]>(dgaa2D);
            IDistribution<double[][,]> dgaa2D2 = Distribution<double>.Array(gaa2D2);
            Assert.Equal(dgaa2D, dgaa2D2);

            Gaussian[][][] gaaa = new Gaussian[][][] { new Gaussian[][] { new Gaussian[] { new Gaussian(0, 1) } } };
            IDistribution<double[][][]> dgaaa = Distribution<double>.Array(gaaa);
            Gaussian[][][] gaaa2 = Distribution.ToArray<Gaussian[][][]>(dgaaa);
            IDistribution<double[][][]> dgaaa2 = Distribution<double>.Array(gaaa2);
            Assert.Equal(dgaaa, dgaaa2);
        }

#pragma warning disable CA2013
        [Fact]
        public void GaussianArrayTest()
        {
            Gaussian g = new Gaussian(0, 1);
            GaussianArray2D a = new GaussianArray2D(new Gaussian(), 3, 2);
            Assert.False(object.ReferenceEquals(a[0], a[1]));
            a.ForEach(delegate (Gaussian item)
            { Assert.True(item.IsUniform()); });
            //GaussianArray a = new GaussianArray(new Gaussian[3,2]);
            //a.ModifyAll(delegate(Gaussian item) { return new Gaussian(); });
            a.SetAllElementsTo(g);
            a[0] = new Gaussian(0, 2);
            Console.WriteLine("a[0] = " + a[0]);
            foreach (Gaussian item in a)
            {
                Console.WriteLine(item);
            }
            Console.WriteLine(a);
            Assert.Equal(g, a[1]);
            Assert.Equal(g, a[0, 1]);
            //Assert.Equal(g, a[new int[] { 0, 1 }]);
            Assert.Equal(1, a.IndexOf(g));
            Assert.Contains(g, a);
            Assert.Equal(1, a.FindIndex(delegate (Gaussian item)
            { return (item.GetVariance() == 1); }));
            GaussianArray2D b = (GaussianArray2D)a.Clone();
            Assert.False(object.ReferenceEquals(b[0], a[0]));
            b.SetTo(a);
            Assert.False(object.ReferenceEquals(b[0], a[0]));
            a.SetToProduct(a, a);
            Assert.True(a[0].MaxDiff(Gaussian.FromMeanAndPrecision(0, 1)) < 1e-10);
            Assert.True(a[1].MaxDiff(Gaussian.FromMeanAndPrecision(0, 2)) < 1e-10);

            DistributionTest(a, new GaussianArray2D(new Gaussian(4, 5), 3, 2));

            a.SetToUniform();
            a.ForEach(delegate (Gaussian item)
            { Assert.True(item.IsUniform()); });
            //UniformTest(a,new double[3,2]);

            b = new GaussianArray2D(new Gaussian(), 3, 3);
            Assert.True(!a.Equals(b));
        }
#pragma warning restore

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

#if false
        [Fact]
        public void GaussianCursorArrayTest()
        {
            VectorGaussian g = new VectorGaussian(3);
            GaussianCursorArray a = new GaussianCursorArray(g, 3, 2);
            int i = 0;
            foreach (VectorGaussian item in a) {
                item.MeanTimesPrecision.SetAllElementsTo(i++);
                item.Precision.SetToIdentity();
            }

            VectorGaussian g2 = new VectorGaussian(3);
            int[] keepAll = new int[] { 0, 1 };
            GaussianCursorArray a2 = new GaussianCursorArray(g2, 3, 2);
            a.ReduceTo(keepAll, a2);
            Console.WriteLine("ReduceTo[0,1]:");
            Console.WriteLine(a2);

            VectorGaussian g3 = new VectorGaussian(3);
            int[] keep1 = new int[] { 0 };
            GaussianCursorArray a3 = new GaussianCursorArray(g3, 3);
            a.ReduceTo(keep1, a3);
            Console.WriteLine("ReduceTo[0]:");
            Console.WriteLine(a3);
            Assert.True(a3[0].MeanTimesPrecision[0] == 1);
            Assert.True(a3[1].MeanTimesPrecision[0] == 5);
            Assert.True(a3[2].MeanTimesPrecision[0] == 9);

            int[] keep2 = new int[] { 1 };
            a3 = new GaussianCursorArray(g3, 2);
            a.ReduceTo(keep2, a3);
            Console.WriteLine("ReduceTo[1]:");
            Console.WriteLine(a3);
            Assert.True(a3[0].MeanTimesPrecision[0] == 6);
            Assert.True(a3[1].MeanTimesPrecision[0] == 9);

            a.ReduceTo(g3);
            Console.WriteLine("ReduceTo[]:");
            Console.WriteLine(g3);
            Assert.True(g3.MeanTimesPrecision[0] == 15);

            a.SetToProduct(a, a);
            Console.WriteLine("SetToProduct:");
            Console.WriteLine(a);

#if false
            a2.Point = a.Sample();
            Console.WriteLine("Sample:");
            Console.WriteLine(a2);
#endif

            a2.SetTo(a);
            Console.WriteLine("SetTo:");
            Console.WriteLine(a);
            Assert.True(a.Equals(a2));
            Assert.True(a.MaxDiff(a2) == 0);

            GaussianCursorArray a3r = a3.Replicate(new int[] { 3, 2 }, new int[] { 1 });
            Console.WriteLine("Replicate[1]:");
            Console.WriteLine(a3r);

            DistributionCursorArray<GaussianCursorArray, Vector[]> jagged;
            try {
                jagged = a.Split(new bool[] { false, false });
                Debug.Fail("Split[F,F] did not fail");
            } catch (ArgumentException ex) {
                Console.WriteLine("Split[F,F] correctly failed with error: " + ex);
            }
            try {
                jagged = a.Split(new bool[] { true, true });
                Debug.Fail("Split[T,T] did not fail");
            } catch (ArgumentException ex) {
                Console.WriteLine("Split[T,T] correctly failed with error: " + ex);
            }
            jagged = a.Split(new bool[] { true, false });
            Console.WriteLine("Split[T,F]:");
            Console.WriteLine(jagged);
            jagged = a.Split(new bool[] { false, true });
            Console.WriteLine("Split[F,T]:");
            Console.WriteLine(jagged);
        }
#endif

        [Fact]
        public void BernoulliTest()
        {
            Assert.Equal(0.0, Bernoulli.LogitProbEqual(0, 0));
            Assert.Equal(0.0, Bernoulli.LogitProbEqual(0, Double.PositiveInfinity));
            Assert.Equal(Double.PositiveInfinity, Bernoulli.LogitProbEqual(Double.PositiveInfinity, Double.PositiveInfinity));
            Assert.Equal(Double.NegativeInfinity, Bernoulli.LogitProbEqual(Double.PositiveInfinity, Double.NegativeInfinity));
            Assert.Equal(Double.PositiveInfinity, Bernoulli.LogitProbEqual(Double.NegativeInfinity, Double.NegativeInfinity));

            Bernoulli b = new Bernoulli(0.1);
            DistributionTest(b, new Bernoulli(0.9));
            PointMassTest(b, true);
            UniformTest(b, true);
            b.SetMean(0.1);
            Assert.True(AbsDiff(0.1, b.GetMean()) < 1e-10);
            b.Point = false;
            Assert.True(AbsDiff(0.0, b.GetMean()) < 1e-10);
            b.SetMean(0.0);
            Assert.True(AbsDiff(0.0, b.GetMean()) < 1e-10);
            Assert.True(b.IsPointMass);
            Assert.False(b.Point);

            // Test the LogitProbEqual calculation
            double logodds1 = -2.0;
            double logodds2 = 0.5;
            Bernoulli b1 = Bernoulli.FromLogOdds(logodds1);
            Bernoulli b2 = Bernoulli.FromLogOdds(logodds2);
            double probtrue1 = b1.GetProbTrue();
            double probtrue2 = b2.GetProbTrue();
            double probx1Eqx2 = (probtrue1 * probtrue2) + (1 - probtrue1) * (1 - probtrue2);
            double logoddsCalc = MMath.Logit(probx1Eqx2);
            double logoddsFunc = Bernoulli.LogitProbEqual(logodds1, logodds2);
            Assert.True(AbsDiff(logoddsCalc, logoddsFunc) < 1e-10);
        }

        [Fact]
        public void DiscreteTest()
        {
            Assert.True(new Discrete(0, 0.71743990380862455, 0.2825600961913754).GetLogAverageOf(Discrete.Uniform(3)) == -System.Math.Log(3));

            Discrete d = new Discrete(0.9, 0.1);
            DistributionTest(d, new Discrete(0.2, 0.8));
            PointMassTest(d, 1);
            UniformTest(d, 1);
            PointMassMomentTest(d, 1, 0.2, 0.1);
            d.SetToUniform();
            GetMomentTest(d, 0.5, 0.25);
            SamplingTest(d, 1);

            Discrete d1 = new Discrete(0.1, 0.9, 0, 0);
            Discrete d2 = new Discrete(0, 0, 0.3, 0.7);
            Assert.True(MMath.AbsDiff(d2.GetAverageLog(d2), 0.3 * System.Math.Log(0.3) + 0.7 * System.Math.Log(0.7), 1 - 6) < 1e-10);
            Discrete d3 = d1 * d2;
            Assert.True(d3.IsZero());
            Assert.False(d3.IsPointMass);
            Discrete zero = new Discrete(0, 0, 0);
            Assert.True(zero.IsZero());
            Assert.False(zero.IsPointMass);
            zero = Discrete.Zero(3);
            Assert.True(zero.IsZero());
            Assert.False(zero.IsPointMass);
            d.SetProbs(Vector.FromArray(0.0, 0.0));
            Assert.True(d.IsZero());
            Assert.False(d.IsPointMass);
            Assert.Equal(double.NegativeInfinity, d.GetLogAverageOf(d));

            d = new Discrete(1.0);
            Assert.Equal(0, d.Sample());
            Assert.Equal(0, d.Sample(4));
            Vector p = d.GetWorkspace();
            p[0] = 2.0;
            d.SetProbs(p);
            Assert.Equal(0.0, d.MaxDiff(Discrete.Uniform(1)));
            d = Discrete.PointMass(0, 1);
            DistributionTest(d, new Discrete(1.0));
            UniformTest(d, 0);
            PointMassMomentTest(d, 0, 0.2, 0.1);
            d.SetToUniform();
            GetMomentTest(d, 0.0, 0.0);

            d = Discrete.Uniform(0);
            DistributionTest(d, new Discrete());

            // Sampling when probability vector is sparse
            double[] pcd1 = new double[] { 0.1, 0.3, 0.4, 0.1, 0.1 };
            double[] pcd2 = new double[] { 0.0, 0.4, 0.6, 0.0, 0.0 };
            Vector[] pcv1 = new Vector[4];
            Vector[] pcv2 = new Vector[4];
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            pcv1[0] = Vector.FromArray(pcd1);
            pcv1[1] = Vector.FromArray(pcd1, Sparsity.Sparse);
            pcv1[2] = Vector.FromArray(pcd1, approxSparsity);
            pcv1[3] = DenseVector.FromArrayReference(3, pcd1, 2);
            pcv2[0] = Vector.FromArray(pcd2);
            pcv2[1] = Vector.FromArray(pcd2, Sparsity.Sparse);
            pcv2[2] = Vector.FromArray(pcd2, approxSparsity);
            pcv2[3] = DenseVector.FromArrayReference(3, pcd2, 2);
            for (int i = 0; i < pcv1.Length; i++)
            {
                SamplingTest(new Discrete(pcv1[i]), 1);
                SamplingTest(new Discrete(pcv2[i]), 1);
            }

            d = new Discrete(1.0);
            Assert.Equal(0, d.GetMedian());
            d = new Discrete(0.1, 0.2, 0.0, 0.7);
            Assert.Equal(3, d.GetMedian());

            // Test partial uniform
            d = new Discrete(SparseVector.FromArray(0.0, 0.5, 0.5, 0.0));
            Assert.True(d.IsPartialUniform());
            d.SetToUniform();
            Assert.True(d.IsPartialUniform());
            d = new Discrete(SparseVector.FromArray(0.0, 0.3, 0.7, 0.0));
            Assert.False(d.IsPartialUniform());
            d.SetToPartialUniformOf(d);
            Assert.True(d.IsPartialUniform());
            d = new Discrete(SparseVector.FromArray(0.0, 0.3, 0.6, 0.1));
            Assert.False(d.IsPartialUniform());
            d.Point = 1;
            Assert.True(d.IsPartialUniform());
            d2 = Discrete.PointMass(2, 4);
            d.SetToPartialUniformOf(d2);
            Assert.True(d.IsPartialUniform());
            Assert.True(d.IsPointMass && d.Point == 2);
        }

        [Fact]
        public void DiscreteTruncateTest()
        {
            Discrete d = new Discrete(SparseVector.FromArray(0.1, 0.0, 0.3, 0.6));
            Discrete truncated = d.Truncate(0, 2);
            Assert.True(truncated.Dimension == 3);
            truncated = d.Truncate(1, 2);
            Assert.True(truncated.IsPointMass);
            Assert.True(truncated.Point == 2);
            truncated = d.Truncate(2, 2);
            Assert.True(truncated.IsPointMass);
            Assert.True(truncated.Point == 2);
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void DirichletTest()
        {
            Vector mTrue = Vector.FromArray(0.1, 0.2, 0.3, 0.4);
            Dirichlet d = new Dirichlet(mTrue * 7);
            Vector vTrue = Vector.Zero(d.Dimension);
            Vector m = Vector.Zero(d.Dimension);
            Vector v = Vector.Zero(d.Dimension);
            d.GetMeanAndVariance(mTrue, vTrue);
            d.SetMeanAndVariance(mTrue, vTrue);
            d.GetMeanAndVariance(m, v);
            Assert.True(mTrue.MaxDiff(m) < 1e-10);
            Assert.True(vTrue.MaxDiff(v) < 1e-10);

            DistributionTest(d, Dirichlet.Symmetric(d.Dimension, 7.7));
            using (TestUtils.TemporarilyAllowDirichletImproperSums)
            {
                DistributionTest(d, Dirichlet.Symmetric(d.Dimension, 7.7));
            }
            PointMassTest(d, mTrue);
            UniformTest(d, mTrue);
            SetMomentTest(d, mTrue, vTrue);
            PointMassMomentTest(d, mTrue, m, vTrue);
            d.SetToUniform();
            GetAndSetMomentTest(d,
                                Vector.Constant(4, 0.25),
                                Vector.Constant(4, 0.25 * 0.75 / 5));

            DirichletSamplingTest(d);

            // Small pseudocount test
            DirichletSamplingTest(new Dirichlet(1e-5, 2e-5, 3e-5, 4e-5));

            // Small/large pseudocount test
            DirichletSamplingTest(new Dirichlet(1e-5, 2e5, 3e5, 4e-5));

            // Sparse vector pseudocount tests
            double[] pcd1 = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2 };
            double[] pcd2 = new double[] { 0.0, 2.3, 3.4, 0.0, 0.0 };
            Vector[] pcv1 = new Vector[4];
            Vector[] pcv2 = new Vector[4];
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            pcv1[0] = Vector.FromArray(pcd1);
            pcv1[1] = Vector.FromArray(pcd1, Sparsity.Sparse);
            pcv1[2] = Vector.FromArray(pcd1, approxSparsity);
            pcv1[3] = DenseVector.FromArrayReference(3, pcd1, 2);
            pcv2[0] = Vector.FromArray(pcd2);
            pcv2[1] = Vector.FromArray(pcd2, Sparsity.Sparse);
            pcv2[2] = Vector.FromArray(pcd2, approxSparsity);
            pcv2[3] = DenseVector.FromArrayReference(3, pcd2, 2);
            for (int i = 0; i < pcv1.Length; i++)
            {
                DirichletSamplingSparseTest(new Dirichlet(pcv1[i]));
                DirichletSamplingSparseTest(new Dirichlet(pcv2[i]));
            }
        }

        private static void DirichletSamplingTest(Dirichlet d)
        {
            Rand.Restart(12347); // Restart random number generator so as to get a consistent test
            Vector sample = Vector.Zero(d.Dimension);
            Vector meanSample = Vector.Zero(d.Dimension);
            Vector mean2Sample = Vector.Zero(d.Dimension);
            Vector mean3Sample = Vector.Zero(d.Dimension);
            Vector meanLogSample = Vector.Zero(d.Dimension);
            Vector temp = Vector.Zero(d.Dimension);
            for (int i = 0; i < nsamples; i++)
            {
                d.Sample(sample);
                meanSample.SetToSum(meanSample, sample);
                mean2Sample.SetToSum(mean2Sample, temp.SetToFunction(sample, x => x * x));
                mean3Sample.SetToSum(mean3Sample, temp.SetToFunction(sample, x => x * x * x));
                meanLogSample.SetToSum(meanLogSample, temp.SetToFunction(sample, System.Math.Log));
            }
            meanSample.Scale(1.0 / nsamples);
            mean2Sample.Scale(1.0 / nsamples);
            mean3Sample.Scale(1.0 / nsamples);
            meanLogSample.Scale(1.0 / nsamples);
            Vector mean = d.GetMean();
            Console.WriteLine("mean = {0} should be {1}", meanSample, mean);
            Assert.True(mean.MaxDiff(meanSample) < 5e-3);
            Vector mean2 = d.GetMeanSquare();
            Console.WriteLine("mean2 = {0} should be {1}", mean2Sample, mean2);
            Assert.True(mean2.MaxDiff(mean2Sample) < 5e-3);
            Vector mean3 = d.GetMeanCube();
            Console.WriteLine("mean3 = {0} should be {1}", mean3Sample, mean3);
            Assert.True(mean3.MaxDiff(mean3Sample) < 5e-3);
            Vector meanLog = d.GetMeanLog();
            Console.WriteLine("meanLog = {0} should be {1}", meanLogSample, meanLog);
            if (meanLog.Min() > -70)
            {
                Assert.True(meanLog.MaxDiff(meanLogSample) < 1e-2);
            }
        }

        private static void DirichletSamplingSparseTest(Dirichlet d)
        {
            Rand.Restart(12347); // Restart random number generator so as to get a consistent test
            Vector v = d.PseudoCount;
            Vector result = null;

            // Check both versions
            for (int pass = 0; pass < 2; pass++)
            {
                if (pass == 0)
                {
                    result = Vector.Constant(v.Count, 1.234, v.Sparsity);
                    d.Sample(result);
                }
                else
                    result = d.Sample();

                // Check that if any pseudo-counts are 0, then the sample is 0
                // Check for non zero that there are no repeat elements in the sample
                Dictionary<double, bool> dic = new Dictionary<double, bool>();
                for (int i = 0; i < v.Count; i++)
                {
                    double r = result[i];
                    if (v[i] == 0.0)
                        Assert.Equal(0.0, r);
                    else
                    {
                        // Here we are testing that a sparse Dirichlet is not generating the same
                        // sample value for all common value indices
                        if (dic.ContainsKey(r))
                            Assert.True(false, "Dirichlet sampler: repeat element");
                        dic[r] = true;
                    }
                }
            }
        }

        [Fact]
        public void DirichletEstimateNewtonTest()
        {
            Vector alphaTrue = Vector.FromArray(0.1, 0.2, 0.3, 0.4);
            alphaTrue.Scale(7);
            DirichletEstimateNewton(alphaTrue);
        }

        private void DirichletEstimateNewton(Vector alphaTrue)
        {
            Dirichlet d = new Dirichlet(alphaTrue);
            Vector meanLog = d.GetMeanLog();
            Console.WriteLine(meanLog);
            d.SetToUniform();
            d.SetMeanLog(meanLog);
            Console.WriteLine("pseudoCount = {0} should be {1}", d.PseudoCount, alphaTrue);
            Assert.True(alphaTrue.MaxDiff(d.PseudoCount) < 1e-6);
        }

        [Fact]
        public void DirichletModeTest()
        {
            Dirichlet d = null;
            Vector v = null;
            for (int iter = 0; iter < 7; iter++)
            {
                if (iter == 0)
                {
                    d = new Dirichlet(0.1, 0.3, 0.4);
                    v = Vector.FromArray(0, 0, 1);
                }
                else if (iter == 1)
                {
                    d = new Dirichlet(1, 0.3, 0.4);
                    v = Vector.FromArray(1, 0, 0);
                }
                else if (iter == 2)
                {
                    d = new Dirichlet(1.1, 0.3, 0.4);
                    v = Vector.FromArray(1, 0, 0);
                }
                else if (iter == 3)
                {
                    d = new Dirichlet(0.3, 1, 1);
                    v = Vector.FromArray(0, 0.5, 0.5);
                }
                else if (iter == 4)
                {
                    d = new Dirichlet(1, 1, 1);
                    v = Vector.FromArray(1.0 / 3, 1.0 / 3, 1.0 / 3);
                }
                else if (iter == 5)
                {
                    d = new Dirichlet(1.3, 1, 1);
                    v = Vector.FromArray(1, 0, 0);
                }
                else if (iter == 6)
                {
                    d = new Dirichlet(3, 4, 5);
                    double denom = 3 + 4 + 5 - 3;
                    v = Vector.FromArray(2.0 / denom, 3.0 / denom, 4.0 / denom);
                }
                Vector mode = d.GetMode();
                Console.WriteLine("{0}: {1} (should be {2})", d, mode, v);
                Assert.True(mode.MaxDiff(v) < 1e-10);
            }
        }

        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        public void BetaTest()
        {
            Beta d = new Beta(0.2, 0.1);
            Assert.Equal(d.GetProbLessThan(0.1), d.GetProbBetween(0, 0.1));
            Assert.Equal(d.GetProbBetween(0, 0.2), d.GetProbBetween(-1, 0.2));
            DistributionTest(d, new Beta(4.4, 3.3));
            using (TestUtils.TemporarilyAllowBetaImproperSums)
            {
                DistributionTest(d, new Beta(4.4, 3.3));
            }
            PointMassTest(d, 0.7);
            UniformTest(d, 0.7);
            PointMassMomentTest(d, 0.6, 0.2, 0.1);
            d.SetToUniform();
            GetAndSetMomentTest(d, 0.5, 1.0 / 12);
            SamplingTest(d, 0.7);
            SamplingTest(new Beta(1e-5, 2e-5), 0.7);
        }

        internal void BetaEstimateNewtonTest()
        {
            Beta d = new Beta(0.2, 0.1);
            double eLogP, eLogOneMinusP;
            d.GetMeanLogs(out eLogP, out eLogOneMinusP);
            Beta d2 = Beta.FromMeanLogs(eLogP, eLogOneMinusP);
            Console.WriteLine("FromMeanLogs = {0} should be {1}", d2, d);
            Assert.True(d.MaxDiff(d2) < 1e-6);
        }

        internal void BetaFromDerivativesTest()
        {
            Beta d = new Beta(0.2, 0.1);
            double x = 0.3;
            double delta = 1e-7;
            double dLogP, ddLogP;
            dLogP = (d.GetLogProb(x + delta) - d.GetLogProb(x - delta)) / (2 * delta);
            //ddLogP = (d.GetLogProb(x + delta) - 2*d.GetLogProb(x) + d.GetLogProb(x - delta)) / (delta*delta);
            ddLogP = -(d.TrueCount - 1) / x / x - (d.FalseCount - 1) / (1 - x) / (1 - x);
            Beta d2 = Beta.FromDerivatives(x, dLogP, ddLogP, false);
            Console.WriteLine("FromDerivatives = {0} should be {1}", d2, d);
            Assert.True(d.MaxDiff(d2) < 1e-6);
        }

        [Fact]
        public void PoissonTest()
        {
            Rand.Restart(12347);
            double rate = 0.3;
            Poisson geo = new Poisson(rate, 0);
            double m = rate / (1 - rate);
            Assert.True(MMath.AbsDiff(m, geo.GetMean()) < 1e-10);
            Assert.True(MMath.AbsDiff(m / (1 - rate), geo.GetVariance()) < 1e-10);
            // m=0.3; x = 0:1000; sum(m.^x*(1-m).*gammaln(x+1))
            Assert.True(MMath.AbsDiff(0.109116895970604, geo.GetMeanLogFactorial()) < 1e-10);

            Poisson d = new Poisson(1.2);
            PointMassTest(d, 3);
            UniformTest(d, 3);
            PointMassMomentTest(d, 3, 0.2, 0.1);
            GetAndSetMomentTest(d, 1.2, 1.2);
            DistributionTest(d, new Poisson(3.4));
            SamplingTest(d, 3);

            rate = 0.5;
            for (int i = 0; i < 6; i++)
            {
                d = new Poisson(rate);
                SamplingTest(d, 3);
                rate *= 10;
                rate += 0.5;
            }
        }

        [Fact]
        public void PoissonTest2()
        {
            Poisson d = new Poisson(1.2, 0.4);
            double sum = 0;
            double sumX = 0, sumX2 = 0;
            double sumLogFact = 0;
            double sumLogFact2 = 0;
            double sumXLogFact = 0;
            for (int i = 0; i < 1000; i++)
            {
                double p = System.Math.Exp(d.GetLogProb(i));
                sum += p;
                sumX += i * p;
                sumX2 += i * i * p;
                double logFact = MMath.GammaLn(i + 1);
                sumLogFact += logFact * p;
                sumLogFact2 += logFact * logFact * p;
                sumXLogFact += i * logFact * p;
            }
            double meanEst = sumX / sum;
            double varEst = sumX2 / sum - meanEst * meanEst;
            double meanLogFactEst = sumLogFact / sum;
            double meanLogFact2Est = sumLogFact2 / sum;
            double meanXLogFactEst = sumXLogFact / sum;
            double Z = System.Math.Exp(Poisson.GetLogNormalizer(d.Rate, d.Precision));
            double mean = System.Math.Exp(Poisson.GetLogPowerSum(d.Rate, d.Precision, 1)) / Z;
            double var = System.Math.Exp(Poisson.GetLogPowerSum(d.Rate, d.Precision, 2)) / Z - mean * mean;
            double meanLogFact = d.GetMeanLogFactorial();
            double meanLogFact2 = Poisson.GetSumLogFactorial2(d.Rate, d.Precision) / Z;
            double meanXLogFact = Poisson.GetSumXLogFactorial(d.Rate, d.Precision) / Z;
            Assert.True(MMath.AbsDiff(mean, meanEst, 1e-6) < 1e-4);
            Assert.True(MMath.AbsDiff(var, varEst, 1e-6) < 1e-4);
            Assert.True(MMath.AbsDiff(meanLogFact, meanLogFactEst, 1e-6) < 1e-4);
            Assert.True(MMath.AbsDiff(meanLogFact2, meanLogFact2Est, 1e-6) < 1e-4);
            Assert.True(MMath.AbsDiff(meanXLogFact, meanXLogFactEst, 1e-6) < 1e-4);
            double varLogFact = meanLogFact2 - meanLogFact * meanLogFact;
            double covXLogFact = meanXLogFact - mean * meanLogFact;

            Poisson d2 = Poisson.FromMeanAndMeanLogFactorial(mean, meanLogFact);
            Console.WriteLine("d2 = {0} should be {1}", d2, d);
            Assert.True(d2.MaxDiff(d) < 1e-4);

            PointMassTest(d, 3);
            UniformTest(d, 3);
            PointMassMomentTest(d, 3, 0.2, 0.1);
            //SetMomentTest(d, 1.2, 3.4);
            DistributionTest(d, new Poisson(3.4, 1.5));
            SamplingTest(d, 3);
        }

        [Fact]
        public void PoissonTest3()
        {
            double mean = 38.914480999999995;
            double meanLogFact = 115.79687593320728;
            // These cannot be satisfied:
            //double mean = 26.503113193371764;
            //double meanLogFact = 77.463522658901041;
            // Poisson expected = new Poisson(1.09452821574556, 0.029552860669053);
            var actual = Poisson.FromMeanAndMeanLogFactorial(mean, meanLogFact);
            double actualMean = actual.GetMean();
            double actualMeanLogFact = actual.GetMeanLogFactorial();
            Assert.True(MMath.AbsDiff(actualMean, mean) < 1e-4);
            Assert.True(MMath.AbsDiff(actualMeanLogFact, meanLogFact) < 1e-4);
        }

        [Fact]
        public void BetaSumTest()
        {
            Beta b1 = new Beta(2, 2);
            Beta b2 = new Beta(3, 3);
            Beta b = new Beta();
            b.SetToSum(0.9, b1, 0.1, b2);
            Console.WriteLine(b);
        }

        [Fact]
        public void DirichletSumTest()
        {
            Beta b1 = new Beta(2, 1);
            Beta b2 = new Beta(4, 3);
            Beta b = new Beta();
            b.SetToSum(0.9, b1, 0.1, b2);

            Dirichlet d1 = new Dirichlet(1.0, 2);
            Dirichlet d2 = new Dirichlet(3.0, 4);
            Dirichlet d = Dirichlet.Uniform(2);
            d.SetToSum(0.9, d1, 0.1, d2);

            Assert.True(MMath.AbsDiff(d.TotalCount, b.TotalCount, 1e-6) < 1e-10);
            Assert.True(d.PseudoCount.MaxDiff(Vector.FromArray(b.FalseCount, b.TrueCount)) < 1e-10);
        }

        [Fact]
        public void DirichletSumMergeTest()
        {
            Dirichlet d1 = new Dirichlet(2.0, 3.5, 5, 7);
            Dirichlet d2 = new Dirichlet(1.0, 4, 5, 7);
            Dirichlet d = Dirichlet.Uniform(d1.Dimension);
            d.SetToSum(0.9, d1, 0.1, d2);

            Dirichlet s1 = new Dirichlet(2.0, 3.5, 12);
            Dirichlet s2 = new Dirichlet(1.0, 4, 12);
            Dirichlet s = Dirichlet.Uniform(s1.Dimension);
            s.SetToSum(0.9, s1, 0.1, s2);

            Vector dMean = d.GetMean();
            Vector sMean = s.GetMean();
            Vector dMean2 = Vector.Zero(sMean.Count);
            dMean2[0] = dMean[0];
            dMean2[1] = dMean[1];
            dMean2[2] = dMean[2] + dMean[3];
            Assert.True(dMean2.MaxDiff(sMean) < 1e-10);
            Assert.True(MMath.AbsDiff(d.TotalCount, s.TotalCount, 1e-6) < 1e-10);
        }

        [Fact]
        public void GaussianSetToRatioProperTest()
        {
            Gaussian numerator = new Gaussian(5, 4);
            Gaussian denominator = new Gaussian(3, 2);
            Gaussian r = new Gaussian();
            r.SetToRatio(numerator, denominator, true);
            Console.WriteLine("ratio: {0}", r);
            Console.WriteLine("ratio*denom: {0} (should be {1})", r * denominator, new Gaussian(5, 2));
            Assert.True(MMath.AbsDiff((r * denominator).GetMean(), numerator.GetMean(), 1e-8) < 1e-10);
        }

        [Fact]
        public void GammaGetProbLessThanTest()
        {
            Assert.Equal(1, Gamma.FromShapeAndRate(1e-72, 1e-260).GetProbLessThan(1e-290));

            // exponential distribution with density exp(-x/m)/m and cdf 1-exp(-x/m)
            double m = 2.3;
            Gamma g = new Gamma(1.0, m);
            double median = -m * System.Math.Log(0.5);
            Assert.Equal(0.5, g.GetProbLessThan(median), 1e-4);
            AssertAlmostEqual(median, g.GetQuantile(0.5));

            g = new Gamma(2, m);
            double probability = g.GetProbLessThan(median);
            double quantile = g.GetQuantile(probability);
            Assert.Equal(median, quantile, 1e-10);
        }

        internal static void AssertAlmostEqual(double x, double y)
        {
            Assert.False(SpecialFunctionsTests.IsErrorSignificant(1e-16, MMath.AbsDiff(x, y)));
        }

        [Fact]
        public void GammaModeTest()
        {
            Gamma g = new Gamma();
            double expected = 0;
            for (int i = 0; i < 7; i++)
            {
                if (i == 0)
                {
                    g = Gamma.FromShapeAndRate(0.1, 0);
                    expected = 0;
                }
                else if (i == 1)
                {
                    g = Gamma.FromShapeAndRate(0.1, 2);
                    expected = 0;
                }
                else if (i == 2)
                {
                    g = Gamma.FromShapeAndRate(1, 2);
                    expected = 0;
                }
                else if (i == 3)
                {
                    g = Gamma.FromShapeAndRate(0, 2);
                    expected = 0;
                }
                else if (i == 4)
                {
                    g = Gamma.FromShapeAndRate(1.1, 2);
                    expected = 0.1 / 2;
                }
                else if (i == 5)
                {
                    g = Gamma.FromShapeAndRate(1, 0);
                    expected = 0;
                }
                else if (i == 6)
                {
                    g = Gamma.FromShapeAndRate(1.1, 0);
                    expected = double.PositiveInfinity;
                }
                double m = g.GetMode();
                Console.WriteLine("{0}: {1} (should be {2})", g, m, expected);
                Assert.True(MMath.AbsDiff(m, expected, 1e-10) < 1e-10);
            }
        }

        [Fact]
        public void GammaSetToRatioTest()
        {
            Gamma numerator = Gamma.FromShapeAndRate(1e20, 1);
            Gamma denominator = numerator;
            Gamma r = numerator / denominator;
            Assert.True(r.IsUniform());
        }

        [Fact]
        public void GammaSetToRatioProperTest()
        {
            foreach (Gamma numerator in new[] {
                Gamma.FromShapeAndRate(0.5, 0.5),
                Gamma.FromShapeAndRate(4, 5),
            })
            {
                foreach (var denominator in new[]
                {
                    Gamma.Uniform(),
                    Gamma.FromShapeAndRate(6, 3),
                    Gamma.FromShapeAndRate(6,7),
                    Gamma.FromShapeAndRate(3,7)
                })
                {
                    Gamma r = new Gamma();
                    r.SetToRatio(numerator, denominator, true);
                    //Trace.WriteLine($"ratio: {r} ratio*denom: {r * denominator} (numerator was {numerator})");
                    Assert.True(r.Shape >= 1);
                    Assert.True(r.Rate >= 0);
                    Assert.True(MMath.AbsDiff((r * denominator).GetMean(), numerator.GetMean(), 1e-8) < 1e-10);
                    // It is counter-intuitive that uniform denominator doesn't return numerator.
                    //if (denominator.IsUniform()) Assert.Equal(numerator, r);
                }
            }
        }

        [Fact]
        public void WishartSetToRatioProperTest()
        {
            int dim = 1;
            Wishart numerator = Wishart.FromShapeAndRate(4, PositiveDefiniteMatrix.IdentityScaledBy(dim, 5));
            foreach (Wishart denominator in new[] {
                Wishart.FromShapeAndRate(6, PositiveDefiniteMatrix.IdentityScaledBy(dim, 3)),
                Wishart.FromShapeAndRate(6, PositiveDefiniteMatrix.IdentityScaledBy(dim, 7)),
                Wishart.FromShapeAndRate(3, PositiveDefiniteMatrix.IdentityScaledBy(dim, 7)),
            })
            {
                Wishart r = new Wishart(dim);
                r.SetToRatio(numerator, denominator, true);
                //Trace.WriteLine($"ratio: {r} ratio*denom: {r * denominator} (numerator was {numerator})");
                Assert.True(r.Shape >= (dim + 1) / 2.0);
                Assert.True((r * denominator).GetMean().MaxDiff(numerator.GetMean()) < 1e-10);
            }
        }

        [Fact]
        public void GammaFromMeanAndMeanLogTest()
        {
            GammaFromMeanAndMeanLog(new Gamma(3, 4));
            GammaFromMeanAndMeanLog(Gamma.PointMass(3));
            Gamma estimated = Gamma.FromMeanAndMeanLog(0, -1e303);
            Assert.True(estimated.IsPointMass && estimated.Point == 0);
        }

        private void GammaFromMeanAndMeanLog(Gamma original)
        {
            double mean = original.GetMean();
            double meanLog = original.GetMeanLog();
            Gamma estimated = Gamma.FromMeanAndMeanLog(mean, meanLog);
            //Console.WriteLine("original = {0}", original);
            //Console.WriteLine("estimated = {0}", estimated);
            Assert.True(original.MaxDiff(estimated) < 1e-10);
        }

        [Fact]
        public void WishartFromMeanAndMeanLogTest()
        {
            PositiveDefiniteMatrix m3 = new PositiveDefiniteMatrix(new double[,] { { 4, 2, 1 }, { 2, 4, 2 }, { 1, 2, 4 } });
            PositiveDefiniteMatrix m2 = new PositiveDefiniteMatrix(new double[,] { { 2, 1 }, { 1, 2 } });
            WishartFromMeanAndMeanLog(new Wishart(3, m2));
            WishartFromMeanAndMeanLog(Wishart.PointMass(m2));
            WishartFromMeanAndMeanLog(new Wishart(4, m3));
            WishartFromMeanAndMeanLog(Wishart.PointMass(m3));
        }

        private void WishartFromMeanAndMeanLog(Wishart original)
        {
            var mean = original.GetMean();
            double meanLog = original.GetMeanLogDeterminant();
            var estimated = Wishart.FromMeanAndMeanLogDeterminant(mean, meanLog);
            Console.WriteLine("original = {0}", original);
            Console.WriteLine("estimated = {0}", estimated);
            Assert.True(original.MaxDiff(estimated) < 1e-10);
        }

        [Fact]
        [Trait("Category", "StringInference")]
        public void StringDistributionTest()
        {
            StringDistribution dist1 = StringDistribution.OneOf("a", "ab", "bcd");
            StringDistribution dist2 = StringDistribution.Capitalized();
            StringDistribution dist3 = StringDistribution.Upper().Append(StringDistribution.Lower());

            stringDistributionTest(dist1, dist2);
            stringDistributionTest(dist1, dist3);
            stringDistributionTest(dist2, dist3);

            StringDistributionPointMassTest(dist1, "ab");
            StringDistributionPointMassTest(dist2, "Abc");
            StringDistributionPointMassTest(dist3, "ABcd");

            InnerProductWithUniformTest(dist1, "ab");
            InnerProductWithUniformTest(dist1, "ad");
            InnerProductWithUniformTest(dist2, "Abc");
            InnerProductWithUniformTest(dist2, "ABbc");
            InnerProductWithUniformTest(dist3, "ABcd");
            InnerProductWithUniformTest(dist3, "cdAB");
        }

        private static void stringDistributionTest(
            StringDistribution distribution1, StringDistribution distribution2)
        {
            ProductWithUniformTest(distribution1);
            ProductWithUniformTest(distribution2);

            SettableToTest(distribution1);
            SettableToTest(distribution2);

            SettableToWeightedSumTest(distribution1, distribution2, false);
        }

        private static void StringDistributionPointMassTest(StringDistribution distribution, string value)
        {
            ProductWithPointMassTest(distribution, value);
            InnerProductPointMassTest(distribution, value);
            PointMassSampleTest(distribution, value);
        }

        #region Generic high-level tests 

        internal static void DistributionTest<T>(T a, T b, bool doSumWithNegativeWeightsTests = true)
            where T : SettableToProduct<T>, SettableToRatio<T>, SettableToPower<T>, SettableToUniform, SettableToWeightedSum<T>,
                ICloneable, Diffable, SettableTo<T>, CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>
        {
            Assert.False(a.Equals(null));
            ProductWithUniformTest(a);
            ProductWithUniformTest(b);

            RatioWithUniformTest(a);
            RatioWithUniformTest(b);
            SettableToRatioTest(a, b);

            SettableToPowerTest(a);
            SettableToPowerTest(b);
            LogAverageOfPowerTest(a, b);

            SettableToTest(a);
            SettableToTest(b);

            SettableToWeightedSumTest(a, b, doSumWithNegativeWeightsTests);
        }

        internal static void PointMassTest<T, DomainType>(T a, DomainType value)
            where T : SettableToProduct<T>, SettableToRatio<T>, SettableToPower<T>, SettableToUniform, SettableToWeightedSum<T>,
                ICloneable, Diffable, SettableTo<T>, HasPoint<DomainType>, CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>, CanGetLogProb<DomainType>,
                CanGetAverageLog<T>, Sampleable<DomainType>
        {
            ProductWithPointMassTest(a, value);
            RatioWithPointMassTest(a, value);
            PointMassPowerTest(a, value);
            AverageLogPointMassTest(a, value);
            InnerProductPointMassTest(a, value);
            PointMassSampleTest(a, value);
            PointMassGetLogProbTest(a, value);
        }

        internal static void UniformTest<T, DomainType>(T a, DomainType value)
            where T : CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>, CanGetAverageLog<T>, SettableToUniform, ICloneable, CanGetLogProb<DomainType>
        {
            InnerProductWithUniformTest(a, value);
            AverageLogUniformTest(a, value);
        }

        /// <summary>
        /// Check that the moments of a point mass distribution can be get and set.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="DomainType"></typeparam>
        /// <typeparam name="MeanType"></typeparam>
        /// <typeparam name="VarianceType"></typeparam>
        /// <param name="a">A distribution to clone.</param>
        /// <param name="value"></param>
        /// <param name="mean">A dummy value used only for type inference.</param>
        /// <param name="variance">A dummy value used only for type inference.</param>
        private static void PointMassMomentTest<T, DomainType, MeanType, VarianceType>(T a, DomainType value, MeanType mean, VarianceType variance)
            where T : HasPoint<DomainType>, CanGetMean<MeanType>, CanGetVariance<VarianceType>, ICloneable
        {
            T b = (T)a.Clone();
            b.Point = value;
            MeanType valueMean = b.GetMean();
            VarianceType zero = (VarianceType)GetZero(b.GetVariance());
            GetMomentTest(b, valueMean, zero);
            if (b is CanSetMeanAndVariance<MeanType, VarianceType>)
            {
                ((CanSetMeanAndVariance<MeanType, VarianceType>)b).SetMeanAndVariance(valueMean, zero);
                GetMomentTest(b, valueMean, zero);
                Assert.True(b.IsPointMass);
                Assert.Equal(value, b.Point);
            }
        }

        private static void PointMassSampleTest<T, DomainType>(T a, DomainType value)
            where T : HasPoint<DomainType>, Sampleable<DomainType>, ICloneable
        {
            T b = (T)a.Clone();
            b.Point = value;
            for (int iter = 0; iter < 3; iter++)
            {
                DomainType sample = b.Sample();
                AssertEqual(sample, value);
            }
        }

        private static void PointMassGetLogProbTest<T, DomainType>(T a, DomainType value)
            where T : HasPoint<DomainType>, CanGetLogProb<DomainType>, ICloneable
        {
            T b = (T)a.Clone();
            b.Point = value;
            Assert.Equal(0.0, b.GetLogProb(value));
            Assert.Equal(double.NegativeInfinity, b.GetLogProb(default(DomainType)));
        }

        private static void AssertEqual(object a, object b)
        {
            // todo: handle types that don't implement value equality
            Assert.True(a.Equals(b));
            Assert.True(a.GetHashCode() == b.GetHashCode());
        }

        private static void SamplingTest<T, DomainType>(T d, DomainType value)
            where T : Sampleable<DomainType>, CanGetMean<double>, CanGetVariance<double>
        {
            Rand.Restart(12347);
            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            for (int iter = 0; iter < 2000000; iter++)
            {
                DomainType sample = d.Sample();
                mva.Add(Convert.ToDouble(sample));
            }
            Console.WriteLine("mean = {0} should be {1}", mva.Mean, d.GetMean());
            Console.WriteLine("variance = {0} should be {1}", mva.Variance, d.GetVariance());
            Assert.True(MMath.AbsDiff(mva.Mean, d.GetMean(), 1e-6) < 1e-2);
            Assert.True(MMath.AbsDiff(mva.Variance, d.GetVariance(), 1e-6) < 1e-2);
        }

        /// <summary>
        /// Clone a distribution, change the moments, and read them back.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="MeanType"></typeparam>
        /// <typeparam name="VarianceType"></typeparam>
        /// <param name="a"></param>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        private static void SetMomentTest<T, MeanType, VarianceType>(T a, MeanType mean, VarianceType variance)
            where T : CanGetMean<MeanType>, CanGetVariance<VarianceType>, CanSetMeanAndVariance<MeanType, VarianceType>, ICloneable
        {
            T b = (T)a.Clone();
            b.SetMeanAndVariance(mean, variance);
            GetMomentTest(b, mean, variance);
        }

        /// <summary>
        /// Check that the moments match those given.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="MeanType"></typeparam>
        /// <typeparam name="VarianceType"></typeparam>
        /// <param name="a"></param>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        private static void GetAndSetMomentTest<T, MeanType, VarianceType>(T a, MeanType mean, VarianceType variance)
            where T : CanGetMean<MeanType>, CanGetVariance<VarianceType>, ICloneable, CanSetMeanAndVariance<MeanType, VarianceType>
        {
            MeanType actualMean = a.GetMean();
            VarianceType actualVariance = a.GetVariance();
            Assert.True(AbsDiff(mean, actualMean) < 1e-10);
            Assert.True(AbsDiff(variance, actualVariance) < 1e-10);
            SetMomentTest(a, mean, variance);
        }

        #endregion

        #region Helper tests

        private static void InnerProductWithUniformTest<DomainType, T>(T a, DomainType value)
            where T : CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>, SettableToUniform, ICloneable, CanGetLogProb<DomainType>
        {
            T uniform = (T)a.Clone();
            uniform.SetToUniform();
            Assert.True(uniform.IsUniform());
            Assert.True(MMath.AbsDiff(a.GetLogAverageOf(uniform), uniform.GetLogProb(value), 1e-6) < 1e-8);
            Assert.True(MMath.AbsDiff(uniform.GetLogAverageOf(a), uniform.GetLogProb(value), 1e-6) < 1e-8);
            Assert.True(MMath.AbsDiff(a.GetLogAverageOfPower(uniform, 1), uniform.GetLogProb(value), 1e-6) < 1e-8);
            Assert.True(MMath.AbsDiff(uniform.GetLogAverageOfPower(a, 1), uniform.GetLogProb(value), 1e-6) < 1e-8);
        }

        private static void AverageLogUniformTest<DomainType, T>(T a, DomainType value)
            where T : CanGetAverageLog<T>, SettableToUniform, ICloneable, CanGetLogProb<DomainType>
        {
            T uniform = (T)a.Clone();
            uniform.SetToUniform();
            Assert.True(uniform.IsUniform());
            Assert.True(MMath.AbsDiff(a.GetAverageLog(uniform), uniform.GetLogProb(value), 1e-6) < 1e-8);
        }

        internal static void ProductWithUniformTest<T>(T a)
            where T : SettableToProduct<T>, SettableToUniform, ICloneable, Diffable
        {
            T uniform = (T)a.Clone();
            uniform.SetToUniform();
            Assert.True(uniform.IsUniform());
            Assert.True(a.MaxDiff(a) == 0.0);
            if (!a.Equals(uniform))
                Assert.True(a.MaxDiff(uniform) > 0.0);
            Assert.True(a.MaxDiff(uniform) == uniform.MaxDiff(a));
            T b = (T)a.Clone();
            b.SetToProduct(a, uniform);
            Assert.True(a.MaxDiff(b) < 1e-10);
            b.SetToProduct(uniform, a);
            Assert.True(a.MaxDiff(b) < 1e-10);
            b.SetToProduct(uniform, uniform);
            Assert.True(uniform.MaxDiff(b) < 1e-10);
        }

        internal static void RatioWithUniformTest<T>(T a)
            where T : SettableToRatio<T>, SettableToUniform, ICloneable, Diffable
        {
            T uniform = (T)a.Clone();
            uniform.SetToUniform();
            T b = (T)a.Clone();
            b.SetToRatio(a, uniform);
            Assert.True(a.MaxDiff(b) < 1e-10);
            b.SetToRatio(uniform, uniform);
            Assert.True(uniform.MaxDiff(b) < 1e-10);
        }

        private static void InnerProductPointMassTest<DomainType, T>(T a, DomainType value)
            where T : HasPoint<DomainType>, CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>, CanGetLogProb<DomainType>, ICloneable
        {
            T pt = (T)a.Clone();
            pt.Point = value;
            Assert.True(MMath.AbsDiff(a.GetLogAverageOf(pt), a.GetLogProb(value), 1e-8) < 1e-8);
            Assert.True(MMath.AbsDiff(pt.GetLogAverageOf(a), a.GetLogProb(value), 1e-8) < 1e-8);
            Assert.True(MMath.AbsDiff(a.GetLogAverageOfPower(pt, 1), a.GetLogProb(value), 1e-8) < 1e-8);
            Assert.True(MMath.AbsDiff(pt.GetLogAverageOfPower(a, 1), a.GetLogProb(value), 1e-8) < 1e-8);
        }

        private static void LogAverageOfPowerTest<T>(T a, T b)
            where T : CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>, SettableToProduct<T>, ICloneable
        {
            for (int n = 0; n < 3; n++)
            {
                LogAverageOfPowerTest(a, b, n);
            }
        }

        private static void LogAverageOfPowerTest<T>(T a, T b, int n)
            where T : CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>, SettableToProduct<T>, ICloneable
        {
            double expected = 0;
            T productPrev = default(T);
            for (int i = 0; i < n; i++)
            {
                T product = (T)a.Clone();
                if (i > 0)
                    product.SetToProduct(productPrev, b);
                expected += product.GetLogAverageOf(b);
                productPrev = product;
            }
            double actual = a.GetLogAverageOfPower(b, n);
            Assert.True(MMath.AbsDiff(expected, actual, 1e-8) < 1e-8);
        }

        private static void AverageLogPointMassTest<DomainType, T>(T a, DomainType value)
            where T : HasPoint<DomainType>, CanGetAverageLog<T>, CanGetLogProb<DomainType>, ICloneable
        {
            T pt = (T)a.Clone();
            pt.Point = value;
            Assert.True(MMath.AbsDiff(a.GetAverageLog(pt), Double.NegativeInfinity, 1e-8) < 1e-8);
            Assert.True(MMath.AbsDiff(pt.GetAverageLog(pt), 0.0, 1e-8) < 1e-8);
            Assert.True(MMath.AbsDiff(pt.GetAverageLog(a), a.GetLogProb(value), 1e-8) < 1e-8);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="DomainType"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="a">A distribution that is everywhere non-zero</param>
        /// <param name="value">Anything other than default(DomainType).</param>
        private static void ProductWithPointMassTest<DomainType, T>(T a, DomainType value)
            where T : SettableToProduct<T>, HasPoint<DomainType>, ICloneable, Diffable
        {
            Assert.False(a.IsPointMass);
            T pt = (T)a.Clone();
            if (object.ReferenceEquals(default(DomainType), null))
            {
                pt.Point = value;
            }
            else
            {
                pt.Point = default(DomainType);
            }
            Assert.True(pt.IsPointMass);
            T b = (T)a.Clone();
            AssertEqual(a, b);
            b.SetToProduct(a, pt);
            Assert.True(pt.MaxDiff(b) < 1e-10);
            b.SetToProduct(pt, a);
            Assert.True(pt.MaxDiff(b) < 1e-10);
            b.SetToProduct(pt, pt);
            Assert.True(pt.MaxDiff(b) < 1e-10);
            T pt2 = (T)a.Clone();
            pt2.Point = pt.Point;
            AssertEqual(pt, pt2);
            pt2.Point = value;
            if (!(a is System.Collections.ICollection))
            {
                Assert.True(pt2.Point.Equals(value));
                if (!pt.Point.Equals(pt2.Point))
                {
                    try
                    {
                        b.SetToProduct(pt, pt2);
                        Assert.True(false, "AllZeroException not thrown");
                    }
                    catch (AllZeroException)
                    {
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="DomainType"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="a">A distribution that is everywhere non-zero</param>
        /// <param name="value">Anything other than default(DomainType).</param>
        private static void RatioWithPointMassTest<DomainType, T>(T a, DomainType value)
            where T : SettableToRatio<T>, SettableToUniform, HasPoint<DomainType>, ICloneable, Diffable
        {
            Assert.False(a.IsPointMass);
            T pt = (T)a.Clone();
            pt.Point = value;
            T b = (T)a.Clone();
            b.SetToRatio(pt, a);
            Assert.True(pt.MaxDiff(b) < 1e-10);
            try
            {
                b.SetToRatio(a, pt);
                Assert.True(false, "DivideByZeroException not thrown");
            }
            catch (DivideByZeroException)
            {
            }
            if (false)
            {
                try
                {
                    b.SetToRatio(pt, pt);
                    Assert.True(false, "DivideByZeroException not thrown");
                }
                catch (DivideByZeroException)
                {
                }
            }
            if (false)
            {
                b.SetToRatio(pt, pt);
                T uniform = (T)a.Clone();
                uniform.SetToUniform();
                Assert.True(uniform.MaxDiff(b) < 1e-10);
            }
            T pt2 = (T)a.Clone();
            pt2.Point = value;
            if (!(a is System.Collections.ICollection))
            {
                if (!pt.Point.Equals(pt2.Point))
                {
                    try
                    {
                        b.SetToRatio(pt, pt2);
                        Assert.True(false, "DivideByZeroException not thrown");
                    }
                    catch (DivideByZeroException)
                    {
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a">A distribution that is everywhere non-zero</param>
        /// <param name="b">A distribution that is everywhere non-zero</param>
        private static void SettableToRatioTest<T>(T a, T b)
            where T : SettableToProduct<T>, SettableToRatio<T>, ICloneable, Diffable, SettableToUniform
        {
            T c = (T)a.Clone();
            c.SetToProduct(a, b);
            if (!b.IsUniform())
                Assert.True(c.MaxDiff(a) > 0.0);
            if (!a.IsUniform())
                Assert.True(c.MaxDiff(b) > 0.0);
            Assert.True(c.MaxDiff(a) == a.MaxDiff(c));
            Assert.True(c.MaxDiff(b) == b.MaxDiff(c));
            c.SetToRatio(c, b);
            Assert.True(a.MaxDiff(c) < 1e-10);
            c.SetToRatio(a, b);
            if (!b.IsUniform())
                Assert.True(c.MaxDiff(a) > 0.0);
            if (!b.IsUniform())
                Assert.True(c.MaxDiff(b) > 0.0);
            Assert.True(c.MaxDiff(a) == a.MaxDiff(c));
            Assert.True(c.MaxDiff(b) == b.MaxDiff(c));
            c.SetToProduct(c, b);
            Assert.True(a.MaxDiff(c) < 1e-10);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a">A distribution that is everywhere non-zero</param>
        private static void SettableToPowerTest<T>(T a)
            where T : SettableToProduct<T>, SettableToRatio<T>, SettableToPower<T>, SettableToUniform, ICloneable, Diffable
        {
            T c = (T)a.Clone();
            c.SetToPower(a, 2.0);
            c.SetToPower(c, 0.5);
            Assert.True(a.MaxDiff(c) < 1e-10);
            c.SetToPower(a, 2.0);
            c.SetToRatio(c, a);
            Assert.True(a.MaxDiff(c) < 1e-10);
            try
            {
                c.SetToPower(a, -1.0);
                c.SetToProduct(c, a);
                c.SetToProduct(c, a);
                Assert.True(a.MaxDiff(c) < 1e-10);
            }
            catch
            {
            }

            c.SetToPower(a, 1.0);
            Assert.True(a.MaxDiff(c) < 1e-12);
            c.SetToPower(a, 0.0);
            Assert.True(c.IsUniform());
        }

        private static void PointMassPowerTest<T, DomainType>(T a, DomainType value)
            where T : HasPoint<DomainType>, SettableToPower<T>, Diffable, SettableToUniform, ICloneable
        {
            T c = (T)a.Clone();
            c.Point = value;
            Assert.True(c.IsPointMass);
            c.SetToPower(c, 0.0);
            Assert.True(c.IsUniform());
            c.Point = value;
            c.SetToPower(c, 0.1);
            Assert.True(c.IsPointMass);
            AssertEqual(c.Point, value);
            try
            {
                c.SetToPower(c, -0.1);
                Assert.True(false, "DivideByZeroException not thrown");
            }
            catch (DivideByZeroException)
            {
            }
        }

        internal static void SettableToTest<T>(T a)
            where T : SettableTo<T>, SettableToUniform, ICloneable
        {
            T b = (T)a.Clone();
            b.SetToUniform();
            b.SetTo(a);
            Assert.True(b.IsUniform() == a.IsUniform());
            // Don't require equality on collections (follows .NET 'standards')
            AssertEqual(a, b);
        }

        private static void SettableToWeightedSumTest<T>(T a, T b, bool doSumWithNegativeWeightsTests = true)
            where T : SettableToWeightedSum<T>, ICloneable, Diffable, SettableToUniform
        {
            T c = (T)a.Clone();
            c.SetToSum(0.5, a, 0.5, b);
            if (!(c is Discrete && ((Discrete)(object)c).Dimension <= 1))
            {
                Assert.False(c.Equals(a));
                Assert.False(c.Equals(b));
            }

            if (doSumWithNegativeWeightsTests)
            {
                c.SetToSum(2.0, c, -1.0, b);
                Assert.True(c.MaxDiff(a) < 1e-10);
            }

            c.SetToUniform();
            c.SetToSum(2.0, a, 0.0, c);
            Assert.True(c.Equals(a));
            c.SetToSum(0.0, c, 2.0, a);
            Assert.True(c.Equals(a));
            c.SetToSum(1.0, c, Double.PositiveInfinity, b);
            Assert.True(c.Equals(b));
            c.SetToSum(Double.PositiveInfinity, a, 1.0, c);
            Assert.True(c.Equals(a));
            c.SetToSum(0.0, a, 0.0, b);
            Assert.True(c.IsUniform());
            try
            {
                c.SetToSum(0.0, a, -1.0, c);
                Assert.True(false, "Did not throw exception");
            }
            catch (ArgumentException)
            {
            }
        }

        private static void GetMomentTest<T, MeanType, VarianceType>(T a, MeanType mean, VarianceType variance)
            where T : CanGetMean<MeanType>, CanGetVariance<VarianceType>
        {
            MeanType actualMean = a.GetMean();
            VarianceType actualVariance = a.GetVariance();
            Assert.True(AbsDiff(mean, actualMean) < 1e-10);
            Assert.True(AbsDiff(variance, actualVariance) < 1e-10);
            if (a is CanGetMeanAndVarianceOut<MeanType, VarianceType>)
            {
                ((CanGetMeanAndVarianceOut<MeanType, VarianceType>)a).GetMeanAndVariance(out actualMean, out actualVariance);
            }
            else if (a is CanGetMeanAndVariance<MeanType, VarianceType>)
            {
                ((CanGetMeanAndVariance<MeanType, VarianceType>)a).GetMeanAndVariance(actualMean, actualVariance);
            }
            else
                return;
            Assert.True(AbsDiff(mean, actualMean) < 1e-10);
            Assert.True(AbsDiff(variance, actualVariance) < 1e-10);
        }

        public static double AbsDiff(object a, object b)
        {
            if (a is double)
                return MMath.AbsDiff((double)a, (double)b, 1e-10);
            else if (a is Vector)
                return ((Vector)a).MaxDiff((Vector)b);
            else if (a is PositiveDefiniteMatrix)
                return ((PositiveDefiniteMatrix)a).MaxDiff((PositiveDefiniteMatrix)b);
            else
                throw new NotImplementedException();
        }

        public static object GetZero(object o)
        {
            if (o is bool)
                return false;
            else if (o is double)
                return 0.0;
            else if (o is Vector)
                return Vector.Zero(((Vector)o).Count);
            else if (o is PositiveDefiniteMatrix)
                return new PositiveDefiniteMatrix(((PositiveDefiniteMatrix)o).Rows, ((PositiveDefiniteMatrix)o).Cols);
            else
                throw new NotImplementedException();
        }

        #endregion
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}