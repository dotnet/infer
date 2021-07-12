using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class GammaOpTests
    {
        [Fact]
        public void GammaRatioOpTest()
        {
            Gamma ratio = new Gamma(2, 3);
            double shape = 4;
            Gamma A = new Gamma(shape, 1);
            Gamma B = new Gamma(5, 6);
            Gamma q = GammaFromShapeAndRateOp_Laplace.Q(ratio, shape, B);
            Gamma bExpected = GammaFromShapeAndRateOp_Laplace.RateAverageConditional(ratio, shape, B, q);
            Gamma rExpected = GammaFromShapeAndRateOp_Laplace.SampleAverageConditional(ratio, shape, B, q);
            q = GammaRatioOp_Laplace.Q(ratio, A, B);
            Gamma bActual = GammaRatioOp_Laplace.BAverageConditional(ratio, A, B, q);
            Gamma rActual = GammaRatioOp_Laplace.RatioAverageConditional(ratio, A, B);
            Console.WriteLine("b = {0} should be {1}", bActual, bExpected);
            Console.WriteLine("ratio = {0} should be {1}", rActual, rExpected);
            Assert.True(bExpected.MaxDiff(bActual) < 1e-4);
            Assert.True(rExpected.MaxDiff(rActual) < 1e-4);
        }

        // Test that the operator behaves correctly for arguments with small variance
        [Fact]
        public void GammaFromShapeAndRateOpTest()
        {
            Assert.False(double.IsNaN(GammaFromShapeAndRateOp_Slow.SampleAverageConditional(Gamma.PointMass(double.Epsilon), 1.0, new Gamma(1, 1)).Rate));
            Assert.False(double.IsNaN(GammaFromShapeAndRateOp_Slow.SampleAverageConditional(Gamma.PointMass(0), 2.0, new Gamma(1, 1)).Rate));
            Assert.False(double.IsNaN(GammaFromShapeAndRateOp_Slow.RateAverageConditional(new Gamma(1, 1), 2.0, Gamma.PointMass(0)).Rate));

            Gamma sample, rate, result;
            double prevDiff;
            double shape = 3;
            rate = Gamma.FromShapeAndRate(4, 5);
            sample = Gamma.PointMass(2);
            result = GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
            Console.WriteLine("{0}: {1}", sample, result);
            prevDiff = double.PositiveInfinity;
            for (int i = 10; i < 50; i++)
            {
                double v = System.Math.Pow(0.1, i);
                sample = Gamma.FromMeanAndVariance(2, v);
                Gamma result2 = GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
                double diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} (diff={2})", sample, result2, diff.ToString("g4"));
                Assert.True(diff <= prevDiff || diff < 1e-10);
                prevDiff = diff;
            }
        }
        [Fact]
        public void GammaFromShapeAndRateOpTest2()
        {
            Gamma sample, rate, result;
            double prevDiff;
            double shape = 3;
            sample = Gamma.FromShapeAndRate(4, 5);
            shape = 0.1;
            rate = Gamma.FromShapeAndRate(0.1, 0.1);
            result = GammaFromShapeAndRateOp_Slow.RateAverageConditional(sample, shape, rate);
            Console.WriteLine(result);

            shape = 3;
            rate = Gamma.PointMass(2);
            result = GammaFromShapeAndRateOp_Slow.RateAverageConditional(sample, shape, rate);
            Console.WriteLine("{0}: {1}", rate, result);
            prevDiff = double.PositiveInfinity;
            for (int i = 10; i < 50; i++)
            {
                double v = System.Math.Pow(0.1, i);
                rate = Gamma.FromMeanAndVariance(2, v);
                Gamma result2 = GammaFromShapeAndRateOp_Slow.RateAverageConditional(sample, shape, rate);
                double diff = result.MaxDiff(result2);
                Console.WriteLine("{0}: {1} (diff={2})", rate, result2, diff.ToString("g4"));
                Assert.True(diff <= prevDiff || diff < 1e-10);
                prevDiff = diff;
            }
        }
        [Fact]
        public void GammaFromShapeAndRateOpTest3()
        {
            Console.WriteLine(GammaFromShapeAndRateOp_Slow.RateAverageConditional(Gamma.FromShapeAndRate(2, 1), 1, Gamma.FromShapeAndRate(1.5, 0.5)));

            Gamma sample = Gamma.FromShapeAndRate(2, 0);
            Gamma rate = Gamma.FromShapeAndRate(4, 1);
            double shape = 1;

            Gamma to_sample2 = GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
            double evExpected = GammaFromShapeAndRateOp_Slow.LogEvidenceRatio(sample, shape, rate, to_sample2);
            Console.WriteLine("sample = {0} to_sample = {1} evidence = {2}", sample, to_sample2, evExpected);
            for (int i = 40; i < 41; i++)
            {
                sample.Rate = System.Math.Pow(0.1, i);
                Gamma to_sample = GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
                double evActual = GammaFromShapeAndRateOp_Slow.LogEvidenceRatio(sample, shape, rate, to_sample);
                Console.WriteLine("sample = {0} to_sample = {1} evidence = {2}", sample, to_sample, evActual);
                Assert.True(to_sample2.MaxDiff(to_sample2) < 1e-4);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-4);
            }
        }
        [Fact]
        public void GammaFromShapeAndRateOpTest4()
        {
            Gamma sample = Gamma.FromShapeAndRate(2, 0);
            Gamma rate = Gamma.FromShapeAndRate(4, 1);
            double shape = 1;

            Gamma rateExpected = GammaFromShapeAndRateOp_Slow.RateAverageConditional(sample, shape, rate);
            Gamma q = GammaFromShapeAndRateOp_Laplace.Q(sample, shape, rate);
            Gamma rateActual = GammaFromShapeAndRateOp_Laplace.RateAverageConditional(sample, shape, rate, q);
            Assert.True(rateExpected.MaxDiff(rateActual) < 1e-4);

            Gamma to_sample2 = GammaFromShapeAndRateOp_Laplace.SampleAverageConditional(sample, shape, rate, q);
            double evExpected = GammaFromShapeAndRateOp_Laplace.LogEvidenceRatio(sample, shape, rate, to_sample2, q);
            Console.WriteLine("sample = {0} to_sample = {1} evidence = {2}", sample, to_sample2, evExpected);
            for (int i = 40; i < 41; i++)
            {
                sample.Rate = System.Math.Pow(0.1, i);
                q = GammaFromShapeAndRateOp_Laplace.Q(sample, shape, rate);
                Gamma to_sample = GammaFromShapeAndRateOp_Laplace.SampleAverageConditional(sample, shape, rate, q);
                double evActual = GammaFromShapeAndRateOp_Laplace.LogEvidenceRatio(sample, shape, rate, to_sample, q);
                Console.WriteLine("sample = {0} to_sample = {1} evidence = {2}", sample, to_sample, evActual);
                Assert.True(to_sample2.MaxDiff(to_sample2) < 1e-4);
                Assert.True(MMath.AbsDiff(evExpected, evActual) < 1e-4);
            }
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void GammaFromShapeAndRateOpTest5()
        {
            Gamma sample;
            Gamma rate;
            double shape = 1;
            Gamma q, sampleExpected, sampleActual;

            sample = Gamma.FromShapeAndRate(101, 6.7234079315458819E-154);
            rate = Gamma.FromShapeAndRate(1, 1);
            q = GammaFromShapeAndRateOp_Laplace.Q(sample, shape, rate);
            Console.WriteLine(q);
            Assert.True(!double.IsNaN(q.Rate));
            sampleExpected = GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
            sampleActual = GammaFromShapeAndRateOp_Laplace.SampleAverageConditional(sample, shape, rate, q);
            Console.WriteLine("sample = {0} should be {1}", sampleActual, sampleExpected);
            Assert.True(sampleExpected.MaxDiff(sampleActual) < 1e-4);

            sample = Gamma.FromShapeAndRate(1.4616957536444839, 6.2203585601953317E+36);
            rate = Gamma.FromShapeAndRate(2.5, 0.99222007168007165);
            sampleExpected = GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
            q = Gamma.FromShapeAndRate(3.5, 0.99222007168007154);
            sampleActual = GammaFromShapeAndRateOp_Laplace.SampleAverageConditional(sample, shape, rate, q);
            Console.WriteLine("sample = {0} should be {1}", sampleActual, sampleExpected);
            Assert.True(sampleExpected.MaxDiff(sampleActual) < 1e-4);

            sample = Gamma.FromShapeAndRate(1.9692446124520258, 1.0717828357423075E+77);
            rate = Gamma.FromShapeAndRate(101.0, 2.1709591889324445E-80);
            sampleExpected = GammaFromShapeAndRateOp_Slow.SampleAverageConditional(sample, shape, rate);
            q = GammaFromShapeAndRateOp_Laplace.Q(sample, shape, rate);
            sampleActual = GammaFromShapeAndRateOp_Laplace.SampleAverageConditional(sample, shape, rate, q);
            Console.WriteLine("sample = {0} should be {1}", sampleActual, sampleExpected);
            Assert.True(sampleExpected.MaxDiff(sampleActual) < 1e-4);

            Assert.Equal(0.0,
                            GammaFromShapeAndRateOp_Laplace.LogEvidenceRatio(Gamma.Uniform(), 4.0, Gamma.PointMass(0.01), Gamma.FromShapeAndRate(4, 0.01),
                                                                             Gamma.PointMass(0.01)));
        }

        // Test inference on a model where precision is scaled.
        internal void GammaProductTest()
        {
            for (int i = 0; i <= 20; i++)
            {
                double minutesPlayed = System.Math.Pow(0.1, i);
                if (i == 20) minutesPlayed = 0;
                var EventCountMean_F = Gaussian.PointMass(0);
                var EventCountPrecision_F = GammaRatioOp.RatioAverageConditional(Gamma.PointMass(1), minutesPlayed);
                var realCount_F = GaussianOp_PointPrecision.SampleAverageConditional(EventCountMean_F, EventCountPrecision_F);
                var realCount_use_B = MaxGaussianOp.BAverageConditional(0.0, 0.0, realCount_F);
                var EventCountPrecision_B = GaussianOp_PointPrecision.PrecisionAverageConditional(realCount_use_B, EventCountMean_F, EventCountPrecision_F);
                var EventsPerMinutePrecision_B = GammaRatioOp.AAverageConditional(EventCountPrecision_B, minutesPlayed);
                Console.WriteLine($"realCount_use_B = {realCount_use_B}, EventCountPrecision_B = {EventCountPrecision_B},  EventsPerMinutePrecision_B = {EventsPerMinutePrecision_B}");
            }
        }

        [Fact]
        public void GammaPowerProductOp_LaplaceTest()
        {
            GammaPowerProductOp_Laplace.ProductAverageConditional(GammaPower.Uniform(-1), new GammaPower(22, 0.04762, -1), new GammaPower(4.984e+10, 9.32e-120, -1), new Gamma(4.984e+10, 9.32e-120), GammaPower.Uniform(-1));
        }

    }
}
