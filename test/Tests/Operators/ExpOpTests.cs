using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Distributions;

using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;

using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class ExpOpTests
    {
        //[Fact]
        //[Trait("Category", "ModifiesGlobals")]
        internal void ExpOp_CompareToSampling()
        {
            int sampleCount = 1_000_000;
            for (int i = 1; i < 10; i++)
            {
                double ve = System.Math.Pow(10, -i);
                Gamma exp = Gamma.FromMeanAndVariance(1, ve);
                Gaussian d = Gaussian.FromMeanAndVariance(0, 1);
                Gaussian to_d = ExpOp.DAverageConditional(exp, d, Gaussian.Uniform());
                var importanceSampler = new ImportanceSampler(sampleCount, d.Sample, x => System.Math.Exp(exp.GetLogProb(System.Math.Exp(x))));
                double mean = importanceSampler.GetExpectation(x => x);
                double Ex2 = importanceSampler.GetExpectation(x => x * x);
                Gaussian to_d_is = new Gaussian(mean, Ex2 - mean * mean) / d;
                var exp2 = Gamma.FromShapeAndRate(exp.Shape - 1, exp.Rate);
                importanceSampler = new ImportanceSampler(sampleCount, exp2.Sample, x => System.Math.Exp(d.GetLogProb(System.Math.Log(x))));
                mean = importanceSampler.GetExpectation(x => System.Math.Log(x));
                Ex2 = importanceSampler.GetExpectation(x => System.Math.Log(x) * System.Math.Log(x));
                Gaussian to_d_is2 = new Gaussian(mean, Ex2 - mean * mean) / d;
                Trace.WriteLine($"{ve}\t{to_d.GetMean()}\t{to_d_is.GetMean()}\t{to_d_is2.GetMean()}");

                Gamma to_exp = ExpOp.ExpAverageConditional(exp, d, Gaussian.Uniform());

                importanceSampler = new ImportanceSampler(sampleCount, d.Sample, x => System.Math.Exp(exp.GetLogProb(System.Math.Exp(x))));
                double Ey = importanceSampler.GetExpectation(x => System.Math.Exp(x));
                double Elogy = importanceSampler.GetExpectation(x => x);
                var to_exp_is1 = Gamma.FromMeanAndMeanLog(Ey, Elogy) / exp;
                importanceSampler = new ImportanceSampler(sampleCount, exp2.Sample, x => System.Math.Exp(d.GetLogProb(System.Math.Log(x))));
                Ey = importanceSampler.GetExpectation(x => x);
                Elogy = importanceSampler.GetExpectation(x => System.Math.Log(x));
                var to_exp_is2 = Gamma.FromMeanAndMeanLog(Ey, Elogy) / exp;
                double is1_mean = to_exp_is1.IsProper() ? to_exp_is1.GetMean() : double.NaN;
                double is2_mean = to_exp_is2.IsProper() ? to_exp_is2.GetMean() : double.NaN;
                Trace.WriteLine($"to_exp: Quad {to_exp.GetMean()} IS1: {is1_mean} IS2: {is2_mean}");
            }
        }

        [Fact]
        public void ExpOpGammaPowerTest()
        {
            Assert.True(!double.IsNaN(ExpOp.ExpAverageConditional(GammaPower.Uniform(-1), Gaussian.FromNatural(0.046634157098979417, 0.00078302234897204242), Gaussian.Uniform()).Rate));
            Assert.True(!double.IsNaN(ExpOp.ExpAverageConditional(GammaPower.Uniform(-1), Gaussian.FromNatural(0.36153121930654075, 0.0005524890062312658), Gaussian.Uniform()).Rate));
            Assert.True(ExpOp.DAverageConditional(GammaPower.PointMass(0, -1), new Gaussian(0, 1), Gaussian.Uniform()).Point < double.MinValue);
            ExpOp.ExpAverageConditional(GammaPower.FromShapeAndRate(-1, 283.673, -1), Gaussian.FromNatural(0.004859823703146038, 6.6322755562737905E-06), Gaussian.FromNatural(0.00075506803981220758, 8.24487022054953E-07));
            GammaPower exp = GammaPower.FromShapeAndRate(0, 0, -1);
            Gaussian[] ds = new[]
            {
                Gaussian.FromNatural(-1.6171314269768655E+308, 4.8976001759138024),
                Gaussian.FromNatural(-0.037020622891705768, 0.00034989765084474117),
                Gaussian.PointMass(double.NegativeInfinity),
            };
            foreach (var d in ds)
            {
                Gaussian to_d = ExpOp.DAverageConditional(exp, d, Gaussian.Uniform());
                Gaussian to_d_slow = ExpOp_Slow.DAverageConditional(exp, d);
                Assert.True(to_d_slow.MaxDiff(to_d) < 1e-10);
                to_d = Gaussian.FromNatural(1, 0);
                GammaPower to_exp = ExpOp.ExpAverageConditional(exp, d, to_d);
                //Trace.WriteLine($"{to_exp}");
            }
            ExpOp.ExpAverageConditional(GammaPower.FromShapeAndRate(-1, 883.22399999999993, -1), Gaussian.FromNatural(0.0072160312702854888, 8.1788482512051846E-06), Gaussian.FromNatural(0.00057861649495666474, 5.6316164560235272E-07));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void ExpOpTest()
        {
            Assert.True(ExpOp.ExpAverageConditional(Gamma.FromShapeAndRate(3.302758272196654, 0.00060601537137241492), Gaussian.FromNatural(55.350150233321628, 6.3510247863590683), Gaussian.FromNatural(27.960892513144643, 3.4099170930572216)).Rate > 0);
            Gamma exp = new Gamma(1, 1);
            Gaussian[] ds = new[]
            {
                Gaussian.FromNatural(-1.6171314269768655E+308, 4.8976001759138024),
                Gaussian.PointMass(double.NegativeInfinity),
            };
            foreach (var d in ds)
            {
                Gamma to_exp = ExpOp.ExpAverageConditional(exp, d, Gaussian.Uniform());
                Gaussian to_d = ExpOp.DAverageConditional(exp, d, Gaussian.Uniform());
                Gaussian to_d_slow = ExpOp_Slow.DAverageConditional(exp, d);
                Trace.WriteLine($"{to_d}");
                Trace.WriteLine($"{to_d_slow}");
                Assert.True(to_d_slow.MaxDiff(to_d) < 1e-10);
            }
        }

        // This test fails because of cancellation in logMeanMinusMd and Gamma.SetToRatio in ExpOp.ExpAverageConditional
        [Fact]
        [Trait("Category", "ModifiesGlobals")]
        [Trait("Category", "OpenBug")]
        public void ExpOp_PointExp()
        {
            using (TestUtils.TemporarilyChangeQuadratureNodeCount(21))
            {
                double vd = 1e-4;
                vd = 1e-3;
                Gaussian d = new Gaussian(0, vd);
                double ve = 2e-3;
                //ve = 1;
                Gaussian uniform = Gaussian.Uniform();
                Gamma expPoint = Gamma.PointMass(2);
                Gamma to_exp_point = ExpOp.ExpAverageConditional(expPoint, d, uniform);
                Gaussian to_d_point = ExpOp.DAverageConditional(expPoint, d, uniform);
                double to_exp_oldError = double.PositiveInfinity;
                double to_d_oldError = double.PositiveInfinity;
                for (int i = 5; i < 100; i++)
                {
                    ve = System.Math.Pow(10, -i);
                    Gamma exp = Gamma.FromMeanAndVariance(2, ve);
                    Gamma to_exp = ExpOp.ExpAverageConditional(exp, d, uniform);
                    Gaussian to_d = ExpOp.DAverageConditional(exp, d, uniform);
                    double to_exp_error = to_exp.MaxDiff(to_exp_point);
                    double to_d_error = System.Math.Abs(to_d.GetMean() - to_d_point.GetMean());
                    Trace.WriteLine($"ve={ve}: to_exp={to_exp} error={to_exp_error} to_d={to_d} error={to_d_error}");
                    Assert.True(to_exp_error <= to_exp_oldError);
                    to_exp_oldError = to_exp_error;
                    Assert.True(to_d_error <= to_d_oldError);
                    to_d_oldError = to_d_error;
                }
                Trace.WriteLine(ExpOp.DAverageConditional(Gamma.FromMeanAndVariance(1, ve), d, uniform));
                using (TestUtils.TemporarilyChangeQuadratureShift(true))
                {
                    Trace.WriteLine(ExpOp.DAverageConditional(Gamma.FromMeanAndVariance(1, ve), d, uniform));
                }
            }
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void ExpOpGammaPower_PointExp()
        {
            double power = -1;
            double vd = 1e-4;
            vd = 1e-3;
            Gaussian d = new Gaussian(0, vd);
            Gaussian uniform = Gaussian.Uniform();
            GammaPower expPoint = GammaPower.PointMass(2, power);
            GammaPower to_exp_point = ExpOp.ExpAverageConditional(expPoint, d, uniform);
            Gaussian to_d_point = ExpOp.DAverageConditional(expPoint, d, uniform);
            double to_exp_oldError = double.PositiveInfinity;
            double to_d_oldError = double.PositiveInfinity;
            for (int i = 0; i < 100; i++)
            {
                double ve = System.Math.Pow(10, -i);
                GammaPower exp = GammaPower.FromMeanAndVariance(2, ve, power);
                GammaPower to_exp = ExpOp.ExpAverageConditional(exp, d, uniform);
                Gaussian to_d = ExpOp.DAverageConditional(exp, d, uniform);
                double to_exp_error = to_exp.MaxDiff(to_exp_point);
                double to_d_error = System.Math.Abs(to_d.GetMean() - to_d_point.GetMean());
                Trace.WriteLine($"ve={ve}: to_exp={to_exp} error={to_exp_error} to_d={to_d} error={to_d_error}");
                Assert.True(to_exp_error <= to_exp_oldError);
                to_exp_oldError = to_exp_error;
                Assert.True(to_d_error <= to_d_oldError);
                to_d_oldError = to_d_error;
            }
        }
    }

    public class ImportanceSampler
    {
        public delegate double Sampler();
        private Vector samples, weights;
        public ImportanceSampler(int N, Sampler sampler, Converter<double, double> weightFunction)
        {
            samples = Vector.Zero(N);
            weights = Vector.Zero(N);
            for (int i = 0; i < N; i++)
            {
                samples[i] = sampler();
                weights[i] = weightFunction(samples[i]);
            }
        }

        public double GetExpectation(Converter<double, double> h)
        {
            var temp = Vector.Zero(samples.Count);
            temp.SetToFunction(samples, h);
            temp.SetToProduct(temp, weights);
            return temp.Sum() / weights.Sum();
        }

    }
}
