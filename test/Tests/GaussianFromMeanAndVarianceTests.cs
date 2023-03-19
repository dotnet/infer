// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Assert = Xunit.Assert;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class GaussianFromMeanAndVarianceTests
    {
        [Fact]
        public void PointVarianceTest()
        {
            Gamma variance = Gamma.PointMass(6.1699356586552062E-08);
            Gaussian mean = Gaussian.PointMass(0.0);
            for (int i = 7; i < 200; i++)
            {
                Gaussian sample = Gaussian.FromNatural(-0.00078333235196782888, System.Math.Pow(10, -i));
                Gamma message = GaussianFromMeanAndVarianceOp.VarianceAverageConditional(sample, mean, variance);
                ////Trace.WriteLine($"{i} {message}");
                Assert.Equal(0, message.Rate);
                Assert.True(message.Shape >= 1);
            }
        }

        [Fact]
        public void PointVarianceTest2()
        {
            Gamma variance = Gamma.PointMass(double.Epsilon);
            Gaussian mean = Gaussian.PointMass(0.0);
            for (int i = 100; i < 324; i++)
            {
                Gaussian sample = Gaussian.FromMeanAndVariance(1.23, System.Math.Pow(10, -i));
                Gamma message = GaussianFromMeanAndVarianceOp.VarianceAverageConditional(sample, mean, variance);
                ////Trace.WriteLine($"{i} {message} {message.Rate}");
                Assert.Equal(0, message.Rate);
                Assert.True(message.Shape >= 1);
            }
        }

        [Fact]
        public void PointVarianceTest3()
        {
            for (int i = 300; i < 324; i++)
            {
                Gamma message = GaussianFromMeanAndVarianceOp.VarianceAverageConditional(Gaussian.FromNatural(-0.00070557425617118793, System.Math.Pow(10, -i)), Gaussian.FromNatural(0, 13466586.033404613), Gamma.PointMass(0.09965));
                ////Trace.WriteLine($"{i} {message} {message.Rate}");
                Assert.True(message.Rate >= 0);
                Assert.True(message.Shape >= 1);
            }
        }

        [Fact]
        public void PointVarianceTest4()
        {
            Gaussian zero = Gaussian.PointMass(0);
            Parallel.ForEach(OperatorTests.DoublesGreaterThanZero(), variance =>
            {
                Gamma varianceDist = Gamma.PointMass(variance);
                foreach (var sample in OperatorTests.Gaussians(10000))
                {
                    Gamma message = GaussianFromMeanAndVarianceOp.VarianceAverageConditional(sample, zero, varianceDist);
                    Gaussian mean = Gaussian.FromNatural(-sample.MeanTimesPrecision, sample.Precision);
                    Gamma message2 = GaussianFromMeanAndVarianceOp.VarianceAverageConditional(zero, mean, varianceDist);
                    Assert.Equal(message, message2);
                }
            });
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void VarianceGammaTimesGaussianMomentsTest()
        {
            double a = 1457.651420562946;
            double m = -55.3161989949422;
            double v = 117.31286883057773;
            double mu, vu;
            GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianMoments5(a, m, v, out mu, out vu);
            Assert.False(double.IsNaN(mu));
            Assert.False(double.IsNaN(vu));
        }

        [Fact]
        [Trait("Category", "OpenBug")]
        public void VarianceGammaTimesGaussianIntegralTest()
        {
            double logZ, mu, vu;
            GaussianFromMeanAndVarianceOp.LaplacianTimesGaussianMoments(-100, 1, out logZ, out mu, out vu);
            Console.WriteLine(logZ);
            //GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianMoments5(1, 1000, 1, out mu, out vu);
            Console.WriteLine(mu);
            Console.WriteLine(vu);
            double[][] vgmoments = GaussianFromMeanAndVarianceOp.NormalVGMomentRatios(10, 1, -6, 1);
            for (int i = 0; i < 10; i++)
            {
                double f = MMath.NormalCdfMomentRatio(i, -6) * MMath.Gamma(i + 1);
                Console.WriteLine("R({0}) = {1}, Zp = {2}", i, f, vgmoments[0][i]);
            }

            // true values computed by tex/factors/matlab/test_variance_gamma2.m
            double scale = 2 * System.Math.Exp(-Gaussian.GetLogProb(2 / System.Math.Sqrt(3), 0, 1));
            Assert.True(MMath.AbsDiff(0.117700554409044, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1, 2, 3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(0.112933034747473, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(2, 2, 3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(0.117331854901251, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.1, 2, 3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(0.115563913123152, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.5, 2, 3) / scale, 1e-20) < 2e-5);

            scale = 2 * System.Math.Exp(-Gaussian.GetLogProb(20 / System.Math.Sqrt(0.3), 0, 1));
            Assert.True(MMath.AbsDiff(1.197359429038085e-009, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1, 20, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(1.239267009054433e-008, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(2, 20, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(1.586340098271600e-009, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.1, 20, 0.3) / scale, 1e-20) < 1e-4);
            Assert.True(MMath.AbsDiff(4.319412089896069e-009, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.5, 20, 0.3) / scale, 1e-20) < 1e-4);

            scale = 2 * System.Math.Exp(-Gaussian.GetLogProb(40 / System.Math.Sqrt(0.3), 0, 1));
            scale = 2 * System.Math.Exp(40 - 0.3 / 2);
            Assert.True(MMath.AbsDiff(2.467941724509690e-018, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1, 40, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(5.022261409377230e-017, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(2, 40, 0.3) / scale, 1e-20) < 1e-5);
            Assert.True(MMath.AbsDiff(3.502361147666615e-018, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.1, 40, 0.3) / scale, 1e-20) < 1e-4);
            Assert.True(MMath.AbsDiff(1.252310352551344e-017, GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.5, 40, 0.3) / scale, 1e-20) < 1e-4);

            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 33.4, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, -33.4, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 58.25, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 50, 0.187) > 0);
            Assert.True(GaussianFromMeanAndVarianceOp.VarianceGammaTimesGaussianIntegral(1.138, 100, 0.187) > 0);
        }

        internal void HeteroscedasticGPR()
        {
            // This model is based on the paper "Most Likely Heteroscedastic Gaussian Process Regression" by Kersting et al, ICML 2007
            // Silverman's motorcycle benchmark dataset
            double[] inputs = new double[]
                {
                2.4, 2.6, 3.2, 3.6, 4, 6.2, 6.6, 6.8, 7.8, 8.199999999999999, 8.800000000000001, 8.800000000000001,
                9.6, 10, 10.2, 10.6, 11, 11.4, 13.2, 13.6, 13.8, 14.6, 14.6, 14.6, 14.6, 14.6, 14.6, 14.8, 15.4, 15.4,
                15.4, 15.4, 15.6, 15.6, 15.8, 15.8, 16, 16, 16.2, 16.2, 16.2, 16.4, 16.4, 16.6, 16.8, 16.8, 16.8, 17.6,
                17.6, 17.6, 17.6, 17.8, 17.8, 18.6, 18.6, 19.2, 19.4, 19.4, 19.6, 20.2, 20.4, 21.2, 21.4, 21.8, 22, 23.2,
                23.4, 24, 24.2, 24.2, 24.6, 25, 25, 25.4, 25.4, 25.6, 26, 26.2, 26.2, 26.4, 27, 27.2, 27.2, 27.2, 27.6,
                28.2, 28.4, 28.4, 28.6, 29.4, 30.2, 31, 31.2, 32, 32, 32.8, 33.4, 33.8, 34.4, 34.8, 35.2, 35.2, 35.4, 35.6,
                35.6, 36.2, 36.2, 38, 38, 39.2, 39.4, 40, 40.4, 41.6, 41.6, 42.4, 42.8, 42.8, 43, 44, 44.4, 45, 46.6, 47.8,
                    47.8, 48.8, 50.6, 52, 53.2, 55, 55, 55.4, 57.6
                };
            double[] outputs = new double[]
                {
                0, -1.3, -2.7, 0, -2.7, -2.7, -2.7, -1.3, -2.7, -2.7, -1.3, -2.7, -2.7, -2.7, -5.4,
                -2.7, -5.4, 0, -2.7, -2.7, 0, -13.3, -5.4, -5.4, -9.300000000000001, -16, -22.8, -2.7, -22.8, -32.1, -53.5,
                -54.9, -40.2, -21.5, -21.5, -50.8, -42.9, -26.8, -21.5, -50.8, -61.7, -5.4, -80.40000000000001, -59, -71,
                -91.09999999999999, -77.7, -37.5, -85.59999999999999, -123.1, -101.9, -99.09999999999999, -104.4, -112.5,
                -50.8, -123.1, -85.59999999999999, -72.3, -127.2, -123.1, -117.9, -134, -101.9, -108.4, -123.1, -123.1, -128.5,
                -112.5, -95.09999999999999, -81.8, -53.5, -64.40000000000001, -57.6, -72.3, -44.3, -26.8, -5.4, -107.1, -21.5,
                -65.59999999999999, -16, -45.6, -24.2, 9.5, 4, 12, -21.5, 37.5, 46.9, -17.4, 36.2, 75, 8.1, 54.9, 48.2, 46.9,
                16, 45.6, 1.3, 75, -16, -54.9, 69.59999999999999, 34.8, 32.1, -37.5, 22.8, 46.9, 10.7, 5.4, -1.3, -21.5, -13.3,
                    30.8, -10.7, 29.4, 0, -10.7, 14.7, -1.3, 0, 10.7, 10.7, -26.8, -14.7, -13.3, 0, 10.7, -14.7, -2.7, 10.7, -2.7, 10.7
                };
            Range j = new Range(inputs.Length);
            Vector[] inputsVec = Util.ArrayInit(inputs.Length, i => Vector.FromArray(inputs[i]));
            VariableArray<Vector> x = Variable.Observed(inputsVec, j).Named("x");
            VariableArray<double> y = Variable.Observed(outputs, j).Named("y");
            // Set up the GP prior, which will be filled in later
            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("prior");
            Variable<SparseGP> prior2 = Variable.New<SparseGP>().Named("prior2");

            // The sparse GP variable - a distribution over functions
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");
            Variable<IFunction> r = Variable<IFunction>.Random(prior2).Named("r");

            Variable<double> mean = Variable.FunctionEvaluate(f, x[j]).Named("mean");
            Variable<double> logVariance = Variable.FunctionEvaluate(r, x[j]).Named("logVariance");
            Variable<double> variance = Variable.Exp(logVariance);
            y[j] = Variable.GaussianFromMeanAndVariance(mean, variance);


            InferenceEngine engine = new InferenceEngine();
            GaussianProcess gp = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(0));
            GaussianProcess gp2 = new GaussianProcess(new ConstantFunction(0), new SquaredExponential(0));
            // Fill in the sparse GP prior
            //Vector[] basis = Util.ArrayInit(120, i => Vector.FromArray(0.5*i));
            Vector[] basis = Util.ArrayInit(60, i => Vector.FromArray(1.0 * i));
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, basis));
            prior2.ObservedValue = new SparseGP(new SparseGPFixed(gp2, basis));
            // Infer the posterior Sparse GP
            SparseGP sgp = engine.Infer<SparseGP>(f);

            // Check that training set is classified correctly
            Console.WriteLine();
            Console.WriteLine("Predictions on training set:");
            for (int i = 0; i < outputs.Length; i++)
            {
                Gaussian post = sgp.Marginal(inputsVec[i]);
                //double postMean = post.GetMean();
                Console.WriteLine("f({0}) = {1}", inputs[i], post);
            }
            //TODO: change path for cross platform using
            using (MatlabWriter writer = new MatlabWriter(@"..\..\HGPR.mat"))
            {
                int n = outputs.Length;
                double[] m = new double[n];
                double[] s = new double[n];
                for (int i = 0; i < n; i++)
                {
                    Gaussian post = sgp.Marginal(inputsVec[i]);
                    double mi, vi;
                    post.GetMeanAndVariance(out mi, out vi);
                    m[i] = mi;
                    s[i] = System.Math.Sqrt(vi);
                }
                writer.Write("mean", m);
                writer.Write("std", s);
            }
        }

        public static double BesselKAsympt(double a, double x)
        {
            double poch1 = a + 0.5;
            double poch2 = 0.5 - a;
            double z = -0.5 / x;
            double term = 1;
            double sum = 0;
            for (int k = 0; k < 1000; k++)
            {
                double oldterm = term;
                sum += term;
                term *= poch1 * poch2 / (k + 1) * z;
                poch1++;
                poch2++;
                // must not sum too many terms
                if (System.Math.Abs(term) > System.Math.Abs(oldterm))
                {
                    Console.WriteLine("BesselKAsympt: {0} terms", k);
                    break;
                }
            }
            return sum * 0.5 * MMath.Sqrt2PI / System.Math.Sqrt(x) * System.Math.Exp(-x);
        }
        public static double BesselKSeries(double a, double x)
        {
            if (a == System.Math.Floor(a))
                throw new Exception("a cannot be integer");
            return 0.5 * System.Math.PI / System.Math.Sin(System.Math.PI * a) * (MMath.BesselI(-a, x) - MMath.BesselI(a, x));
        }
        public static double[] xBesselKDerivativesAt0(int n, double a)
        {
            double[] derivs = new double[n + 1];
            derivs[0] = MMath.Gamma(System.Math.Abs(a)) * System.Math.Pow(2, a - 0.5) / System.Math.Sqrt(System.Math.PI);
            if (n > 0)
            {
                derivs[1] = derivs[0];
                if (n > 1)
                {
                    double[] derivs2 = xBesselKDerivativesAt0(n - 2, a - 1);
                    for (int i = 0; i < n - 2; i++)
                    {
                        derivs[i + 2] = derivs[i + 1] - (i + 1) * derivs2[i];
                    }
                }
            }
            return derivs;
        }
        internal void BesselKTest()
        {
            if (true)
            {
                // 12.695270349135780322468450599556863540081194237115
                Console.WriteLine(BesselKSeries(1.1, 0.1));
                Console.WriteLine(BesselKAsympt(1.1, 0.1));
                // 0.000018836375374259574280782608722012995073702923939575
                Console.WriteLine(BesselKSeries(1.1, 10));
                // requires approx x terms
                Console.WriteLine(BesselKAsympt(1.1, 10));
                Console.WriteLine(BesselKSeries(0.5, 10));
                //return;
            }
            if (true)
            {
                double[] derivs = xBesselKDerivativesAt0(10, 0.5);
                for (int i = 0; i < derivs.Length; i++)
                {
                    Console.WriteLine(derivs[i]);
                }
                return;
            }
        }
        internal void GaussianFromMeanAndVarianceTest()
        {
            double mm = 4, vm = 3;
            double mx = 0, vx = 1;
            double b = 3;
            for (int i = 1; i < 10; i++)
            {
                double a = i * 0.5;
                GaussianFromMeanAndVarianceTest(mm, vm, mx, vx, a, b);
                if (true)
                {
                    // test extreme variances
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e6, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e20, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, Double.PositiveInfinity, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e6, mx, vx, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e20, mx, vx, a, b);
                }
            }
        }
        internal void GaussianFromMeanAndVarianceTest2()
        {
            double mm = 4, vm = .3;
            double mx = 0, vx = .001;
            double b = 3;

            for (int i = 1; i < 10; i++)
            {
                //double a = i*0.5;
                double a = 1;
                GaussianFromMeanAndVarianceTest(mm, vm * System.Math.Pow(2, -i), mx, vx, a, b);
                if (true)
                {
                    // test extreme variances
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e6, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm, mx, vx * 1e20, a, b);
                    //GaussianFromMeanAndVarianceTest(mm, vm, mx, Double.PositiveInfinity, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e6, mx, vx, a, b);
                    GaussianFromMeanAndVarianceTest(mm, vm * 1e20, mx, vx, a, b);
                }
            }
        }
        private void GaussianFromMeanAndVarianceTest(double mm, double vm, double mx, double vx, double a, double b)
        {
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(mm, vm).Named("mean");
            Variable<double> variance = Variable.GammaFromShapeAndRate(a, b).Named("variance");
            Variable<double> x = Variable.GaussianFromMeanAndVariance(mean, variance).Named("x");
            Variable.ConstrainEqualRandom(x, new Gaussian(mx, vx));
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            double evExpected;
            Gaussian xExpected;
            Gamma vExpected;
            if (a == 1 || a == 2)
            {
                double c = System.Math.Sqrt(2 * b);
                double m = c * (mx - mm);
                double v = c * c * (vx + vm);
                double Z, mu, m2u;
                VarianceGammaTimesGaussianMoments(a, m, v, out Z, out mu, out m2u);
                evExpected = System.Math.Log(Z * c);
                double vu = m2u - mu * mu;
                double r = Double.IsPositiveInfinity(vx) ? 1.0 : vx / (vx + vm);
                double mp = r * (mu / c + mm) + (1 - r) * mx;
                double vp = r * r * vu / (c * c) + r * vm;
                xExpected = new Gaussian(mp, vp);
                double Zplus1, Zplus2;
                VarianceGammaTimesGaussianMoments(a + 1, m, v, out Zplus1, out mu, out m2u);
                VarianceGammaTimesGaussianMoments(a + 2, m, v, out Zplus2, out mu, out m2u);
                double vmp = a / b * Zplus1 / Z;
                double vm2p = a * (a + 1) / (b * b) * Zplus2 / Z;
                double vvp = vm2p - vmp * vmp;
                vExpected = Gamma.FromMeanAndVariance(vmp, vvp);
            }
            else
            {
                int n = 1000000;
                GaussianEstimator est = new GaussianEstimator();
                GammaEstimator vEst = new GammaEstimator();
                Gaussian xLike = new Gaussian(mx, vx);
                for (int i = 0; i < n; i++)
                {
                    double m = Gaussian.Sample(mm, 1 / vm);
                    double v = Rand.Gamma(a) / b;
                    double xSample = Gaussian.Sample(m, 1 / v);
                    double weight = System.Math.Exp(xLike.GetLogProb(xSample));
                    est.Add(xSample, weight);
                    vEst.Add(v, weight);
                }
                evExpected = System.Math.Log(est.mva.Count / n);
                xExpected = est.GetDistribution(new Gaussian());
                vExpected = vEst.GetDistribution(new Gamma());
            }
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} should be {1}", evActual, evExpected);
            Gaussian xActual = engine.Infer<Gaussian>(x);
            Console.WriteLine("x = {0} should be {1}", xActual, xExpected);
            Gamma vActual = engine.Infer<Gamma>(variance);
            Console.WriteLine("variance = {0} should be {1}", vActual, vExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-4);
            Assert.True(xExpected.MaxDiff(xActual) < 1e-4);
            Assert.True(vExpected.MaxDiff(vActual) < 1e-4);
        }

        private VariableArray<double> GaussianFromMeanAndVarianceSampleSizeModel(Variable<double> input, Range n, bool variance, double noisePrecision = 0)
        {
            var meanVar = Variable.GaussianFromMeanAndVariance(0, 10);
            var b = Variable.Array<double>(n);
            var cleanX = Variable.Array<double>(n);
            if (variance)
                cleanX[n] = Variable.GaussianFromMeanAndVariance(meanVar, input).ForEach(n);
            else
                cleanX[n] = Variable.GaussianFromMeanAndPrecision(meanVar, input).ForEach(n);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(cleanX[n], noisePrecision);

            return x;
        }

        // Commented out since this test is very slow and does not have any assertions
        //[Fact]
        internal void GaussianFromMeanAndVarianceSampleSize()
        {

            double mean = 2.0, variance = 2.0, noisePrecision = 10.0;

            var Nvar = Variable.Observed<int>(1);
            var n = new Range(Nvar);

            var variance_EP = Variable.GammaFromShapeAndRate(1, 1);
            var x_ep = GaussianFromMeanAndVarianceSampleSizeModel(variance_EP, n, true, noisePrecision);
            var ie_EP = new InferenceEngine();
            ie_EP.ShowWarnings = false;
            ie_EP.ShowProgress = false;

            var ev = Variable.Bernoulli(.5);
            var modelBlock = Variable.If(ev);
            var variance_MC = Variable.GammaFromShapeAndRate(1, 1);
            var x_mc = GaussianFromMeanAndVarianceSampleSizeModel(variance_MC, n, true, noisePrecision);
            modelBlock.CloseBlock();
            var ie_MC = new InferenceEngine();
            ie_MC.ShowWarnings = false;
            ie_MC.ShowProgress = false;

            var precision_VMP = Variable.GammaFromShapeAndRate(1, 1);
            var x_vmp = GaussianFromMeanAndVarianceSampleSizeModel(precision_VMP, n, false, noisePrecision);
            var ie_VMP = new InferenceEngine(new VariationalMessagePassing());
            ie_VMP.ShowWarnings = false;
            ie_VMP.ShowProgress = false;

            Console.WriteLine("N EP MC VMP EMP");
            for (int j = 0; j < 10; j++)
            {
                Rand.Restart(2);
                //int N = (int)Math.Pow(10, j + 1);
                int N = 10 * (j + 1);

                var data = Enumerable.Range(0, N).Select(i => Gaussian.Sample(mean, 1.0 / variance) + Gaussian.Sample(0, noisePrecision)).ToArray();

                Nvar.ObservedValue = N;
                x_ep.ObservedValue = data;
                x_mc.ObservedValue = data;
                x_vmp.ObservedValue = data;

                double epMean = double.NaN;
                try
                {
                    epMean = ie_EP.Infer<Gamma>(variance_EP).GetMean();
                }
                catch
                {
                }
                ;

                double vmpMean = ie_VMP.Infer<Gamma>(precision_VMP).GetMeanInverse();
                double varianceSample = 2;
                var ge = new GammaEstimator();

                Converter<double, double> f = vari =>
                {
                    variance_MC.ObservedValue = vari;
                    return ie_MC.Infer<Bernoulli>(ev).LogOdds;
                };
                int burnin = 1000, thin = 5, numSamples = 1000;
                var samples = new double[numSamples];
                for (int i = 0; i < burnin; i++)
                    varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                for (int i = 0; i < numSamples; i++)
                {
                    for (int k = 0; k < thin; k++)
                        varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                    samples[i] = varianceSample;
                    ge.Add(varianceSample);
                }
                double mcMean = samples.Sum() / numSamples;
                var empiricalMean = data.Sum() / data.Length;
                var empVar = data.Sum(o => o * o) / data.Length - empiricalMean * empiricalMean;
                Console.WriteLine(N + " " + epMean + " " + mcMean + " " + vmpMean + " " + empVar);
            }
        }

        private VariableArray<double> GaussianPlusNoiseModel(Variable<double> varianceOrPrecision, Range n, bool isVariance, VariableArray<double> noisePrecision)
        {
            var mean = Variable.GaussianFromMeanAndVariance(0, 100);
            var b = Variable.Array<double>(n);
            var cleanX = Variable.Array<double>(n);
            if (isVariance)
                cleanX[n] = Variable.GaussianFromMeanAndVariance(mean, varianceOrPrecision).ForEach(n);
            else
                cleanX[n] = Variable.GaussianFromMeanAndPrecision(mean, varianceOrPrecision).ForEach(n);
            var x = Variable.Array<double>(n);
            x[n] = Variable.GaussianFromMeanAndPrecision(cleanX[n], noisePrecision[n]);
            return x;
        }


        // Commented out since this test is very slow and does not have any assertions
        //[Fact]
        internal void GaussianFromMeanAndVarianceVaryNoise()
        {

            double mean = 2.0, variance = 2.0;

            var Nvar = Variable.Observed<int>(1);
            var n = new Range(Nvar);

            var noisePrecision = Variable.Array<double>(n);

            var variance_EP = Variable.GammaFromShapeAndRate(1, 1);
            var x_ep = GaussianPlusNoiseModel(variance_EP, n, true, noisePrecision);
            var ie_EP = new InferenceEngine();
            ie_EP.ShowWarnings = false;
            ie_EP.ShowProgress = false;

            var ev = Variable.Bernoulli(.5);
            var modelBlock = Variable.If(ev);
            var variance_MC = Variable.GammaFromShapeAndRate(1, 1);
            var x_mc = GaussianPlusNoiseModel(variance_MC, n, true, noisePrecision);
            modelBlock.CloseBlock();
            var ie_MC = new InferenceEngine();
            ie_MC.ShowWarnings = false;
            ie_MC.ShowProgress = false;

            var precision_VMP = Variable.GammaFromShapeAndRate(1, 1);
            var x_vmp = GaussianPlusNoiseModel(precision_VMP, n, false, noisePrecision);
            var ie_VMP = new InferenceEngine(new VariationalMessagePassing());
            ie_VMP.ShowWarnings = false;
            ie_VMP.ShowProgress = false;

            Console.WriteLine("N EP MC VMP EMP");
            for (int j = 0; j < 10; j++)
            {
                Rand.Restart(2);
                //int N = (int)Math.Pow(10, j + 1);
                int N = 10 * (j + 1);

                var noiseP = Enumerable.Range(0, N).Select(i =>
                {
                    if (i % 3 == 0)
                        return .1;
                    if (i % 3 == 1)
                        return 1;
                    else
                        return 10;
                }).ToArray();

                var data = noiseP.Select(i => Gaussian.Sample(mean, 1.0 / variance) + Gaussian.Sample(0, i)).ToArray();

                Nvar.ObservedValue = N;
                noisePrecision.ObservedValue = noiseP;
                x_ep.ObservedValue = data;
                x_mc.ObservedValue = data;
                x_vmp.ObservedValue = data;

                double epMean = double.NaN;
                try
                {
                    epMean = ie_EP.Infer<Gamma>(variance_EP).GetMean();
                }
                catch
                {
                }
                ;

                double vmpMean = ie_VMP.Infer<Gamma>(precision_VMP).GetMeanInverse();
                double varianceSample = 2;
                var ge = new GammaEstimator();

                Converter<double, double> f = vari =>
                {
                    variance_MC.ObservedValue = vari;
                    return ie_MC.Infer<Bernoulli>(ev).LogOdds;
                };
                int burnin = 1000, thin = 5, numSamples = 1000;
                var samples = new double[numSamples];
                for (int i = 0; i < burnin; i++)
                    varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                for (int i = 0; i < numSamples; i++)
                {
                    for (int k = 0; k < thin; k++)
                        varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                    samples[i] = varianceSample;
                    ge.Add(varianceSample);
                }
                double mcMean = samples.Sum() / numSamples;
                var empiricalMean = data.Sum() / data.Length;
                var empVar = data.Sum(o => o * o) / data.Length - empiricalMean * empiricalMean;
                Console.WriteLine(N + " " + epMean + " " + mcMean + " " + vmpMean + " " + empVar);
            }
        }

        // Commented out since this test is very slow and does not have any assertions
        //[Fact]
        internal void GaussianFromMeanAndVarianceVaryTruePrec()
        {

            double mean = 2.0;

            var Nvar = Variable.Observed<int>(50);
            var n = new Range(Nvar);

            var noisePrecision = 10;

            var variancePriors = new double[] { 1, 5, 10 }.Select(i => Gamma.FromShapeAndRate(i, i)).ToArray();

            var varPriorVar = Variable.Observed(new Gamma());

            var variance_EP = Variable<double>.Random(varPriorVar);
            var x_ep = GaussianFromMeanAndVarianceSampleSizeModel(variance_EP, n, true, noisePrecision);
            var ie_EP = new InferenceEngine();
            ie_EP.ShowWarnings = false;
            ie_EP.ShowProgress = false;

            var ev = Variable.Bernoulli(.5);
            var modelBlock = Variable.If(ev);
            var variance_MC = Variable<double>.Random(varPriorVar);
            var x_mc = GaussianFromMeanAndVarianceSampleSizeModel(variance_MC, n, true, noisePrecision);
            modelBlock.CloseBlock();
            var ie_MC = new InferenceEngine();
            ie_MC.ShowWarnings = false;
            ie_MC.ShowProgress = false;

            var precision_VMP = Variable.GammaFromShapeAndRate(1, 1);
            var x_vmp = GaussianFromMeanAndVarianceSampleSizeModel(precision_VMP, n, false, noisePrecision);
            var ie_VMP = new InferenceEngine(new VariationalMessagePassing());
            ie_VMP.ShowWarnings = false;
            ie_VMP.ShowProgress = false;

            var ie_EP_prec = new InferenceEngine();
            ie_EP_prec.ShowWarnings = false;
            ie_EP_prec.ShowProgress = false;

            Console.WriteLine("var " + variancePriors.Select(i => "EP" + i.Shape).Aggregate((i, j) => i + " " + j) + " "
                      + variancePriors.Select(i => "MC" + i.Shape).Aggregate((i, j) => i + " " + j) + " EPp VMP EMP");
            for (int j = 0; j < 10; j++)
            {
                Rand.Restart(2);
                //int N = (int)Math.Pow(10, j + 1);
                double variance = System.Math.Exp(-5 + j);

                var data = Enumerable.Range(0, Nvar.ObservedValue).Select(i => Gaussian.Sample(mean, 1.0 / variance) + Gaussian.Sample(0, noisePrecision)).ToArray();

                x_ep.ObservedValue = data;
                x_mc.ObservedValue = data;
                x_vmp.ObservedValue = data;

                var epMeans = variancePriors.Select(i =>
                {
                    double res = double.NaN;
                    varPriorVar.ObservedValue = i;
                    try
                    {
                        res = ie_EP.Infer<Gamma>(variance_EP).GetMean();
                    }
                    catch
                    {
                    }
                    ;
                    return res;
                }).ToArray();

                var mcMeans = variancePriors.Select(p =>
                {
                    double varianceSample = 2;
                    var ge = new GammaEstimator();
                    varPriorVar.ObservedValue = p;
                    Converter<double, double> f = vari =>
                    {
                        variance_MC.ObservedValue = vari;
                        return ie_MC.Infer<Bernoulli>(ev).LogOdds;
                    };
                    int burnin = 1000, thin = 5, numSamples = 1000;
                    var samples = new double[numSamples];
                    for (int i = 0; i < burnin; i++)
                        varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                    for (int i = 0; i < numSamples; i++)
                    {
                        for (int k = 0; k < thin; k++)
                            varianceSample = NonconjugateVMP2Tests.SliceSampleUnivariate(varianceSample, f, lower_bound: 0);
                        samples[i] = varianceSample;
                        ge.Add(varianceSample);
                    }
                    return samples.Sum() / numSamples;
                }).ToArray();


                double epMean2 = double.NaN;
                try
                {
                    epMean2 = ie_EP_prec.Infer<Gamma>(precision_VMP).GetMeanInverse();
                }
                catch
                {
                }
                ;

                double vmpMean = ie_VMP.Infer<Gamma>(precision_VMP).GetMeanInverse();

                var empiricalMean = data.Sum() / data.Length;
                var empVar = data.Sum(o => o * o) / data.Length - empiricalMean * empiricalMean;
                Console.Write("{0:G3} ", variance);
                foreach (var i in epMeans)
                    Console.Write("{0:G3} ", i);
                foreach (var i in mcMeans)
                    Console.Write("{0:G3} ", i);
                Console.WriteLine("{0:G3} {1:G3} {2:G3}", epMean2, vmpMean, empVar);
            }
        }



        private static void VarianceGammaTimesGaussianMoments(double a, double m, double v, out double sum, out double moment1, out double moment2)
        {
            int n = 1000000;
            double lowerBound = -20;
            double upperBound = 20;
            double range = upperBound - lowerBound;
            sum = 0;
            double sumx = 0, sumx2 = 0;
            for (int i = 0; i < n; i++)
            {
                double x = range * i / (n - 1) + lowerBound;
                double absx = System.Math.Abs(x);
                double diff = x - m;
                double logp = -0.5 * diff * diff / v - absx;
                double p = System.Math.Exp(logp);
                if (a == 1)
                { // do nothing
                }
                else if (a == 2)
                {
                    p *= 1 + absx;
                }
                else if (a == 3)
                {
                    p *= 3 + 3 * absx + absx * absx;
                }
                else if (a == 4)
                {
                    p *= 15 + 15 * absx + 6 * absx * absx + absx * absx * absx;
                }
                else
                    throw new ArgumentException("a is not in {1,2,3,4}");
                sum += p;
                sumx += x * p;
                sumx2 += x * x * p;
            }
            moment1 = sumx / sum;
            moment2 = sumx2 / sum;
            sum /= MMath.Gamma(a) * System.Math.Pow(2, a);
            sum /= MMath.Sqrt2PI * System.Math.Sqrt(v);
            double inc = range / (n - 1);
            sum *= inc;
        }
    }
}
