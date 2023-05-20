// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

using Microsoft.ML.Probabilistic.Math;

using Xunit;

using Assert = Xunit.Assert;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class RandomTests
    {
        public const double TOLERANCE = 4.0e-2;
        public const int nsamples = 100000;

        [Fact]
        public void RandSample_HasCorrectDistribution()
        {
            // Uniform integers
            Vector hist = Vector.Zero(4);
            for (int i = 0; i < nsamples; i++)
            {
                int x = Rand.Int(4);
                hist[x]++;
            }

            hist *= (1.0 / nsamples);
            Vector unif = Vector.Constant(4, 0.25);
            double error = hist.MaxDiff(unif);
            if (error > TOLERANCE)
            {
                Assert.True(false, $"Uniform: error={error}");
            }

            hist.SetAllElementsTo(0);
            double[] init1 = new double[] { 0.1, 0.2, 0.3, 0.4 };
            Vector p = Vector.FromArray(init1);
            for (int i = 0; i < nsamples; i++)
            {
                int x = Rand.Sample(p);
                hist[x]++;
            }
            hist *= (1.0 / nsamples);
            error = hist.MaxDiff(p);
            if (error > TOLERANCE)
            {
                Assert.True(false, $"Sample([0.1,0.2,0.3,0.4]) error={error}");
            }
        }

        [Fact]
        public void RandSample_NeverDrawsZero()
        {
            const int nsamples = 1_000_000;
            double[] init1 = new double[] { 0, double.Epsilon };
            Vector p = Vector.FromArray(init1);
            for (int i = 0; i < nsamples; i++)
            {
                int x = Rand.Sample(p);
                Assert.Equal(1, x);
                x = Rand.Sample((IList<double>)p, p[1]);
                Assert.Equal(1, x);
            }
        }

        [Fact]
        public void RandRestart()
        {
            Rand.Restart(7);
            double first = Rand.Double();
            Rand.Restart(7);
            double second = Rand.Double();
            Assert.Equal(first, second);
        }

        [Fact]
        public void RandNormal()
        {
            double sum = 0, sum2 = 0;
            for (int i = 0; i < nsamples; i++)
            {
                double x = Rand.Normal();
                sum += x;
                sum2 += x * x;
            }
            double m = sum / nsamples;
            double v = sum2 / nsamples - m * m;
            // the sample mean has stddev = 1/sqrt(n)
            double dError = System.Math.Abs(m);
            if (dError > 4 / System.Math.Sqrt(nsamples))
            {
                Assert.True(false, $"m: error = {dError}");
            }

            // the sample variance is Gamma(n/2,n/2) whose stddev = sqrt(2/n)
            dError = System.Math.Abs(v - 1.0);
            if (dError > 4 * System.Math.Sqrt(2.0 / nsamples))
            {
                Assert.True(false, $"v: error = {dError}");
            }
        }

        [Fact]
        public void RandNormalGreaterThanTest()
        {
            RandNormalGreaterThan(double.NegativeInfinity);
            RandNormalGreaterThan(-1);
            RandNormalGreaterThan(0);
            RandNormalGreaterThan(1);
            RandNormalGreaterThan(10);
            RandNormalGreaterThan(100);
            RandNormalGreaterThan(1000);
            RandNormalGreaterThan(10000);
        }

        private void RandNormalGreaterThan(double lowerBound)
        {
            new Probabilistic.Distributions.TruncatedGaussian(0, 1, lowerBound, double.PositiveInfinity).GetMeanAndVariance(out double meanExpected, out double varianceExpected);

            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            for (int i = 0; i < nsamples; i++)
            {
                double x = Rand.NormalGreaterThan(lowerBound);
                mva.Add(x);
            }
            double m = mva.Mean;
            double v = mva.Variance;
            // the sample mean has stddev = 1/sqrt(n)
            double dError = System.Math.Abs(m - meanExpected);
            if (dError > 4 / System.Math.Sqrt(nsamples))
            {
                Assert.True(false, $"m: error = {dError}");
            }

            // the sample variance is Gamma(n/2,n/2) whose stddev = sqrt(2/n)
            dError = System.Math.Abs(v - varianceExpected);
            if (dError > 4 * System.Math.Sqrt(2.0 / nsamples))
            {
                Assert.True(false, $"v: error = {dError}");
            }
        }

        [Fact]
        public void RandNormalBetweenTest()
        {
            RandNormalBetween(-0.1, 0.2);
            RandNormalBetween(-3, 1);
            RandNormalBetween(-10, 11);
            RandNormalBetween(100, 100.1);
            RandNormalBetween(-1000.1, -1000);
        }

        private void RandNormalBetween(double lowerBound, double upperBound)
        {
            new Probabilistic.Distributions.TruncatedGaussian(0, 1, lowerBound, upperBound).GetMeanAndVariance(out double meanExpected, out double varianceExpected);

            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            for (int i = 0; i < nsamples; i++)
            {
                double x = Rand.NormalBetween(lowerBound, upperBound);
                mva.Add(x);
            }
            double m = mva.Mean;
            double v = mva.Variance;
            // the sample mean has stddev = 1/sqrt(n)
            double dError = System.Math.Abs(m - meanExpected);
            if (dError > 4 / System.Math.Sqrt(nsamples))
            {
                Assert.True(false, $"m: error = {dError}");
            }

            // the sample variance is Gamma(n/2,n/2) whose stddev = sqrt(2/n)
            dError = System.Math.Abs(v - varianceExpected);
            if (dError > 4 * System.Math.Sqrt(2.0 / nsamples))
            {
                Assert.True(false, $"v: error = {dError}");
            }
        }

        [Fact]
        public void RandGammaTest()
        {
            RandGamma(0.7);
            RandGamma(0.01);
        }

        private void RandGamma(double a)
        {
            double sum = 0;
            double sum2 = 0;
            for (int i = 0; i < nsamples; i++)
            {
                double x = Rand.Gamma(a);
                sum += x;
                sum2 += x * x;
            }
            double m = sum / nsamples;
            double v = sum2 / nsamples - m * m;

            // sample mean has stddev = sqrt(a/n)
            double dError = System.Math.Abs(m - a);
            if (dError > 4 * System.Math.Sqrt(a / nsamples))
            {
                Assert.True(false, $"m: error = {dError}");
            }

            dError = System.Math.Abs(v - a);
            if (dError > TOLERANCE)
            {
                Assert.True(false, $"v: error = {dError}");
            }
        }

        [Fact]
        public void RandNormal2()
        {
            double[] m_init = { 1, 2, 3 };
            Vector mt = Vector.FromArray(m_init);

            double[,] v_init = { { 2, 1, 1 }, { 1, 2, 1 }, { 1, 1, 2 } };
            PositiveDefiniteMatrix vt = new PositiveDefiniteMatrix(v_init);

            int d = mt.Count;
            Vector x = Vector.Zero(d);
            VectorMeanVarianceAccumulator mva = new VectorMeanVarianceAccumulator(d);
            for (int i = 0; i < nsamples; i++)
            {
                Rand.Normal(mt, vt, x);
                mva.Add(x);
            }
            Vector m = mva.Mean;
            Matrix v = mva.Variance;
            double dError = m.MaxDiff(mt);
            if (dError > TOLERANCE)
            {
                Assert.True(false, $"m: error = {dError}");
            }

            dError = v.MaxDiff(vt);
            if (dError > TOLERANCE)
            {
                Assert.True(false, $"v: error = {dError}");
            }
        }

        [Fact]
        public void RandNormalP()
        {
            double[] m_init = { 1, 2, 3 };
            Vector mt = Vector.FromArray(m_init);

            double[,] v_init = { { 2, 1, 1 }, { 1, 2, 1 }, { 1, 1, 2 } };
            PositiveDefiniteMatrix vt = new PositiveDefiniteMatrix(v_init);
            PositiveDefiniteMatrix pt = ((PositiveDefiniteMatrix)vt.Clone()).SetToInverse(vt);

            int d = mt.Count;
            Vector x = Vector.Zero(d);
            VectorMeanVarianceAccumulator mva = new VectorMeanVarianceAccumulator(d);
            for (int i = 0; i < nsamples; i++)
            {
                Rand.NormalP(mt, pt, x);
                mva.Add(x);
            }
            Vector m = mva.Mean;
            PositiveDefiniteMatrix v = mva.Variance;

            double dError = m.MaxDiff(mt);
            if (dError > TOLERANCE)
            {
                Assert.True(false, $"m: error = {dError}");
            }

            dError = v.MaxDiff(vt);
            if (dError > TOLERANCE)
            {
                Assert.True(false, $"v: error = {dError}");
            }
        }

        [Fact]
        public void RandWishart()
        {
            // multivariate Gamma
            double a = 2.7;
            int d = 3;
            PositiveDefiniteMatrix mTrue = new PositiveDefiniteMatrix(d, d);
            mTrue.SetToIdentity();
            mTrue.SetToProduct(mTrue, a);
            LowerTriangularMatrix L = new LowerTriangularMatrix(d, d);
            PositiveDefiniteMatrix X = new PositiveDefiniteMatrix(d, d);
            PositiveDefiniteMatrix m = new PositiveDefiniteMatrix(d, d);
            m.SetAllElementsTo(0);
            double s = 0;
            for (int i = 0; i < nsamples; i++)
            {
                Rand.Wishart(a, L);
                X.SetToProduct(L, L.Transpose());
                m += X;
                s += X.LogDeterminant();
            }
            double sTrue = 0;
            for (int i = 0; i < d; i++) sTrue += MMath.Digamma(a - i * 0.5);
            m.Scale(1.0 / nsamples);
            s /= nsamples;

            double dError = m.MaxDiff(mTrue);
            if (dError > TOLERANCE)
            {
                Assert.True(false, $"Wishart({a}) mean: (should be {a}*I), error = {dError}");
            }
            if (System.Math.Abs(s - sTrue) > TOLERANCE)
            {
                Assert.True(false, $"E[logdet]: {s} (should be {sTrue})");
            }
        }

        [Fact]
        public void RandBinomialTest()
        {
            RandBinomial(0.3, 0);
            RandBinomial(0.3, 1);
            RandBinomial(0.3, 2);
            RandBinomial(0.3, 10);
            RandBinomial(0.3, 100);
            RandBinomial(0.3, 1000);
            RandBinomial(1, 200);
        }

        private void RandBinomial(double p, int n)
        {
            double meanExpected = p * n;
            double varianceExpected = p * (1 - p) * n;
            double sum = 0;
            double sum2 = 0;
            for (int i = 0; i < nsamples; i++)
            {
                double x = Rand.Binomial(n, p);
                sum += x;
                sum2 += x * x;
            }
            double mean = sum / nsamples;
            double variance = sum2 / nsamples - mean * mean;
            double error = MMath.AbsDiff(meanExpected, mean, 1e-6);
            if (error > System.Math.Sqrt(varianceExpected) * 5)
            {
                Assert.True(false, $"Binomial({p},{n}) mean = {mean} should be {meanExpected}, error = {error}");
            }
            error = MMath.AbsDiff(varianceExpected, variance, 1e-6);
            if (error > System.Math.Sqrt(varianceExpected) * 5)
            {
                Assert.True(false, $"Binomial({p},{n}) variance = {variance} should be {varianceExpected}, error = {error}");
            }
        }

        [Fact]
        public void RandMultinomialTest()
        {
            RandMultinomial(DenseVector.Zero(0), 0);
            RandMultinomial(DenseVector.Zero(0), 10);
            RandMultinomial(DenseVector.FromArray(1.0), 0);
            RandMultinomial(DenseVector.FromArray(1.0), 10);
            RandMultinomial(DenseVector.FromArray(0.2, 0.8), 0);
            RandMultinomial(DenseVector.FromArray(0.2, 0.8), 10);
            RandMultinomial(DenseVector.FromArray(0.2, 0.3, 0.5), 0);
            RandMultinomial(DenseVector.FromArray(0.2, 0.3, 0.5), 1);
            RandMultinomial(DenseVector.FromArray(0.2, 0.3, 0.5), 10);
        }

        private void RandMultinomial(DenseVector p, int n)
        {
            Vector meanExpected = p * n;
            Vector varianceExpected = p * (1 - p) * n;
            DenseVector sum = DenseVector.Zero(p.Count);
            DenseVector sum2 = DenseVector.Zero(p.Count);
            for (int i = 0; i < nsamples; i++)
            {
                int[] x = Rand.Multinomial(n, p);
                for (int dim = 0; dim < x.Length; dim++)
                {
                    sum[dim] += x[dim];
                    sum2[dim] += x[dim] * x[dim];
                }
            }
            Vector mean = sum * (1.0 / nsamples);
            Vector variance = sum2 * (1.0 / nsamples) - mean * mean;
            for (int dim = 0; dim < p.Count; dim++)
            {
                double error = MMath.AbsDiff(meanExpected[dim], mean[dim], 1e-6);
                ////Console.WriteLine("Multinomial({0},{1})     mean[{5}] = {2} should be {3}, error = {4}", p, n, mean[dim], meanExpected[dim], error, dim);
                Assert.True(error < TOLERANCE);
                error = MMath.AbsDiff(varianceExpected[dim], variance[dim], 1e-6);
                ////Console.WriteLine("Multinomial({0},{1}) variance[{5}] = {2} should be {3}, error = {4}", p, n, variance[dim], varianceExpected[dim], error, dim);
                Assert.True(error < TOLERANCE);
            }
        }

        [Fact]
        public void RandInt_ThrowsIfEqualArguments()
        {
            Assert.Throws<ArgumentException>(() =>
            {
                Rand.Int(9, 9);
            });
        }

        [Fact]
        public void MeanVarianceAccumulator_Add_Infinity()
        {
            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            mva.Add(double.PositiveInfinity);
            mva.Add(0.0);
            Assert.True(double.IsPositiveInfinity(mva.Mean));
            Assert.True(double.IsPositiveInfinity(mva.Variance));
        }

        [Fact]
        public void MeanVarianceAccumulator_Add_ZeroWeight()
        {
            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            mva.Add(4.5, 0.0);
            Assert.Equal(4.5, mva.Mean);
            mva.Add(4.5);
            mva.Add(double.PositiveInfinity, 0.0);
            Assert.Equal(4.5, mva.Mean);
        }
    }
}