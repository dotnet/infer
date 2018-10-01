// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Math = System.Math;
    using Assert = AssertHelper;

    public class QuantileTests
    {
        /// <summary>
        /// Test that QuantileEstimator can handle billions of items.
        /// </summary>
        //[TestMethod, TestCategory("Performance")]
        internal void QuantileEstimatorBillionsTest()
        {
            long n = 100000;
            double maximumError = 1e-2;
            QuantileEstimator est = new QuantileEstimator(maximumError);
            for (int i = 0; i < n; i++)
            {
                if (i % 100 == 0)
                    Console.WriteLine(i);
                for (int j = 0; j < n; j++)
                {
                    est.Add(i + j);
                }
            }
            long expectedCount = n * n;
            ulong actualCount = est.GetCount();
            Console.WriteLine($"count = {actualCount} should be {expectedCount}");
            Assert.Equal(expectedCount, actualCount, maximumError * expectedCount);
            double expectedMedian = n - 2;
            double actualMedian = est.GetQuantile(0.5);
            Console.WriteLine($"median = {actualMedian} should be {expectedMedian}");
            Assert.Equal(expectedMedian, actualMedian, maximumError * expectedMedian);
        }

        /// <summary>
        /// Test that the median is equal to the middle item in an odd number of items.
        /// </summary>
        [Fact]
        public void QuantileEstimator_MedianTest()
        {
            double middle = 3.4;
            double[] x = { 1.2, middle, 5.6 };
            var outer = new OuterQuantiles(x);
            Assert.Equal(outer.GetQuantile(0.5), middle);
            var inner = new InnerQuantiles(3, outer);
            Assert.Equal(inner.GetQuantile(0.5), middle);
            var est = new QuantileEstimator(0.01);
            est.AddRange(x);
            Assert.Equal(est.GetQuantile(0.5), middle);
        }

        [Fact]
        public void QuantileEstimatorDeflationTest()
        {
            double maximumError = 0.05;
            int n = 1000;
            // g1 has weight 1/3, g2 has weight 2/3
            Gaussian g1 = new Gaussian(2, 3);
            Gaussian g2 = new Gaussian(5, 1);
            var est = new QuantileEstimator(maximumError);
            List<double> x = new List<double>();
            for (int i = 0; i < n; i++)
            {
                double sample = g1.Sample();
                x.Add(sample);
                est.Add(sample);
            }
            est.Deflate();
            for (int i = 0; i < n; i++)
            {
                double sample = g2.Sample();
                x.Add(sample);
                x.Add(sample);
                est.Add(sample);
            }
            CheckProbLessThan(est, x, maximumError);
        }

        [Fact]
        public void QuantileEstimatorMergingTest()
        {
            double maximumError = 0.05;
            // draw many samples from N(m,v)
            Rand.Restart(0);
            double m = 2;
            double stddev = 3;
            Gaussian prior = new Gaussian(m, stddev * stddev);
            var est = new QuantileEstimator(maximumError);
            List<double> x = new List<double>();
            int batchCount = 10;
            for (int batch = 0; batch < batchCount; batch++)
            {
                var est2 = new QuantileEstimator(maximumError);
                int n = ((1 << 5) - 1) * 46 + 1;
                for (int i = 0; i < n; i++)
                {
                    double sample = prior.Sample();
                    x.Add(sample);
                    est2.Add(sample);
                }
                est.Add(est2);
            }
            CheckProbLessThan(est, x, maximumError);
        }

        [Fact]
        public void QuantileEstimator_DuplicationTest()
        {
            double middle = 3.4;
            double[] x = { 1.2, middle, middle, middle, 5.6 };
            var outer = new OuterQuantiles(x);
            Assert.Equal(0.25, outer.GetProbLessThan(middle));
            Assert.Equal(outer.GetQuantile(0.3), middle);
            Assert.Equal(outer.GetQuantile(0.5), middle);
            Assert.Equal(outer.GetQuantile(0.7), middle);
            CheckGetQuantile(outer, outer);
            var inner = new InnerQuantiles(7, outer);
            Assert.Equal(0.25, inner.GetProbLessThan(middle));
            Assert.Equal(outer.GetQuantile(0.3), middle);
            Assert.Equal(outer.GetQuantile(0.5), middle);
            Assert.Equal(outer.GetQuantile(0.7), middle);
            CheckGetQuantile(inner, inner, 100/8, 100*7/8);
            var est = new QuantileEstimator(0.01);
            est.AddRange(x);
            Assert.Equal(0.25, est.GetProbLessThan(middle));
            Assert.Equal(est.GetQuantile(0.3), middle);
            Assert.Equal(est.GetQuantile(0.5), middle);
            Assert.Equal(est.GetQuantile(0.7), middle);
            CheckGetQuantile(est, est);
        }

        [Fact]
        public void QuantileEstimator_DoubleDuplicationTest()
        {
            double first = 1;
            double second = 2;
            double between = (first + second) / 2;
            double next = second + MMath.Ulp(second);
            double[] x = { first, first, second, second };
            // quantiles are 0, 1/3, 2/3, 1
            var outer = new OuterQuantiles(x);
            Assert.Equal(0.0, outer.GetProbLessThan(first));
            Assert.Equal(first, outer.GetQuantile(0.0));
            Assert.Equal(0.5, outer.GetProbLessThan(between));
            Assert.Equal(between, outer.GetQuantile(0.5));
            Assert.Equal(2.0 / 3, outer.GetProbLessThan(second));
            Assert.Equal(second, outer.GetQuantile(2.0 / 3));
            Assert.Equal(1.0, outer.GetProbLessThan(next));
            Assert.Equal(next, outer.GetQuantile(1.0));
            CheckGetQuantile(outer, outer);
            var inner = new InnerQuantiles(5, outer);
            CheckGetQuantile(inner, inner, (int)Math.Ceiling(100.0/6), (int)Math.Floor(100.0*5/6));
            var est = new QuantileEstimator(0.01);
            est.Add(first, 2);
            est.Add(second, 2);
            Assert.Equal(0.0, est.GetProbLessThan(first));
            Assert.Equal(first, est.GetQuantile(0.0));
            Assert.Equal(0.5, est.GetProbLessThan(between));
            Assert.Equal(between, est.GetQuantile(0.5));
            Assert.Equal(2.0 / 3, est.GetProbLessThan(second));
            Assert.Equal(second, est.GetQuantile(2.0 / 3));
            Assert.Equal(1.0, est.GetProbLessThan(next));
            Assert.Equal(next, est.GetQuantile(1.0));
            CheckGetQuantile(est, est);
        }

        /// <summary>
        /// Tests a tricky case with duplicated data.
        /// </summary>
        [Fact]
        public void QuantileEstimator_DoubleDuplicationTest2()
        {
            var data = new double[]
            {
                0.16659357378138889, // 0
                0.70210023978217528, // 0.25
                0.70210023978217528, // 0.5
                0.70319732172768734, // 0.75
                0.70319732172768734  // 1
            };
            var est = new QuantileEstimator(0.01);
            est.AddRange(data);
            Assert.Equal(data[4], est.GetQuantile(0.76));
            Assert.Equal(data[2], est.GetQuantile(0.3));
            CheckGetQuantile(est, est);
            var outer = new OuterQuantiles(data);
            Assert.Equal(data[4], outer.GetQuantile(0.76));
            Assert.Equal(data[2], outer.GetQuantile(0.3));
            CheckGetQuantile(outer, outer);
            var inner = new InnerQuantiles(7, outer);
            CheckGetQuantile(inner, inner, (int)Math.Ceiling(100.0/8), (int)Math.Floor(100.0*7/8));
        }

        internal void CheckGetQuantile(CanGetQuantile canGetQuantile, CanGetProbLessThan canGetProbLessThan, int minPercent = 0, int maxPercent = 100)
        {
            for (int i = minPercent; i < maxPercent; i++)
            {
                // probability = 1.0 is excluded
                double probability = i / 100.0;
                double x = canGetQuantile.GetQuantile(probability);
                double probLessThanX = canGetProbLessThan.GetProbLessThan(x);
                Assert.True(probLessThanX <= probability);
                double next = x + MMath.Ulp(x);
                double probLessThanNext = canGetProbLessThan.GetProbLessThan(next);
                Assert.True(probLessThanNext > probability);
            }
        }

        [Fact]
        public void QuantileEstimator_AllEqualTest()
        {
            double middle = 3.4;
            double next = middle + MMath.Ulp(middle);
            double[] x = { middle, middle, middle };
            foreach (int weight in new[] { 1, 2, 3 })
            {
                var outer = new OuterQuantiles(x);
                Assert.Equal(0.0, outer.GetProbLessThan(middle));
                Assert.Equal(outer.GetQuantile(0.0), middle);
                Assert.Equal(1.0, outer.GetProbLessThan(next));
                Assert.Equal(outer.GetQuantile(1.0), next);
                var est = new QuantileEstimator(0.01);
                foreach (var item in x)
                {
                    est.Add(item, weight);
                }
                Assert.Equal(0.0, est.GetProbLessThan(middle));
                Assert.Equal(est.GetQuantile(0.0), middle);
                Assert.Equal(1.0, est.GetProbLessThan(next));
                Assert.Equal(est.GetQuantile(1.0), next);
            }
        }

        [Fact]
        public void QuantileEstimatorTest()
        {
            foreach (int n in new[] { 20, 10000 })
            {
                QuantileEstimator(0.05, n);
                QuantileEstimator(0.01, n);
                QuantileEstimator(0.005, n);
                QuantileEstimator(0.001, n);
            }
        }

        private void QuantileEstimator(double maximumError, int n)
        {
            // draw many samples from N(m,v)
            Rand.Restart(0);
            double m = 2;
            double stddev = 3;
            Gaussian prior = new Gaussian(m, stddev * stddev);
            var est = new QuantileEstimator(maximumError);
            List<double> x = new List<double>();
            for (int i = 0; i < n; i++)
            {
                double sample = prior.Sample();
                x.Add(sample);
                est.Add(sample);
            }
            CheckProbLessThan(est, x, (n < 40) ? 0 : maximumError);
        }

        private void CheckProbLessThan(CanGetProbLessThan canGetProbLessThan, List<double> x, double maximumError)
        {
            x.Sort();
            var sortedData = new OuterQuantiles(x.ToArray());
            // check that quantiles match within the desired accuracy
            var min = MMath.Min(x);
            var max = MMath.Max(x);
            var range = max - min;
            var margin = range * 0.01;
            var testPoints = EpTests.linspace(min - margin, max + margin, 100);
            double maxError = 0;
            foreach (var testPoint in testPoints)
            {
                var trueRank = sortedData.GetProbLessThan(testPoint);
                var estRank = canGetProbLessThan.GetProbLessThan(testPoint);
                var error = System.Math.Abs(trueRank - estRank);
                maxError = System.Math.Max(maxError, error);
            }
            Console.WriteLine($"max rank error = {maxError}");
            Assert.True(maxError <= 2 * maximumError);
        }

        [Fact]
        public void QuantileTest()
        {
            // draw many samples from N(m,v)
            Rand.Restart(0);
            int n = 10000;
            double m = 2;
            double stddev = 3;
            Gaussian prior = new Gaussian(m, stddev * stddev);
            List<double> x = new List<double>();
            for (int i = 0; i < n; i++)
            {
                x.Add(prior.Sample());
            }
            x.Sort();
            var sortedData = new OuterQuantiles(x.ToArray());

            // compute quantiles
            var quantiles = new InnerQuantiles(100, sortedData);

            // loop over x's and compare true quantile rank
            var testPoints = EpTests.linspace(MMath.Min(x) - stddev, MMath.Max(x) + stddev, 100);
            double maxError = 0;
            foreach (var testPoint in testPoints)
            {
                var trueRank = MMath.NormalCdf((testPoint - m) / stddev);
                var estRank = quantiles.GetProbLessThan(testPoint);
                var error = System.Math.Abs(trueRank - estRank);
                //Trace.WriteLine($"{testPoint} trueRank={trueRank} estRank={estRank} error={error}");
                Assert.True(error < 0.02);
                maxError = System.Math.Max(maxError, error);

                double estQuantile = quantiles.GetQuantile(estRank);
                error = MMath.AbsDiff(estQuantile, testPoint, 1e-8);
                //Trace.WriteLine($"{testPoint} estRank={estRank} estQuantile={estQuantile} error={error}");
                Assert.True(error < 1e-8);

                estRank = sortedData.GetProbLessThan(testPoint);
                error = System.Math.Abs(trueRank - estRank);
                //Trace.WriteLine($"{testPoint} trueRank={trueRank} estRank={estRank} error={error}");
                Assert.True(error < 0.02);
            }
            //Trace.WriteLine($"max rank error = {maxError}");
        }
    }
}
