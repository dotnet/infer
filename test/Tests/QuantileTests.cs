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
        [Fact]
        public void QuantileEstimator_SinglePointIsMedian()
        {
            QuantileEstimator est = new QuantileEstimator(0.1);
            double point = 2;
            est.Add(point);
            Assert.Equal(point, est.GetQuantile(0.5));

            OuterQuantiles outer = new OuterQuantiles(new[] { point });
            Assert.Equal(point, outer.GetQuantile(0.5));

            InnerQuantiles inner = new InnerQuantiles(new[] { point });
            Assert.Equal(point, inner.GetQuantile(0.5));
        }

        /// <summary>
        /// Test a tricky case.
        /// </summary>
        [Fact]
        public void InnerQuantileTest()
        {
            double[] quantiles = { -2.3396737042130806, -2.1060851851919309, -1.8587796492436919, -1.7515040214502977, -1.6631549706936311, -1.5649421094540212, -1.4760970897199182, -1.4120516891795316, -1.3472276831887715, -1.2800915764085863, -1.2315546431485036, -1.1733035015194753, -1.1275506999997809, -1.0868191452824896, -1.0423720676050061, -1.0030087867587449, -0.96427545374863111, -0.917480799606264, -0.88868683894166878, -0.85040868414900372, -0.80942702953353063, -0.78299794937710787, -0.74791530550923879, -0.71057667829968463, -0.6764786230399974, -0.64937712088706545, -0.61647747819758114, -0.585418062478127, -0.55212155586237877, -0.52794712262708809, -0.49602391921870309, -0.4699661621821, -0.44707572988386468, -0.41779003649017038, -0.38751278424822111, -0.3659754249474671, -0.33671101603741, -0.30844051169056652, -0.28736460398884939, -0.26394181175383763, -0.2339421108026867, -0.20421395179821347, -0.17975005820876525, -0.15495505128166037, -0.12881080807789203, -0.10882854018038969, -0.080502768973386082, -0.054592779524389491, -0.030746556623353873, 0.0010699779508669754, 0.018476164506076323, 0.042997842018717161, 0.068170326454891988, 0.098939711480485845, 0.12364243085219064, 0.14897752107634207, 0.17232065117344095, 0.19510514320430472, 0.21967681963331126, 0.25144866739098226, 0.26627058021030359, 0.28976112810281413, 0.325183138022793, 0.34611510490686043, 0.37135045464414679, 0.40484250840269187, 0.423660564514518, 0.45260008550109493, 0.47897070643517381, 0.513466904702678, 0.54074552445523427, 0.56782579073247685, 0.59191546380311844, 0.630594130276651, 0.66170186000470765, 0.69059427870805967, 0.72267836185626344, 0.75388989983592025, 0.78095231060517345, 0.81945443104186122, 0.85806474163877222, 0.88543000730858912, 0.9254742516670329, 0.96663287584250224, 1.0081099518226813, 1.0414716524617549, 1.0873521052324735, 1.138068925150572, 1.1769604530537776, 1.2209510765755074, 1.2805602443304192, 1.3529085306332467, 1.4111760504339896, 1.4822842454846386, 1.5518312997748602, 1.6439254270476189, 1.7357210363862619, 1.9281504259252962, 2.064331420559117, 2.3554568165928291, };
            InnerQuantiles inner = new InnerQuantiles(quantiles);
            inner.GetQuantile(0.49471653842100138);
            inner.GetQuantile((double)quantiles.Length / (quantiles.Length + 1));
        }

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
            double left = 1.2;
            double middle = 3.4;
            double right = 5.6;
            double[] x = { left, middle, right };
            var outer = new OuterQuantiles(x);
            Assert.Equal(middle, outer.GetQuantile(0.5));
            var inner = new InnerQuantiles(3, outer);
            Assert.Equal(middle, inner.GetQuantile(0.5));
            inner = new InnerQuantiles(x);
            CheckGetQuantile(inner, inner, 25, 75);
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
            double next = MMath.NextDouble(second);
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
                double next = MMath.NextDouble(x);
                double probLessThanNext = canGetProbLessThan.GetProbLessThan(next);
                Assert.True(probLessThanNext > probability);
            }
        }

        [Fact]
        public void QuantileEstimator_AllEqualTest()
        {
            double middle = 3.4;
            double next = MMath.NextDouble(middle);
            double[] x = { middle, middle, middle };
            var outer = new OuterQuantiles(x);
            Assert.Equal(0.0, outer.GetProbLessThan(middle));
            Assert.Equal(middle, outer.GetQuantile(0.0));
            Assert.Equal(middle, outer.GetQuantile(0.75));
            Assert.Equal(1.0, outer.GetProbLessThan(next));
            Assert.Equal(next, outer.GetQuantile(1.0));
            foreach (int weight in new[] { 1, 2, 3 })
            {
                var est = new QuantileEstimator(0.01);
                foreach (var item in x)
                {
                    est.Add(item, weight);
                }
                Assert.Equal(0.0, est.GetProbLessThan(middle));
                Assert.Equal(middle, est.GetQuantile(0.0));
                Assert.Equal(1.0, est.GetProbLessThan(next));
                Assert.Equal(next, est.GetQuantile(1.0));
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
