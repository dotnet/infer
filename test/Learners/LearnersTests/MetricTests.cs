// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Xunit;
    using Assert = AssertHelper;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Collections;



    /// <summary>
    /// Tests for metric computation.
    /// </summary>
    public class MetricTests
    {
        /// <summary>
        /// The tolerance for metric value comparisons.
        /// </summary>
        private const double Tolerance = 1e-9;

        /// <summary>
        /// Tests the point estimator for <see cref="Bernoulli"/> distributions.
        /// </summary>
        [Fact]
        public void TestBernoulliPointEstimator()
        {
            var distribution = new Bernoulli(0.25);

            Assert.False(distribution.GetMode());
            Assert.False(PointEstimator.GetEstimate(distribution, Metrics.ZeroOneError)); // Mode
            Assert.False(distribution.GetMean() >= 0.5);
            Assert.False(PointEstimator.GetEstimate(distribution, Metrics.SquaredError)); // Mean
            Assert.False(PointEstimator.GetEstimate(distribution, Metrics.AbsoluteError)); // Median = Mode

            distribution = new Bernoulli(0.5);
            Assert.True(distribution.GetMode()); 
            Assert.True(PointEstimator.GetEstimate(distribution, Metrics.ZeroOneError)); // Mode
            Assert.True(distribution.GetMean() >= 0.5); 
            Assert.True(PointEstimator.GetEstimate(distribution, Metrics.SquaredError)); // Mean
            Assert.True(PointEstimator.GetEstimate(distribution, Metrics.AbsoluteError)); // Median = Mode

            distribution = new Bernoulli(0.6);
            Assert.True(distribution.GetMode());
            Assert.True(PointEstimator.GetEstimate(distribution, Metrics.ZeroOneError)); // Mode
            Assert.True(distribution.GetMean() >= 0.5);
            Assert.True(PointEstimator.GetEstimate(distribution, Metrics.SquaredError)); // Mean
            Assert.True(PointEstimator.GetEstimate(distribution, Metrics.AbsoluteError)); // Median = Mode

            Assert.Throws<ArgumentNullException>(() => PointEstimator.GetEstimate(distribution, null));

            // Test generic representation of distribution 
            var genericDistribution = new Dictionary<bool, double> { { true, 0.25 }, { false, 0.75 } }; // Order matters!
            Assert.False(PointEstimator.GetEstimate(genericDistribution, Metrics.ZeroOneError)); // Mode
            Assert.False(PointEstimator.GetEstimate(genericDistribution, Metrics.SquaredError)); // Mean
            Assert.False(PointEstimator.GetEstimate(genericDistribution, Metrics.AbsoluteError)); // Median = Mode

            genericDistribution = new Dictionary<bool, double> { { true, 0.5 }, { false, 0.5 } };
            Assert.True(PointEstimator.GetEstimate(genericDistribution, Metrics.ZeroOneError)); // Mode
            Assert.True(PointEstimator.GetEstimate(genericDistribution, Metrics.SquaredError)); // Mean
            Assert.True(PointEstimator.GetEstimate(genericDistribution, Metrics.AbsoluteError)); // Median = Mode

            genericDistribution = new Dictionary<bool, double> { { true, 0.6 }, { false, 0.4 } };
            Assert.True(PointEstimator.GetEstimate(genericDistribution, Metrics.ZeroOneError)); // Mode
            Assert.True(PointEstimator.GetEstimate(genericDistribution, Metrics.SquaredError)); // Mean
            Assert.True(PointEstimator.GetEstimate(genericDistribution, Metrics.AbsoluteError)); // Median = Mode

            Assert.Throws<ArgumentNullException>(() => PointEstimator.GetEstimate(distribution, null));
        }

        /// <summary>
        /// Tests the point estimator for <see cref="Discrete"/> distributions.
        /// </summary>
        [Fact]
        public void TestDiscretePointEstimator()
        {
            var distribution = new Discrete(1, 1, 4, 2);
            Assert.Equal(2, distribution.GetMode());
            Assert.Equal(2, PointEstimator.GetEstimate(distribution, Metrics.ZeroOneError)); // Mode
            Assert.Equal(2, Convert.ToInt32(distribution.GetMean()));
            Assert.Equal(2, PointEstimator.GetEstimate(distribution, Metrics.SquaredError)); // Mean
            Assert.Equal(2, distribution.GetMedian());
            Assert.Equal(2, PointEstimator.GetEstimate(distribution, Metrics.AbsoluteError)); // Median
            Assert.Equal(2, PointEstimator.GetEstimate(distribution, this.LinearLossFunction())); // Median

            Assert.Equal(1, PointEstimator.GetEstimate(distribution, this.LinearLossFunction(3))); // 1st quartile
            Assert.Equal(2, PointEstimator.GetEstimate(distribution, this.LinearLossFunction(1, 3))); // 3rd quartile
            Assert.Equal(3, PointEstimator.GetEstimate(distribution, this.LinearLossFunction(1, 4))); // 4th quintile

            distribution = new Discrete(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
            Assert.Equal(0, distribution.GetMode());
            Assert.Equal(4.5, distribution.GetMean(), 1e-10);
            int median = distribution.GetMedian();
            Assert.True(median == 4 || median == 5);
            Assert.Equal(3, PointEstimator.GetEstimate(distribution, this.LinearLossFunction(3, 2))); // 2nd quintile
            Assert.Equal(7, PointEstimator.GetEstimate(distribution, this.LinearLossFunction(1, 3))); // 3rd quartile
            Assert.Equal(9, PointEstimator.GetEstimate(distribution, this.LinearLossFunction(1, 999))); // 999th permille

            Assert.Throws<ArgumentNullException>(() => PointEstimator.GetEstimate(null, Metrics.AbsoluteError));
            Assert.Throws<ArgumentNullException>(() => PointEstimator.GetEstimate(distribution, null));

            // Test generic representation of distribution
            var genericDistribution = new Dictionary<int, double> { { 0, 1 }, { 1, 1 }, { 2, 4 }, { 3, 2 } }; 
            Assert.Equal(2, genericDistribution.GetMode());
            Assert.Equal(2, PointEstimator.GetEstimate(genericDistribution, Metrics.ZeroOneError)); // Mode
            Assert.Equal(2, PointEstimator.GetEstimate(genericDistribution, Metrics.SquaredError)); // Mean
            Assert.Equal(2, PointEstimator.GetEstimate(genericDistribution, Metrics.AbsoluteError)); // Median
            Assert.Equal(2, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction())); // Median
            Assert.Equal(1, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction(3))); // 1st quartile
            Assert.Equal(2, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction(1, 3))); // 3rd quartile
            Assert.Equal(3, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction(1, 4))); // 4th quintile

            genericDistribution = new Dictionary<int, double> { { 0, 0.1 }, { 1, 0.1 }, { 2, 0.1 }, { 3, 0.1 }, { 4, 0.1 }, { 5, 0.1 }, { 6, 0.1 }, { 7, 0.1 }, { 8, 0.1 }, { 9, 0.1 } };
            Assert.Equal(0, genericDistribution.GetMode());
            Assert.Equal(0, PointEstimator.GetEstimate(genericDistribution, Metrics.ZeroOneError)); // Mode
            Assert.Equal(4, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction())); // Median
            Assert.Equal(3, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction(3, 2))); // 2nd quintile
            Assert.Equal(7, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction(1, 3))); // 3rd quartile
            Assert.Equal(9, PointEstimator.GetEstimate(genericDistribution, this.LinearLossFunction(1, 999))); // 999th permille

            Assert.Throws<ArgumentNullException>(() => PointEstimator.GetEstimate(null, Metrics.AbsoluteError));
            Assert.Throws<ArgumentNullException>(() => PointEstimator.GetEstimate(genericDistribution, null));
        }

        /// <summary>
        /// Tests the zero-one error function.
        /// </summary>
        [Fact]
        public void TestZeroOneError()
        {
            Assert.Equal(0, Metrics.ZeroOneError(1, 1), Tolerance);
            Assert.Equal(0, Metrics.ZeroOneError(0, 0), Tolerance);
            Assert.Equal(1, Metrics.ZeroOneError(3, 1), Tolerance);
            Assert.Equal(1, Metrics.ZeroOneError(-5, -2), Tolerance);
            Assert.Equal(1, Metrics.ZeroOneError(3, -1), Tolerance);

            Assert.Equal(0, Metrics.ZeroOneError("A", "A"), Tolerance);
            Assert.Equal(1, Metrics.ZeroOneError("A", "B"), Tolerance);

            Assert.Equal(0, Metrics.ZeroOneError(true, true), Tolerance);
            Assert.Equal(1, Metrics.ZeroOneError(false, true), Tolerance);
            Assert.Equal(0, Metrics.ZeroOneError(false, false), Tolerance);
            Assert.Equal(1, Metrics.ZeroOneError(true, false), Tolerance);

            Assert.Throws<ArgumentNullException>(() => Metrics.ZeroOneError("A", null));
            Assert.Throws<ArgumentNullException>(() => Metrics.ZeroOneError(null, "B"));
        }

        /// <summary>
        /// Tests the squared error function.
        /// </summary>
        [Fact]
        public void TestSquaredError()
        {
            Assert.Equal(0, Metrics.SquaredError(1, 1), Tolerance);
            Assert.Equal(0, Metrics.SquaredError(0, 0), Tolerance);
            Assert.Equal(4, Metrics.SquaredError(3, 1), Tolerance);
            Assert.Equal(9, Metrics.SquaredError(-5, -2), Tolerance);
            Assert.Equal(16, Metrics.SquaredError(3, -1), Tolerance);

            Assert.Equal(0, Metrics.SquaredError(true, true), Tolerance);
            Assert.Equal(1, Metrics.SquaredError(false, true), Tolerance);
            Assert.Equal(0, Metrics.SquaredError(false, false), Tolerance);
            Assert.Equal(1, Metrics.SquaredError(true, false), Tolerance);
        }

        /// <summary>
        /// Tests the absolute error function.
        /// </summary>
        [Fact]
        public void TestAbsoluteError()
        {
            Assert.Equal(0, Metrics.AbsoluteError(1, 1), Tolerance);
            Assert.Equal(0, Metrics.AbsoluteError(0, 0), Tolerance);
            Assert.Equal(2, Metrics.AbsoluteError(3, 1), Tolerance);
            Assert.Equal(3, Metrics.AbsoluteError(-5, -2), Tolerance);
            Assert.Equal(4, Metrics.AbsoluteError(3, -1), Tolerance);

            Assert.Equal(0, Metrics.AbsoluteError(true, true), Tolerance);
            Assert.Equal(1, Metrics.AbsoluteError(false, true), Tolerance);
            Assert.Equal(0, Metrics.AbsoluteError(false, false), Tolerance);
            Assert.Equal(1, Metrics.AbsoluteError(true, false), Tolerance);
        }

        /// <summary>
        /// Tests the negative log probability computation.
        /// </summary>
        [Fact]
        public void TestNegativeLogProbability()
        {
            var bernoulliDistribution = new Bernoulli(0.25);
            Assert.Equal(Math.Log(4) - Math.Log(3), Metrics.NegativeLogProbability(false, bernoulliDistribution), Tolerance);
            Assert.Equal(Math.Log(4), Metrics.NegativeLogProbability(true, bernoulliDistribution), Tolerance);

            var discreteDistribution = new Discrete(1.0, 1.0, 4.0, 2.0);
            Assert.Equal(Math.Log(8), Metrics.NegativeLogProbability(0, discreteDistribution), Tolerance);
            Assert.Equal(Math.Log(8), Metrics.NegativeLogProbability(1, discreteDistribution), Tolerance);
            Assert.Equal(Math.Log(2), Metrics.NegativeLogProbability(2, discreteDistribution), Tolerance);
            Assert.Equal(Math.Log(4), Metrics.NegativeLogProbability(3, discreteDistribution), Tolerance);

            var labelDistribution = new Dictionary<string, double> { { "A", 0.125 }, { "B", 0.125 }, { "C", 0.5 }, { "D", 0.25 } };
            Assert.Equal(Math.Log(8), Metrics.NegativeLogProbability("A", labelDistribution), Tolerance);
            Assert.Equal(Math.Log(8), Metrics.NegativeLogProbability("B", labelDistribution), Tolerance);
            Assert.Equal(Math.Log(2), Metrics.NegativeLogProbability("C", labelDistribution), Tolerance);
            Assert.Equal(Math.Log(4), Metrics.NegativeLogProbability("D", labelDistribution), Tolerance);

            Assert.Throws<ArgumentException>(() => Metrics.NegativeLogProbability("E", labelDistribution));

            Assert.Throws<ArgumentNullException>(() => Metrics.NegativeLogProbability("A", null));
            Assert.Throws<ArgumentNullException>(() => Metrics.NegativeLogProbability(null, labelDistribution));

            var labelDictionary = new Dictionary<string, double> { { "A", 0.125 }, { "B", 0.125 }, { "C", 0.5 }, { "D", 0.26 } };
            Assert.Throws<ArgumentException>(() => Metrics.NegativeLogProbability("D", labelDictionary));

            var negativelabelDictionary = new Dictionary<string, double> { { "A", 0.5 }, { "B", 0.5 }, { "C", 0.5 }, { "D", -0.5 } };
            Assert.Throws<ArgumentException>(() => Metrics.NegativeLogProbability("D", negativelabelDictionary));

            var shortLabelDictionary = new Dictionary<string, double> { { "A", 1 } };
            Assert.Throws<ArgumentException>(() => Metrics.NegativeLogProbability("A", shortLabelDictionary));
        }

        /// <summary>
        /// Tests the DCG computation.
        /// </summary>
        [Fact]
        public void TestDcg()
        {
            double[] gains = { 3, 1, 2, 2 };
            double dcg = Metrics.Dcg(gains, Metrics.LinearDiscountFunc);
            double trueDcg = (3.0 / 1.0) + (1.0 / 2.0) + (2.0 / 3.0) + (2.0 / 4.0);
            Assert.Equal(trueDcg, dcg, Tolerance);

            // DCG is defined even for empty sequences
            Assert.Equal(0, Metrics.Dcg(new double[0]));
        }

        /// <summary>
        /// Tests the normalized DCG computation.
        /// </summary>
        [Fact]
        public void TestNormalizedDcg()
        {
            double[] gains = { 3, 1, 2, 2 };
            double[] bestGains = gains.OrderByDescending(x => x).ToArray();
            double[] worstGains = new double[gains.Length];
            
            // Normal test
            double ndcg = Metrics.Ndcg(gains, bestGains, Metrics.LinearDiscountFunc);
            const double TrueDcg = (3.0 / 1.0) + (1.0 / 2.0) + (2.0 / 3.0) + (2.0 / 4.0);
            const double BestDcg = (3.0 / 1.0) + (2.0 / 2.0) + (2.0 / 3.0) + (1.0 / 4.0);
            const double TrueNdcg = TrueDcg / BestDcg;
            Assert.Equal(TrueNdcg, ndcg, Tolerance);

            // Test for the extreme cases
            Assert.Equal(0.0, Metrics.Ndcg(worstGains, bestGains));
            Assert.Equal(1.0, Metrics.Ndcg(bestGains, bestGains));
        }

        /// <summary>
        /// Tests if the normalized DCG equals to 1 for the best possible ordering.
        /// </summary>
        [Fact]
        public void TestNormalizedDcgBest()
        {
            double[] gains = { 7, 5, 4, 4, 4, 1 };
            Assert.Equal(1, Metrics.Ndcg(gains, gains, Metrics.LinearDiscountFunc), Tolerance);
            Assert.Equal(1, Metrics.Ndcg(gains, gains, Metrics.LogarithmicDiscountFunc), Tolerance);
        }

        /// <summary>
        /// Tests the graded average precision computation.
        /// </summary>
        [Fact]
        public void TestGap()
        {
            double[] relevances = { 3, 1, 2, 2 };
            double gap = Metrics.GradedAveragePrecision(relevances);
            const double TrueGap =
                ((3.0 / 1.0) + ((1.0 + 1.0) / 2.0) + ((2.0 + 1.0 + 2.0) / 3.0) + ((2.0 + 1.0 + 2.0 + 2.0) / 4.0)) / (3.0 + 1.0 + 2.0 + 2.0);
            Assert.Equal(TrueGap, gap, Tolerance);
        }

        /// <summary>
        /// Tests the computation of the precision-recall curve.
        /// </summary>
        [Fact]
        public void TestPrecisionRecallCurve()
        {
            // General
            var expectedCurve = new[]
                                    {
                                        new PrecisionRecall(1, 0), 
                                        new PrecisionRecall(0, 0),
                                        new PrecisionRecall(0.5, 0.5), 
                                        new PrecisionRecall(2 / (double)3, 1),
                                        new PrecisionRecall(0.5, 1)
                                    };
            var computedCurve = Metrics.PrecisionRecallCurve(new[] { 1, 2 }, new Dictionary<int, double> { { 3, 1 }, { 1, 0.5 }, { 2, 0.25 }, { 4, 0 } }).ToArray();
            Assert.Equal(expectedCurve.Length, computedCurve.Length);
            for (int i = 0; i < expectedCurve.Length; i++)
            {
                Assert.Equal(expectedCurve[i], computedCurve[i]);
            }

            // No instance scores
            expectedCurve = new[] { new PrecisionRecall(1, 0),  };
            computedCurve = Metrics.PrecisionRecallCurve(new[] { 1 }, new Dictionary<int, double>()).ToArray();
            Assert.Equal(expectedCurve.Length, computedCurve.Length);
            for (int i = 0; i < expectedCurve.Length; i++)
            {
                Assert.Equal(expectedCurve[i], computedCurve[i]);
            }

            // No negative instance scores
            expectedCurve = new[] { new PrecisionRecall(1, 0), new PrecisionRecall(1, 1) };
            computedCurve = Metrics.PrecisionRecallCurve(new[] { 1 }, new Dictionary<int, double> { { 1, 1 } }).ToArray();
            Assert.Equal(expectedCurve.Length, computedCurve.Length);
            for (int i = 0; i < expectedCurve.Length; i++)
            {
                Assert.Equal(expectedCurve[i], computedCurve[i]);
            }

            // Duplicate positive instances
            computedCurve = Metrics.PrecisionRecallCurve(new[] { 1, 1 }, new Dictionary<int, double> { { 1, 1 } }).ToArray();
            Assert.Equal(expectedCurve.Length, computedCurve.Length);
            for (int i = 0; i < expectedCurve.Length; i++)
            {
                Assert.Equal(expectedCurve[i], computedCurve[i]);
            }

            // No positive instance scores
            Assert.Throws<ArgumentException>(() => Metrics.PrecisionRecallCurve(new int[] { }, new Dictionary<int, double> { { 1, 1 } }));

            // Null checks
            Assert.Throws<ArgumentNullException>(() => Metrics.PrecisionRecallCurve(null, new Dictionary<int, double> { { 1, 1 } }));
            Assert.Throws<ArgumentNullException>(() => Metrics.PrecisionRecallCurve(new[] { 1 }, null));
        }

        /// <summary>
        /// Tests the computation of the receiver operating characteristic curve.
        /// </summary>
        [Fact]
        public void TestReceiverOperatingCharacteristicCurve()
        {
            // Duplicate instance scores, duplicate positive instances
            var expectedCurve = new[]
                                    {
                                        new FalseAndTruePositiveRate(0, 0), 
                                        new FalseAndTruePositiveRate(0.5, 1),
                                        new FalseAndTruePositiveRate(1, 1)
                                    };
            var computedCurve = Metrics.ReceiverOperatingCharacteristicCurve(new[] { 1, 1, 2 }, new Dictionary<int, double> { { 1, 0.5 }, { 2, 0.5 }, { 3, 0.5 }, { 4, 0 } }).ToArray();
            foreach (var tuple in computedCurve)
            {
                Console.WriteLine(tuple);
            }

            Xunit.Assert.Equal(expectedCurve, computedCurve);

            // No positive instance scores
            Assert.Throws<ArgumentException>(() => Metrics.ReceiverOperatingCharacteristicCurve(new int[] { }, new Dictionary<int, double> { { 1, 1 } }));

            // No negative instance scores
            Assert.Throws<ArgumentException>(() => Metrics.ReceiverOperatingCharacteristicCurve(new[] { 1 }, new Dictionary<int, double> { { 1, 1 } }));
            
            // No instance scores
            Assert.Throws<ArgumentException>(() => Metrics.ReceiverOperatingCharacteristicCurve(new[] { 1 }, new Dictionary<int, double>()));
            
            // Null checks
            Assert.Throws<ArgumentNullException>(() => Metrics.ReceiverOperatingCharacteristicCurve(null, new Dictionary<int, double> { { 1, 1 } }));
            Assert.Throws<ArgumentNullException>(() => Metrics.ReceiverOperatingCharacteristicCurve(new[] { 1 }, null));
        }

        /// <summary>
        /// Tests the computation of the area under the receiver operating characteristic curve.
        /// </summary>
        [Fact]
        public void TestAreaUnderRocCurve()
        {
            // Duplicate instance scores, duplicate positive instances
            Assert.Equal(0.75, Metrics.AreaUnderRocCurve(new[] { 1, 1, 2 }, new Dictionary<int, double> { { 1, 0.5 }, { 2, 0.5 }, { 3, 0.5 }, { 4, 0 } }));
            Assert.Equal(0.5, Metrics.AreaUnderRocCurve(new[] { 1, 1, 2 }, new Dictionary<int, double> { { 1, 0.5 }, { 2, 0.5 }, { 3, 1 }, { 4, 0 } }));

            // No positive instance scores
            Assert.Throws<ArgumentException>(() => Metrics.AreaUnderRocCurve(new int[] { }, new Dictionary<int, double> { { 1, 1 } }));

            // No negative instance scores
            Assert.Throws<ArgumentException>(() => Metrics.AreaUnderRocCurve(new[] { 1 }, new Dictionary<int, double> { { 1, 1 } }));

            // No instance scores
            Assert.Throws<ArgumentException>(() => Metrics.AreaUnderRocCurve(new[] { 1 }, new Dictionary<int, double>()));

            // Null checks
            Assert.Throws<ArgumentNullException>(() => Metrics.AreaUnderRocCurve(null, new Dictionary<int, double> { { 1, 1 } }));
            Assert.Throws<ArgumentNullException>(() => Metrics.AreaUnderRocCurve(new[] { 1 }, null));
        }

        /// <summary>
        /// Tests the cosine similarity computation.
        /// </summary>
        [Fact]
        public void TestCosineSimilarity()
        {
            double[] values1 = { 3, 1, 2, 2 };
            double[] values2 = { 1, 1, 2, -2 };
            double sim = Metrics.CosineSimilarity(Vector.FromArray(values1), Vector.FromArray(values2));
            double trueSim = 4 / Math.Sqrt(18) / Math.Sqrt(10);
            Assert.Equal(trueSim, sim, Tolerance);
        }

        /// <summary>
        /// Tests the Pearson's correlation computation.
        /// </summary>
        [Fact]
        public void TestPearsonCorrelation()
        {
            double[] values1 = { 3, 1, 2, 2 };
            double[] values2 = { 1, 2, 1, 0 };
            double correlation = Metrics.PearsonCorrelation(Vector.FromArray(values1), Vector.FromArray(values2));
            const double TrueCorrelation = -0.5;
            Assert.Equal(TrueCorrelation, correlation, Tolerance);
        }

        /// <summary>
        /// Tests the normalized Euclidean distance based similarity computation.
        /// </summary>
        [Fact]
        public void TestNormalizedEuclideanDistance()
        {
            double[] values1 = { 3, 1, 2, 2 };
            double[] values2 = { 1, 2, 1, 0 };
            double distance = Metrics.NormalizedEuclideanSimilarity(Vector.FromArray(values1), Vector.FromArray(values2));
            double trueDistance = 1.0 / (1.0 + Math.Sqrt(2.5));
            Assert.Equal(trueDistance, distance, Tolerance);
        }

        /// <summary>
        /// Tests the normalized Manhattan distance based similarity computation.
        /// </summary>
        [Fact]
        public void TestNormalizedManhattanDistance()
        {
            double[] values1 = { 3, 1, 2, 2 };
            double[] values2 = { 1, 2, 1, 0 };
            double distance = Metrics.NormalizedManhattanSimilarity(Vector.FromArray(values1), Vector.FromArray(values2));
            const double TrueDistance = 0.4;
            Assert.Equal(TrueDistance, distance, Tolerance);
        }

        /// <summary>
        /// Tests if invalid calls to NDCG computation are handled correctly.
        /// </summary>
        [Fact]
        public void TestInvalidNdcgComputationCalls()
        {
            Assert.Throws<ArgumentNullException>(() => Metrics.LinearNdcg(null, new double[0])); // Null arguments 1
            Assert.Throws<ArgumentNullException>(() => Metrics.LinearNdcg(new double[0], null)); // Null arguments 2
            Assert.Throws<ArgumentNullException>(() => Metrics.LinearNdcg(null, null)); // Null arguments 3
            Assert.Throws<ArgumentException>(() => Metrics.LinearNdcg(new double[0], new double[0])); // Empty gain lists
            Assert.Throws<ArgumentException>(() => Metrics.LinearNdcg(new[] { 1.0 }, new[] { 0.0 })); // Denominator is zero 1
            Assert.Throws<ArgumentException>(() => Metrics.LinearNdcg(new[] { 0.0 }, new[] { 0.0 })); // Denominator is zero 2
            Assert.Throws<ArgumentException>(() => Metrics.LinearNdcg(new[] { 1.0, 1.0 }, new[] { 2.0, -4.0 })); // Denominator is zero 3
            Assert.Throws<ArgumentException>(() => Metrics.LinearNdcg(new[] { 1.0, 1.0 }, new[] { 1.0, -4.0 })); // NDCG is negative
            Assert.Throws<ArgumentException>(() => Metrics.LinearNdcg(new[] { 1.0, 2.0 }, new[] { 1.0, 1.0 })); // NDCG is greater than 1
        }

        #region Helper methods

        /// <summary>
        /// Gets the linear loss function for given parameters a and b. 
        /// Using this loss function yields the b / (a + b) quantile as an estimate of the posterior distribution. 
        /// For example, choosing a = b yields the median and is equivalent to the absolute loss. 
        /// </summary>
        /// <param name="a">The parameter a. Defaults to 1.</param>
        /// <param name="b">The parameter b. Defaults to 1.</param>
        /// <returns>The linear loss function determined by a and b.</returns>
        private Func<int, int, double> LinearLossFunction(double a = 1, double b = 1)
        {
            if (a <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(a), $"The parameter {nameof(a)} needs to be strictly positive.");
            }

            if (b <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(b), $"The parameter {nameof(b)} needs to be strictly positive.");
            }

            return (truth, estimate) =>
            {
                double difference = truth - estimate;
                return difference > 0 ? a * difference : -b * difference;
            };
        }

        #endregion
    }
}
