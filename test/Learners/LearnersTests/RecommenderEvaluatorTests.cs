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
    using Microsoft.ML.Probabilistic.Learners.Mappings;

    using DummyFeatureStorage = System.Int16;
    using Instance = System.Tuple<string, string, int, System.Collections.Generic.IDictionary<int, double>, double>;
    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;
    

    /// <summary>
    /// Tests recommender evaluation routines.
    /// </summary>
    public class RecommenderEvaluatorTests
    {
        /// <summary>
        /// Tolerance for metric value comparison.
        /// </summary>
        private const double CompareEps = 1e-9;

        /// <summary>
        /// The dataset to test evaluation on.
        /// </summary>
        private IEnumerable<Instance> dataset;

        /// <summary>
        /// The rating info for <see cref="dataset"/>.
        /// </summary>
        private IStarRatingInfo<double> starRatingInfo;

        /// <summary>
        /// The evaluation engine.
        /// </summary>
        private StarRatingRecommenderEvaluator<IEnumerable<Instance>, string, string, double> evaluator;

        /// <summary>
        /// The prediction dictionary built from <see cref="dataset"/>.
        /// </summary>
        private IDictionary<string, IDictionary<string, int>> predictions;

        /// <summary>
        /// The uncertain prediction dictionary built from <see cref="dataset"/>.
        /// </summary>
        private IDictionary<string, IDictionary<string, RatingDistribution>> uncertainPredictions;

        /// <summary>
        /// Prepares the environment (dataset, predictions, evaluation engine etc) before each test.
        /// </summary>
        public RecommenderEvaluatorTests()
        {
            this.dataset = new[]
            {
                // user, item, predicted rating, prediction distribution, true rating (data domain)
                Tuple.Create<string, string, int, RatingDistribution, double>(
                    "A", "a", 4, new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.2 }, { 2, 0.0 }, { 3, 0.0 }, { 4, 0.6 }, { 5, 0.2 } }, 1.1),
                Tuple.Create<string, string, int, RatingDistribution, double>(
                    "B", "a", 1, new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.5 }, { 2, 0.0 }, { 3, 0.0 }, { 4, 0.5 }, { 5, 0.0 } }, 4.1),
                Tuple.Create<string, string, int, RatingDistribution, double>(
                    "D", "b", 2, new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 1.0 }, { 3, 0.0 }, { 4, 0.0 }, { 5, 0.0 } }, 1.9),
                Tuple.Create<string, string, int, RatingDistribution, double>(
                    "A", "b", 5, new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 0.0 }, { 3, 0.0 }, { 4, 0.2 }, { 5, 0.8 } }, 5.3),
                Tuple.Create<string, string, int, RatingDistribution, double>(
                    "A", "c", 3, new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 0.0 }, { 3, 0.6 }, { 4, 0.2 }, { 5, 0.2 } }, 4.7),
                Tuple.Create<string, string, int, RatingDistribution, double>(
                    "A", "e", 1, new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.5 }, { 2, 0.0 }, { 3, 0.0 }, { 4, 0.5 }, { 5, 0.0 } }, 3.6),
                Tuple.Create<string, string, int, RatingDistribution, double>(
                    "B", "c", 2, new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 0.8 }, { 3, 0.2 }, { 4, 0.0 }, { 5, 0.0 } }, 3.1)
            };

            var recommenderMapping = new StarRatingRecommenderMapping();
            var evaluatorMapping = recommenderMapping.ForEvaluation();

            this.starRatingInfo = recommenderMapping.GetRatingInfo(null);
            this.evaluator = new StarRatingRecommenderEvaluator<IEnumerable<Instance>, string, string, double>(evaluatorMapping);
            this.predictions = BuildPredictionDictionary(this.dataset, i => i.Item3);
            this.uncertainPredictions = BuildPredictionDictionary(this.dataset, i => i.Item4);
        }

        #region Tests for exact rating prediction metrics

        /// <summary>
        /// Tests the metric aggregation in <see cref="RecommenderMetricAggregationMethod.PerUserFirst"/> mode.
        /// </summary>
        [Fact]
        public void TestPerUserMetricAggregation()
        {
            double perUserMae = this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, this.predictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst);
            const double TruePerUserMae = (((3.0 + 0.0 + 2.0 + 3.0) / 4.0) + ((3.0 + 1.0) / 2.0) + (0.0 / 1.0)) / 3.0;
            Assert.Equal(TruePerUserMae, perUserMae, CompareEps);

            double perUserExpectedMae = this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, this.uncertainPredictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst);
            const double TruePerUserExpectedMae = (((2.6 + 0.2 + 1.4 + 1.5) / 4.0) + ((1.5 + 0.8) / 2.0) + (0.0 / 3.0)) / 3.0;
            Assert.Equal(TruePerUserExpectedMae, perUserExpectedMae, CompareEps);
        }

        /// <summary>
        /// Tests the mean absolute error computation for ratings in model domain.
        /// </summary>
        [Fact]
        public void TestModelDomainMeanAbsoluteError()
        {
            double mae = this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, this.predictions, Metrics.AbsoluteError);
            const double TrueMae = (3.0 + 0.0 + 2.0 + 3.0 + 3.0 + 1.0 + 0.0) / 7.0;
            Assert.Equal(TrueMae, mae, CompareEps);
        }

        /// <summary>
        /// Tests the mean squared error computation for ratings in model domain.
        /// </summary>
        [Fact]
        public void TestModelDomainMeanSquaredError()
        {
            double mse = this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, this.predictions, Metrics.SquaredError);
            const double TrueMse = (9.0 + 0.0 + 4.0 + 9.0 + 9.0 + 1.0 + 0.0) / 7.0;
            Assert.Equal(TrueMse, mse, CompareEps);
        }

        /// <summary>
        /// Tests the mean absolute error computation for ratings in data domain.
        /// </summary>
        [Fact]
        public void TestDataDomainMeanAbsoluteError()
        {
            double mae = this.evaluator.RatingPredictionMetric(this.dataset, this.predictions, Metrics.AbsoluteError);
            const double TrueMae = (2.9 + 0.3 + 1.7 + 2.6 + 3.1 + 1.1 + 0.1) / 7.0;
            Assert.Equal(TrueMae, mae, CompareEps);
        }

        /// <summary>
        /// Tests the mean squared error computation for ratings in data domain.
        /// </summary>
        [Fact]
        public void TestDataDomainMeanSquaredError()
        {
            double mse = this.evaluator.RatingPredictionMetric(this.dataset, this.predictions, Metrics.SquaredError);
            const double TrueMse = (8.41 + 0.09 + 2.89 + 6.76 + 9.61 + 1.21 + 0.01) / 7.0;
            Assert.Equal(TrueMse, mse, CompareEps);
        }

        /// <summary>
        /// Tests the probability calibration plot building.
        /// </summary>
        [Fact]
        public void TestProbabilityCalibrationPlot()
        {
            const int Bins = 3;
            double[] plot = this.evaluator.ProbabilityCalibrationPlot(this.dataset, this.uncertainPredictions, 4, Bins);

            Assert.Equal(Bins, plot.Length);
            Assert.False(double.IsNaN(plot[0]));
            Assert.False(double.IsNaN(plot[1]));
            Assert.True(double.IsNaN(plot[2]));
            Assert.Equal(0, plot[0], CompareEps);
            Assert.Equal(2.0 / 3.0, plot[1], CompareEps);
        }

        /// <summary>
        /// Tests the probability calibration error computation.
        /// </summary>
        [Fact]
        public void TestProbabilityCalibrationError()
        {
            double error = this.evaluator.ProbabilityCalibrationError(this.dataset, this.uncertainPredictions, 5, 2);
            const double TrueError = ((0.25 - (1.0 / 6.0)) + (1.0 - 0.75)) / 2.0;
            Assert.Equal(TrueError, error);
        }

        /// <summary>
        /// Tests the confusion matrix building.
        /// </summary>
        [Fact]
        public void TestConfusionMatrix()
        {
            var trueConfusion = new RatingMatrix(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating);
            trueConfusion[1, 4] = 3.0 / 12.0;
            trueConfusion[2, 2] = 1.0 / 3.0;
            trueConfusion[2, 3] = 1.0 / 6.0;
            trueConfusion[3, 5] = 1.0 / 12.0;
            trueConfusion[4, 1] = 1.0 / 12.0;
            trueConfusion[5, 5] = 1.0 / 12.0;
            RatingMatrix confusion = this.evaluator.ConfusionMatrix(this.dataset, this.predictions, RecommenderMetricAggregationMethod.PerUserFirst);

            Assert.Equal(trueConfusion.MaxRating, confusion.MaxRating);
            for (int i = 1; i <= confusion.MaxRating; ++i)
            {
                for (int j = 1; j <= confusion.MaxRating; ++j)
                {
                    Assert.Equal(trueConfusion[i, j], confusion[i, j], CompareEps);
                }
            }
        }

        /// <summary>
        /// Tests the weighted confusion computation by computing the same values as regular rating prediction metrics.
        /// </summary>
        [Fact]
        public void TestWeightedConfusion()
        {
            Assert.Equal(
                this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, this.predictions, Metrics.AbsoluteError),
                this.evaluator.WeightedConfusion(this.dataset, this.predictions, RatingMatrix.AbsoluteErrorLossMatrix(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating)),
                CompareEps);

            Assert.Equal(
                this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, this.predictions, Metrics.SquaredError),
                this.evaluator.WeightedConfusion(this.dataset, this.predictions, RatingMatrix.SquaredErrorLossMatrix(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating)),
                CompareEps);

            Assert.Equal(
                this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, this.predictions, Metrics.ZeroOneError),
                this.evaluator.WeightedConfusion(this.dataset, this.predictions, RatingMatrix.ZeroOneErrorLossMatrix(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating)),
                CompareEps);
        }

        #endregion

        #region Tests for uncertain rating prediction metrics

        /// <summary>
        /// Tests the expected mean absolute error computation.
        /// </summary>
        [Fact]
        public void TestExpectedMeanAbsoluteError()
        {
            double expectedMae = this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, this.uncertainPredictions, Metrics.AbsoluteError);
            const double TrueExpectedMae = (2.6 + 0.2 + 1.4 + 1.5 + 1.5 + 0.8 + 0.0) / 7.0;
            Assert.Equal(TrueExpectedMae, expectedMae, CompareEps);
        }

        /// <summary>
        /// Tests the expected mean squared error computation.
        /// </summary>
        [Fact]
        public void TestExpectedMeanSquaredError()
        {
            double expectedMse = this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, this.uncertainPredictions, Metrics.SquaredError);
            const double TrueExpectedMse = (8.6 + 0.2 + 2.6 + 4.5 + 4.5 + 0.8 + 0.0) / 7.0;
            Assert.Equal(TrueExpectedMse, expectedMse, CompareEps);
        }

        /// <summary>
        /// Tests the expected weighted confusion computation by computing the same values as regular rating prediction metrics.
        /// </summary>
        [Fact]
        public void TestExpectedWeightedConfusion()
        {
            double expectedAbsoluteErrorAsWeightedConfusion = this.evaluator.ExpectedWeightedConfusion(
                this.dataset,
                this.uncertainPredictions,
                RatingMatrix.AbsoluteErrorLossMatrix(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating));
            Assert.Equal(
                this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, this.uncertainPredictions, Metrics.AbsoluteError),
                expectedAbsoluteErrorAsWeightedConfusion,
                CompareEps);

            double expectedSquaredErrorAsWeightedConfusion = this.evaluator.ExpectedWeightedConfusion(
                this.dataset,
                this.uncertainPredictions,
                RatingMatrix.SquaredErrorLossMatrix(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating));
            Assert.Equal(
                this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, this.uncertainPredictions, Metrics.SquaredError),
                expectedSquaredErrorAsWeightedConfusion,
                CompareEps);

            double expectedZeroOneErrorAsWeightedConfusion = this.evaluator.ExpectedWeightedConfusion(
                this.dataset,
                this.uncertainPredictions,
                RatingMatrix.ZeroOneErrorLossMatrix(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating));
            Assert.Equal(
                this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, this.uncertainPredictions, Metrics.ZeroOneError),
                expectedZeroOneErrorAsWeightedConfusion,
                CompareEps);
        }

        /// <summary>
        /// Tests the mean negative log-probability computation.
        /// </summary>
        [Fact]
        public void TestMeanNegativeLogProbability()
        {
            double negativeLogProb = this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, this.uncertainPredictions, Metrics.NegativeLogProbability);
            double trueLogProb = (Math.Log(0.2) + Math.Log(0.8) + Math.Log(0.2) + Math.Log(0.5) + Math.Log(0.5) + Math.Log(0.2) + Math.Log(1.0)) / 7.0;
            Assert.Equal(-trueLogProb, negativeLogProb, CompareEps);
        }

        #endregion

        #region Tests for item recommendation metrics

        /// <summary>
        /// Tests the mean DCG computation for various number of recommended items.
        /// Linear position discount is used for simplicity.
        /// </summary>
        /// <remarks>Original ratings are used as gains.</remarks>
        [Fact]
        public void TestMeanDcg()
        {
            Func<IEnumerable<double>, double> linearDcg = list => Metrics.Dcg(list, Metrics.LinearDiscountFunc);
            
            double expectedDcg = (5.3 + 3.1 + 1.9) / 3.0;
            Assert.Equal(
                expectedDcg,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 1), linearDcg),
                CompareEps);

            expectedDcg += (1.1 + 4.1) / (3.0 * 2.0);
            Assert.Equal(
                expectedDcg,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 2), linearDcg),
                CompareEps);

            expectedDcg += 4.7 / (3.0 * 3.0);
            Assert.Equal(
                expectedDcg,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 3), linearDcg),
                CompareEps);

            expectedDcg += 3.6 / (3.0 * 4.0);
            Assert.Equal(
                expectedDcg,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 4), linearDcg),
                CompareEps);

            Assert.Equal(
                expectedDcg,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 5), linearDcg),
                CompareEps);
        }

        /// <summary>
        /// Tests the mean normalized DCG computation for various number of recommended items.
        /// Linear position discount is used for simplicity.
        /// </summary>
        /// <remarks>Original ratings are used as gains.</remarks>
        [Fact]
        public void TestMeanNdcg()
        {
            Func<IEnumerable<double>, IEnumerable<double>, double> linearNdcg =
                (list, bestList) => Metrics.Ndcg(list, bestList, Metrics.LinearDiscountFunc);

            Assert.Equal(
                ((5.3 / 5.3) + (3.1 / 4.1) + (1.9 / 1.9)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 1), linearNdcg),
                CompareEps);
            Assert.Equal(
                (((5.3 + (1.1 / 2.0)) / (5.3 + (4.7 / 2.0))) + ((3.1 + (4.1 / 2.0)) / (4.1 + (3.1 / 2.0))) + (1.9 / 1.9)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 2), linearNdcg),
                CompareEps);
            Assert.Equal(
                (((5.3 + (1.1 / 2.0) + (4.7 / 3.0)) / (5.3 + (4.7 / 2.0) + (3.6 / 3.0))) + ((3.1 + (4.1 / 2.0)) / (4.1 + (3.1 / 2.0))) + (1.9 / 1.9)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 3), linearNdcg),
                CompareEps);
            Assert.Equal(
                (((5.3 + (1.1 / 2.0) + (4.7 / 3.0) + (3.6 / 4.0)) / (5.3 + (4.7 / 2.0) + (3.6 / 3.0) + (1.1 / 4.0))) + ((3.1 + (4.1 / 2.0)) / (4.1 + (3.1 / 2.0))) + (1.9 / 1.9)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 4), linearNdcg),
                CompareEps);
            Assert.Equal(
                (((5.3 + (1.1 / 2.0) + (4.7 / 3.0) + (3.6 / 4.0)) / (5.3 + (4.7 / 2.0) + (3.6 / 3.0) + (1.1 / 4.0))) + ((3.1 + (4.1 / 2.0)) / (4.1 + (3.1 / 2.0))) + (1.9 / 1.9)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 5), linearNdcg),
                CompareEps);
        }

        /// <summary>
        /// Tests the mean graded average precision computation for various number of recommended items.
        /// </summary>
        /// <remarks>Rounded ratings are used as gains.</remarks>
        [Fact]
        public void TestMeanGap()
        {
            Assert.Equal(
                ((5.0 / 5.0) + (3.0 / 3.0) + (2.0 / 2.0)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 1), Metrics.GradedAveragePrecision, Math.Round),
                CompareEps);
            Assert.Equal(
                (((5.0 + ((1.0 + 1.0) / 2.0)) / (5.0 + 1.0)) + ((3.0 + ((3.0 + 4.0) / 2.0)) / (3.0 + 4.0)) + (2.0 / 2.0)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 2), Metrics.GradedAveragePrecision, Math.Round),
                CompareEps);
            Assert.Equal(
                (((5.0 + ((1.0 + 1.0) / 2.0) + ((5.0 + 1.0 + 5.0) / 3.0)) / (5.0 + 1.0 + 5.0)) + ((3.0 + ((3.0 + 4.0) / 2.0)) / (3.0 + 4.0)) + (2.0 / 2.0)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 3), Metrics.GradedAveragePrecision, Math.Round),
                CompareEps);
            Assert.Equal(
                (((5.0 + ((1.0 + 1.0) / 2.0) + ((5.0 + 1.0 + 5.0) / 3.0) + ((4.0 + 1.0 + 4.0 + 4.0) / 4.0)) / (5.0 + 1.0 + 5.0 + 4.0)) + ((3.0 + ((3.0 + 4.0) / 2.0)) / (3.0 + 4.0)) + (2.0 / 2.0)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 4), Metrics.GradedAveragePrecision, Math.Round),
                CompareEps);
            Assert.Equal(
                (((5.0 + ((1.0 + 1.0) / 2.0) + ((5.0 + 1.0 + 5.0) / 3.0) + ((4.0 + 1.0 + 4.0 + 4.0) / 4.0)) / (5.0 + 1.0 + 5.0 + 4.0)) + ((3.0 + ((3.0 + 4.0) / 2.0)) / (3.0 + 4.0)) + (2.0 / 2.0)) / 3.0,
                this.evaluator.ItemRecommendationMetric(this.dataset, BuildRecommendationDictionary(this.dataset, 5), Metrics.GradedAveragePrecision, Math.Round),
                CompareEps);
        }

        #endregion

        #region Tests for related users/items prediction metrics

        /// <summary>
        /// Tests the DCG computation for the prediction of 1 related user.
        /// </summary>
        /// <remarks>Ratings are rounded before computing distance.</remarks>
        [Fact]
        public void TestRelatedUsersDcgAt1()
        {
            var relatedUserPredictions = new Dictionary<string, IEnumerable<string>> { { "A", new[] { "B" } } };

            double ndcg = this.evaluator.RelatedUsersMetric(
                this.dataset, relatedUserPredictions, 1, Metrics.LinearNdcg, Metrics.CosineSimilarity, Math.Round);
            double dcg = this.evaluator.RelatedUsersMetric(
                this.dataset, relatedUserPredictions, 1, Metrics.LinearDcg, Metrics.CosineSimilarity, Math.Round);
            double expectedDcg = 19.0 / 5.0 / Math.Sqrt(26.0);
            const double BestDcg = 1;

            Assert.Equal(expectedDcg, dcg, CompareEps);
            Assert.Equal(expectedDcg / BestDcg, ndcg, CompareEps);
        }

        /// <summary>
        /// Tests the DCG computation for the prediction of 2 related users.
        /// </summary>
        /// <remarks>Ratings are rounded before computing distance.</remarks>
        [Fact]
        public void TestRelatedUsersDcgAt2()
        {
            var relatedUserPredictions = new Dictionary<string, IEnumerable<string>> { { "A", new[] { "B", "D" } } };

            double ndcg = this.evaluator.RelatedUsersMetric(
                this.dataset, relatedUserPredictions, 1, Metrics.LinearNdcg, Metrics.CosineSimilarity, Math.Round);
            double dcg = this.evaluator.RelatedUsersMetric(
                this.dataset, relatedUserPredictions, 1, Metrics.LinearDcg, Metrics.CosineSimilarity, Math.Round);
            double expectedDcg = (19.0 / (5.0 * Math.Sqrt(26.0))) + 0.5;
            double bestDcg = 1.0 + ((0.5 * 19.0) / (5.0 * Math.Sqrt(26.0)));

            Assert.Equal(expectedDcg, dcg, CompareEps);
            Assert.Equal(expectedDcg / bestDcg, ndcg, CompareEps);
        }

        /// <summary>
        /// Tests the DCG computation for the prediction of 1 related item.
        /// </summary>
        /// <remarks>Ratings are rounded before computing distance.</remarks>
        [Fact]
        public void TestRelatedItemsDcgAt1()
        {
            var relatedItemPredictions = new Dictionary<string, IEnumerable<string>> { { "a", new[] { "c" } } };

            double ndcg = this.evaluator.RelatedItemsMetric(
                this.dataset, relatedItemPredictions, 1, Metrics.LinearNdcg, Metrics.CosineSimilarity, Math.Round);
            double dcg = this.evaluator.RelatedItemsMetric(
                this.dataset, relatedItemPredictions, 1, Metrics.LinearDcg, Metrics.CosineSimilarity, Math.Round);
            double expectedDcg = 1.0 / Math.Sqrt(2);
            const double BestDcg = 1;

            Assert.Equal(expectedDcg, dcg, CompareEps);
            Assert.Equal(expectedDcg / BestDcg, ndcg, CompareEps);
        }

        /// <summary>
        /// Tests the DCG computation for the prediction of 2 related items.
        /// </summary>
        /// <remarks>Ratings are rounded before computing distance.</remarks>
        [Fact]
        public void TestRelatedItemsDcgAt2()
        {
            var relatedItemPredictions = new Dictionary<string, IEnumerable<string>> { { "a", new[] { "c", "b" } } };

            double ndcg = this.evaluator.RelatedItemsMetric(
                this.dataset, relatedItemPredictions, 1, Metrics.LinearNdcg, Metrics.CosineSimilarity, Math.Round);
            double dcg = this.evaluator.RelatedItemsMetric(
                this.dataset, relatedItemPredictions, 1, Metrics.LinearDcg, Metrics.CosineSimilarity, Math.Round);
            double expectedDcg = (1.0 / Math.Sqrt(2.0)) + 0.5;
            const double BestDcg = 1.0 + 0.5;

            Assert.Equal(expectedDcg, dcg, CompareEps);
            Assert.Equal(expectedDcg / BestDcg, ndcg, CompareEps);
        }

        #endregion

        #region Tests for error handling

        /// <summary>
        /// Tests if empty rating predictions are handled correctly.
        /// </summary>
        [Fact]
        public void TestEmptyRatingPredictions()
        {
            // No predictions to evaluate
            var testPredictions = new Dictionary<string, IDictionary<string, int>>();
            var testUncertainPredictions = new Dictionary<string, IDictionary<string, RatingDistribution>>();
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, testPredictions, Metrics.AbsoluteError));
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, testPredictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst));
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, testUncertainPredictions, Metrics.AbsoluteError));
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, testUncertainPredictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst));

            // There is one user presented but still no predictions to evaluate
            testPredictions.Add("A", new Dictionary<string, int>());
            testUncertainPredictions.Add("A", new Dictionary<string, RatingDistribution>());
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, testPredictions, Metrics.AbsoluteError));
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, testPredictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst));
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, testUncertainPredictions, Metrics.AbsoluteError));
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, testUncertainPredictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst));

            // Now we add one more user with predictions
            testPredictions.Add("B", new Dictionary<string, int> { { "a", 4 } });
            testUncertainPredictions.Add(
                "B", new Dictionary<string, RatingDistribution> { { "a", new SortedDictionary<int, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 0.0 }, { 3, 0.0 }, { 4, 1.0 }, { 5, 0.0 } } } });

            // If there are some predictions, default aggregation makes sense
            Assert.Equal(0.0, this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, testPredictions, Metrics.AbsoluteError));
            Assert.Equal(0.0, this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, testUncertainPredictions, Metrics.AbsoluteError));

            // Per-user aggregation still doesn't make sense
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetric(this.dataset, testPredictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst));
            Assert.Throws<ArgumentException>(() => this.evaluator.ModelDomainRatingPredictionMetricExpectation(this.dataset, testUncertainPredictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst));
        }

        /// <summary>
        /// Tests if empty item recommendations are handled correctly.
        /// </summary>
        [Fact]
        public void TestEmptyItemRecommendations()
        {
            // No predictions to evaluate
            var testRecommendations = new Dictionary<string, IEnumerable<string>>();
            Assert.Throws<ArgumentException>(() => this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Dcg));
            Assert.Throws<ArgumentException>(() => this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Ndcg));

            // One user with empty recommendation list is presented
            testRecommendations.Add("A", new List<string>());
            Assert.Equal(0.0, this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Dcg)); // DCG makes sense
            Assert.Throws<ArgumentException>(() => this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Ndcg)); // NDCG doesn't
        }

        /// <summary>
        /// Tests if empty related user predictions are handled correctly.
        /// </summary>
        [Fact]
        public void TestEmptyRelatedUserPredictions()
        {
            // No predictions to evaluate
            var testRelatedUsers = new Dictionary<string, IEnumerable<string>>();
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedUsersMetric(this.dataset, testRelatedUsers, 1, Metrics.Dcg, Metrics.NormalizedManhattanSimilarity));
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedUsersMetric(this.dataset, testRelatedUsers, 1, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity));

            // One user with empty list of related users is presented
            testRelatedUsers.Add("A", new List<string>());
            Assert.Equal(
                0.0, this.evaluator.RelatedUsersMetric(this.dataset, testRelatedUsers, 1, Metrics.Dcg, Metrics.NormalizedManhattanSimilarity)); // DCG makes sense
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedUsersMetric(this.dataset, testRelatedUsers, 1, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity)); // NDCG doesn't
        }

        /// <summary>
        /// Tests if empty related item predictions are handled correctly.
        /// </summary>
        [Fact]
        public void TestEmptyRelatedItemPredictions()
        {
            // No predictions to evaluate
            var testRelatedItems = new Dictionary<string, IEnumerable<string>>();
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedItemsMetric(this.dataset, testRelatedItems, 1, Metrics.Dcg, Metrics.NormalizedManhattanSimilarity));
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedItemsMetric(this.dataset, testRelatedItems, 1, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity));

            // One item with empty list of related items is presented
            testRelatedItems.Add("a", new List<string>());
            Assert.Equal(
                0.0, this.evaluator.RelatedItemsMetric(this.dataset, testRelatedItems, 1, Metrics.Dcg, Metrics.NormalizedManhattanSimilarity)); // DCG makes sense
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedItemsMetric(this.dataset, testRelatedItems, 1, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity)); // NDCG doesn't
        }

        /// <summary>
        /// Tests if recommended item can be presented in the recommendation list no more than once.
        /// </summary>
        [Fact]
        public void TestRecommendItemMoreThanOnce()
        {
            var recommendationsForA = new List<string>();
            var testRecommendations = new Dictionary<string, IEnumerable<string>> { { "A", recommendationsForA } };
            this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Dcg); // Must not throw
            recommendationsForA.Add("a");
            this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Dcg); // Must not throw
            recommendationsForA.Add("b");
            this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Dcg); // Must not throw
            recommendationsForA.Add("a");
            Assert.Throws<ArgumentException>(
                () => this.evaluator.ItemRecommendationMetric(this.dataset, testRecommendations, Metrics.Dcg)); // Must throw, 'a' is recommended twice
        }

        /// <summary>
        /// Tests if a user can be presented in the related user list more than once.
        /// </summary>
        [Fact]
        public void TestReturnRelatedUserMoreThanOnce()
        {
            var relatedUsersForA = new List<string>();
            var testPredictions = new Dictionary<string, IEnumerable<string>> { { "A", relatedUsersForA } };
            this.evaluator.RelatedUsersMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            relatedUsersForA.Add("B");
            this.evaluator.RelatedUsersMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            relatedUsersForA.Add("D");
            this.evaluator.RelatedUsersMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            relatedUsersForA.Add("B");
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedUsersMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity)); // Must throw, B is presented twice
        }

        /// <summary>
        /// Tests if an item can be presented in the related item list more than once.
        /// </summary>
        [Fact]
        public void TestReturnRelatedItemMoreThanOnce()
        {
            var relatedItemsForA = new List<string>();
            var testPredictions = new Dictionary<string, IEnumerable<string>> { { "a", relatedItemsForA } };
            this.evaluator.RelatedItemsMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            relatedItemsForA.Add("b");
            this.evaluator.RelatedItemsMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            relatedItemsForA.Add("c");
            this.evaluator.RelatedItemsMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            relatedItemsForA.Add("b");
            Assert.Throws<ArgumentException>(() => this.evaluator.RelatedItemsMetric(this.dataset, testPredictions, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity)); // Must throw, b is presented twice
        }

        /// <summary>
        /// Tests if the query user can be returned in the related user list.
        /// </summary>
        [Fact]
        public void TestQueryUserInRelatedUserList()
        {
            var testPredictionsGood = new Dictionary<string, IEnumerable<string>> { { "A", new[] { "B" } } };
            this.evaluator.RelatedUsersMetric(this.dataset, testPredictionsGood, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            var testPredictionsBad = new Dictionary<string, IEnumerable<string>> { { "A", new[] { "A" } } };
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedUsersMetric(this.dataset, testPredictionsBad, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity)); // Must throw, A is being related to itself
        }

        /// <summary>
        /// Tests if the query item can be returned in the related item list.
        /// </summary>
        [Fact]
        public void TestQueryItemInRelatedItemList()
        {
            var testPredictionsGood = new Dictionary<string, IEnumerable<string>> { { "a", new[] { "b" } } };
            this.evaluator.RelatedItemsMetric(this.dataset, testPredictionsGood, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity); // Must not throw
            var testPredictionsBad = new Dictionary<string, IEnumerable<string>> { { "a", new[] { "a" } } };
            Assert.Throws<ArgumentException>(
                () => this.evaluator.RelatedItemsMetric(this.dataset, testPredictionsBad, 1, Metrics.Dcg, Metrics.NormalizedEuclideanSimilarity)); // Must throw, A is being related to itself
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Builds a prediction dictionary from a dataset.
        /// </summary>
        /// <typeparam name="T">The type of the prediction.</typeparam>
        /// <param name="dataset">The dataset.</param>
        /// <param name="predictionExtractor">The function to extract prediction from a dataset line.</param>
        /// <returns>The prediction dictionary.</returns>
        private static IDictionary<string, IDictionary<string, T>> BuildPredictionDictionary<T>(
            IEnumerable<Instance> dataset,
            Func<Instance, T> predictionExtractor)
        {
            var result = new Dictionary<string, IDictionary<string, T>>();
            foreach (var instance in dataset)
            {
                string user = instance.Item1;
                string item = instance.Item2;
                T prediction = predictionExtractor(instance);

                IDictionary<string, T> itemToPrediction;
                if (!result.TryGetValue(user, out itemToPrediction))
                {
                    itemToPrediction = new Dictionary<string, T>();
                    result.Add(user, itemToPrediction);
                }

                itemToPrediction.Add(item, prediction);
            }

            return result;
        }

        /// <summary>
        /// Builds a recommendation dictionary from a dataset.
        /// </summary>
        /// <param name="dataset">The dataset.</param>
        /// <param name="recommendationCount">The number of recommendations per user.</param>
        /// <returns>The recommendation dictionary.</returns>
        private static IDictionary<string, IEnumerable<string>> BuildRecommendationDictionary(
            IEnumerable<Instance> dataset, int recommendationCount)
        {
            return dataset.GroupBy(instance => instance.Item1).ToDictionary(
                grouping => grouping.Key,
                grouping => grouping.Select(instance => Tuple.Create(instance.Item2, instance.Item3))
                                    .OrderByDescending(itemRating => itemRating.Item2)
                                    .Take(recommendationCount)
                                    .Select(itemRating => itemRating.Item1));
        }

        #endregion

        #region StarRatingRecommenderMapping implementation

        /// <summary>
        /// Represents a testing star rating recommender mapping
        /// </summary>
        private class StarRatingRecommenderMapping : IStarRatingRecommenderMapping<IEnumerable<Instance>, Instance, string, string, double, DummyFeatureStorage, double[]>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<Instance> GetInstances(IEnumerable<Instance> instanceSource)
            {
                return instanceSource;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public string GetUser(IEnumerable<Instance> instanceSource, Instance instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public string GetItem(IEnumerable<Instance> instanceSource, Instance instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public double GetRating(IEnumerable<Instance> instanceSource, Instance instance)
            {
                return instance.Item5;
            }

            /// <summary>
            /// Provides an array of feature values for a given user.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The array of features for <paramref name="user"/>.</returns>
            public double[] GetUserFeatures(short featureSource, string user)
            {
                return null;
            }

            /// <summary>
            /// Provides an array of feature values for a given item.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The array of features for <paramref name="item"/>.</returns>
            public double[] GetItemFeatures(short featureSource, string item)
            {
                return null;
            }

            /// <summary>
            /// Provides the object describing how ratings provided by the instance source map to stars.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
            public IStarRatingInfo<double> GetRatingInfo(IEnumerable<Instance> instanceSource)
            {
                return new RoundingStarRatingInfo(1, 5);
            }
        }

        #endregion
    }
}
