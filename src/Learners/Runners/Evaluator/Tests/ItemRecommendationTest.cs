// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Learners.Mappings;

    using Evaluator = StarRatingRecommenderEvaluator<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, User, Item, int>;
    using Recommender = IRecommender<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, User, Item, int, System.Collections.Generic.IDictionary<int, double>, DummyFeatureSource>;

    /// <summary>
    /// The test of item recommendation.
    /// </summary>
    internal class ItemRecommendationTest : RecommenderTest
    {
        /// <summary>
        /// The number of items to recommend to a user.
        /// </summary>
        private readonly int maxRecommendedItemCount;

        /// <summary>
        /// The minimum size of the recommendation pool for a single user.
        /// </summary>
        private readonly int minRecommendationPoolSize;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="ItemRecommendationTest"/> class.
        /// </summary>
        /// <param name="maxRecommendedItemCount">The number of items to recommend to a user.</param>
        /// <param name="minRecommendationPoolSize">
        /// If a user has less than <paramref name="minRecommendationPoolSize"/> possible items to recommend,
        /// it will be skipped.
        /// </param>
        public ItemRecommendationTest(int maxRecommendedItemCount, int minRecommendationPoolSize)
        {
            this.maxRecommendedItemCount = maxRecommendedItemCount;
            this.minRecommendationPoolSize = minRecommendationPoolSize;
        }

        /// <summary>
        /// Executes the test for a given recommender and a test dataset.
        /// </summary>
        /// <param name="recommender">The recommender to evaluate.</param>
        /// <param name="evaluator">The evaluator.</param>
        /// <param name="testDataset">The test dataset.</param>
        /// <param name="predictionTime">When the method returns, this parameter contains the total prediction time.</param>
        /// <param name="evaluationTime">When the method returns, this parameter contains the total evaluation time.</param>
        /// <param name="metrics">When the method returns, this parameter contains names and values of the computed metrics.</param>
        public override void Execute(
            Recommender recommender,
            Evaluator evaluator,
            SplitInstanceSource<RecommenderDataset> testDataset,
            out TimeSpan predictionTime,
            out TimeSpan evaluationTime,
            out MetricValueDistributionCollection metrics)
        {
            var predictionTimer = Stopwatch.StartNew();
            IDictionary<User, IEnumerable<Item>> recommendations = evaluator.RecommendRatedItems(
                recommender, testDataset, this.maxRecommendedItemCount, this.minRecommendationPoolSize);
            predictionTime = predictionTimer.Elapsed;

            var evaluationTimer = Stopwatch.StartNew();
            metrics = new MetricValueDistributionCollection();
            Func<int, double> ratingToGainConverter = r => r - testDataset.InstanceSource.StarRatingInfo.MinStarRating + 1; // Prevent non-positive gains
            metrics.Add("Recommendation NDCG", new MetricValueDistribution(evaluator.ItemRecommendationMetric(testDataset, recommendations, Metrics.Ndcg, ratingToGainConverter)));
            metrics.Add("Recommendation GAP", new MetricValueDistribution(evaluator.ItemRecommendationMetric(testDataset, recommendations, Metrics.GradedAveragePrecision, ratingToGainConverter)));
            evaluationTime = evaluationTimer.Elapsed;
        }
    }
}