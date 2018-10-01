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
    /// The test of related user prediction.
    /// </summary>
    internal class RelatedUserPredictionTest : RecommenderTest
    {
        /// <summary>
        /// The maximum number of related users to be returned for a given user.
        /// </summary>
        private readonly int maxRelatedUserCount;

        /// <summary>
        /// The minimum number of common ratings two users should have to be considered a relation.
        /// </summary>
        private readonly int minCommonRatingCount;

        /// <summary>
        /// The minimum size of the related user pool for a single user.
        /// </summary>
        private readonly int minRelatedUserPoolSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="RelatedUserPredictionTest"/> class.
        /// </summary>
        /// <param name="maxRelatedUserCount">The maximum number of related users to be returned for a given user.</param>
        /// <param name="minCommonRatingCount">The minimum number of common ratings two users should have to be considered a relation.</param>
        /// <param name="minRelatedUserPoolSize">
        /// If a user has less than <paramref name="minRelatedUserPoolSize"/> possible items to recommend,
        /// it will be skipped.
        /// </param>
        public RelatedUserPredictionTest(int maxRelatedUserCount, int minCommonRatingCount, int minRelatedUserPoolSize)
        {
            this.maxRelatedUserCount = maxRelatedUserCount;
            this.minCommonRatingCount = minCommonRatingCount;
            this.minRelatedUserPoolSize = minRelatedUserPoolSize;
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
            Stopwatch predictionTimer = Stopwatch.StartNew();
            IDictionary<User, IEnumerable<User>> predictions = evaluator.FindRelatedUsersWhoRatedSameItems(
                recommender, testDataset, this.maxRelatedUserCount, this.minCommonRatingCount, this.minRelatedUserPoolSize);
            predictionTime = predictionTimer.Elapsed;

            Stopwatch evaluationTimer = Stopwatch.StartNew();
            metrics = new MetricValueDistributionCollection();
            metrics.Add(
                "Related users L1 Sim NDCG",
                new MetricValueDistribution(evaluator.RelatedUsersMetric(testDataset, predictions, this.minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity)));
            metrics.Add(
                "Related users L2 Sim NDCG",
                new MetricValueDistribution(evaluator.RelatedUsersMetric(testDataset, predictions, this.minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedEuclideanSimilarity)));
            evaluationTime = evaluationTimer.Elapsed;
        }
    }
}
