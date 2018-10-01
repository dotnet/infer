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
    /// The test of related item prediction.
    /// </summary>
    internal class RelatedItemPredictionTest : RecommenderTest
    {
        /// <summary>
        /// The maximum number of related items to be returned for a given item.
        /// </summary>
        private readonly int maxRelatedItemCount;

        /// <summary>
        /// The minimum number of common ratings two items should have to be considered a relation.
        /// </summary>
        private readonly int minCommonRatingCount;

        /// <summary>
        /// The minimum size of the related user pool for a single user.
        /// </summary>
        private readonly int minRelatedItemPoolSize;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="RelatedItemPredictionTest"/> class.
        /// </summary>
        /// <param name="maxRelatedItemCount">The maximum number of related items to be returned for a given item.</param>
        /// <param name="minCommonRatingCount">The minimum number of common ratings two items should have to be considered a relation.</param>
        /// <param name="minRelatedItemPoolSize">
        /// If an item has less than <paramref name="minRelatedItemPoolSize"/> possible related items,
        /// it will be skipped.
        /// </param>
        public RelatedItemPredictionTest(int maxRelatedItemCount, int minCommonRatingCount, int minRelatedItemPoolSize)
        {
            this.maxRelatedItemCount = maxRelatedItemCount;
            this.minCommonRatingCount = minCommonRatingCount;
            this.minRelatedItemPoolSize = minRelatedItemPoolSize;
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
            IDictionary<Item, IEnumerable<Item>> predictions = evaluator.FindRelatedItemsRatedBySameUsers(
                recommender, testDataset, this.maxRelatedItemCount, this.minCommonRatingCount, this.minRelatedItemPoolSize);
            predictionTime = predictionTimer.Elapsed;

            Stopwatch evaluationTimer = Stopwatch.StartNew();
            metrics = new MetricValueDistributionCollection();
            metrics.Add(
                "Related items L1 Sim NDCG",
                new MetricValueDistribution(evaluator.RelatedItemsMetric(testDataset, predictions, this.minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity)));
            metrics.Add(
                "Related items L2 Sim NDCG",
                new MetricValueDistribution(evaluator.RelatedItemsMetric(testDataset, predictions, this.minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedEuclideanSimilarity)));
            evaluationTime = evaluationTimer.Elapsed;
        }
    }
}