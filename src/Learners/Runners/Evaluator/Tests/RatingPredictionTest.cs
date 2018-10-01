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
    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;
    using Recommender = IRecommender<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, User, Item, int, System.Collections.Generic.IDictionary<int, double>, DummyFeatureSource>;

    /// <summary>
    /// The test of rating prediction.
    /// </summary>
    internal class RatingPredictionTest : RecommenderTest
    {
        /// <summary>
        /// Indicates whether the uncertain prediction metrics should also be computed.
        /// </summary>
        private readonly bool computeUncertainPredictionMetrics;

        /// <summary>
        /// Initializes a new instance of the <see cref="RatingPredictionTest"/> class.
        /// </summary>
        /// <param name="computeUncertainPredictionMetrics">Whether the uncertain prediction metrics should also be computed.</param>
        public RatingPredictionTest(bool computeUncertainPredictionMetrics)
        {
            this.computeUncertainPredictionMetrics = computeUncertainPredictionMetrics;
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
            IDictionary<User, IDictionary<Item, int>> predictions = recommender.Predict(testDataset);
            IDictionary<User, IDictionary<Item, RatingDistribution>> uncertainPredictions =
                this.computeUncertainPredictionMetrics ? recommender.PredictDistribution(testDataset) : null;
            predictionTime = predictionTimer.Elapsed;

            var evaluationTimer = Stopwatch.StartNew();
            metrics = new MetricValueDistributionCollection();
            metrics.Add("MAE", new MetricValueDistribution(evaluator.ModelDomainRatingPredictionMetric(testDataset, predictions, Metrics.AbsoluteError)));
            metrics.Add("Per-user MAE", new MetricValueDistribution(evaluator.ModelDomainRatingPredictionMetric(testDataset, predictions, Metrics.AbsoluteError, RecommenderMetricAggregationMethod.PerUserFirst)));
            metrics.Add("RMSE", new MetricValueDistribution(Math.Sqrt(evaluator.ModelDomainRatingPredictionMetric(testDataset, predictions, Metrics.SquaredError))));
            metrics.Add("Per-user RMSE", new MetricValueDistribution(Math.Sqrt(evaluator.ModelDomainRatingPredictionMetric(testDataset, predictions, Metrics.SquaredError, RecommenderMetricAggregationMethod.PerUserFirst))));
            if (this.computeUncertainPredictionMetrics)
            {
                metrics.Add("Expected MAE", new MetricValueDistribution(evaluator.ModelDomainRatingPredictionMetricExpectation(testDataset, uncertainPredictions, Metrics.AbsoluteError)));
            }

            evaluationTime = evaluationTimer.Elapsed;
        }
    }
}