// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;

    using Microsoft.ML.Probabilistic.Learners.Mappings;

    using Evaluator = StarRatingRecommenderEvaluator<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, User, Item, int>;
    using Recommender = IRecommender<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, User, Item, int, System.Collections.Generic.IDictionary<int, double>, DummyFeatureSource>;

    /// <summary>
    /// The base class for various recommender tests.
    /// </summary>
    public abstract class RecommenderTest
    {
        /// <summary>
        /// This function should be implemented in the derived classes to execute the test for a given recommender and a test dataset.
        /// </summary>
        /// <param name="recommender">The recommender to evaluate.</param>
        /// <param name="evaluator">The evaluator.</param>
        /// <param name="testDataset">The test dataset.</param>
        /// <param name="predictionTime">When the method returns, this parameter contains the total prediction time.</param>
        /// <param name="evaluationTime">When the method returns, this parameter contains the total evaluation time.</param>
        /// <param name="metrics">When the method returns, this parameter contains names and values of the computed metrics.</param>
        public abstract void Execute(
            Recommender recommender,
            Evaluator evaluator,
            SplitInstanceSource<RecommenderDataset> testDataset,
            out TimeSpan predictionTime,
            out TimeSpan evaluationTime,
            out MetricValueDistributionCollection metrics);
    }
}
