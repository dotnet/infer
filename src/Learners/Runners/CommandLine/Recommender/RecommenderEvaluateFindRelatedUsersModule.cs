// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections.Generic;
    using System.IO;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// A command-line module to evaluate item recommendation.
    /// </summary>
    internal class RecommenderEvaluateFindRelatedUsersModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string testDatasetFile = string.Empty;
            string predictionsFile = string.Empty;
            string reportFile = string.Empty;
            int minCommonRatingCount = 5;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--test-data", "FILE", "Test dataset used to obtain ground truth", v => testDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "Predictions to evaluate", v => predictionsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--report", "FILE", "Evaluation report file", v => reportFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--min-common-items", "NUM", "Minimum number of items that the query user and the related user should have rated in common; defaults to 5", v => minCommonRatingCount = v, CommandLineParameterType.Optional);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(testDatasetFile);
            IDictionary<User, IEnumerable<User>> relatedUsers = RecommenderPersistenceUtils.LoadRelatedUsers(predictionsFile);

            var evaluatorMapping = Mappings.StarRatingRecommender.ForEvaluation();
            var evaluator = new StarRatingRecommenderEvaluator<RecommenderDataset, User, Item, int>(evaluatorMapping);
            using (var writer = new StreamWriter(reportFile))
            {
                writer.WriteLine(
                    "L1 Sim NDCG: {0:0.000}",
                    evaluator.RelatedUsersMetric(testDataset, relatedUsers, minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedManhattanSimilarity));
                writer.WriteLine(
                    "L2 Sim NDCG: {0:0.000}",
                    evaluator.RelatedUsersMetric(testDataset, relatedUsers, minCommonRatingCount, Metrics.Ndcg, Metrics.NormalizedEuclideanSimilarity));
            }

            return true;
        }
    }
}
