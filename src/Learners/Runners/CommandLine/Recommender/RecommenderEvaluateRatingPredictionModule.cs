// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.IO;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// A command-line module to evaluate rating prediction.
    /// </summary>
    internal class RecommenderEvaluateRatingPredictionModule : CommandLineModule
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

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--test-data", "FILE", "Test dataset used to obtain ground truth", v => testDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "Predictions to evaluate", v => predictionsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--report", "FILE", "Evaluation report file", v => reportFile = v, CommandLineParameterType.Required);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(testDatasetFile);
            IDictionary<User, IDictionary<Item, int>> ratingPredictions = RecommenderPersistenceUtils.LoadPredictedRatings(predictionsFile);

            var evaluatorMapping = Mappings.StarRatingRecommender.ForEvaluation();
            var evaluator = new StarRatingRecommenderEvaluator<RecommenderDataset, User, Item, int>(evaluatorMapping);
            using (var writer = new StreamWriter(reportFile))
            {
                writer.WriteLine(
                    "Mean absolute error: {0:0.000}",
                    evaluator.RatingPredictionMetric(testDataset, ratingPredictions, Metrics.AbsoluteError));
                writer.WriteLine(
                    "Root mean squared error: {0:0.000}",
                    Math.Sqrt(evaluator.RatingPredictionMetric(testDataset, ratingPredictions, Metrics.SquaredError)));
            }

            return true;
        }
    }
}
