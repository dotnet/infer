// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Learners.Mappings;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;
    
    /// <summary>
    /// A command-line module to find related items given a trained recommender model and a dataset.
    /// </summary>
    internal class RecommenderFindRelatedItemsModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string datasetFile = string.Empty;
            string trainedModelFile = string.Empty;
            string predictionsFile = string.Empty;
            int maxRelatedItemCount = 5;
            int minCommonRatingCount = 5;
            int minRelatedItemPoolSize = 5;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--data", "FILE", "Dataset to make predictions for", v => datasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--model", "FILE", "File with trained model", v => trainedModelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "File with generated predictions", v => predictionsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--max-items", "NUM", "Maximum number of related items for a single item; defaults to 5", v => maxRelatedItemCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--min-common-users", "NUM", "Minimum number of users that the query item and the related item should have been rated by in common; defaults to 5", v => minCommonRatingCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--min-pool-size", "NUM", "Minimum size of the related item pool for a single item; defaults to 5", v => minRelatedItemPoolSize = v, CommandLineParameterType.Optional);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(datasetFile);

            var trainedModel = MatchboxRecommender.LoadBackwardCompatible<RecommenderDataset, RatedUserItem, User, Item, int, DummyFeatureSource>(trainedModelFile, Mappings.StarRatingRecommender);
            var evaluator = new RecommenderEvaluator<RecommenderDataset, User, Item, int, int, RatingDistribution>(
                Mappings.StarRatingRecommender.ForEvaluation());
            IDictionary<Item, IEnumerable<Item>> relatedItems = evaluator.FindRelatedItemsRatedBySameUsers(
                trainedModel, testDataset, maxRelatedItemCount, minCommonRatingCount, minRelatedItemPoolSize);
            RecommenderPersistenceUtils.SaveRelatedItems(predictionsFile, relatedItems);

            return true;
        }
    }
}
