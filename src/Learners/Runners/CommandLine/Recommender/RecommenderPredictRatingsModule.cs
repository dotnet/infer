// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections.Generic;
    using System.Linq;
    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// A command-line module to predict ratings given a trained recommender model and a dataset.
    /// </summary>
    internal class RecommenderPredictRatingsModule : CommandLineModule
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
            
            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--data", "FILE", "Dataset to make predictions for", v => datasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--model", "FILE", "File with trained model", v => trainedModelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "File with generated predictions", v => predictionsFile = v, CommandLineParameterType.Required);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset testDataset = RecommenderDataset.Load(datasetFile);

            var idToUser = testDataset.Users.ToDictionary(x => x.Id);
            var idToItem = testDataset.Items.ToDictionary(x => x.Id);
            // TODO: CHECK int type parameters.
            var trainedModel = MatchboxRecommender.Load<RecommenderDataset, int, User, Item, int, RatingDistribution, DummyFeatureSource>(
                trainedModelFile,
                readUser: x => idToUser[x.ReadString()],
                readItem: x => idToItem[x.ReadString()],
                writeUser: (writer, x) => writer.Write(x.Id),
                writeItem: (writer, x) => writer.Write(x.Id));
            IDictionary<User, IDictionary<Item, int>> predictions = trainedModel.Predict(testDataset);
            RecommenderPersistenceUtils.SavePredictedRatings(predictionsFile, predictions);

            return true;
        }
    }
}
