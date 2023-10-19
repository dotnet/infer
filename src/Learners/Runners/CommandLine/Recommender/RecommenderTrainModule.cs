// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    /// <summary>
    /// A command-line module to train a recommendation model on a given dataset.
    /// </summary>
    internal class RecommenderTrainModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string trainingDatasetFile = string.Empty;
            string trainedModelFile = string.Empty;
            int traitCount = MatchboxRecommenderTrainingSettings.TraitCountDefault;
            int iterationCount = MatchboxRecommenderTrainingSettings.IterationCountDefault;
            int batchCount = MatchboxRecommenderTrainingSettings.BatchCountDefault;
            bool useUserFeatures = MatchboxRecommenderTrainingSettings.UseUserFeaturesDefault;
            bool useItemFeatures = MatchboxRecommenderTrainingSettings.UseItemFeaturesDefault;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--training-data", "FILE", "Training dataset", v => trainingDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--trained-model", "FILE", "Trained model file", v => trainedModelFile = v, CommandLineParameterType.Required);

            parser.RegisterParameterHandler("--traits", "NUM", "Number of traits (defaults to " + traitCount + ")", v => traitCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--iterations", "NUM", "Number of inference iterations (defaults to " + iterationCount + ")", v => iterationCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--batches", "NUM", "Number of batches to split the training data into (defaults to " + batchCount + ")", v => batchCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--use-user-features", "Use user features in the model (defaults to " + useUserFeatures + ")", () => useUserFeatures = true);
            parser.RegisterParameterHandler("--use-item-features", "Use item features in the model (defaults to " + useItemFeatures + ")", () => useItemFeatures = true);

            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            RecommenderDataset trainingDataset = RecommenderDataset.Load(trainingDatasetFile);
            var recommender = MatchboxRecommender.Create(Mappings.StarRatingRecommender);
            recommender.Settings.Training.TraitCount = traitCount;
            recommender.Settings.Training.IterationCount = iterationCount;
            recommender.Settings.Training.BatchCount = batchCount;
            recommender.Settings.Training.UseUserFeatures = useUserFeatures;
            recommender.Settings.Training.UseItemFeatures = useItemFeatures;
            recommender.Train(trainingDataset);
            recommender.SaveForwardCompatible(trainedModelFile);

            return true;
        }
    }
}