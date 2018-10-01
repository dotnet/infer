// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// A command-line module to split a given recommendation dataset into training and test parts.
    /// </summary>
    internal class RecommenderSplitDataModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string inputDatasetFile = string.Empty;
            string outputTrainingDatasetFile = string.Empty;
            string outputTestDatasetFile = string.Empty;
            double trainingOnlyUserFraction = 0.5;
            double testUserRatingTrainingFraction = 0.25;
            double coldUserFraction = 0;
            double coldItemFraction = 0;
            double ignoredUserFraction = 0;
            double ignoredItemFraction = 0;
            bool removeOccasionalColdItems = false;
            
            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--input-data", "FILE", "Dataset to split", v => inputDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--output-data-train", "FILE", "Training part of the split dataset", v => outputTrainingDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--output-data-test", "FILE", "Test part of the split dataset", v => outputTestDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--training-users", "NUM", "Fraction of training-only users; defaults to 0.5", (double v) => trainingOnlyUserFraction = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--test-user-training-ratings", "NUM", "Fraction of test user ratings for training; defaults to 0.25", (double v) => testUserRatingTrainingFraction = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--cold-users", "NUM", "Fraction of cold (test-only) users; defaults to 0", (double v) => coldUserFraction = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--cold-items", "NUM", "Fraction of cold (test-only) items; defaults to 0", (double v) => coldItemFraction = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--ignored-users", "NUM", "Fraction of ignored users; defaults to 0", (double v) => ignoredUserFraction = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--ignored-items", "NUM", "Fraction of ignored items; defaults to 0", (double v) => ignoredItemFraction = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--remove-occasional-cold-items", "Remove occasionally produced cold items", () => removeOccasionalColdItems = true);

            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            var splittingMapping = Mappings.StarRatingRecommender.SplitToTrainTest(
                trainingOnlyUserFraction,
                testUserRatingTrainingFraction,
                coldUserFraction,
                coldItemFraction,
                ignoredUserFraction,
                ignoredItemFraction,
                removeOccasionalColdItems);

            var inputDataset = RecommenderDataset.Load(inputDatasetFile);
            var outputTrainingDataset = new RecommenderDataset(
                splittingMapping.GetInstances(SplitInstanceSource.Training(inputDataset)),
                inputDataset.StarRatingInfo);
            outputTrainingDataset.Save(outputTrainingDatasetFile);
            var outputTestDataset = new RecommenderDataset(
                splittingMapping.GetInstances(SplitInstanceSource.Test(inputDataset)),
                inputDataset.StarRatingInfo);
            outputTestDataset.Save(outputTestDatasetFile);
            
            return true;
        }
    }
}
