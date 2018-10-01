// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Linq;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// A command-line module to generate negative data for a positive-only recommender dataset.
    /// </summary>
    internal class RecommenderGenerateNegativeDataModule : CommandLineModule
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
            string outputDatasetFile = string.Empty;
            
            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--input-data", "FILE", "Input dataset, treated as if all the ratings are positive", v => inputDatasetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--output-data", "FILE", "Output dataset with both posisitve and negative data", v => outputDatasetFile = v, CommandLineParameterType.Required);
            
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            var generatorMapping = Mappings.StarRatingRecommender.WithGeneratedNegativeData();

            var inputDataset = RecommenderDataset.Load(inputDatasetFile);
            var outputDataset = new RecommenderDataset(
                generatorMapping.GetInstances(inputDataset).Select(i => new RatedUserItem(i.User, i.Item, i.Rating)),
                generatorMapping.GetRatingInfo(inputDataset));
            outputDataset.Save(outputDatasetFile);

            return true;
        }
    }
}
