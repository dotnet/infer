// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    /// <summary>
    /// Generates algorithms for inference tasks on the Matchbox model.
    /// </summary>
    internal static class AlgorithmFactories
    {
        /// <summary>
        /// Generates the rating prediction algorithm for the Matchbox model.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated algorithm to.</param>
        public static void GenerateRatingPredictionAlgorithm(string generatedSourceFolder)
        {
            var variablesToInfer = Models.BuildModel(
                buildTrainingModel: false,
                breakTraitSymmetry: false,
                usePreAdjustedUserParameters: true,
                usePreAdjustedItemParameters: true);

            const string ModelName = "MatchboxRatingPrediction";
            var engine = CreateInferenceEngine(ModelName, generatedSourceFolder);
            engine.GetCompiledInferenceAlgorithm(variablesToInfer);
        }

        /// <summary>
        /// Generates the community training algorithm for the Matchbox model.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated algorithm to.</param>
        public static void GenerateCommunityTrainingAlgorithm(string generatedSourceFolder)
        {
            var variablesToInfer = Models.BuildModel(
                buildTrainingModel: true,
                breakTraitSymmetry: true,
                usePreAdjustedUserParameters: false,
                usePreAdjustedItemParameters: false);

            const string ModelName = "MatchboxCommunityTraining";
            var engine = CreateInferenceEngine(ModelName, generatedSourceFolder);
            // these options should only be set for training
            engine.Compiler.UseSpecialFirstIteration = true;
            engine.Compiler.AllowSerialInitialisers = engine.Compiler.UseSpecialFirstIteration;
            engine.GetCompiledInferenceAlgorithm(variablesToInfer);
        }

        /// <summary>
        /// Creates an inference engine.
        /// </summary>
        /// <param name="modelName">The name of the model which will use the created engine.</param>
        /// <param name="generatedSourceFolder">The folder to drop the generated algorithm to.</param>
        /// <returns>The created engine.</returns>
        private static InferenceEngine CreateInferenceEngine(string modelName, string generatedSourceFolder)
        {
            const string modelNamespace = "Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal";
            var engine = new InferenceEngine { ModelNamespace = modelNamespace, ModelName = modelName, ShowProgress = false };

            // Set compiler options
            engine.Compiler.UseSerialSchedules = true;
            engine.Compiler.ReturnCopies = true;
            engine.Compiler.FreeMemory = false;
            engine.Compiler.WriteSourceFiles = true;
            engine.Compiler.GeneratedSourceFolder = generatedSourceFolder;
            engine.Compiler.GenerateInMemory = true;
            engine.Compiler.AddComments = false;  // avoid irrelevant code changes

            return engine;
        }
    }
}