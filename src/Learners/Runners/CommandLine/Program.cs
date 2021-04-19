// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;

    /// <summary>
    /// The program.
    /// </summary>
    public static class Program
    {
        /// <summary>
        /// The entry point for the program.
        /// </summary>
        /// <param name="args">The array of command-line arguments.</param>
        public static void Main(string[] args)
        {
            // Matchbox recommender
            var recommenderModuleSelector = new CommandLineModuleSelector();
            recommenderModuleSelector.RegisterModule("SplitData", new RecommenderSplitDataModule());
            recommenderModuleSelector.RegisterModule("GenerateNegativeData", new RecommenderGenerateNegativeDataModule());
            recommenderModuleSelector.RegisterModule("Train", new RecommenderTrainModule());
            recommenderModuleSelector.RegisterModule("PredictRatings", new RecommenderPredictRatingsModule());
            recommenderModuleSelector.RegisterModule("RecommendItems", new RecommenderRecommendItemsModule());
            recommenderModuleSelector.RegisterModule("FindRelatedUsers", new RecommenderFindRelatedUsersModule());
            recommenderModuleSelector.RegisterModule("FindRelatedItems", new RecommenderFindRelatedItemsModule());
            recommenderModuleSelector.RegisterModule("EvaluateRatingPrediction", new RecommenderEvaluateRatingPredictionModule());
            recommenderModuleSelector.RegisterModule("EvaluateItemRecommendation", new RecommenderEvaluateItemRecommendationModule());
            recommenderModuleSelector.RegisterModule("EvaluateFindRelatedUsers", new RecommenderEvaluateFindRelatedUsersModule());
            recommenderModuleSelector.RegisterModule("EvaluateFindRelatedItems", new RecommenderEvaluateFindRelatedItemsModule());

            // Binary Bayes point machine classifier
            var binaryBayesPointMachineModuleSelector = new CommandLineModuleSelector();
            binaryBayesPointMachineModuleSelector.RegisterModule("Train", new BinaryBayesPointMachineClassifierTrainingModule());
            binaryBayesPointMachineModuleSelector.RegisterModule("TrainIncremental", new BinaryBayesPointMachineClassifierIncrementalTrainingModule());
            binaryBayesPointMachineModuleSelector.RegisterModule("Predict", new BinaryBayesPointMachineClassifierPredictionModule());
            binaryBayesPointMachineModuleSelector.RegisterModule("CrossValidate", new BinaryBayesPointMachineClassifierCrossValidationModule());
            binaryBayesPointMachineModuleSelector.RegisterModule("SampleWeights", new BinaryBayesPointMachineClassifierSampleWeightsModule());
            binaryBayesPointMachineModuleSelector.RegisterModule("DiagnoseTrain", new BinaryBayesPointMachineClassifierTrainingDiagnosticsModule());

            // Multi-class Bayes point machine classifier
            var multiclassBayesPointMachineModuleSelector = new CommandLineModuleSelector();
            multiclassBayesPointMachineModuleSelector.RegisterModule("Train", new MulticlassBayesPointMachineClassifierTrainingModule());
            multiclassBayesPointMachineModuleSelector.RegisterModule("TrainIncremental", new MulticlassBayesPointMachineClassifierIncrementalTrainingModule());
            multiclassBayesPointMachineModuleSelector.RegisterModule("Predict", new MulticlassBayesPointMachineClassifierPredictionModule());
            multiclassBayesPointMachineModuleSelector.RegisterModule("CrossValidate", new MulticlassBayesPointMachineClassifierCrossValidationModule());
            multiclassBayesPointMachineModuleSelector.RegisterModule("SampleWeights", new MulticlassBayesPointMachineClassifierSampleWeightsModule());
            multiclassBayesPointMachineModuleSelector.RegisterModule("DiagnoseTrain", new MulticlassBayesPointMachineClassifierTrainingDiagnosticsModule());

            // Classifier
            var classifierModuleSelector = new CommandLineModuleSelector();
            classifierModuleSelector.RegisterModule("Evaluate", new ClassifierEvaluationModule());
            classifierModuleSelector.RegisterModule("BinaryBayesPointMachine", binaryBayesPointMachineModuleSelector);
            classifierModuleSelector.RegisterModule("MulticlassBayesPointMachine", multiclassBayesPointMachineModuleSelector);

            // Modules
            var moduleSelector = new CommandLineModuleSelector();
            moduleSelector.RegisterModule("Recommender", recommenderModuleSelector);
            moduleSelector.RegisterModule("Classifier", classifierModuleSelector);

            try
            {
                bool success = moduleSelector.Run(args, Environment.GetCommandLineArgs()[0]);
                Environment.Exit(success ? 0 : 1);
            }
            catch (Exception e)
            {
                Console.WriteLine("An error has occured in one of the modules. {0}", e.Message);
            }
        }
    }
}
