// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections.Generic;

    /// <summary>
    /// A command-line module to predict labels given a trained binary Bayes point machine classifier model and a test set.
    /// </summary>
    internal class BinaryBayesPointMachineClassifierPredictionModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string testSetFile = string.Empty;
            string modelFile = string.Empty;
            string predictionsFile = string.Empty;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--test-set", "FILE", "File with test data", v => testSetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--model", "FILE", "File with a trained binary Bayes point machine model", v => modelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "File to store predictions for the test data", v => predictionsFile = v, CommandLineParameterType.Required);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            var testSet = ClassifierPersistenceUtils.LoadLabeledFeatureValues(testSetFile);
            BayesPointMachineClassifierModuleUtilities.WriteDataSetInfo(testSet);            

            var classifier =
                BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string>(modelFile, Mappings.Classifier);

            // Predict labels
            var predictions = classifier.PredictDistribution(testSet);

            // Write labels to file
            ClassifierPersistenceUtils.SaveLabelDistributions(predictionsFile, predictions);

            return true;
        }
    }
}
