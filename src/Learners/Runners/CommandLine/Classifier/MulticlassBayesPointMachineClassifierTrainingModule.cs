// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Linq;

    /// <summary>
    /// A command-line module to train a multi-class Bayes point machine classifier on some given data.
    /// </summary>
    internal class MulticlassBayesPointMachineClassifierTrainingModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string trainingSetFile = string.Empty;
            string modelFile = string.Empty;
            int iterationCount = BayesPointMachineClassifierTrainingSettings.IterationCountDefault;
            int batchCount = BayesPointMachineClassifierTrainingSettings.BatchCountDefault;
            bool computeModelEvidence = BayesPointMachineClassifierTrainingSettings.ComputeModelEvidenceDefault;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--training-set", "FILE", "File with training data", v => trainingSetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--model", "FILE", "File to store the trained multi-class Bayes point machine model", v => modelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--iterations", "NUM", "Number of training algorithm iterations (defaults to " + iterationCount + ")", v => iterationCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--batches", "NUM", "Number of batches to split the training data into (defaults to " + batchCount + ")", v => batchCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--compute-evidence", "Compute model evidence (defaults to " + computeModelEvidence + ")", () => computeModelEvidence = true);
            
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            var trainingSet = ClassifierPersistenceUtils.LoadLabeledFeatureValues(trainingSetFile);
            BayesPointMachineClassifierModuleUtilities.WriteDataSetInfo(trainingSet);            

            var featureSet = trainingSet.Count > 0 ? trainingSet.First().FeatureSet : null;
            var mapping = new ClassifierMapping(featureSet);
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);
            classifier.Settings.Training.IterationCount = iterationCount;
            classifier.Settings.Training.BatchCount = batchCount;
            classifier.Settings.Training.ComputeModelEvidence = computeModelEvidence;

            classifier.Train(trainingSet);

            if (classifier.Settings.Training.ComputeModelEvidence)
            {
                Console.WriteLine("Log evidence = {0,10:0.0000}", classifier.LogModelEvidence);
            }

            classifier.SaveForwardCompatible(modelFile);

            return true;
        }
    }
}