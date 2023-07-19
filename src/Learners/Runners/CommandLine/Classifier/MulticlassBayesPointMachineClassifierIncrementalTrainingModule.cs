// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections.Generic;

    /// <summary>
    /// A command-line module to incrementally train a multi-class Bayes point machine classifier on some given data.
    /// </summary>
    internal class MulticlassBayesPointMachineClassifierIncrementalTrainingModule : CommandLineModule
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
            string inputModelFile = string.Empty;
            string outputModelFile = string.Empty;
            int iterationCount = BayesPointMachineClassifierTrainingSettings.IterationCountDefault;
            int batchCount = BayesPointMachineClassifierTrainingSettings.BatchCountDefault;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--training-set", "FILE", "File with training data", v => trainingSetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--input-model", "FILE", "File with the trained multi-class Bayes point machine model", v => inputModelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--model", "FILE", "File to store the incrementally trained multi-class Bayes point machine model", v => outputModelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--iterations", "NUM", "Number of training algorithm iterations (defaults to " + iterationCount + ")", v => iterationCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--batches", "NUM", "Number of batches to split the training data into (defaults to " + batchCount + ")", v => batchCount = v, CommandLineParameterType.Optional);

            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            var trainingSet = ClassifierPersistenceUtils.LoadLabeledFeatureValues(trainingSetFile);
            BayesPointMachineClassifierModuleUtilities.WriteDataSetInfo(trainingSet);            

            var formatter = SerializationUtils.GetJsonFormatter();
            var classifier = BayesPointMachineClassifier.LoadMulticlassClassifier<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string, IDictionary<string, double>>(inputModelFile, formatter);
            classifier.Settings.Training.IterationCount = iterationCount;
            classifier.Settings.Training.BatchCount = batchCount;

            classifier.TrainIncremental(trainingSet);

            classifier.Save(outputModelFile, formatter);

            return true;
        }
    }
}