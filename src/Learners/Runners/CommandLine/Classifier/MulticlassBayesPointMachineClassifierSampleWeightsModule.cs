// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections.Generic;

    /// <summary>
    /// A command-line module to sample weights from a trained multi-class Bayes point machine classifier model.
    /// </summary>
    internal class MulticlassBayesPointMachineClassifierSampleWeightsModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string modelFile = string.Empty;
            string samplesFile = string.Empty;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--model", "FILE", "File with a trained multi-class Bayes point machine model", v => modelFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--samples", "FILE", "File to store samples of the weights", v => samplesFile = v, CommandLineParameterType.Required);
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            var classifier =
                BayesPointMachineClassifier.LoadBackwardCompatibleBinaryClassifier<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string>(modelFile, Mappings.Classifier);

            BayesPointMachineClassifierModuleUtilities.SampleWeights(classifier, samplesFile);

            return true;
        }
    }
}
