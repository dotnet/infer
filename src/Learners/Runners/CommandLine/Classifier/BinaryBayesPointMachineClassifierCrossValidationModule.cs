// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// A command-line module to cross-validate a binary Bayes point machine classifier on given data.
    /// </summary>
    internal class BinaryBayesPointMachineClassifierCrossValidationModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string dataSetFile = string.Empty;
            string resultsFile = string.Empty;
            int crossValidationFoldCount = 5;
            int iterationCount = BayesPointMachineClassifierTrainingSettings.IterationCountDefault;
            int batchCount = BayesPointMachineClassifierTrainingSettings.BatchCountDefault;
            bool computeModelEvidence = BayesPointMachineClassifierTrainingSettings.ComputeModelEvidenceDefault;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--data-set", "FILE", "File with training data", v => dataSetFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--results", "FILE", "File with cross-validation results", v => resultsFile = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--folds", "NUM", "Number of cross-validation folds (defaults to " + crossValidationFoldCount + ")", v => crossValidationFoldCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--iterations", "NUM", "Number of training algorithm iterations (defaults to " + iterationCount + ")", v => iterationCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--batches", "NUM", "Number of batches to split the training data into (defaults to " + batchCount + ")", v => batchCount = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--compute-evidence", "Compute model evidence (defaults to " + computeModelEvidence + ")", () => computeModelEvidence = true);

            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            // Load and shuffle data
            var dataSet = ClassifierPersistenceUtils.LoadLabeledFeatureValues(dataSetFile);
            BayesPointMachineClassifierModuleUtilities.WriteDataSetInfo(dataSet);

            Rand.Restart(562);
            Rand.Shuffle(dataSet);

            // Create evaluator 
            var evaluatorMapping = Mappings.Classifier.ForEvaluation();
            var evaluator = new ClassifierEvaluator<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string>(evaluatorMapping);

            // Create performance metrics
            var accuracy = new List<double>();
            var negativeLogProbability = new List<double>();
            var auc = new List<double>();
            var evidence = new List<double>();
            var iterationCounts = new List<double>();
            var trainingTime = new List<double>();

            // Run cross-validation
            int validationSetSize = dataSet.Count / crossValidationFoldCount;
            Console.WriteLine("Running {0}-fold cross-validation on {1}", crossValidationFoldCount, dataSetFile);

            // TODO: Use chained mapping to implement cross-validation
            for (int fold = 0; fold < crossValidationFoldCount; fold++)
            {
                // Construct training and validation sets for fold
                int validationSetStart = fold * validationSetSize;
                int validationSetEnd = (fold + 1 == crossValidationFoldCount)
                                           ? dataSet.Count
                                           : (fold + 1) * validationSetSize;

                var trainingSet = new List<LabeledFeatureValues>();
                var validationSet = new List<LabeledFeatureValues>();

                for (int instance = 0; instance < dataSet.Count; instance++)
                {
                    if (validationSetStart <= instance && instance < validationSetEnd)
                    {
                        validationSet.Add(dataSet[instance]);
                    }
                    else
                    {
                        trainingSet.Add(dataSet[instance]);
                    }
                }

                // Print info
                Console.WriteLine("   Fold {0} [validation set instances {1} - {2}]", fold + 1, validationSetStart, validationSetEnd - 1);

                // Create classifier
                var classifier = BayesPointMachineClassifier.CreateBinaryClassifier(Mappings.Classifier);
                classifier.Settings.Training.IterationCount = iterationCount;
                classifier.Settings.Training.BatchCount = batchCount;
                classifier.Settings.Training.ComputeModelEvidence = computeModelEvidence;

                int currentIterationCount = 0;
                classifier.IterationChanged += (sender, eventArgs) => { currentIterationCount = eventArgs.CompletedIterationCount; };

                // Train classifier
                var stopWatch = new Stopwatch();
                stopWatch.Start();
                classifier.Train(trainingSet);
                stopWatch.Stop();

                // Produce predictions
                var predictions = classifier.PredictDistribution(validationSet).ToList();
                var predictedLabels = predictions.Select(
                    prediction => prediction.Aggregate((aggregate, next) => next.Value > aggregate.Value ? next : aggregate).Key).ToList();

                // Iteration count
                iterationCounts.Add(currentIterationCount);

                // Training time
                trainingTime.Add(stopWatch.ElapsedMilliseconds);

                // Compute accuracy
                accuracy.Add(1 - (evaluator.Evaluate(validationSet, predictedLabels, Metrics.ZeroOneError) / predictions.Count));

                // Compute mean negative log probability
                negativeLogProbability.Add(evaluator.Evaluate(validationSet, predictions, Metrics.NegativeLogProbability) / predictions.Count);

                // Compute M-measure (averaged pairwise AUC)
                auc.Add(evaluator.AreaUnderRocCurve(validationSet, predictions));

                // Compute log evidence if desired
                evidence.Add(computeModelEvidence ? classifier.LogModelEvidence : double.NaN);

                // Persist performance metrics
                Console.WriteLine(
                    "      Accuracy = {0,5:0.0000}   NegLogProb = {1,5:0.0000}   AUC = {2,5:0.0000}{3}   Iterations = {4}   Training time = {5}",
                    accuracy[fold],
                    negativeLogProbability[fold],
                    auc[fold],
                    computeModelEvidence ? string.Format("   Log evidence = {0,5:0.0000}", evidence[fold]) : string.Empty,
                    iterationCounts[fold],
                    BayesPointMachineClassifierModuleUtilities.FormatElapsedTime(trainingTime[fold]));

                BayesPointMachineClassifierModuleUtilities.SavePerformanceMetrics(
                    resultsFile, accuracy, negativeLogProbability, auc, evidence, iterationCounts, trainingTime);
            }

            return true;
        }
    }
}