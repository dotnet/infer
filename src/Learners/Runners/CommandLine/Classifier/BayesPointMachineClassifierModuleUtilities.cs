// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Utilities for the Bayes point machine classifier modules.
    /// </summary>
    internal static class BayesPointMachineClassifierModuleUtilities
    {
        /// <summary>
        /// Diagnoses the Bayes point machine classifier on the specified data set.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine classifier.</param>
        /// <param name="trainingSet">The dataset.</param>
        /// <param name="maxParameterChangesFileName">The name of the file to store the maximum parameter differences.</param>
        /// <param name="modelFileName">The name of the file to store the trained Bayes point machine model.</param>
        public static void DiagnoseClassifier<TTrainingSettings>(
            IBayesPointMachineClassifier<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string, IDictionary<string, double>, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<string>> classifier,
            IList<LabeledFeatureValues> trainingSet,
            string maxParameterChangesFileName,
            string modelFileName)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Create prior distributions over weights
            int classCount = trainingSet[0].LabelDistribution.LabelSet.Count;
            int featureCount = trainingSet[0].GetDenseFeatureVector().Count;
            var priorWeightDistributions = Util.ArrayInit(classCount, c => Util.ArrayInit(featureCount, f => new Gaussian(0.0, 1.0)));

            // Create IterationChanged handler
            var watch = new Stopwatch();
            classifier.IterationChanged += (sender, eventArgs) =>
                {
                    watch.Stop();
                    double maxParameterChange = MaxDiff(eventArgs.WeightPosteriorDistributions, priorWeightDistributions);

                    if (!string.IsNullOrEmpty(maxParameterChangesFileName))
                    {
                        SaveMaximumParameterDifference(
                            maxParameterChangesFileName,
                            eventArgs.CompletedIterationCount,
                            maxParameterChange,
                            watch.ElapsedMilliseconds);
                    }

                    Console.WriteLine(
                        "[{0}] Iteration {1,-4}   dp = {2,-20}   dt = {3,5}ms",
                        DateTime.Now.ToLongTimeString(),
                        eventArgs.CompletedIterationCount,
                        maxParameterChange,
                        watch.ElapsedMilliseconds);

                    // Copy weight marginals
                    for (int c = 0; c < eventArgs.WeightPosteriorDistributions.Count; c++)
                    {
                        for (int f = 0; f < eventArgs.WeightPosteriorDistributions[c].Count; f++)
                        {
                            priorWeightDistributions[c][f] = eventArgs.WeightPosteriorDistributions[c][f];        
                        }
                    }
                    
                    watch.Restart();
                };

            // Write file header
            if (!string.IsNullOrEmpty(maxParameterChangesFileName))
            {
                using (var writer = new StreamWriter(maxParameterChangesFileName))
                {
                    writer.WriteLine("# time, # iteration, # maximum absolute parameter difference, # iteration time in milliseconds");
                }
            }

            // Train the Bayes point machine classifier
            Console.WriteLine("[{0}] Starting training...", DateTime.Now.ToLongTimeString());
            watch.Start();

            classifier.Train(trainingSet);

            // Compute evidence
            if (classifier.Settings.Training.ComputeModelEvidence)
            {
                Console.WriteLine("Log evidence = {0,10:0.0000}", classifier.LogModelEvidence);
            }

            // Save trained model
            if (!string.IsNullOrEmpty(modelFileName))
            {
                classifier.SaveForwardCompatible(modelFileName);
            }
        }

        /// <summary>
        /// Samples weights from the learned weight distribution of the Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine used to sample weights from.</param>
        /// <param name="samplesFile">The name of the file to which the weights will be written.</param>
        public static void SampleWeights<TTrainingSettings>(
            IBayesPointMachineClassifier<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string, IDictionary<string, double>, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<string>> classifier, 
            string samplesFile)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            // Sample weights
            var samples = SampleWeights(classifier);

            // Write samples to file
            if (!string.IsNullOrEmpty(samplesFile))
            {
                ClassifierPersistenceUtils.SaveVectors(samplesFile, samples);
            }
        }

        /// <summary>
        /// Saves the performance metrics to a file with the specified name.
        /// </summary>
        /// <param name="fileName">The name of the file to save the metrics to.</param>
        /// <param name="accuracy">The accuracy.</param>
        /// <param name="negativeLogProbability">The mean negative log probability.</param>
        /// <param name="auc">The AUC.</param>
        /// <param name="evidence">The model's log evidence.</param>
        /// <param name="iterationCount">The number of training iterations.</param>
        /// <param name="trainingTime">The training time in milliseconds.</param>
        public static void SavePerformanceMetrics(
            string fileName,
            ICollection<double> accuracy,
            IEnumerable<double> negativeLogProbability,
            IEnumerable<double> auc,
            IEnumerable<double> evidence,
            IEnumerable<double> iterationCount,
            IEnumerable<double> trainingTime)
        {
            using (var writer = new StreamWriter(fileName))
            {
                // Write header
                for (int fold = 0; fold < accuracy.Count; fold++)
                {
                    if (fold == 0)
                    {
                        writer.Write("# ");
                    }

                    writer.Write("Fold {0}, ", fold + 1);
                }

                writer.WriteLine("Mean, Standard deviation");
                writer.WriteLine();

                // Write metrics
                SaveSinglePerformanceMetric(writer, "Accuracy", accuracy);
                SaveSinglePerformanceMetric(writer, "Mean negative log probability", negativeLogProbability);
                SaveSinglePerformanceMetric(writer, "AUC", auc);
                SaveSinglePerformanceMetric(writer, "Log evidence", evidence);
                SaveSinglePerformanceMetric(writer, "Training time", trainingTime);
                SaveSinglePerformanceMetric(writer, "Iteration count", iterationCount);
            }
        }

        /// <summary>
        /// Converts elapsed time in milliseconds into a human readable format.
        /// </summary>
        /// <param name="elapsedMilliseconds">The elapsed time in milliseconds.</param>
        /// <returns>A human readable string of specified time.</returns>
        public static string FormatElapsedTime(double elapsedMilliseconds)
        {
            TimeSpan time = TimeSpan.FromMilliseconds(elapsedMilliseconds);

            string formattedTime = time.Hours > 0 ? string.Format("{0}:", time.Hours) : string.Empty;
            formattedTime += time.Hours > 0 ? string.Format("{0:D2}:", time.Minutes) : time.Minutes > 0 ? string.Format("{0}:", time.Minutes) : string.Empty;
            formattedTime += time.Hours > 0 || time.Minutes > 0 ? string.Format("{0:D2}.{1:D3}", time.Seconds, time.Milliseconds) : string.Format("{0}.{1:D3} seconds", time.Seconds, time.Milliseconds);

            return formattedTime;
        }

        /// <summary>
        /// Writes key statistics of the specified data set to console.
        /// </summary>
        /// <param name="dataSet">The data set.</param>
        public static void WriteDataSetInfo(IList<LabeledFeatureValues> dataSet)
        {
            Console.WriteLine(
                "Data set contains {0} instances, {1} classes and {2} features.",
                dataSet.Count,
                dataSet.Count > 0 ? dataSet.First().LabelDistribution.LabelSet.Count : 0,
                dataSet.Count > 0 ? dataSet.First().FeatureSet.Count : 0);
        }

        #region Helper methods

        /// <summary>
        /// Writes a single performance metric to the specified writer.
        /// </summary>
        /// <param name="writer">The writer to write the metrics to.</param>
        /// <param name="description">The metric description.</param>
        /// <param name="metric">The metric.</param>
        private static void SaveSinglePerformanceMetric(TextWriter writer, string description, IEnumerable<double> metric)
        {
            // Write description
            writer.WriteLine("# " + description);

            // Write metric
            var mva = new MeanVarianceAccumulator();
            foreach (double value in metric)
            {
                writer.Write("{0}, ", value);
                mva.Add(value);
            }

            writer.WriteLine("{0}, {1}", mva.Mean, Math.Sqrt(mva.Variance));
            writer.WriteLine();
        }

        /// <summary>
        /// Saves the maximum absolute difference between two given weight distributions to a file with the specified name.
        /// </summary>
        /// <param name="fileName">The name of the file to save the maximum absolute difference between weight distributions to.</param>
        /// <param name="iteration">The inference algorithm iteration.</param>
        /// <param name="maxParameterChange">The maximum absolute difference in any parameter of two weight distributions.</param>
        /// <param name="elapsedMilliseconds">The elapsed milliseconds.</param>
        private static void SaveMaximumParameterDifference(string fileName, int iteration, double maxParameterChange, long elapsedMilliseconds)
        {
            using (var writer = new StreamWriter(fileName, true))
            {
                writer.WriteLine("{0}, {1}, {2}, {3}", DateTime.Now.ToLongTimeString(), iteration, maxParameterChange, elapsedMilliseconds);
            }
        }

        /// <summary>
        /// Computes the maximum difference in any parameter of two Gaussian distributions.
        /// </summary>
        /// <param name="first">The first Gaussian.</param>
        /// <param name="second">The second Gaussian.</param>
        /// <returns>The maximum absolute difference in any parameter.</returns>
        /// <remarks>This difference computation is based on mean and variance instead of mean*precision and precision.</remarks>
        private static double MaxDiff(IReadOnlyList<IReadOnlyList<Gaussian>> first, Gaussian[][] second)
        {
            int classCount = first.Count;
            int featureCount = first[0].Count;

            double maxDiff = double.NegativeInfinity;
            for (int c = 0; c < classCount; c++)
            {
                for (int f = 0; f < featureCount; f++)
                {
                    double firstMean, firstVariance, secondMean, secondVariance;
                    first[c][f].GetMeanAndVariance(out firstMean, out firstVariance);
                    second[c][f].GetMeanAndVariance(out secondMean, out secondVariance);
                    double meanDifference = Math.Abs(firstMean - secondMean);
                    double varianceDifference = Math.Abs(firstVariance - secondVariance);

                    if (meanDifference > maxDiff)
                    {
                        maxDiff = Math.Abs(meanDifference);
                    }

                    if (Math.Abs(varianceDifference) > maxDiff)
                    {
                        maxDiff = Math.Abs(varianceDifference);
                    }
                }
            }

            return maxDiff;
        }

        /// <summary>
        /// Samples weights from the learned weight distribution of the Bayes point machine classifier.
        /// </summary>
        /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
        /// <param name="classifier">The Bayes point machine used to sample weights from.</param>
        /// <returns>The samples from the weight distribution of the Bayes point machine classifier.</returns>
        private static IEnumerable<Vector> SampleWeights<TTrainingSettings>(
             IBayesPointMachineClassifier<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string, IDictionary<string, double>, TTrainingSettings, IBayesPointMachineClassifierPredictionSettings<string>> classifier)
            where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        {
            Debug.Assert(classifier != null, "The classifier must not be null.");

            IReadOnlyList<IReadOnlyList<Gaussian>> weightPosteriorDistributions = classifier.WeightPosteriorDistributions;
            int classCount = weightPosteriorDistributions.Count < 2 ? 2 : weightPosteriorDistributions.Count;
            int featureCount = weightPosteriorDistributions[0].Count;

            var samples = new Vector[classCount - 1];
            for (int c = 0; c < classCount - 1; c++)
            {
                var sample = Vector.Zero(featureCount);
                for (int f = 0; f < featureCount; f++)
                {
                    sample[f] = weightPosteriorDistributions[c][f].Sample();
                }

                samples[c] = sample;
            }

            return samples;
        }

        #endregion
    }
}
