// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Provides utility functions for inference algorithms.
    /// </summary>
    internal static class InferenceAlgorithmUtilities
    {
        #region Training algorithm factories

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a binary Bayes point machine classifier 
        /// with factorized weight distributions.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the generated training algorithm expects features in a sparse representation.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        public static IGeneratedAlgorithm CreateBinaryTrainingAlgorithm(
            bool computeModelEvidence,
            bool useSparseFeatures,
            bool useCompoundWeightPriorDistributions)
        {
            return useSparseFeatures ? 
                CreateSparseBinaryTrainingAlgorithm(computeModelEvidence, useCompoundWeightPriorDistributions) :
                CreateDenseBinaryTrainingAlgorithm(computeModelEvidence, useCompoundWeightPriorDistributions);
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a multi-class Bayes point machine classifier 
        /// with factorized weight distributions.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the generated training algorithm expects features in a sparse representation.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        public static IGeneratedAlgorithm CreateMulticlassTrainingAlgorithm(
            bool computeModelEvidence,
            bool useSparseFeatures,
            bool useCompoundWeightPriorDistributions)
        {
            return useSparseFeatures ? 
                CreateSparseMulticlassTrainingAlgorithm(computeModelEvidence, useCompoundWeightPriorDistributions) :
                CreateDenseMulticlassTrainingAlgorithm(computeModelEvidence, useCompoundWeightPriorDistributions);
        }

        #endregion

        #region Prediction algorithm factories

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a binary Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the generated prediction algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        public static IGeneratedAlgorithm CreateBinaryPredictionAlgorithm(bool useSparseFeatures)
        {
            return useSparseFeatures ? CreateSparseBinaryPredictionAlgorithm() : CreateDenseBinaryPredictionAlgorithm();
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a multi-class Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the generated prediction algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        public static IGeneratedAlgorithm CreateMulticlassPredictionAlgorithm(bool useSparseFeatures)
        {
            return useSparseFeatures ? CreateSparseMulticlassPredictionAlgorithm() : CreateDenseMulticlassPredictionAlgorithm();
        }

        #endregion

        #region Evidence corrections

        /// <summary>
        /// Computes the logarithm of the model evidence contribution of the Replicate operator for all batches.
        /// </summary>
        /// <typeparam name="TDistribution">The type of the distribution.</typeparam>
        /// <param name="batchOutputMessages">The output messages for all batches.</param>
        /// <returns>The logarithm of the computed model evidence correction.</returns>
        public static double ComputeLogEvidenceCorrectionReplicateAllBatches<TDistribution>(IList<TDistribution> batchOutputMessages)
            where TDistribution : CanGetLogAverageOf<TDistribution>, SettableTo<TDistribution>, SettableToProduct<TDistribution>, ICloneable, SettableToUniform
        {
            Debug.Assert(batchOutputMessages != null, "The batch output messages must not be null.");
            Debug.Assert(batchOutputMessages.Count > 1, "There need to be output messages for at least two batches.");

            var prior = (TDistribution)batchOutputMessages[0].Clone();
            prior.SetToUniform();
            return UsesEqualDefOp.LogEvidenceRatio1(batchOutputMessages, prior);
        }

        #endregion

        #region Assertions

        /// <summary>
        /// Asserts that the specified feature values and indexes are valid.
        /// </summary>
        /// <param name="isExpectedSparse">True, if the features are expected to be sparse and false otherwise.</param>
        /// <param name="expectedFeatureCount">The expected number of features.</param>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        public static void CheckFeatures(bool isExpectedSparse, int expectedFeatureCount, double[][] featureValues, int[][] featureIndexes)
        {
            Debug.Assert(featureValues != null, "The feature values must not be null.");
            Debug.Assert(featureValues.Length > 0, "The feature values must not be empty.");
            Debug.Assert(featureValues.Any(values => values != null), "The feature values must not be null for any instance.");
            Debug.Assert(
                featureValues.Any(values => !values.Any(value => double.IsInfinity(value) || double.IsNaN(value))),
                "A feature value must not be infinite or NaN.");

            if (isExpectedSparse)
            {
                CheckSparseFeatures(expectedFeatureCount, featureValues, featureIndexes);
            }
            else
            {
                CheckDenseFeatures(expectedFeatureCount, featureValues, featureIndexes);
            }
        }

        /// <summary>
        /// Asserts that the specified feature values and indexes are valid sparse features.
        /// </summary>
        /// <param name="expectedFeatureCount">The expected number of features.</param>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        public static void CheckSparseFeatures(int expectedFeatureCount, double[][] featureValues, int[][] featureIndexes)
        {
            Debug.Assert(featureIndexes != null, "The feature indexes must not be null.");
            Debug.Assert(featureIndexes.Length > 0, "The feature indexes must not be empty.");
            Debug.Assert(featureIndexes.Length == featureValues.Length, "There must be the same number of feature values and indexes.");
            Debug.Assert(featureIndexes.All(indexes => indexes != null), "The feature indexes must not be null for any instance.");

            for (int instance = 0; instance < featureIndexes.Length; instance++)
            {
                Debug.Assert(featureIndexes[instance].All(index => index >= 0), "The feature indexes must not be negative.");
                Debug.Assert(featureValues[instance].Length == featureIndexes[instance].Length, "Each single instances must have the same number of feature values and feature indexes.");
                Debug.Assert(featureIndexes[instance].All(index => index < expectedFeatureCount), "No index must be greater than the total number of features.");

                var featureIndexSet = new HashSet<int>(featureIndexes[instance]);
                Debug.Assert(featureIndexSet.Count == featureIndexes[instance].Length, "The feature indexes must not contain duplicates.");
            }
        }

        /// <summary>
        /// Asserts that the specified feature values and indexes are valid dense features.
        /// </summary>
        /// <param name="expectedFeatureCount">The expected number of features.</param>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        public static void CheckDenseFeatures(int expectedFeatureCount, double[][] featureValues, int[][] featureIndexes)
        {
            Debug.Assert(featureIndexes == null, "The feature indexes must be null.");
            Debug.Assert(featureValues.Any(values => values.Length == expectedFeatureCount), "All instances must have the same number of feature values.");
        }

        /// <summary>
        /// Asserts that the specified labels are valid.
        /// </summary>
        /// <param name="labels">The labels.</param>
        public static void CheckLabels(bool[] labels)
        {
            Debug.Assert(labels != null, "The labels must not be null.");
            Debug.Assert(labels.Length > 0, "There must be at least one label.");
        }

        /// <summary>
        /// Asserts that the specified labels are valid.
        /// </summary>
        /// <param name="labels">The labels.</param>
        /// <param name="expectedClassCount">The expected number of classes.</param>
        public static void CheckLabels(int[] labels, int expectedClassCount)
        {
            Debug.Assert(labels != null, "The labels must not be null.");
            Debug.Assert(labels.Length > 0, "There must be at least one label.");
            Debug.Assert(labels.All(l => l >= 0), "A label must not be negative.");
            Debug.Assert(labels.All(l => l < expectedClassCount), "A label must be smaller than " + expectedClassCount + ".");
        }

        /// <summary>
        /// Asserts that the specified number of classes is greater than 1.
        /// </summary>
        /// <param name="classCount">The number of classes.</param>
        public static void CheckClassCount(int classCount)
        {
            Debug.Assert(classCount > 1, "There must be at least two classes.");
        }

        /// <summary>
        /// Asserts that the specified number of features is not negative.
        /// </summary>
        /// <param name="featureCount">The number of features.</param>
        public static void CheckFeatureCount(int featureCount)
        {
            Debug.Assert(featureCount >= 0, "The number of features must not be negative.");
        }

        /// <summary>
        /// Asserts that the specified batch is not negative and smaller than the specified batch count.
        /// </summary>
        /// <param name="batchNumber">The batch number.</param>
        /// <param name="batchCount">The total number of batches. Defaults to 1.</param>
        public static void CheckBatchNumber(int batchNumber, int batchCount = 1)
        {
            Debug.Assert(batchNumber >= 0, "The batch must not be negative.");
            Debug.Assert(batchNumber < batchCount, "The batch must not be greater than the total number of batches.");
        }

        /// <summary>
        /// Asserts that the specified batch count is greater than 0.
        /// </summary>
        /// <param name="batchCount">The total number of batches.</param>
        public static void CheckBatchCount(int batchCount)
        {
            Debug.Assert(batchCount > 0, "The number of batches must be 1 or greater.");
        }

        /// <summary>
        /// Asserts that the specified number of iterations is not negative.
        /// </summary>
        /// <param name="iterationCount">The number of iterations.</param>
        public static void CheckIterationCount(int iterationCount)
        {
            Debug.Assert(iterationCount >= 0, "The number of iterations must not be negative.");
        }

        /// <summary>
        /// Asserts that the specified variance is not negative.
        /// </summary>
        /// <param name="variance">The variance.</param>
        public static void CheckVariance(double variance)
        {
            Debug.Assert(variance >= 0, "The variance must not be negative.");
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Counts the number of instances for which a feature has value zero.
        /// </summary>
        /// <param name="featureCount">The total number of features.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <returns>An array of counts, one per instance, containing the number of times a feature has value zero.</returns>
        public static double[] CountZeroFeatureValueInstances(int featureCount, int[][] featureIndexes)
        {
            Debug.Assert(featureCount >= 0, "The number of features must not be negative.");
            Debug.Assert(featureIndexes != null, "The feature indexes must not be null.");

            var counts = new double[featureCount];
            for (int feature = 0; feature < featureCount; feature++)
            {
                counts[feature] = featureIndexes.Length;
            }

            for (int instance = 0; instance < featureIndexes.Length; instance++)
            {
                for (int index = 0; index < featureIndexes[instance].Length; index++)
                {
                    counts[featureIndexes[instance][index]]--;
                }
            }

            return counts;
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a binary Bayes point machine classifier 
        /// with factorized weight distributions on features in a dense representation.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        private static IGeneratedAlgorithm CreateDenseBinaryTrainingAlgorithm(
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            return CreateTrainingAlgorithm(
                computeModelEvidence,
                useCompoundWeightPriorDistributions,
                typeof(GaussianDenseBinaryBpmTraining_EP),
                typeof(CompoundDenseBinaryBpmTraining_EP),
                typeof(GaussianDenseBinaryBpmTrainingEvidence_EP),
                typeof(CompoundDenseBinaryBpmTrainingEvidence_EP));
                }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a binary Bayes point machine classifier 
        /// with factorized weight distributions on features in a sparse representation.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        private static IGeneratedAlgorithm CreateSparseBinaryTrainingAlgorithm(
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            return CreateTrainingAlgorithm(
                computeModelEvidence,
                useCompoundWeightPriorDistributions,
                typeof(GaussianSparseBinaryBpmTraining_EP),
                typeof(CompoundSparseBinaryBpmTraining_EP),
                typeof(GaussianSparseBinaryBpmTrainingEvidence_EP),
                typeof(CompoundSparseBinaryBpmTrainingEvidence_EP));
                }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a multi-class Bayes point machine classifier 
        /// with factorized weight distributions on features in a dense representation.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        private static IGeneratedAlgorithm CreateDenseMulticlassTrainingAlgorithm(
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            return CreateTrainingAlgorithm(
                computeModelEvidence,
                useCompoundWeightPriorDistributions,
                typeof(GaussianDenseMulticlassBpmTraining_EP),
                typeof(CompoundDenseMulticlassBpmTraining_EP),
                typeof(GaussianDenseMulticlassBpmTrainingEvidence_EP),
                typeof(CompoundDenseMulticlassBpmTrainingEvidence_EP));
                }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a multi-class Bayes point machine classifier 
        /// with factorized weight distributions on features in a sparse representation.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        private static IGeneratedAlgorithm CreateSparseMulticlassTrainingAlgorithm(
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            return CreateTrainingAlgorithm(
                computeModelEvidence,
                useCompoundWeightPriorDistributions,
                typeof(GaussianSparseMulticlassBpmTraining_EP),
                typeof(CompoundSparseMulticlassBpmTraining_EP),
                typeof(GaussianSparseMulticlassBpmTrainingEvidence_EP),
                typeof(CompoundSparseMulticlassBpmTrainingEvidence_EP));
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a binary Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a dense representation.
        /// </summary>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        private static IGeneratedAlgorithm CreateDenseBinaryPredictionAlgorithm()
        {
            return new GaussianDenseBinaryBpmPrediction_EP();
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a binary Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a dense representation.
        /// </summary>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        private static IGeneratedAlgorithm CreateSparseBinaryPredictionAlgorithm()
        {
            return new GaussianSparseBinaryBpmPrediction_EP();
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a multi-class Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a dense representation.
        /// </summary>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        private static IGeneratedAlgorithm CreateDenseMulticlassPredictionAlgorithm()
        {
            return new GaussianDenseMulticlassBpmPrediction_EP();
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a multi-class Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a sparse representation.
        /// </summary>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        private static IGeneratedAlgorithm CreateSparseMulticlassPredictionAlgorithm()
        {
            return new GaussianSparseMulticlassBpmPrediction_EP();
        }

        /// <summary>
        /// Creates a specified training algorithm.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the selected training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the selected training algorithm uses compound prior distributions over weights. Otherwise <see cref="Gaussian"/> 
        /// prior distributions are used.
        /// </param>
        /// <param name="gaussianTrainingAlgorithmType">
        /// The type of a training algorithm for a Bayes point machine classifier with <see cref="Gaussian"/> prior distributions which does not compute evidence.
        /// </param>
        /// <param name="compoundTrainingAlgorithmType">
        /// The type of a training algorithm for a Bayes point machine classifier with compound prior distributions which does not compute evidence.
        /// </param>
        /// <param name="gaussianEvidenceTrainingAlgorithmType">
        /// The type of a training algorithm for a Bayes point machine classifier with <see cref="Gaussian"/> prior distributions which computes evidence.
        /// </param>
        /// <param name="compoundEvidenceTrainingAlgorithmType">
        /// The type of a training algorithm for a Bayes point machine classifier with compound prior distributions which computes evidence.
        /// </param>
        /// <returns>The selected <see cref="IGeneratedAlgorithm"/> for training.</returns>
        private static IGeneratedAlgorithm CreateTrainingAlgorithm(
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions,
            Type gaussianTrainingAlgorithmType,
            Type compoundTrainingAlgorithmType,
            Type gaussianEvidenceTrainingAlgorithmType,
            Type compoundEvidenceTrainingAlgorithmType)
        {
            Type trainingAlgorithmType;

            if (computeModelEvidence)
        {
                trainingAlgorithmType = useCompoundWeightPriorDistributions ? compoundEvidenceTrainingAlgorithmType : gaussianEvidenceTrainingAlgorithmType;
            }
            else
            {
                trainingAlgorithmType = useCompoundWeightPriorDistributions ? compoundTrainingAlgorithmType : gaussianTrainingAlgorithmType;
            }

            return (IGeneratedAlgorithm)Activator.CreateInstance(trainingAlgorithmType);
        }

        #endregion
    }
}
