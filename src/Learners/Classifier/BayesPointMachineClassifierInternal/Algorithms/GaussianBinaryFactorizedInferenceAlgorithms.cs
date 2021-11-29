// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Serialization;

    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gaussian, double>;

    /// <summary>
    /// A binary Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over factorized weights.
    /// <para>
    /// The factorized weight distributions can be interpreted as a <see cref="VectorGaussian"/> distribution 
    /// with diagonal covariance matrix, which ignores possible correlations between weights.
    /// </para>
    /// </summary>
    [Serializable]
    internal class GaussianBinaryFactorizedInferenceAlgorithms : BinaryFactorizedInferenceAlgorithms
    {
        #region Fields and constructors

        /// <summary>
        /// The current custom binary serialization version of the <see cref="GaussianBinaryFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The prior distributions over weights for the training algorithm.
        /// </summary>
        private readonly GaussianArray weightPriorDistributions;

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBinaryFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        /// <param name="weightPriorVariance">The variance of the prior distributions over weights. Defaults to 1.</param>
        public GaussianBinaryFactorizedInferenceAlgorithms(bool computeModelEvidence, bool useSparseFeatures, int featureCount, double weightPriorVariance = 1.0)
            : base(computeModelEvidence, useSparseFeatures, featureCount)
        {
            InferenceAlgorithmUtilities.CheckVariance(weightPriorVariance);

            // Set the prior distributions over weights
            this.weightPriorDistributions = new GaussianArray(new Gaussian(0.0, weightPriorVariance), featureCount);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBinaryFactorizedInferenceAlgorithms"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the inference algorithm from.</param>
        public GaussianBinaryFactorizedInferenceAlgorithms(IReader reader)
            : base(reader)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.weightPriorDistributions = reader.ReadGaussianArray();
            }
        }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the state of the inference algorithms using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the inference algorithms to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
            writer.Write(this.weightPriorDistributions);
        }

        #endregion

        #region Inference

        /// <summary>
        /// Runs the generated training algorithm for the specified features and labels.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <param name="labels">The labels.</param>
        /// <param name="iterationCount">The number of iterations to run the training algorithm for.</param>
        /// <param name="batchNumber">
        /// An optional batch number. Defaults to 0 and is used only if the training data is divided into batches.
        /// </param>
        protected override void TrainInternal(double[][] featureValues, int[][] featureIndexes, bool[] labels, int iterationCount, int batchNumber = 0)
        {
            // Update the prior distributions over weights
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightPriors, this.weightPriorDistributions);

            // Run training
            base.TrainInternal(featureValues, featureIndexes, labels, iterationCount, batchNumber);
        }

        #endregion

        #region Inference algorithm generation

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains the Bayes point machine classifier.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the generated training algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for training.</returns>
        protected override IGeneratedAlgorithm CreateTrainingAlgorithm(bool computeModelEvidence, bool useSparseFeatures)
        {
            return InferenceAlgorithmUtilities.CreateBinaryTrainingAlgorithm(
                computeModelEvidence,
                useSparseFeatures,
                useCompoundWeightPriorDistributions: false);
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from the Bayes point machine classifier.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the generated prediction algorithm expects features in a sparse representation.</param>
        /// <returns>The created <see cref="IGeneratedAlgorithm"/> for prediction.</returns>
        protected override IGeneratedAlgorithm CreatePredictionAlgorithm(bool useSparseFeatures)
        {
            return InferenceAlgorithmUtilities.CreateBinaryPredictionAlgorithm(useSparseFeatures);
        }

        #endregion
    }
}