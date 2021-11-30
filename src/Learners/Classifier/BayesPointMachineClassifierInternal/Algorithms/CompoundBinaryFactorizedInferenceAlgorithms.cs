// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    using GammaArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gamma, double>;
    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gaussian, double>;

    /// <summary>
    /// A binary Bayes point machine classifier with compound prior distributions over factorized weights.
    /// <para>
    /// The factorized weight distributions can be interpreted as a <see cref="VectorGaussian"/> distribution 
    /// with diagonal covariance matrix, which ignores possible correlations between weights.
    /// </para>
    /// </summary>
    [Serializable]
    internal class CompoundBinaryFactorizedInferenceAlgorithms : BinaryFactorizedInferenceAlgorithms
    {
        #region Fields and constructors

        /// <summary>
        /// The current custom binary serialization version of the <see cref="CompoundBinaryFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The constraint distributions for the weight precision rates.
        /// </summary>
        private GammaArray weightPrecisionRateConstraints;

        /// <summary>
        /// The marginal distributions over weight precision rates divided by their prior distributions in the training algorithm.
        /// </summary>
        private GammaArray weightPrecisionRateMarginalsDividedByPriors;

        /// <summary>
        /// The per-batch output messages for the weight precision rates in the training algorithm.
        /// </summary>
        private GammaArray[] batchWeightPrecisionRateOutputMessages;

        /// <summary>
        /// Initializes a new instance of the <see cref="CompoundBinaryFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        public CompoundBinaryFactorizedInferenceAlgorithms(bool computeModelEvidence, bool useSparseFeatures, int featureCount)
            : base(computeModelEvidence, useSparseFeatures, featureCount)
        {
            // Set the marginal distributions over weight precision rates divided by their prior distributions
            // to uniform Gamma distributions (no constraints)
            this.weightPrecisionRateMarginalsDividedByPriors = new GammaArray(Gamma.Uniform(), featureCount);

            // Set the constraint distributions over weight precision rates to uniform Gamma distributions (no constraints)
            this.weightPrecisionRateConstraints = new GammaArray(Gamma.Uniform(), featureCount);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CompoundBinaryFactorizedInferenceAlgorithms"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the inference algorithm from.</param>
        public CompoundBinaryFactorizedInferenceAlgorithms(IReader reader)
            : base(reader)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.weightPrecisionRateMarginalsDividedByPriors = reader.ReadGammaArray();
                this.weightPrecisionRateConstraints = reader.ReadGammaArray();
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
            writer.Write(this.weightPrecisionRateMarginalsDividedByPriors);
            writer.Write(this.weightPrecisionRateConstraints);
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
            if (this.UseSparseFeatures)
            {
                InferenceAlgorithmUtilities.CheckSparseFeatures(this.FeatureCount, featureValues, featureIndexes);

                // Observe the number of instances with zero-value features to anchor the compound prior distributions
                this.TrainingAlgorithm.SetObservedValue(
                    InferenceQueryVariableNames.ZeroFeatureValueInstanceCounts, 
                    InferenceAlgorithmUtilities.CountZeroFeatureValueInstances(this.FeatureCount, featureIndexes));
            }

            if (this.BatchCount > 1)
            {
                // Compute the constraint distributions for the weight precision rates for the given batch
                this.weightPrecisionRateConstraints.SetToRatio(
                    this.weightPrecisionRateMarginalsDividedByPriors, this.batchWeightPrecisionRateOutputMessages[batchNumber], false);
            }
            else
            {
                // Required for incremental training
                this.weightPrecisionRateConstraints = this.weightPrecisionRateMarginalsDividedByPriors;
            }

            // Set the constraint distributions for the weight precision rates
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightPrecisionRateConstraints, this.weightPrecisionRateConstraints);

            // Run training
            base.TrainInternal(featureValues, featureIndexes, labels, iterationCount, batchNumber);

            // Update the marginal distributions over weight precision rates divided by their prior distributions
            this.weightPrecisionRateMarginalsDividedByPriors.SetTo(this.TrainingAlgorithm.Marginal<GammaArray>(
                InferenceQueryVariableNames.WeightPrecisionRates, QueryTypes.MarginalDividedByPrior.Name));

            if (this.BatchCount > 1)
            {
                // Update the output messages for the weight precision rates for the given batch
                this.batchWeightPrecisionRateOutputMessages[batchNumber].SetToRatio(
                    this.weightPrecisionRateMarginalsDividedByPriors, this.weightPrecisionRateConstraints, false);
            }
        }

        /// <summary>
        /// Creates uniform output messages for all training data batches.
        /// </summary>
        /// <param name="batchCount">The number of batches.</param>
        /// <returns>
        /// An array of uniform output messages, one per training data batch, and null if there is only one single batch.
        /// </returns>
        protected override GaussianArray[] CreateUniformBatchOutputMessages(int batchCount)
        {
            // Create uniform output messages for weight precision rates as required by the compound prior distributions
            this.batchWeightPrecisionRateOutputMessages = 
                batchCount > 1 ? Util.ArrayInit(batchCount, b => new GammaArray(Gamma.Uniform(), this.FeatureCount)) : null;

            return base.CreateUniformBatchOutputMessages(batchCount);
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
                useCompoundWeightPriorDistributions: true);
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

        #region Evidence computation

        /// <summary>
        /// Computes the logarithm of the model evidence corrections required in batched training.
        /// </summary>
        /// <returns>The logarithm of the model evidence corrections.</returns>
        /// <remarks>
        /// When the training data is split into several batches, it is necessary to eliminate evidence contributions
        /// which would otherwise be double counted. In essence, evidence corrections remove the superfluous contributions 
        /// of factors which are shared across data batches, such as priors and constraints. To compute the evidence 
        /// contributions of the factors shared across batches, one can compute the evidence on an empty batch.
        /// </remarks>
        protected override double ComputeLogEvidenceCorrection()
        {
            // Correct the below base correction and add the missing evidence contribution 
            // for the Replicate operator on the weight precision rates for all batches
            double logModelEvidenceCorrection = 
                InferenceAlgorithmUtilities.ComputeLogEvidenceCorrectionReplicateAllBatches(this.batchWeightPrecisionRateOutputMessages);

            // Compute base evidence correction
            logModelEvidenceCorrection += base.ComputeLogEvidenceCorrection();

            return logModelEvidenceCorrection;
        }

        /// <summary>
        /// Computes the logarithm of the model evidence contribution of an empty batch.
        /// </summary>
        /// <returns>The logarithm of the computed model evidence contribution of an empty batch.</returns>
        protected override double ComputeLogEvidenceContributionEmptyBatch()
        {
            // Update the constraints on the distributions over weight precision rates
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.WeightPrecisionRateConstraints, this.weightPrecisionRateMarginalsDividedByPriors);

            return base.ComputeLogEvidenceContributionEmptyBatch();
        }

        /// <summary>
        /// Sets the training algorithm to use empty training data.
        /// </summary>
        protected override void ObserveEmptyTrainingData()
        {
            base.ObserveEmptyTrainingData();

            if (this.UseSparseFeatures)
            {
                // Compound prior prior anchoring for sparse features
                this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.ZeroFeatureValueInstanceCounts, new double[this.FeatureCount]);
            }
        }

        #endregion
    }
}