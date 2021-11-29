// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.IO;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// An abstract multi-class Bayes point machine classifier with factorized weight distributions.
    /// <para>
    /// The factorized weight distributions can be interpreted as a <see cref="VectorGaussian"/> distribution 
    /// with diagonal covariance matrix, which ignores possible correlations between weights.
    /// </para>
    /// </summary>
    [Serializable]
    internal abstract class MulticlassFactorizedInferenceAlgorithms : InferenceAlgorithms<GaussianMatrix, int, Discrete>
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="MulticlassFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The number of classes that the inference algorithms use.
        /// </summary>
        private readonly int classCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassFactorizedInferenceAlgorithms"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the training algorithm computes evidence.</param>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        /// <param name="classCount">The number of classes that the inference algorithms use.</param>
        protected MulticlassFactorizedInferenceAlgorithms(bool computeModelEvidence, bool useSparseFeatures, int featureCount, int classCount)
            : base(computeModelEvidence, useSparseFeatures, featureCount)
        {
            InferenceAlgorithmUtilities.CheckClassCount(classCount);
            this.classCount = classCount;

            // Set the marginal distributions over weights divided by their prior distributions
            // to uniform distributions in the training algorithm (no constraints)
            this.WeightMarginalsDividedByPriors = new GaussianMatrix(new GaussianArray(Gaussian.Uniform(), featureCount), classCount);

            // Set the constraint distributions over weights to uniform distributions (no constraints)
            this.WeightConstraints = new GaussianMatrix(new GaussianArray(Gaussian.Uniform(), featureCount), classCount);

            this.InitializeInferenceAlgorithms();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassFactorizedInferenceAlgorithms"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the inference algorithm from.</param>
        protected MulticlassFactorizedInferenceAlgorithms(IReader reader)
            : base(reader)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.WeightMarginals = reader.ReadGaussianMatrix();
                this.WeightMarginalsDividedByPriors = reader.ReadGaussianMatrix();
                this.WeightConstraints = reader.ReadGaussianMatrix();

                this.classCount = reader.ReadInt32();
            }

            this.InitializeInferenceAlgorithms();
        }

        /// <summary>
        /// Gets the distributions over weights as factorized <see cref="Gaussian"/> distributions.
        /// </summary>
        public override IReadOnlyList<IReadOnlyList<Gaussian>> WeightDistributions
        {
            get
            {
                if (this.ReadOnlyWeightMarginals == null)
                {
                    var readOnlyWeights = new IReadOnlyList<Gaussian>[this.classCount];
                    for (int c = 0; c < this.classCount; c++)
                    {
                        readOnlyWeights[c] = new ReadOnlyCollection<Gaussian>(this.WeightMarginals[c]);
                    }

                    this.ReadOnlyWeightMarginals = new ReadOnlyCollection<IReadOnlyList<Gaussian>>(readOnlyWeights);
                }

                return this.ReadOnlyWeightMarginals;
            }
        }

        /// <summary>
        /// Sets the labels of the training algorithm to the specified labels.
        /// </summary>
        protected override int[] Labels
        {
            set
            {
                InferenceAlgorithmUtilities.CheckLabels(value, this.classCount);
                this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.Labels, value);
            }
        }

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the state of the inference algorithms using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the inference algorithms to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);

            writer.Write(this.WeightMarginals);
            writer.Write(this.WeightMarginalsDividedByPriors);
            writer.Write(this.WeightConstraints);

            writer.Write(this.classCount);
        }

        #endregion

        /// <summary>
        /// Creates uniform output messages for all training data batches.
        /// </summary>
        /// <param name="batchCount">The number of batches.</param>
        /// <returns>
        /// An array of uniform output messages, one per training data batch, and null if there is only one single batch.
        /// </returns>
        protected override GaussianMatrix[] CreateUniformBatchOutputMessages(int batchCount)
        {
            return batchCount > 1 ? Util.ArrayInit(batchCount, b => new GaussianMatrix(new GaussianArray(Gaussian.Uniform(), this.FeatureCount), this.classCount)) : null;
        }

        /// <summary>
        /// Copies the specified distribution over labels.
        /// </summary>
        /// <param name="labelDistribution">The distribution over labels to be copied.</param>
        /// <returns>The copy of the specified label distribution.</returns>
        protected override Discrete CopyLabelDistribution(Discrete labelDistribution)
        {
            return new Discrete(labelDistribution);
        }

        /// <summary>
        /// Initializes the inference algorithms for training and prediction.
        /// </summary>
        private void InitializeInferenceAlgorithms()
        {
            this.TrainingAlgorithm.SetObservedValue(InferenceQueryVariableNames.ClassCount, this.classCount);
            this.PredictionAlgorithm.SetObservedValue(InferenceQueryVariableNames.ClassCount, this.classCount);

            // Create uniform constraint distributions for the weights of the prediction algorithm (no constraints)
            this.PredictionAlgorithm.SetObservedValue(
                InferenceQueryVariableNames.WeightConstraints, 
                new GaussianMatrix(new GaussianArray(Gaussian.Uniform(), this.FeatureCount), this.classCount));
        }

        /// <summary>
        /// Performs all actions required after deserialization.
        /// </summary>
        /// <param name="context">The streaming context.</param>
        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            this.InitializeInferenceAlgorithms();
        }
    }
}