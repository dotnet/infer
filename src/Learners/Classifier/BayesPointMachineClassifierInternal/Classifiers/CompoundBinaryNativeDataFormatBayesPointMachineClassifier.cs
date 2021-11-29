// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A binary Bayes point machine classifier with compound prior distributions over weights
    /// which operates on data in the native format of the underlying Infer.NET model.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    [Serializable]
    [SerializationVersion(6)]
    internal class CompoundBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource> :
        BinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, BayesPointMachineClassifierTrainingSettings>
    {
        /// <summary>
        /// The current custom binary serialization version of the
        /// <see cref="CompoundBinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="CompoundBinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource}"/> 
        /// class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        public CompoundBinaryNativeDataFormatBayesPointMachineClassifier(IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
            : base(mapping)
        {
            this.Settings = new BinaryBayesPointMachineClassifierSettings<bool>(() => this.IsTrained);
        }

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="CompoundBinaryNativeDataFormatBayesPointMachineClassifier{TInstanceSource, TInstance, TLabelSource}"/> 
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the binary Bayes point machine classifier from.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        public CompoundBinaryNativeDataFormatBayesPointMachineClassifier(IReader reader, IBayesPointMachineClassifierMapping<TInstanceSource, TInstance, TLabelSource, bool> mapping)
            : base(reader, mapping)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                // Deserialize settings and inference algorithms
                this.Settings = new BinaryBayesPointMachineClassifierSettings<bool>(reader, () => this.IsTrained);

                if (this.IsTrained)
                {
                    this.InferenceAlgorithms = new CompoundBinaryFactorizedInferenceAlgorithms(reader);
                }
            }
        }

        /// <summary>
        /// Saves the state of the Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the Bayes point machine classifier to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);

            // Write settings
            this.Settings.SaveForwardCompatible(writer);

            // Write inference algorithms
            if (this.IsTrained)
            {
                this.InferenceAlgorithms.SaveForwardCompatible(writer);
            }
        }

        /// <summary>
        /// Creates the inference algorithms for the specified feature representation.
        /// </summary>
        /// <param name="useSparseFeatures">If true, the inference algorithms expect features in a sparse representation.</param>
        /// <param name="featureCount">The number of features that the inference algorithms use.</param>
        /// <returns>The inference algorithms for the specified feature representation.</returns>
        protected override IInferenceAlgorithms<bool, Bernoulli> CreateInferenceAlgorithms(bool useSparseFeatures, int featureCount)
        {
            return new CompoundBinaryFactorizedInferenceAlgorithms(this.Settings.Training.ComputeModelEvidence, useSparseFeatures, featureCount);
        }
    }
}