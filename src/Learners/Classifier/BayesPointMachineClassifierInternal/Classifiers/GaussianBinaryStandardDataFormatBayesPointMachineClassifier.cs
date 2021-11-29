// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A binary Bayes point machine classifier with Gaussian prior distributions over weights 
    /// which operates on data in the standard format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of the instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of the label source.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    [SerializationVersion(6)]
    internal class GaussianBinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel> :
        BinaryStandardDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource, TLabel, GaussianBayesPointMachineClassifierTrainingSettings>
    {
        /// <summary>
        /// The current custom binary serialization version of the
        /// <see cref="GaussianBinaryStandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TLabel}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="GaussianBinaryStandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TLabel}"/> class.
        /// </summary>
        /// <param name="standardMapping">The mapping used for accessing data in the standard format.</param>
        public GaussianBinaryStandardDataFormatBayesPointMachineClassifier(IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> standardMapping)
            : base(standardMapping)
        {
            this.Classifier = new GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(this.NativeMapping);
            this.Settings = new GaussianBinaryBayesPointMachineClassifierSettings<TLabel>(() => this.Classifier.IsTrained);
        }

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="GaussianBinaryStandardDataFormatBayesPointMachineClassifier{TInstanceSource,TInstance,TLabelSource,TLabel}"/>
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the binary Bayes point machine classifier from.</param>
        /// <param name="standardMapping">The mapping used for accessing data in the standard format.</param>
        public GaussianBinaryStandardDataFormatBayesPointMachineClassifier(IReader reader, IClassifierMapping<TInstanceSource, TInstance, TLabelSource, TLabel, Vector> standardMapping)
            : base(reader, standardMapping)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                // Deserialization of classifier and settings
                this.Classifier = new GaussianBinaryNativeDataFormatBayesPointMachineClassifier<TInstanceSource, TInstance, TLabelSource>(reader, this.NativeMapping);
                this.Settings = new GaussianBinaryBayesPointMachineClassifierSettings<TLabel>(reader, () => this.Classifier.IsTrained);
            }
        }

        /// <summary>
        /// Saves the state of the Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the Bayes point machine classifier to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
            this.Classifier.SaveForwardCompatible(writer);
            this.Settings.SaveForwardCompatible(writer);
        }
    }
}