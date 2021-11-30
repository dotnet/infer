// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings of the multi-class Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    internal class GaussianMulticlassBayesPointMachineClassifierSettings<TLabel> :
        BayesPointMachineClassifierSettings<TLabel, GaussianBayesPointMachineClassifierTrainingSettings, MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>>
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="GaussianMulticlassBayesPointMachineClassifierSettings{TLabel}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianMulticlassBayesPointMachineClassifierSettings{TLabel}"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the multi-class Bayes point machine classifier is trained.</param>
        internal GaussianMulticlassBayesPointMachineClassifierSettings(Func<bool> isTrained)
        {
            this.Training = new GaussianBayesPointMachineClassifierTrainingSettings(isTrained);
            this.Prediction = new MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianMulticlassBayesPointMachineClassifierSettings{TLabel}"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the settings of the multi-class Bayes point machine classifier from.</param>
        /// <param name="isTrained">Indicates whether the multi-class Bayes point machine classifier is trained.</param>
        internal GaussianMulticlassBayesPointMachineClassifierSettings(IReader reader, Func<bool> isTrained) : base(reader)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.Training = new GaussianBayesPointMachineClassifierTrainingSettings(reader, isTrained);
                this.Prediction = new MulticlassBayesPointMachineClassifierPredictionSettings<TLabel>(reader);
            }
        }

        /// <summary>
        /// Saves the settings of the multi-class Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the settings to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
            this.Training.SaveForwardCompatible(writer);
            this.Prediction.SaveForwardCompatible(writer);
        }
    }
}