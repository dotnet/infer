// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings of the binary Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    public class BinaryBayesPointMachineClassifierSettings<TLabel> : 
        BayesPointMachineClassifierSettings<TLabel, BayesPointMachineClassifierTrainingSettings, BinaryBayesPointMachineClassifierPredictionSettings<TLabel>>
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="BinaryBayesPointMachineClassifierSettings{TLabel}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryBayesPointMachineClassifierSettings{TLabel}"/> class. 
        /// </summary>
        /// <param name="isTrained">Indicates whether the binary Bayes point machine classifier is trained.</param>
        internal BinaryBayesPointMachineClassifierSettings(Func<bool> isTrained)
        {
            this.Training = new BayesPointMachineClassifierTrainingSettings(isTrained);
            this.Prediction = new BinaryBayesPointMachineClassifierPredictionSettings<TLabel>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryBayesPointMachineClassifierSettings{TLabel}"/> class 
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the settings of the binary Bayes point machine classifier from.</param>
        /// <param name="isTrained">Indicates whether the binary Bayes point machine classifier is trained.</param>
        internal BinaryBayesPointMachineClassifierSettings(IReader reader, Func<bool> isTrained) : base(reader)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.Training = new BayesPointMachineClassifierTrainingSettings(reader, isTrained);
                this.Prediction = new BinaryBayesPointMachineClassifierPredictionSettings<TLabel>(reader);
            }
        }

        /// <summary>
        /// Saves the settings of the binary Bayes point machine classifier using the specified writer to a binary stream.
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