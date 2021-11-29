// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings for the binary Bayes point machine classifier which affect prediction.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <remarks>
    /// These settings can be modified after training.
    /// </remarks>
    [Serializable]
    public class BinaryBayesPointMachineClassifierPredictionSettings<TLabel> : BayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// The current serialization version of <see cref="BinaryBayesPointMachineClassifierPredictionSettings{TLabel}"/>.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryBayesPointMachineClassifierPredictionSettings{TLabel}"/> class.
        /// </summary>
        public BinaryBayesPointMachineClassifierPredictionSettings()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryBayesPointMachineClassifierPredictionSettings{TLabel}"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the prediction settings from.</param>
        public BinaryBayesPointMachineClassifierPredictionSettings(IReader reader) : base(reader)
        {
            reader.ReadSerializationVersion(CustomSerializationVersion);
            
            // Nothing to deserialize
        }

        /// <summary>
        /// Saves the prediction settings of the binary Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the prediction settings to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
        }
    }
}