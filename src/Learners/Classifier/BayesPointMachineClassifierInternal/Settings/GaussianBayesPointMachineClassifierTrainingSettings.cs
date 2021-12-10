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
    /// Settings which affect training of the Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
    /// </summary>
    [Serializable]
    public class GaussianBayesPointMachineClassifierTrainingSettings : BayesPointMachineClassifierTrainingSettings
    {
        /// <summary>
        /// The current serialization version of <see cref="GaussianBayesPointMachineClassifierTrainingSettings"/>.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBayesPointMachineClassifierTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal GaussianBayesPointMachineClassifierTrainingSettings(Func<bool> isTrained) 
            : base(isTrained)
        {
            this.Advanced = new GaussianBayesPointMachineClassifierAdvancedTrainingSettings(isTrained);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBayesPointMachineClassifierTrainingSettings"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the training settings from.</param>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal GaussianBayesPointMachineClassifierTrainingSettings(IReader reader, Func<bool> isTrained)
            : base(reader, isTrained)
        {
            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.Advanced = new GaussianBayesPointMachineClassifierAdvancedTrainingSettings(reader, isTrained);
            }
        }

        /// <summary>
        /// Gets the advanced settings of the Bayes point machine classifier.
        /// </summary>
        public GaussianBayesPointMachineClassifierAdvancedTrainingSettings Advanced { get; private set; }

        /// <summary>
        /// Saves the training settings of a Bayes point machine classifier with <see cref="Gaussian"/> 
        /// prior distributions over weights using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the training settings to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
            this.Advanced.SaveForwardCompatible(writer);
        }
    }
}
