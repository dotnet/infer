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
    /// Advanced training settings for a Bayes point machine classifier with <see cref="Gaussian"/> prior distributions over weights.
    /// </summary>
    /// <remarks>
    /// These settings cannot be modified after training.
    /// </remarks>
    [Serializable]
    public class GaussianBayesPointMachineClassifierAdvancedTrainingSettings : ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="GaussianBayesPointMachineClassifierAdvancedTrainingSettings"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the <see cref="GaussianBayesPointMachineClassifierAdvancedTrainingSettings"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("EB90956E-3FB4-4D1D-82A0-6EEF70F1DD3C");

        /// <summary>
        /// Guards the training settings of the Bayes point machine classifier from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Gets or sets the variance of the prior distributions over weights of the Bayes point machine classifier.
        /// </summary>
        private double weightPriorVariance;

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBayesPointMachineClassifierAdvancedTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal GaussianBayesPointMachineClassifierAdvancedTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.weightPriorVariance = 1.0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="GaussianBayesPointMachineClassifierAdvancedTrainingSettings"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the advanced training settings from.</param>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal GaussianBayesPointMachineClassifierAdvancedTrainingSettings(IReader reader, Func<bool> isTrained)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            reader.VerifySerializationGuid(
                this.customSerializationGuid, 
                "The binary stream does not contain the advanced training settings of an Infer.NET Bayes point machine classifier.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.trainingSettingsGuard = new SettingsGuard(reader, isTrained);

                this.weightPriorVariance = reader.ReadDouble();
            }
        }

        /// <summary>
        /// Gets or sets the variance of the prior distributions over weights of the Bayes point machine classifier.
        /// </summary>
        public double WeightPriorVariance
        {
            get
            {
                return this.weightPriorVariance;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();

                if (value < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The variance of the prior distributions over weights must not be negative.");
                }

                if (double.IsPositiveInfinity(value))
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The variance of the prior distributions over weights must not be infinite.");
                }

                if (double.IsNaN(value))
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The variance of the prior distributions over weights must be a number.");
                }

                this.weightPriorVariance = value;
            }
        }

        /// <summary>
        /// Saves the advanced training settings of a Bayes point machine classifier with <see cref="Gaussian"/> 
        /// prior distributions over weights using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the advanced training settings to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);
            this.trainingSettingsGuard.SaveForwardCompatible(writer);
            writer.Write(this.weightPriorVariance);
        }
    }
}