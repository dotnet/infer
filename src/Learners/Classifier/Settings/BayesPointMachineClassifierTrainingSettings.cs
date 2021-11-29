// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Settings for the Bayes point machine classifier which affect training.
    /// </summary>
    [Serializable]
    public class BayesPointMachineClassifierTrainingSettings : ICustomSerializable
    {
        /// <summary>
        /// The default value indicating whether model evidence is computed during training.
        /// </summary>
        public const bool ComputeModelEvidenceDefault = false;
        
        /// <summary>
        /// The default number of iterations of the training algorithm.
        /// </summary>
        public const int IterationCountDefault = 30;

        /// <summary>
        /// The default number of batches the training data is split into.
        /// </summary>
        public const int BatchCountDefault = 1;

        /// <summary>
        /// The current custom binary serialization version of the <see cref="BayesPointMachineClassifierTrainingSettings"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the <see cref="BayesPointMachineClassifierTrainingSettings"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("EF0C41F2-5455-49B6-8460-6ECAD9FCC50C");

        /// <summary>
        /// Guards the training settings of the Bayes point machine classifier from being changed after training.
        /// </summary>
        private readonly SettingsGuard trainingSettingsGuard;

        /// <summary>
        /// Indicates whether model evidence is computed during training.
        /// </summary>
        private bool computeModelEvidence;

        /// <summary>
        /// The number of iterations of the training algorithm.
        /// </summary>
        private int iterationCount;

        /// <summary>
        /// The number of batches the training data is split into.
        /// </summary>
        private int batchCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="BayesPointMachineClassifierTrainingSettings"/> class.
        /// </summary>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal BayesPointMachineClassifierTrainingSettings(Func<bool> isTrained)
        {
            this.trainingSettingsGuard = new SettingsGuard(isTrained, "This setting cannot be changed after training.");

            this.computeModelEvidence = ComputeModelEvidenceDefault;
            this.iterationCount = IterationCountDefault;
            this.batchCount = BatchCountDefault;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BayesPointMachineClassifierTrainingSettings"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the training settings from.</param>
        /// <param name="isTrained">Indicates whether the Bayes point machine classifier is trained.</param>
        internal BayesPointMachineClassifierTrainingSettings(IReader reader, Func<bool> isTrained)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The binary stream does not contain the training settings of an Infer.NET Bayes point machine classifier.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.trainingSettingsGuard = new SettingsGuard(reader, isTrained);

                this.computeModelEvidence = reader.ReadBoolean();
                this.iterationCount = reader.ReadInt32();
                this.batchCount = reader.ReadInt32();
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether model evidence is computed during training.
        /// </summary>
        /// <remarks>
        /// This setting cannot be modified after training.
        /// </remarks>
        public bool ComputeModelEvidence
        {
            get
            {
                return this.computeModelEvidence;
            }

            set
            {
                this.trainingSettingsGuard.OnSettingChanging();
                this.computeModelEvidence = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of iterations of the training algorithm.
        /// </summary>
        public int IterationCount
        {
            get
            {
                return this.iterationCount;
            }

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The number of iterations of the training algorithm must be positive.");
                }

                this.iterationCount = value;
            }
        }

        /// <summary>
        /// Gets or sets the number of batches the training data is split into.
        /// </summary>
        public int BatchCount
        {
            get
            {
                return this.batchCount;
            }

            set
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(value), "The number of batches must be positive.");
                }

                this.batchCount = value;
            }
        }

        /// <summary>
        /// Saves the training settings of the Bayes point machine classifier using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the training settings to.</param>
        public virtual void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);

            this.trainingSettingsGuard.SaveForwardCompatible(writer);
            
            writer.Write(this.computeModelEvidence);
            writer.Write(this.iterationCount);
            writer.Write(this.batchCount);
        }
    }
}
